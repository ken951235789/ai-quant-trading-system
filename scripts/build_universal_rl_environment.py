"""從多個 Step 3 特徵 CSV 建立通用強化學習環境。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_quant_trading.reinforcement_learning import (  # noqa: E402
    RLSplitConfig,
    UniversalPortfolioTradingEnv,
    build_expert_profile,
    prepare_universal_rl_dataset,
    run_environment_diagnostic,
    save_universal_rl_environment,
    validate_expert_interval,
)
from ai_quant_trading.features import build_feature_dataset  # noqa: E402
from ai_quant_trading.features.builder import infer_annualization_periods  # noqa: E402
from ai_quant_trading.ai_pipeline import load_ai_pipeline_config  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="建立長期／短期跨市場 PPO／SAC 環境")
    parser.add_argument("--inputs", nargs="+", required=True, help="至少兩個 OHLCV 或特徵 CSV")
    parser.add_argument(
        "--expert-kind",
        choices=["general", "long_term", "short_term"],
        default="general",
        help="一般、長期或短期交易專家",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/rl/environments",
        help="RL 環境輸出資料夾",
    )
    parser.add_argument("--history-years", type=int)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--initial-capital", type=float, default=50_000)
    parser.add_argument("--max-position-fraction", type=float)
    parser.add_argument(
        "--allow-short",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否允許負目標倉位；短期專家預設啟用",
    )
    parser.add_argument("--max-short-fraction", type=float)
    parser.add_argument("--short-borrow-rate-annual", type=float)
    parser.add_argument("--max-drawdown", type=float)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--slippage-rate", type=float, default=0.0005)
    parser.add_argument("--episode-length", type=int)
    parser.add_argument(
        "--transformer-checkpoint",
        help="要隨 RL 環境封裝的 Transformer best_model.pt",
    )
    parser.add_argument(
        "--finbert-scored-news",
        default="data/processed/sentiment/finbert_news_latest.csv",
        help="模擬／實盤時使用的 FinBERT 新聞分數檔",
    )
    return parser


def main(arguments: list[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    if len(args.inputs) < 2:
        raise ValueError("通用環境至少需要兩個特徵 CSV")
    frames: dict[str, pd.DataFrame] = {}
    sources: dict[str, str] = {}
    intervals: set[str] = set()
    profile = build_expert_profile(
        args.expert_kind,
        initial_capital=args.initial_capital,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
    )
    history_years = args.history_years or (10 if args.expert_kind == "long_term" else 5)
    for input_text in args.inputs:
        path = Path(input_text).resolve()
        frame = pd.read_csv(path)
        if "sma_200" not in frame.columns or "short_rsi_2" not in frame.columns:
            frame = build_feature_dataset(
                frame,
                target_horizon=1,
                annualization_periods=infer_annualization_periods(frame),
                drop_na=True,
            )
        timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        latest = timestamps.max()
        if pd.isna(latest):
            raise ValueError(f"{path.name} 沒有有效 timestamp")
        frame = frame.loc[
            timestamps >= latest - pd.DateOffset(years=history_years)
        ].reset_index(drop=True)
        first = frame.iloc[0]
        interval = str(first.get("interval", "unknown"))
        market = f"{first.get('exchange', 'unknown')}:{first.get('symbol', path.stem)}:{interval}"
        if market in frames:
            raise ValueError(f"重複市場：{market}")
        frames[market] = frame
        sources[market] = str(path)
        intervals.add(interval)
    if len(intervals) != 1:
        raise ValueError("所有市場必須使用相同 K 線週期")
    validate_expert_interval(args.expert_kind, next(iter(intervals)))

    split = RLSplitConfig(args.train_fraction, args.validation_fraction, 20)
    allow_short = (
        profile.environment.allow_short
        if args.allow_short is None
        else bool(args.allow_short)
    )
    max_short_fraction = (
        args.max_short_fraction
        if args.max_short_fraction is not None
        else profile.environment.max_short_fraction
    )
    if allow_short and max_short_fraction <= 0:
        max_short_fraction = 0.05
    env_config = replace(
        profile.environment,
        max_position_fraction=(
            args.max_position_fraction
            if args.max_position_fraction is not None
            else profile.environment.max_position_fraction
        ),
        max_drawdown_limit=(
            args.max_drawdown
            if args.max_drawdown is not None
            else profile.environment.max_drawdown_limit
        ),
        allow_short=allow_short,
        max_short_fraction=max_short_fraction if allow_short else 0.0,
        short_borrow_rate_annual=(
            args.short_borrow_rate_annual
            if args.short_borrow_rate_annual is not None
            else profile.environment.short_borrow_rate_annual
        ),
        episode_length=(
            args.episode_length
            if args.episode_length is not None
            else profile.environment.episode_length
        ),
    )
    ai_config = load_ai_pipeline_config(PROJECT_ROOT / "data/config/ai_pipeline.json")
    if ai_config.ppo_use_transformer and not args.transformer_checkpoint:
        raise ValueError("AI 管線已啟用 Transformer，請提供 --transformer-checkpoint")
    finbert_path = Path(args.finbert_scored_news)
    if ai_config.ppo_use_finbert and not finbert_path.exists():
        raise FileNotFoundError(f"找不到 FinBERT 新聞分數：{finbert_path}")
    dataset = prepare_universal_rl_dataset(
        frames,
        split,
        expert_kind=args.expert_kind,
        use_finbert=ai_config.ppo_use_finbert,
        use_transformer=ai_config.ppo_use_transformer,
    )
    environment = UniversalPortfolioTradingEnv(
        {name: market.train for name, market in dataset.markets.items()},
        dataset.feature_columns,
        env_config,
        selection_mode="cycle",
    )
    diagnostic = run_environment_diagnostic(environment)
    paths = save_universal_rl_environment(
        dataset,
        env_config,
        args.output_dir,
        source_paths=sources,
        diagnostic=diagnostic,
        transformer_checkpoint=args.transformer_checkpoint,
        finbert_scored_news_path=args.finbert_scored_news,
        finbert_config=asdict(ai_config.finbert),
    )
    print(f"通用環境：{paths.run_dir}")
    print(f"市場數：{len(dataset.markets)}")
    print(f"共同特徵數：{len(dataset.feature_columns)}")
    print(f"交易專家：{profile.label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
