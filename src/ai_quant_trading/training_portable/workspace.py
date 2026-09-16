"""在獨立訓練電腦準備 BTC 資料與強化學習環境。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import pandas as pd

from ai_quant_trading.data_collection.collectors import (
    collect_binance_futures_timeframe_bundle,
)
from ai_quant_trading.features.builder import build_features_from_csv_batch
from ai_quant_trading.features.multitimeframe import (
    BTC_MULTITIMEFRAME_INTERVALS,
    build_multitimeframe_feature_datasets,
)
from ai_quant_trading.reinforcement_learning import (
    PortfolioTradingEnv,
    RLSplitConfig,
    build_expert_profile,
    prepare_rl_dataset,
    run_environment_diagnostic,
    save_rl_environment,
)
from ai_quant_trading.reinforcement_learning.feature_contract import (
    attach_expected_return, build_expected_return_contract,
)
from ai_quant_trading.reinforcement_learning.universal import build_compact_short_term_frame
from ai_quant_trading.transformer import (
    infer_transformer_context_frame,
    transformer_oos_provenance,
)


DataProgress = Callable[[str, int, int], None]


@dataclass(frozen=True, slots=True)
class BTCTrainingDataResult:
    """一次 BTC 多週期資料準備工作的輸出。"""

    raw_files: tuple[Path, ...]
    feature_files: tuple[Path, ...]
    multitimeframe_file: Path
    rows: int


@dataclass(frozen=True, slots=True)
class BTCEnvironmentSettings:
    """BTC 15m SAC/PPO 訓練環境的風控與資料切分設定。"""

    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    initial_capital: float = 1_000.0
    max_position_fraction: float = 0.50
    max_short_fraction: float = 0.50
    leverage: float = 2.0
    max_drawdown_limit: float = 0.10
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0005
    risk_per_trade: float = 0.0025
    max_margin_fraction: float = 0.20
    daily_loss_limit: float = 0.01
    max_consecutive_losses: int = 3
    take_profit_distance: float = 0.015
    episode_length: int = 2_880
    expected_return_horizon: int = 5
    feature_budget: int = 112

    def __post_init__(self) -> None:
        if not 0 < self.train_fraction < 1:
            raise ValueError("訓練集比例必須介於 0 與 1 之間")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("驗證集比例必須介於 0 與 1 之間")
        if self.train_fraction + self.validation_fraction >= 0.9:
            raise ValueError("測試集至少必須保留 10%")
        if self.initial_capital <= 0:
            raise ValueError("初始資金必須大於 0")
        if not 0 < self.max_position_fraction <= 1:
            raise ValueError("最大多頭部位必須介於 0 與 1 之間")
        if not 0 < self.max_short_fraction <= 1:
            raise ValueError("最大空頭部位必須介於 0 與 1 之間")
        if not 1 <= self.leverage <= 3:
            raise ValueError("研究環境槓桿限制為 1 至 3 倍")
        if self.episode_length < 32:
            raise ValueError("Episode 至少需要 32 根 K 線")
        build_expected_return_contract(self.expected_return_horizon)
        if not 80 <= self.feature_budget <= 120:
            raise ValueError("短線特徵預算必須介於 80 與 120")


def prepare_btc_training_data(
    project_root: str | Path,
    *,
    start: str,
    end: str | None = None,
    include_derivatives_context: bool = True,
    progress_callback: DataProgress | None = None,
) -> BTCTrainingDataResult:
    """下載 BTC 永續多週期 K 線，建立技術特徵及因果對齊資料。"""
    root = Path(project_root).resolve()
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    collection = collect_binance_futures_timeframe_bundle(
        ["BTC/USDT"],
        raw_dir,
        intervals=BTC_MULTITIMEFRAME_INTERVALS,
        start=start,
        end=end,
        limit=1_500,
        include_derivatives_context=include_derivatives_context,
        progress_callback=progress_callback,
    )
    feature_artifacts = build_features_from_csv_batch(
        collection.ohlcv_files,
        output_dir=processed_dir,
        target_horizon=1,
        drop_na=False,
    )
    mtf_artifacts = build_multitimeframe_feature_datasets(
        [artifact.output_path for artifact in feature_artifacts],
        "15m",
        processed_dir,
    )
    artifact = mtf_artifacts[0]
    return BTCTrainingDataResult(
        raw_files=tuple(collection.ohlcv_files),
        feature_files=tuple(item.output_path for item in feature_artifacts),
        multitimeframe_file=artifact.output_path,
        rows=artifact.rows,
    )


def build_btc_rl_environment(
    project_root: str | Path,
    feature_path: str | Path,
    transformer_checkpoint: str | Path,
    settings: BTCEnvironmentSettings | None = None,
) -> Path:
    """以 Transformer 歷史輸出建立可直接訓練 SAC/PPO 的 BTC 環境。"""
    root = Path(project_root).resolve()
    source = Path(feature_path).resolve()
    checkpoint = Path(transformer_checkpoint).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"找不到 BTC 多週期特徵：{source}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"找不到 Transformer checkpoint：{checkpoint}")

    config = settings or BTCEnvironmentSettings()
    frame = pd.read_csv(source)
    if frame.empty:
        raise ValueError("BTC 多週期特徵檔是空的")
    first = frame.iloc[0]
    identity = (
        str(first.get("exchange", "")).lower(),
        str(first.get("symbol", "")).upper(),
        str(first.get("interval", "")).lower(),
    )
    if identity != ("binance_futures", "BTC/USDT", "15m"):
        raise ValueError("可攜訓練中心只接受 Binance Futures BTC/USDT 15m 多週期資料")

    provenance = transformer_oos_provenance(checkpoint, frame)
    inferred = infer_transformer_context_frame(checkpoint, frame, device="auto")
    inferred = inferred.copy()
    inferred["timestamp"] = pd.to_datetime(inferred["timestamp"], utc=True, errors="coerce")
    safe_start = pd.Timestamp(str(provenance["safe_rl_start_exclusive"]))
    inferred = inferred.loc[
        (inferred["timestamp"] > safe_start)
        & (pd.to_numeric(inferred["transformer_available"], errors="coerce") >= 0.5)
    ].reset_index(drop=True)
    if inferred.empty:
        raise ValueError("Transformer 校準截止時間之後沒有可供 RL 使用的樣本外預測")
    inferred = attach_expected_return(inferred, build_expected_return_contract(config.expected_return_horizon))
    inferred, available_features = build_compact_short_term_frame(inferred, maximum_features=config.feature_budget)
    if not available_features:
        raise ValueError("沒有可供 SAC/PPO 使用的模型特徵")

    split = RLSplitConfig(
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
        min_rows_per_split=20,
    )
    profile = build_expert_profile("short_term", initial_capital=config.initial_capital)
    env_config = replace(
        profile.environment,
        initial_capital=config.initial_capital,
        fee_rate=config.fee_rate,
        slippage_rate=config.slippage_rate,
        max_position_fraction=config.max_position_fraction,
        allow_short=True,
        max_short_fraction=config.max_short_fraction,
        leverage=config.leverage,
        max_drawdown_limit=config.max_drawdown_limit,
        risk_per_trade=config.risk_per_trade,
        max_margin_fraction=config.max_margin_fraction,
        daily_loss_limit=config.daily_loss_limit,
        max_consecutive_losses=config.max_consecutive_losses,
        take_profit_distance=config.take_profit_distance,
        episode_length=config.episode_length,
        random_start=True,
    )
    dataset = prepare_rl_dataset(inferred, available_features, split)
    environment = PortfolioTradingEnv(dataset.train, dataset.feature_columns, env_config)
    diagnostic = run_environment_diagnostic(environment)
    paths = save_rl_environment(
        dataset,
        env_config,
        root / "data" / "processed" / "rl" / "environments",
        source_path=source,
        diagnostic=diagnostic,
        transformer_checkpoint=checkpoint,
        transformer_provenance=provenance,
    )
    return paths.run_dir
