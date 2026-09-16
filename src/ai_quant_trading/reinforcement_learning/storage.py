"""保存 Step 9 強化學習環境設定與時序資料。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import pandas as pd

from ai_quant_trading.data_collection.assets import infer_asset_class, market_file_symbol_slug
from ai_quant_trading.reinforcement_learning.config import PortfolioEnvConfig
from ai_quant_trading.reinforcement_learning.dataset import PreparedRLDataset


@dataclass(frozen=True, slots=True)
class RLArtifactPaths:
    """一個可供未來 PPO／SAC 使用的完整環境成品。"""

    run_dir: Path
    metadata_json: Path
    train_csv: Path
    validation_csv: Path
    test_csv: Path
    diagnostic_csv: Path


def save_rl_environment(
    dataset: PreparedRLDataset,
    env_config: PortfolioEnvConfig,
    output_dir: str | Path,
    *,
    source_path: str | Path | None = None,
    diagnostic: pd.DataFrame | None = None,
    transformer_checkpoint: str | Path | None = None,
    finbert_scored_news_path: str | Path | None = None,
    finbert_config: dict[str, object] | None = None,
    transformer_provenance: dict[str, object] | None = None,
    transformer_crossfit_summary: str | Path | None = None,
) -> RLArtifactPaths:
    """保存標準化資料、切分、Reward 設定與診斷結果，不執行模型訓練。"""
    transformer_coverage = float(
        dataset.availability_coverage.get("transformer_available", 0.0)
    )
    uses_transformer = bool(dataset.expected_return_contract) or transformer_coverage > 0
    if transformer_checkpoint is not None and not transformer_provenance:
        raise ValueError("帶 Transformer 的 RL 環境缺少時間隔離來源證明")
    if uses_transformer and not dataset.expected_return_contract:
        raise ValueError("新 Transformer RL 環境必須明確指定 expected_return_contract，不可猜測預測週期")
    if uses_transformer and transformer_checkpoint is None and transformer_crossfit_summary is None:
        raise ValueError("Transformer RL 訓練資料必須提供 checkpoint provenance 或 cross-fit 證明")
    expected_return_available = (
        "expected_return" in dataset.frame
        and pd.to_numeric(dataset.frame["expected_return"], errors="coerce").notna().any()
    )
    if env_config.minimum_gross_target_cost_multiple > 0 and not expected_return_available:
        raise ValueError("啟用毛利成本倍數閘門時，RL 資料必須提供 expected_return")
    if (
        env_config.minimum_net_risk_reward > 0
        and env_config.take_profit_distance is None
        and not expected_return_available
    ):
        raise ValueError(
            "啟用淨風報比閘門時，必須提供 expected_return 或固定 take_profit_distance"
        )
    first = dataset.frame.iloc[0]
    symbol = str(first.get("symbol", "unknown"))
    exchange = str(first.get("exchange", "unknown")).lower()
    interval = str(first.get("interval", "unknown"))
    asset_class = infer_asset_class(exchange, symbol)
    slug = market_file_symbol_slug(symbol, asset_class)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = Path(output_dir) / f"{timestamp}_{asset_class}_{slug}_{interval}"
    run_dir.mkdir(parents=True, exist_ok=False)

    ai_dir = run_dir / "ai"
    transformer_relative_path: str | None = None
    crossfit_relative_path: str | None = None
    finbert_relative_path: str | None = None
    finbert_runtime_path: str | None = None
    if transformer_checkpoint is not None:
        source_checkpoint = Path(transformer_checkpoint).resolve()
        if not source_checkpoint.is_file():
            raise FileNotFoundError(f"找不到 Transformer 模型：{source_checkpoint}")
        ai_dir.mkdir(parents=True, exist_ok=True)
        target_checkpoint = ai_dir / "transformer_model.pt"
        shutil.copy2(source_checkpoint, target_checkpoint)
        transformer_relative_path = str(target_checkpoint.relative_to(run_dir))
    if transformer_crossfit_summary is not None:
        source_crossfit = Path(transformer_crossfit_summary).resolve()
        if not source_crossfit.is_file():
            raise FileNotFoundError(f"找不到 Transformer cross-fit 摘要：{source_crossfit}")
        crossfit_payload = json.loads(source_crossfit.read_text(encoding="utf-8"))
        plan = dict(crossfit_payload.get("plan", {}))
        if (
            crossfit_payload.get("status") != "complete"
            or not bool(plan.get("final_holdout_sealed", False))
        ):
            raise ValueError("Transformer cross-fit 尚未完成或 final holdout 未封存")
        ai_dir.mkdir(parents=True, exist_ok=True)
        target_crossfit = ai_dir / "transformer_crossfit.json"
        shutil.copy2(source_crossfit, target_crossfit)
        crossfit_relative_path = str(target_crossfit.relative_to(run_dir))
    if finbert_scored_news_path is not None:
        source_news = Path(finbert_scored_news_path).resolve()
        if not source_news.is_file():
            raise FileNotFoundError(f"找不到 FinBERT 新聞分數：{source_news}")
        # 環境內副本用於重現訓練；原始 latest 路徑供執行期讀取新資料。
        finbert_runtime_path = str(source_news)
        ai_dir.mkdir(parents=True, exist_ok=True)
        target_news = ai_dir / "finbert_news.csv"
        shutil.copy2(source_news, target_news)
        finbert_relative_path = str(target_news.relative_to(run_dir))

    metadata_json = run_dir / "environment.json"
    train_csv = run_dir / "train.csv"
    validation_csv = run_dir / "validation.csv"
    test_csv = run_dir / "test.csv"
    diagnostic_csv = run_dir / "diagnostic.csv"
    dataset.train.to_csv(train_csv, index=False, encoding="utf-8")
    dataset.validation.to_csv(validation_csv, index=False, encoding="utf-8")
    dataset.test.to_csv(test_csv, index=False, encoding="utf-8")
    (diagnostic if diagnostic is not None else pd.DataFrame()).to_csv(
        diagnostic_csv, index=False, encoding="utf-8"
    )

    def ai_coverage(column: str) -> float:
        return float(dataset.availability_coverage.get(column, 0.0))

    payload = {
        "status": "environment_ready",
        "training_started": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_path": str(source_path) if source_path is not None else None,
        "source": {
            "exchange": exchange,
            "symbol": symbol,
            "interval": interval,
            "start": str(dataset.frame.iloc[0]["timestamp"]),
            "end": str(dataset.frame.iloc[-1]["timestamp"]),
        },
        "feature_columns": dataset.feature_columns,
        "feature_contract": dataset.feature_contract,
        "feature_transform": (
            "universal_ratios" if all(column.startswith("u_") for column in dataset.feature_columns)
            else "technical_indicators"
        ),
        "observation_size": (
            len(dataset.feature_columns)
            + (5 if env_config.include_position_context else 3)
            + (7 if env_config.include_risk_context else 0)
            + (2 if env_config.include_trade_plan_context else 0)
        ),
        "action": {
            "type": (
                "normalized_continuous_target_exposure"
                if env_config.normalized_action_space
                else "continuous_target_position_fraction"
            ),
            "minimum": -1.0 if env_config.normalized_action_space and env_config.allow_short else (
                -env_config.max_short_fraction
            ),
            "maximum": 1.0 if env_config.normalized_action_space else (
                env_config.max_position_fraction
            ),
            "ppo_discrete_mapping": {
                "0": "維持",
                "1": "最大允許多單",
                "2": "最大允許空單",
                "3": "平倉",
            },
        },
        "reward": {
            "formula": (
                "log_return - drawdown_penalty - turnover_penalty - "
                "downside_penalty - concentration_penalty - risk_termination_penalty"
            ),
            "next_open_execution": True,
        },
        "execution": {
            "mode": env_config.execution_mode,
            "isolated_margin": env_config.execution_mode == "perpetual",
            "leverage": env_config.leverage,
            "max_leverage": env_config.max_leverage,
            "max_margin_fraction": env_config.max_margin_fraction,
            "risk_per_trade": env_config.risk_per_trade,
            "daily_loss_limit": env_config.daily_loss_limit,
            "max_consecutive_losses": env_config.max_consecutive_losses,
        },
        "expert": {
            "kind": env_config.expert_kind,
            "rebalance_deadband": env_config.rebalance_deadband,
            "minimum_holding_bars": env_config.minimum_holding_bars,
            "soft_drawdown_limit": env_config.soft_drawdown_limit,
            "allow_short": env_config.allow_short,
            "max_short_fraction": env_config.max_short_fraction,
        },
        "ai_context": {
            "finbert_enabled": finbert_relative_path is not None,
            "transformer_enabled": uses_transformer,
            "runtime_ready": (
                (not uses_transformer or transformer_relative_path is not None)
                and (finbert_scored_news_path is None or finbert_relative_path is not None)
            ),
            "finbert_scored_news_path": finbert_relative_path,
            "finbert_runtime_news_path": finbert_runtime_path,
            "finbert_config": finbert_config or {},
            "transformer_checkpoint": transformer_relative_path,
            "transformer_crossfit_summary": crossfit_relative_path,
            "transformer_provenance": transformer_provenance or {},
            "expected_return_contract": (
                dataset.expected_return_contract
                if uses_transformer else None
            ),
            "coverage": {
                "aggregate": {
                    "finbert": ai_coverage("finbert_available"),
                    "transformer": ai_coverage("transformer_available"),
                }
            },
        },
        "environment_config": env_config.to_dict(),
        "split_config": dataset.split_config.to_dict(),
        "split_rows": {
            "train": len(dataset.train),
            "validation": len(dataset.validation),
            "test": len(dataset.test),
        },
        "normalization": {
            "fit_on": "train_only",
            "clip": [-10.0, 10.0],
            "mean": {key: float(value) for key, value in dataset.feature_mean.items()},
            "std": {key: float(value) for key, value in dataset.feature_std.items()},
        },
    }
    metadata_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return RLArtifactPaths(
        run_dir,
        metadata_json,
        train_csv,
        validation_csv,
        test_csv,
        diagnostic_csv,
    )


def list_rl_environments(output_dir: str | Path) -> list[Path]:
    """列出已完成且尚未訓練的 RL 環境成品。"""
    root = Path(output_dir)
    if not root.exists():
        return []
    paths = [path for path in root.iterdir() if (path / "environment.json").exists()]
    return sorted(paths, key=lambda path: path.stat().st_mtime, reverse=True)
