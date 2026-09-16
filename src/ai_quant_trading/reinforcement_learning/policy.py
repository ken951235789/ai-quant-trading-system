"""載入 PPO／SAC 成品並把最新市場資料轉成目標持倉。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from ai_quant_trading.features import build_feature_dataset
from ai_quant_trading.features.builder import infer_annualization_periods
from ai_quant_trading.market_clock import completed_bars_only
from ai_quant_trading.operations.integrity import verify_artifact_manifest
from ai_quant_trading.reinforcement_learning.config import PortfolioEnvConfig
from ai_quant_trading.reinforcement_learning.feature_contract import (
    attach_expected_return,
    resolve_expected_return_contract,
)
from ai_quant_trading.reinforcement_learning.universal import add_universal_rl_features
from ai_quant_trading.sentiment import FinBERTConfig, aggregate_finbert_features
from ai_quant_trading.sentiment.news import load_scored_news
from ai_quant_trading.trading import assess_model_input_health
from ai_quant_trading.transformer import (
    infer_latest_transformer_context,
    infer_transformer_context_frame,
    transformer_checkpoint_sequence_length,
)


TransformerInferenceMode = Literal["latest", "full"]
POLICY_FEATURE_WARMUP_BARS = 250


@dataclass(slots=True)
class LoadedRLPolicy:
    """可重建 observation 的本機可信任 RL 模型成品。"""

    model: Any
    training_dir: Path
    environment_dir: Path
    training_metadata: dict[str, Any]
    environment_metadata: dict[str, Any]
    feature_columns: list[str]
    env_config: PortfolioEnvConfig

    @property
    def universal(self) -> bool:
        return (
            self.environment_metadata.get("environment_kind") == "universal"
            or self.environment_metadata.get("feature_transform") == "universal_ratios"
            or all(column.startswith("u_") for column in self.feature_columns)
        )


@dataclass(frozen=True, slots=True)
class RLTargetSignal:
    """最新已收盤 K 線的 RL 目標持倉。"""

    timestamp: str
    symbol: str
    target_fraction: float
    close: float
    atr: float | None
    model_input_health: dict[str, object] | None = None


def _environment_dir(training_dir: Path, metadata: dict[str, Any]) -> Path:
    local = training_dir.parent.parent
    if (local / "environment.json").exists():
        return local
    configured = Path(str(metadata.get("environment_dir", "")))
    if (configured / "environment.json").exists():
        return configured
    raise FileNotFoundError("找不到 RL 訓練對應的 environment.json")


def _model_path(training_dir: Path, metadata: dict[str, Any]) -> Path:
    candidates = [
        training_dir / "best_model" / "best_model.zip",
        training_dir / "final_model.zip",
        Path(str(metadata.get("selected_model", ""))),
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"RL 訓練資料夾沒有可載入模型：{training_dir}")


def _resolve_ai_context_path(environment_dir: Path, configured: object) -> Path:
    """解析可攜式 AI 成品路徑，專案搬家後仍優先找到環境內檔案。"""
    text = str(configured or "").strip()
    if not text:
        raise FileNotFoundError("RL 環境沒有記錄 AI 成品路徑")
    configured_path = Path(text)
    if configured_path.is_absolute() and configured_path.exists():
        return configured_path
    candidates = [environment_dir / configured_path, Path.cwd() / configured_path]
    candidates.extend(parent / configured_path for parent in environment_dir.parents)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"找不到 RL 環境需要的 AI 成品：{text}")


def _attach_runtime_ai_context(
    frame: pd.DataFrame,
    policy: LoadedRLPolicy,
    *,
    transformer_mode: TransformerInferenceMode = "latest",
) -> pd.DataFrame:
    """用環境保存的同一組 FinBERT 與 Transformer 設定重建最新特徵。"""
    context = dict(policy.environment_metadata.get("ai_context", {}))
    result = frame
    if bool(context.get("finbert_enabled", False)):
        scored_path = _runtime_finbert_news_path(policy, context)
        scored_news = load_scored_news(scored_path)
        config_values = dict(context.get("finbert_config", {}))
        result = aggregate_finbert_features(
            result,
            scored_news,
            FinBERTConfig(**config_values),
        )
    if bool(context.get("transformer_enabled", False)):
        checkpoint = _resolve_ai_context_path(
            policy.environment_dir,
            context.get("transformer_checkpoint"),
        )
        if transformer_mode == "full":
            result = infer_transformer_context_frame(
                checkpoint,
                result,
                device="auto",
                batch_size=512,
            )
        else:
            result = infer_latest_transformer_context(checkpoint, result, device="auto")
        available = pd.to_numeric(result.get("transformer_available", pd.Series(dtype=float)), errors="coerce")
        if available.empty or not np.isfinite(available.iloc[-1]) or float(available.iloc[-1]) < 0.5:
            raise ValueError("Transformer 最新預測不可用，禁止以全零特徵交易")
        result = attach_expected_return(result, resolve_expected_return_contract(context))
        _ensure_latest_expected_return(result)
    return result


def _ensure_latest_expected_return(frame: pd.DataFrame) -> None:
    """不可用缺值或無限大預測繞過成本與風報比閘門。"""
    latest = pd.to_numeric(frame["expected_return"], errors="coerce").iloc[-1]
    if not np.isfinite(latest):
        raise ValueError("模型指定的最新 expected_return 無效，禁止沿用其他預測週期")


def _runtime_finbert_news_path(
    policy: LoadedRLPolicy,
    context: dict[str, Any],
) -> Path:
    """優先讀取持續更新的新聞檔，舊環境則自動尋找專案的 latest 檔。"""
    configured = str(context.get("finbert_runtime_news_path") or "").strip()
    if configured:
        try:
            candidate = _resolve_ai_context_path(policy.environment_dir, configured)
        except FileNotFoundError:
            candidate = None
        if candidate is not None and candidate.is_file():
            return candidate

    for root in (policy.environment_dir, *policy.environment_dir.parents):
        candidate = root / "data" / "processed" / "sentiment" / "finbert_news_latest.csv"
        if candidate.is_file():
            return candidate.resolve()

    # 保留環境建立時的快照作為離線重現與斷網降級來源。
    return _resolve_ai_context_path(
        policy.environment_dir,
        context.get("finbert_scored_news_path"),
    )


def _runtime_inference_window(
    frame: pd.DataFrame,
    since_timestamp: str | pd.Timestamp | None,
    sequence_length: int,
) -> tuple[pd.DataFrame, TransformerInferenceMode]:
    """停機漏掉多根 K 線時，保留指標與 Transformer 所需暖機資料。"""
    if since_timestamp is None:
        return frame, "latest"
    since = pd.to_datetime(since_timestamp, utc=True, errors="coerce")
    if pd.isna(since):
        return frame, "latest"
    newer_positions = np.flatnonzero(frame["timestamp"].to_numpy() > since)
    if len(newer_positions) <= 1:
        return frame, "latest"
    warmup = max(POLICY_FEATURE_WARMUP_BARS, sequence_length - 1)
    start = max(0, int(newer_positions[0]) - warmup)
    return frame.iloc[start:].reset_index(drop=True), "full"


def load_rl_policy(training_dir: str | Path, *, device: str = "cpu") -> LoadedRLPolicy:
    """載入本機 PPO／SAC；移動專案後優先使用訓練資料夾內相對路徑。"""
    root = Path(training_dir).resolve()
    # SB3 zip 含 Python 序列化內容；任何介面都必須先驗證本機訓練成品。
    verify_artifact_manifest(root, required=True)
    training_json = root / "training.json"
    if not training_json.exists():
        raise FileNotFoundError(f"缺少 RL training.json：{training_json}")
    training_metadata = json.loads(training_json.read_text(encoding="utf-8"))
    environment_dir = _environment_dir(root, training_metadata)
    environment_metadata = json.loads(
        (environment_dir / "environment.json").read_text(encoding="utf-8")
    )
    features = [str(column) for column in environment_metadata.get("feature_columns", [])]
    if not features:
        raise ValueError("RL environment.json 缺少 feature_columns")
    raw_config = dict(environment_metadata.get("environment_config", {}))
    if "include_position_context" not in raw_config:
        stored_size = int(environment_metadata.get("observation_size", len(features) + 3))
        raw_config["include_position_context"] = stored_size >= len(features) + 5
    env_config = PortfolioEnvConfig(**raw_config)
    algorithm = str(training_metadata.get("training_config", {}).get("algorithm", "")).lower()
    if algorithm == "ppo":
        from stable_baselines3 import PPO

        model = PPO.load(_model_path(root, training_metadata), device=device)
    elif algorithm == "sac":
        from stable_baselines3 import SAC

        model = SAC.load(_model_path(root, training_metadata), device=device)
    else:
        raise ValueError(f"不支援的 RL 演算法：{algorithm}")
    return LoadedRLPolicy(
        model,
        root,
        environment_dir,
        training_metadata,
        environment_metadata,
        features,
        env_config,
    )


def prepare_rl_policy_market_frame(
    frame: pd.DataFrame,
    policy: LoadedRLPolicy,
    *,
    since_timestamp: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """從 OHLCV 或 Step 3 資料重建 RL 需要的最新完整特徵。"""
    completed = completed_bars_only(frame).reset_index(drop=True)
    if completed.empty:
        raise ValueError("沒有已收盤 K 線可供 RL 判斷")
    already_prepared = set(policy.feature_columns).issubset(completed.columns)
    if already_prepared:
        if "u_transformer_available" in policy.feature_columns:
            latest_available = float(
                pd.to_numeric(completed["u_transformer_available"], errors="coerce").iloc[-1]
            )
            if not np.isfinite(latest_available) or latest_available < 0.5:
                raise ValueError("Transformer 最新預測不可用，禁止以全零特徵交易")
        context = dict(policy.environment_metadata.get("ai_context", {}))
        if bool(context.get("transformer_enabled", False)):
            contract = resolve_expected_return_contract(context)
            if str(contract["source_column"]) in completed:
                completed = attach_expected_return(completed, contract)
            elif "expected_return" not in completed:
                raise ValueError("已準備的 RL 資料缺少模型指定的 expected_return")
            elif context.get("expected_return_contract") is not None and completed.attrs.get(
                "expected_return_contract"
            ) != contract:
                raise ValueError("已準備的 RL 資料無法驗證 expected_return 來源，請重新建立推論資料")
            _ensure_latest_expected_return(completed)
        return completed

    transformer_mode: TransformerInferenceMode = "latest"
    context = dict(policy.environment_metadata.get("ai_context", {}))
    if bool(context.get("transformer_enabled", False)):
        checkpoint = _resolve_ai_context_path(
            policy.environment_dir,
            context.get("transformer_checkpoint"),
        )
        completed, transformer_mode = _runtime_inference_window(
            completed,
            since_timestamp,
            transformer_checkpoint_sequence_length(checkpoint),
        )

    if not set(policy.feature_columns).issubset(completed.columns):
        completed = build_feature_dataset(
            completed,
            target_horizon=1,
            annualization_periods=infer_annualization_periods(completed),
            drop_na=False,
        )
    if bool(context.get("finbert_enabled", False)) or bool(
        context.get("transformer_enabled", False)
    ):
        completed = _attach_runtime_ai_context(
            completed,
            policy,
            transformer_mode=transformer_mode,
        )
    if policy.universal:
        completed = add_universal_rl_features(completed, feature_columns=policy.feature_columns)
    missing = [column for column in policy.feature_columns if column not in completed.columns]
    if missing:
        preview = "、".join(missing[:12])
        suffix = "..." if len(missing) > 12 else ""
        raise ValueError(f"RL 最新資料缺少 {len(missing)} 個特徵：{preview}{suffix}")
    return completed


def latest_rl_target(
    frame: pd.DataFrame,
    policy: LoadedRLPolicy,
    *,
    cash_ratio: float,
    position_ratio: float,
    drawdown: float = 0.0,
    unrealized_return: float = 0.0,
    holding_bars: int = 0,
    equity_ratio: float = 1.0,
    realized_pnl_ratio: float = 0.0,
    daily_return: float = 0.0,
    consecutive_losses: int = 0,
    margin_ratio: float = 0.0,
    stop_distance: float = 0.0,
    liquidation_distance: float = 0.0,
    entry_price_distance: float = 0.0,
    take_profit_distance: float = 0.0,
) -> RLTargetSignal:
    """使用訓練期 mean/std 與目前帳戶狀態建立單一最新 observation。"""
    prepared = prepare_rl_policy_market_frame(frame, policy)
    feature_frame = prepared[policy.feature_columns].apply(pd.to_numeric, errors="coerce")
    valid = feature_frame.notna().all(axis=1)
    if not valid.any() or not bool(valid.iloc[-1]):
        raise ValueError("RL 最新已收盤 K 線沒有完整特徵，禁止沿用舊訊號")
    normalization = dict(policy.environment_metadata.get("normalization", {}))
    means = pd.Series(dict(normalization.get("mean", {})), dtype=float)
    stds = pd.Series(dict(normalization.get("std", {})), dtype=float).replace(0, 1.0)
    if not set(policy.feature_columns).issubset(means.index) or not set(
        policy.feature_columns
    ).issubset(stds.index):
        raise ValueError("RL environment.json 缺少完整標準化參數")
    model_input_health = assess_model_input_health(
        feature_frame,
        policy.feature_columns,
        means,
        stds,
        lookback_rows=20,
    )
    if not model_input_health.ready:
        raise ValueError("RL 模型輸入健康檢查失敗：" + "；".join(model_input_health.reasons))
    market = (
        (feature_frame.iloc[-1] - means[policy.feature_columns]) / stds[policy.feature_columns]
    ).clip(-10.0, 10.0)
    portfolio = [
        float(
            np.clip(
                cash_ratio,
                0.0,
                1.0 + policy.env_config.max_short_fraction,
            )
        ),
        float(
            np.clip(
                position_ratio,
                -policy.env_config.max_short_fraction,
                policy.env_config.max_position_fraction,
            )
        ),
        float(np.clip(drawdown, -1.0, 0.0)),
    ]
    if policy.env_config.include_position_context:
        portfolio.extend(
            [
                float(np.clip(unrealized_return, -1.0, 1.0)),
                float(
                    np.clip(
                        holding_bars / policy.env_config.holding_period_reference,
                        0.0,
                        1.0,
                    )
                ),
            ]
        )
    if policy.env_config.include_risk_context:
        loss_scale = max(policy.env_config.max_consecutive_losses, 1)
        portfolio.extend(
            [
                float(np.clip(equity_ratio, 0.0, 10.0)),
                float(np.clip(realized_pnl_ratio, -1.0, 10.0)),
                float(np.clip(daily_return, -1.0, 1.0)),
                float(np.clip(consecutive_losses / loss_scale, 0.0, 1.0)),
                float(np.clip(margin_ratio, 0.0, 1.0)),
                float(np.clip(stop_distance, 0.0, 1.0)),
                float(np.clip(liquidation_distance, 0.0, 1.0)),
            ]
        )
    if policy.env_config.include_trade_plan_context:
        portfolio.extend(
            [
                float(np.clip(entry_price_distance, -1.0, 1.0)),
                float(np.clip(take_profit_distance, 0.0, 1.0)),
            ]
        )
    observation = np.concatenate(
        [market.to_numpy(dtype=np.float32), np.asarray(portfolio, dtype=np.float32)]
    )
    if tuple(observation.shape) != tuple(policy.model.observation_space.shape):
        raise ValueError(
            f"RL observation 維度 {observation.shape} 與模型 "
            f"{policy.model.observation_space.shape} 不相容"
        )
    action, _ = policy.model.predict(observation, deterministic=True)
    action_value = float(np.asarray(action, dtype=float).reshape(-1)[0])
    model_action_space = getattr(policy.model, "action_space", None)
    if hasattr(model_action_space, "n"):
        discrete = int(action_value)
        target = {
            0: float(position_ratio),
            1: policy.env_config.max_position_fraction,
            2: (-policy.env_config.max_short_fraction if policy.env_config.allow_short else 0.0),
            3: 0.0,
        }.get(discrete, 0.0)
    elif policy.env_config.normalized_action_space:
        if abs(action_value) <= policy.env_config.neutral_action_threshold:
            action_value = 0.0
        target = (
            action_value * policy.env_config.max_position_fraction
            if action_value >= 0
            else action_value * policy.env_config.max_short_fraction
        )
    else:
        target = action_value
    row = prepared.iloc[-1]
    if not np.isfinite(target):
        target = 0.0
    if model_input_health.severe_drift:
        reducing = target * position_ratio >= 0 and abs(target) <= abs(position_ratio)
        if not reducing:
            target = 0.0 if target * position_ratio < 0 else float(position_ratio)
    elif model_input_health.risk_multiplier < 1.0:
        reducing = target * position_ratio >= 0 and abs(target) <= abs(position_ratio)
        if not reducing:
            # 中度漂移不直接停機，但只允許以訓練分布可信度逐步增加曝險。
            if target * position_ratio < 0 and abs(position_ratio) > 1e-12:
                target = 0.0
            else:
                target = position_ratio + (
                    target - position_ratio
                ) * model_input_health.risk_multiplier
    risk_increase = (
        not (abs(target) <= abs(position_ratio) and target * position_ratio >= 0)
        and abs(target) > 1e-12
    )
    if risk_increase:
        event_blackout = pd.to_numeric(row.get("event_blackout"), errors="coerce")
        if pd.notna(event_blackout) and float(event_blackout) >= 0.5:
            target = 0.0
        atr = pd.to_numeric(row.get("atr_14"), errors="coerce")
        close = pd.to_numeric(row.get("close"), errors="coerce")
        if (
            policy.env_config.max_atr_fraction is not None
            and pd.notna(atr)
            and pd.notna(close)
            and float(close) > 0
            and float(atr) / float(close) > policy.env_config.max_atr_fraction
        ):
            target = 0.0
        expected_return = pd.to_numeric(
            row.get("expected_return"),
            errors="coerce",
        )
        spread_bps = pd.to_numeric(row.get("spread_bps"), errors="coerce")
        spread_rate = (
            float(spread_bps) / 10_000
            if pd.notna(spread_bps) and float(spread_bps) >= 0
            else policy.env_config.spread_rate
        )
        if (
            policy.env_config.max_spread_bps is not None
            and spread_rate * 10_000 > policy.env_config.max_spread_bps
        ):
            target = 0.0
        if (
            policy.env_config.max_slippage_bps is not None
            and policy.env_config.slippage_rate * 10_000 > policy.env_config.max_slippage_bps
        ):
            target = 0.0
        if policy.env_config.minimum_gross_target_cost_multiple > 0 and pd.notna(expected_return):
            round_trip_cost = (
                2 * policy.env_config.fee_rate + spread_rate + 2 * policy.env_config.slippage_rate
            )
            required_return = policy.env_config.minimum_gross_target_cost_multiple * round_trip_cost
            direction = 1.0 if target >= 0 else -1.0
            if float(expected_return) * direction < required_return:
                target = 0.0
        if policy.env_config.minimum_net_risk_reward > 0:
            direction = 1.0 if target >= 0 else -1.0
            gross_target = (
                policy.env_config.take_profit_distance
                if policy.env_config.take_profit_distance is not None
                else float(expected_return) * direction
                if pd.notna(expected_return)
                else 0.0
            )
            round_trip_cost = (
                2 * policy.env_config.fee_rate + spread_rate + 2 * policy.env_config.slippage_rate
            )
            planned_stop = max(
                stop_distance,
                policy.env_config.minimum_stop_distance,
            )
            net_risk_reward = (gross_target - round_trip_cost) / (planned_stop + round_trip_cost)
            if net_risk_reward < policy.env_config.minimum_net_risk_reward:
                target = 0.0
    atr_value = row.get("atr_14")
    return RLTargetSignal(
        pd.Timestamp(row["timestamp"]).isoformat(),
        str(row.get("symbol", "")),
        float(
            np.clip(
                target,
                -policy.env_config.max_short_fraction,
                policy.env_config.max_position_fraction,
            )
        ),
        float(row["close"]),
        None if pd.isna(atr_value) else float(atr_value),
        model_input_health.to_dict(),
    )
