"""PPO 與 Transformer 的歷史回測工具。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

from ai_quant_trading.reinforcement_learning import (
    PortfolioEnvConfig,
    evaluate_rl_model,
    load_rl_policy,
)
from ai_quant_trading.transformer import infer_transformer_context_frame


DatasetSplit = Literal["train", "validation", "test", "all"]


@dataclass(frozen=True, slots=True)
class TransformerSignalConfig:
    """把 Transformer 預測轉成透明、可重現的多空目標部位。"""

    horizon: int = 5
    minimum_return: float = 0.0003
    minimum_regime_probability: float = 0.35
    maximum_uncertainty: float = 0.80
    volatility_multiple: float = 1.0
    max_long_fraction: float = 1.0
    allow_short: bool = True
    max_short_fraction: float = 0.75

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("預測週期必須大於 0")
        if self.minimum_return < 0:
            raise ValueError("最小預測報酬不可小於 0")
        if not 0 <= self.minimum_regime_probability <= 1:
            raise ValueError("行情機率門檻必須介於 0 與 1")
        if not 0 <= self.maximum_uncertainty <= 1:
            raise ValueError("不確定性上限必須介於 0 與 1")
        if self.volatility_multiple <= 0:
            raise ValueError("波動調整倍數必須大於 0")
        if not 0 < self.max_long_fraction <= 1:
            raise ValueError("最大多頭部位必須大於 0 且不超過 1")
        if not 0 <= self.max_short_fraction <= 1:
            raise ValueError("最大空頭部位必須介於 0 與 1")
        if self.allow_short and self.max_short_fraction <= 0:
            raise ValueError("啟用作空時最大空頭部位必須大於 0")


@dataclass(slots=True)
class ModelBacktestResult:
    """模型回測的逐根結果、指標與資料契約。"""

    model_kind: str
    evaluation: pd.DataFrame
    metrics: dict[str, float | int]
    source_frame: pd.DataFrame
    metadata: dict[str, Any]


class _ActionSequencePolicy:
    """依序提供已由 Transformer 產生的目標部位。"""

    def __init__(self, actions: np.ndarray) -> None:
        self.actions = np.asarray(actions, dtype=np.float32)
        self.index = 0

    def predict(self, _observation: np.ndarray, *, deterministic: bool) -> tuple[np.ndarray, None]:
        del deterministic
        value = float(self.actions[self.index]) if self.index < len(self.actions) else 0.0
        self.index += 1
        return np.array([value], dtype=np.float32), None


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 格式不正確：{path}")
    return payload


def _is_btc_metadata(payload: dict[str, Any]) -> bool:
    source = dict(payload.get("environment_source") or payload.get("source") or {})
    symbol = str(source.get("symbol", "")).upper()
    if symbol:
        return "BTC" in symbol
    return any("BTC" in str(item.get("symbol", "")).upper() for item in payload.get("sources", []))


def list_ppo_backtest_runs(rl_dir: str | Path) -> list[Path]:
    """列出有模型成品、已完成且屬於 BTC 的 PPO／SAC 訓練。"""
    root = Path(rl_dir)
    if not root.exists():
        return []
    runs: list[Path] = []
    for training_json in root.rglob("training.json"):
        try:
            payload = _read_json(training_json)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        algorithm = str(dict(payload.get("training_config", {})).get("algorithm", ""))
        has_model = (training_json.parent / "final_model.zip").is_file() or (
            training_json.parent / "best_model" / "best_model.zip"
        ).is_file()
        if (
            payload.get("status") == "complete"
            and algorithm.lower() in {"ppo", "sac"}
            and has_model
            and _is_btc_metadata(payload)
        ):
            runs.append(training_json.parent)
    return sorted(runs, key=lambda path: path.stat().st_mtime, reverse=True)


def list_transformer_backtest_runs(transformer_dir: str | Path) -> list[Path]:
    """只列出有 checkpoint、已完成且屬於 BTC 的 Transformer 訓練。"""
    root = Path(transformer_dir)
    if not root.exists():
        return []
    runs: list[Path] = []
    for training_json in root.rglob("training.json"):
        try:
            payload = _read_json(training_json)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        model_name = str(dict(payload.get("artifacts", {})).get("model", "best_model.pt"))
        if (
            payload.get("status") == "complete"
            and (training_json.parent / model_name).is_file()
            and _is_btc_metadata(payload)
        ):
            runs.append(training_json.parent)
    return sorted(runs, key=lambda path: path.stat().st_mtime, reverse=True)


def _canonical_history(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"歷史資料缺少欄位：{missing}")
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(
        result["timestamp"], utc=True, errors="coerce", format="mixed"
    )
    return (
        result.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )


def load_ppo_history(
    training_dir: str | Path,
    split: Literal["train", "validation", "test"],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """讀取 RL 訓練時保存且已使用 train-only 統計量標準化的資料。"""
    run_dir = Path(training_dir).resolve()
    training = _read_json(run_dir / "training.json")
    configured = Path(str(training.get("environment_dir", "")))
    local = run_dir.parent.parent
    environment_dir = local if (local / "environment.json").exists() else configured
    metadata_path = environment_dir / "environment.json"
    if not metadata_path.exists():
        raise FileNotFoundError("找不到 RL 對應的 environment.json")
    environment = _read_json(metadata_path)
    source_symbol = str(dict(environment.get("source", {})).get("symbol", "")).upper()
    if not _is_btc_metadata(training) and "BTC" not in source_symbol:
        raise ValueError("目前 AI 回測只接受 BTC 模型")
    csv_path = environment_dir / f"{split}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"找不到 RL {split} 歷史資料：{csv_path}")
    return _canonical_history(pd.read_csv(csv_path)), training, environment


def filter_history(
    frame: pd.DataFrame,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """依 UTC 時間裁切歷史區間，至少保留兩根 K 線。"""
    result = _canonical_history(frame)
    if start is not None:
        result = result.loc[result["timestamp"] >= pd.to_datetime(start, utc=True, errors="raise")]
    if end is not None:
        result = result.loc[result["timestamp"] <= pd.to_datetime(end, utc=True, errors="raise")]
    result = result.reset_index(drop=True)
    if len(result) < 2:
        raise ValueError("回測區間至少需要兩根 K 線")
    return result


def _attach_market_context(
    evaluation: pd.DataFrame,
    source: pd.DataFrame,
    initial_capital: float,
) -> pd.DataFrame:
    result = evaluation.copy()
    result["timestamp"] = pd.to_datetime(
        result["timestamp"], utc=True, errors="coerce", format="mixed"
    )
    prices = source[["timestamp", "close"]].copy()
    prices["timestamp"] = pd.to_datetime(
        prices["timestamp"], utc=True, errors="coerce", format="mixed"
    )
    result = result.merge(prices, on="timestamp", how="left")
    first_close = float(pd.to_numeric(source["close"], errors="coerce").iloc[0])
    result["benchmark_equity"] = (
        initial_capital * pd.to_numeric(result["close"], errors="coerce") / first_close
    )
    return result


def _benchmark_metrics(metrics: dict[str, float | int]) -> dict[str, float | int]:
    """只保留跨策略比較時真正有解釋力的樣本外指標。"""
    keys = (
        "final_equity",
        "total_return",
        "max_drawdown",
        "profit_factor",
        "expectancy",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "win_rate",
        "trades",
        "fee_to_initial_capital",
        "liquidation_count",
    )
    return {key: metrics[key] for key in keys if key in metrics}


def _reference_actions(
    frame: pd.DataFrame,
    env_config: PortfolioEnvConfig,
    *,
    kind: Literal["fixed_ema", "random"],
    seed: int,
) -> np.ndarray:
    """建立只依當下與過去價格計算的固定或隨機基準動作。"""
    long_action = 1.0 if env_config.normalized_action_space else env_config.max_position_fraction
    short_action = (
        -1.0
        if env_config.normalized_action_space and env_config.allow_short
        else -env_config.max_short_fraction
        if env_config.allow_short
        else 0.0
    )
    if kind == "random":
        rng = np.random.default_rng(seed)
        choices = np.array([short_action, 0.0, long_action], dtype=np.float32)
        return rng.choice(choices, size=len(frame), replace=True).astype(np.float32)

    close = pd.to_numeric(frame["close"], errors="coerce")
    ema_20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
    ema_50 = close.ewm(span=50, adjust=False, min_periods=20).mean()
    ema_200 = close.ewm(span=200, adjust=False, min_periods=20).mean()
    long_condition = (ema_20 > ema_50) & (close > ema_200)
    short_condition = (ema_20 < ema_50) & (close < ema_200)
    return np.select(
        [long_condition, short_condition],
        [long_action, short_action],
        default=0.0,
    ).astype(np.float32)


def _rl_backtest_benchmarks(
    model_evaluation: pd.DataFrame,
    frame: pd.DataFrame,
    feature_columns: list[str],
    env_config: PortfolioEnvConfig,
    model_metrics: dict[str, float | int],
    *,
    seed: int,
) -> dict[str, dict[str, float | int]]:
    """以完全相同的歷史區間比較模型、固定規則、隨機策略與成本影響。"""
    comparison_config = replace(
        env_config,
        minimum_gross_target_cost_multiple=0.0,
        minimum_net_risk_reward=0.0,
        initial_capital_randomization=0.0,
        slippage_randomization=0.0,
    )
    benchmarks: dict[str, dict[str, float | int]] = {
        "model_with_costs": _benchmark_metrics(model_metrics)
    }
    for name, kind in (("fixed_ema", "fixed_ema"), ("random", "random")):
        actions = _reference_actions(
            frame,
            comparison_config,
            kind=kind,  # type: ignore[arg-type]
            seed=seed,
        )
        _, metrics = evaluate_rl_model(
            _ActionSequencePolicy(actions),
            frame,
            feature_columns,
            comparison_config,
            deterministic=True,
            seed=seed,
        )
        benchmarks[name] = _benchmark_metrics(metrics)

    close = pd.to_numeric(frame["close"], errors="coerce")
    buy_hold_equity = env_config.initial_capital * close / float(close.iloc[0])
    buy_hold_drawdown = buy_hold_equity / buy_hold_equity.cummax() - 1.0
    benchmarks["buy_and_hold"] = {
        "final_equity": float(buy_hold_equity.iloc[-1]),
        "total_return": float(close.iloc[-1] / close.iloc[0] - 1.0),
        "max_drawdown": abs(float(buy_hold_drawdown.min())),
        "trades": 1,
        "liquidation_count": 0,
    }

    no_cost_frame = frame.copy()
    if "spread_bps" in no_cost_frame:
        no_cost_frame["spread_bps"] = 0.0
    if "funding_rate" in no_cost_frame:
        no_cost_frame["funding_rate"] = 0.0
    no_cost_config = replace(
        comparison_config,
        fee_rate=0.0,
        slippage_rate=0.0,
        spread_rate=0.0,
        short_borrow_rate_annual=0.0,
        liquidation_fee_rate=0.0,
        normalized_action_space=False,
        neutral_action_threshold=0.0,
    )
    replay_actions = pd.to_numeric(model_evaluation["target_fraction"], errors="coerce").fillna(0.0)
    _, no_cost_metrics = evaluate_rl_model(
        _ActionSequencePolicy(replay_actions.to_numpy(dtype=np.float32)),
        no_cost_frame,
        feature_columns,
        no_cost_config,
        deterministic=True,
        seed=seed,
    )
    benchmarks["model_without_costs"] = _benchmark_metrics(no_cost_metrics)
    return benchmarks


def run_rl_backtest(
    training_dir: str | Path,
    frame: pd.DataFrame,
    *,
    initial_capital: float,
    fee_rate: float,
    slippage_rate: float,
    short_borrow_rate_annual: float,
    deterministic: bool = True,
    seed: int = 42,
    device: str = "cpu",
) -> ModelBacktestResult:
    """使用保存的 PPO 或 SAC 與原始環境契約重跑指定歷史區間。"""
    policy = load_rl_policy(training_dir, device=device)
    missing = sorted(set(policy.feature_columns).difference(frame.columns))
    if missing:
        raise ValueError(f"RL 歷史資料缺少模型特徵：{missing[:12]}")
    env_config = replace(
        policy.env_config,
        initial_capital=float(initial_capital),
        fee_rate=float(fee_rate),
        slippage_rate=float(slippage_rate),
        short_borrow_rate_annual=float(short_borrow_rate_annual),
        episode_length=None,
        random_start=False,
    )
    evaluation, metrics = evaluate_rl_model(
        policy.model,
        frame,
        policy.feature_columns,
        env_config,
        deterministic=deterministic,
        seed=seed,
    )
    benchmarks = _rl_backtest_benchmarks(
        evaluation,
        frame,
        policy.feature_columns,
        env_config,
        metrics,
        seed=seed,
    )
    evaluation = _attach_market_context(evaluation, frame, initial_capital)
    ai_context = dict(policy.environment_metadata.get("ai_context", {}))
    uses_transformer = bool(
        ai_context.get("transformer_enabled", False)
        or "u_transformer_available" in policy.feature_columns
        or "transformer_available" in policy.feature_columns
    )
    algorithm = str(
        dict(policy.training_metadata.get("training_config", {})).get("algorithm", "rl")
    ).lower()
    return ModelBacktestResult(
        algorithm,
        evaluation,
        metrics,
        frame,
        {
            "training_dir": str(Path(training_dir).resolve()),
            "environment_dir": str(policy.environment_dir),
            "feature_count": len(policy.feature_columns),
            "uses_transformer": uses_transformer,
            "execution_config": env_config.to_dict(),
            "benchmarks": benchmarks,
        },
    )


def run_ppo_backtest(
    training_dir: str | Path,
    frame: pd.DataFrame,
    **kwargs: Any,
) -> ModelBacktestResult:
    """相容舊呼叫名稱；實際會依 training.json 自動辨識 PPO 或 SAC。"""
    return run_rl_backtest(training_dir, frame, **kwargs)


def _resolve_transformer_source(
    run_dir: Path,
    source: dict[str, Any],
    project_root: str | Path,
) -> Path:
    configured = Path(str(source.get("path", "")))
    project = Path(project_root).resolve()
    candidates = [configured]
    if configured.name:
        candidates.extend(
            [
                project / "data" / "processed" / configured.name,
                run_dir.parent.parent / configured.name,
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    if configured.name:
        matches = list((project / "data" / "processed").rglob(configured.name))
        if matches:
            return matches[0].resolve()
    raise FileNotFoundError(f"找不到 Transformer 訓練來源：{configured}")


def load_transformer_history(
    training_dir: str | Path,
    project_root: str | Path,
    *,
    source_index: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any], Path, Path]:
    """載入 Transformer 訓練時的 BTC 來源與 checkpoint。"""
    run_dir = Path(training_dir).resolve()
    metadata = _read_json(run_dir / "training.json")
    sources = list(metadata.get("sources", []))
    if not sources:
        raise ValueError("Transformer training.json 沒有來源資料")
    if not 0 <= source_index < len(sources):
        raise IndexError("Transformer 來源索引超出範圍")
    source = dict(sources[source_index])
    if "BTC" not in str(source.get("symbol", "")).upper():
        raise ValueError("目前 AI 回測只接受 BTC 模型")
    source_path = _resolve_transformer_source(run_dir, source, project_root)
    frame = _canonical_history(pd.read_csv(source_path))
    start = pd.to_datetime(source.get("start_at"), utc=True, errors="coerce")
    end = pd.to_datetime(source.get("end_at"), utc=True, errors="coerce")
    if not pd.isna(start):
        frame = frame.loc[frame["timestamp"] >= start]
    if not pd.isna(end):
        frame = frame.loc[frame["timestamp"] <= end]
    expected_rows = int(source.get("rows", 0) or 0)
    if expected_rows and len(frame) > expected_rows:
        frame = frame.tail(expected_rows)
    model_name = str(dict(metadata.get("artifacts", {})).get("model", "best_model.pt"))
    checkpoint = run_dir / model_name
    if not checkpoint.is_file():
        raise FileNotFoundError(f"找不到 Transformer checkpoint：{checkpoint}")
    return frame.reset_index(drop=True), metadata, checkpoint, source_path


def transformer_split(
    frame: pd.DataFrame,
    metadata: dict[str, Any],
    split: DatasetSplit,
) -> pd.DataFrame:
    """依訓練時完全相同的時間比例取得 Transformer 資料區段。"""
    if split == "all":
        return frame.reset_index(drop=True).copy()
    config = dict(metadata.get("training_config", {}))
    train_fraction = float(config.get("train_fraction", 0.60))
    validation_fraction = float(config.get("validation_fraction", 0.20))
    train_cut = int(len(frame) * train_fraction)
    validation_cut = int(len(frame) * (train_fraction + validation_fraction))
    bounds = {
        "train": (0, train_cut),
        "validation": (train_cut, validation_cut),
        "test": (validation_cut, len(frame)),
    }
    start, stop = bounds[split]
    result = frame.iloc[start:stop].reset_index(drop=True)
    if len(result) < 2:
        raise ValueError(f"Transformer {split} 區段資料不足")
    return result


def infer_transformer_history(
    checkpoint: str | Path,
    frame: pd.DataFrame,
    *,
    device: str = "auto",
    batch_size: int = 512,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> pd.DataFrame:
    """對完整歷史資料做因果序列推論，再交由 split 函式裁切。"""
    inferred = infer_transformer_context_frame(
        checkpoint,
        frame,
        device=device,
        batch_size=batch_size,
        progress_callback=progress_callback,
    )
    return _canonical_history(inferred)


def build_transformer_actions(
    frame: pd.DataFrame,
    config: TransformerSignalConfig,
) -> pd.Series:
    """使用方向、行情機率、不確定性和預測波動率計算目標部位。"""
    return_column = f"transformer_return_{config.horizon}"
    required = {
        return_column,
        "transformer_volatility",
        "transformer_bear_probability",
        "transformer_bull_probability",
        "transformer_uncertainty",
        "transformer_available",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Transformer 推論資料缺少欄位：{missing}")
    predicted_return = pd.to_numeric(frame[return_column], errors="coerce").fillna(0.0)
    volatility = (
        pd.to_numeric(frame["transformer_volatility"], errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    uncertainty = (
        pd.to_numeric(frame["transformer_uncertainty"], errors="coerce").fillna(1.0).clip(0.0, 1.0)
    )
    bull = (
        pd.to_numeric(frame["transformer_bull_probability"], errors="coerce")
        .fillna(0.0)
        .clip(0.0, 1.0)
    )
    bear = (
        pd.to_numeric(frame["transformer_bear_probability"], errors="coerce")
        .fillna(0.0)
        .clip(0.0, 1.0)
    )
    available = pd.to_numeric(frame["transformer_available"], errors="coerce").fillna(0.0) >= 0.5
    direction = np.sign(predicted_return.to_numpy(dtype=float))
    confidence = np.where(direction >= 0, bull, bear)
    eligible = (
        available.to_numpy()
        & (predicted_return.abs().to_numpy() >= config.minimum_return)
        & (confidence >= config.minimum_regime_probability)
        & (uncertainty.to_numpy() <= config.maximum_uncertainty)
    )
    risk_unit = np.maximum(
        volatility.to_numpy(dtype=float) * config.volatility_multiple,
        max(config.minimum_return, 1e-8),
    )
    strength = np.clip(predicted_return.abs().to_numpy(dtype=float) / risk_unit, 0.0, 1.0)
    certainty = np.clip(1.0 - uncertainty.to_numpy(dtype=float), 0.0, 1.0)
    targets = np.where(
        direction >= 0,
        config.max_long_fraction * strength * certainty,
        -config.max_short_fraction * strength * certainty,
    )
    if not config.allow_short:
        targets = np.maximum(targets, 0.0)
    return pd.Series(
        np.where(eligible, targets, 0.0),
        index=frame.index,
        name="transformer_target_fraction",
    )


def _transformer_prediction_metrics(
    frame: pd.DataFrame,
    actions: pd.Series,
    horizon: int,
) -> dict[str, float | int]:
    predicted = pd.to_numeric(frame[f"transformer_return_{horizon}"], errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    actual = close.shift(-horizon) / close - 1.0
    available = pd.to_numeric(frame["transformer_available"], errors="coerce").fillna(0.0) >= 0.5
    valid = available & predicted.notna() & actual.notna()
    if not valid.any():
        return {
            "prediction_samples": 0,
            "direction_accuracy": 0.0,
            "return_mae": 0.0,
            "return_correlation": 0.0,
            "signal_coverage": float((actions.abs() > 1e-8).mean()),
        }
    predicted_valid = predicted.loc[valid]
    actual_valid = actual.loc[valid]
    correlation = predicted_valid.corr(actual_valid)
    return {
        "prediction_samples": int(valid.sum()),
        "direction_accuracy": float((np.sign(predicted_valid) == np.sign(actual_valid)).mean()),
        "return_mae": float((predicted_valid - actual_valid).abs().mean()),
        "return_correlation": float(correlation) if np.isfinite(correlation) else 0.0,
        "signal_coverage": float((actions.abs() > 1e-8).mean()),
    }


def run_transformer_backtest(
    frame: pd.DataFrame,
    signal_config: TransformerSignalConfig,
    *,
    initial_capital: float,
    fee_rate: float,
    slippage_rate: float,
    short_borrow_rate_annual: float,
    rebalance_deadband: float,
    maximum_drawdown: float,
    seed: int = 42,
) -> ModelBacktestResult:
    """把 Transformer 預測轉成多空部位並使用 RL 同一套撮合環境回測。"""
    if "transformer_available" not in frame:
        raise ValueError("指定資料尚未完成 Transformer 歷史推論")
    available = pd.to_numeric(frame["transformer_available"], errors="coerce").fillna(0.0)
    usable = frame.loc[available >= 0.5].reset_index(drop=True)
    if len(usable) < 2:
        raise ValueError("指定區間沒有足夠的 Transformer 有效預測")
    actions = build_transformer_actions(usable, signal_config)
    env_config = PortfolioEnvConfig(
        initial_capital=float(initial_capital),
        fee_rate=float(fee_rate),
        slippage_rate=float(slippage_rate),
        max_position_fraction=signal_config.max_long_fraction,
        allow_short=signal_config.allow_short,
        max_short_fraction=(signal_config.max_short_fraction if signal_config.allow_short else 0),
        short_borrow_rate_annual=float(short_borrow_rate_annual),
        drawdown_penalty=0.0,
        turnover_penalty=0.0,
        max_drawdown_limit=float(maximum_drawdown),
        episode_length=None,
        random_start=False,
        rebalance_deadband=float(rebalance_deadband),
        risk_termination_penalty=0.0,
    )
    evaluation, metrics = evaluate_rl_model(
        _ActionSequencePolicy(actions.to_numpy()),
        usable,
        ["transformer_available"],
        env_config,
        deterministic=True,
        seed=seed,
    )
    decision = usable.iloc[:-1].reset_index(drop=True)
    return_column = f"transformer_return_{signal_config.horizon}"
    close = pd.to_numeric(usable["close"], errors="coerce")
    actual_return = close.shift(-signal_config.horizon) / close - 1.0
    evaluation["decision_timestamp"] = decision["timestamp"].to_numpy()
    evaluation["predicted_return"] = pd.to_numeric(
        decision[return_column], errors="coerce"
    ).to_numpy()
    evaluation["actual_future_return"] = actual_return.iloc[:-1].to_numpy()
    evaluation["bull_probability"] = pd.to_numeric(
        decision["transformer_bull_probability"], errors="coerce"
    ).to_numpy()
    evaluation["bear_probability"] = pd.to_numeric(
        decision["transformer_bear_probability"], errors="coerce"
    ).to_numpy()
    evaluation["uncertainty"] = pd.to_numeric(
        decision["transformer_uncertainty"], errors="coerce"
    ).to_numpy()
    metrics.update(_transformer_prediction_metrics(usable, actions, signal_config.horizon))
    evaluation = _attach_market_context(evaluation, usable, initial_capital)
    return ModelBacktestResult(
        "transformer",
        evaluation,
        metrics,
        usable,
        {"signal_config": asdict(signal_config), "execution_config": env_config.to_dict()},
    )


def save_latest_model_backtest(
    result: ModelBacktestResult,
    output_dir: str | Path,
    *,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """每種模型只保留最近一次結果，避免歷史回測無限累積。"""
    run_dir = Path(output_dir) / f"latest_{result.model_kind}"
    run_dir.mkdir(parents=True, exist_ok=True)
    evaluation_path = run_dir / "evaluation.csv"
    result.evaluation.to_csv(evaluation_path, index=False, encoding="utf-8")
    payload = {
        "schema_version": 1,
        "model_kind": result.model_kind,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(result.evaluation),
        "metrics": result.metrics,
        "metadata": {**result.metadata, **(extra_metadata or {})},
        "artifacts": {"evaluation": evaluation_path.name},
    }
    (run_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return run_dir
