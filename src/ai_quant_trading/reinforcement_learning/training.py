"""以 Stable-Baselines3 訓練、評估並保存 PPO／SAC 模型。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np
import pandas as pd

from ai_quant_trading.features.builder import infer_annualization_periods
from ai_quant_trading.reinforcement_learning.config import PortfolioEnvConfig
from ai_quant_trading.reinforcement_learning.device import (
    TrainingBackendStatus,
    inspect_training_backend,
    resolve_training_device,
)
from ai_quant_trading.reinforcement_learning.environment import (
    DiscretePortfolioActionWrapper,
    PortfolioTradingEnv,
    UniversalPortfolioTradingEnv,
)
from ai_quant_trading.reinforcement_learning.preflight import (
    assess_pretraining_readiness,
    ensure_pretraining_integrity,
)
from ai_quant_trading.reinforcement_learning.training_config import RLTrainingConfig
from ai_quant_trading.performance import configure_torch_cpu_threads


ProgressCallback = Callable[[dict[str, object]], None]
FrameCollection = pd.DataFrame | dict[str, pd.DataFrame]


@dataclass(frozen=True, slots=True)
class RLTrainingPaths:
    """一次強化學習訓練的模型、紀錄與評估檔案。"""

    run_dir: Path
    metadata_json: Path
    progress_json: Path
    final_model: Path
    best_model: Path
    checkpoints_dir: Path
    logs_dir: Path
    validation_csv: Path
    test_csv: Path


@dataclass(frozen=True, slots=True)
class RLTrainingResult:
    """已完成訓練的路徑、裝置與樣本外指標。"""

    paths: RLTrainingPaths
    config: RLTrainingConfig
    requested_device: str
    resolved_device: str
    duration_seconds: float
    metrics: dict[str, dict[str, float | int]]
    selected_model: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_environment_artifact(
    environment_dir: str | Path,
) -> tuple[
    dict[str, Any],
    list[str],
    PortfolioEnvConfig,
    FrameCollection,
    FrameCollection,
    FrameCollection,
]:
    root = Path(environment_dir)
    metadata_path = root / "environment.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"強化學習環境缺少檔案：{metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_columns = [str(column) for column in metadata.get("feature_columns", [])]
    if not feature_columns:
        raise ValueError("environment.json 缺少 feature_columns")
    raw_env_config = dict(metadata.get("environment_config", {}))
    if "include_position_context" not in raw_env_config:
        stored_size = int(metadata.get("observation_size", len(feature_columns) + 3))
        raw_env_config["include_position_context"] = stored_size >= len(feature_columns) + 5
    env_config = PortfolioEnvConfig(**raw_env_config)
    if metadata.get("environment_kind") == "universal":
        market_payload = dict(metadata.get("markets", {}))
        if len(market_payload) < 2:
            raise ValueError("通用 environment.json 至少需要兩個 markets")
        collections: dict[str, dict[str, pd.DataFrame]] = {
            "train": {},
            "validation": {},
            "test": {},
        }
        for market, values in market_payload.items():
            files = dict(values.get("files", {}))
            for split_name in collections:
                relative = files.get(split_name)
                path = root / str(relative or "")
                if not relative or not path.exists():
                    raise FileNotFoundError(f"{market} 缺少 {split_name} 資料：{path}")
                collections[split_name][str(market)] = pd.read_csv(path)
        return (
            metadata,
            feature_columns,
            env_config,
            collections["train"],
            collections["validation"],
            collections["test"],
        )

    required = [root / "train.csv", root / "validation.csv", root / "test.csv"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"強化學習環境缺少檔案：{missing}")
    return (
        metadata,
        feature_columns,
        env_config,
        pd.read_csv(root / "train.csv"),
        pd.read_csv(root / "validation.csv"),
        pd.read_csv(root / "test.csv"),
    )


def assess_rl_environment_preflight(environment_dir: str | Path):
    """載入既有環境並回傳與正式訓練相同的訓練前報告。"""
    metadata, features, env_config, train, validation, test = _load_environment_artifact(
        environment_dir
    )
    return assess_pretraining_readiness(
        metadata=metadata,
        feature_columns=features,
        env_config=env_config,
        train=train,
        validation=validation,
        test=test,
    )


def _create_training_paths(environment_dir: Path, algorithm: str) -> RLTrainingPaths:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = environment_dir / "training" / f"{timestamp}_{algorithm}"
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    checkpoints_dir.mkdir(parents=True, exist_ok=False)
    logs_dir.mkdir(parents=True, exist_ok=False)
    return RLTrainingPaths(
        run_dir=run_dir,
        metadata_json=run_dir / "training.json",
        progress_json=run_dir / "progress.json",
        final_model=run_dir / "final_model.zip",
        best_model=run_dir / "best_model" / "best_model.zip",
        checkpoints_dir=checkpoints_dir,
        logs_dir=logs_dir,
        validation_csv=run_dir / "validation_evaluation.csv",
        test_csv=run_dir / "test_evaluation.csv",
    )


def _activation_class(name: str) -> type[Any]:
    import torch.nn as nn

    classes = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
    }
    return classes[name]


def build_algorithm_parameters(
    config: RLTrainingConfig,
    action_dimension: int,
) -> tuple[dict[str, Any], Any | None]:
    """將完整設定轉成 SB3 建構參數與 SAC Action Noise。"""
    action_noise = None
    common: dict[str, Any] = {
        "learning_rate": config.learning_rate,
        "gamma": config.gamma,
        "batch_size": config.batch_size,
        "stats_window_size": config.stats_window_size,
        "policy_kwargs": {
            "net_arch": list(config.net_arch),
            "activation_fn": _activation_class(config.activation_fn),
        },
        "verbose": 0,
        "seed": config.seed,
    }
    if config.algorithm == "ppo":
        common.update(
            {
                "n_steps": config.ppo_n_steps,
                "n_epochs": config.ppo_n_epochs,
                "gae_lambda": config.ppo_gae_lambda,
                "clip_range": config.ppo_clip_range,
                "clip_range_vf": config.ppo_clip_range_vf,
                "normalize_advantage": config.ppo_normalize_advantage,
                "ent_coef": config.ppo_ent_coef,
                "vf_coef": config.ppo_vf_coef,
                "max_grad_norm": config.ppo_max_grad_norm,
                "use_sde": config.ppo_use_sde,
                "sde_sample_freq": config.ppo_sde_sample_freq,
                "target_kl": config.ppo_target_kl,
            }
        )
        return common, None

    if config.sac_action_noise != "none" and config.sac_action_noise_sigma > 0:
        from stable_baselines3.common.noise import (
            NormalActionNoise,
            OrnsteinUhlenbeckActionNoise,
        )

        mean = np.zeros(action_dimension, dtype=np.float32)
        sigma = np.full(action_dimension, config.sac_action_noise_sigma, dtype=np.float32)
        noise_class = (
            NormalActionNoise
            if config.sac_action_noise == "normal"
            else OrnsteinUhlenbeckActionNoise
        )
        action_noise = noise_class(mean=mean, sigma=sigma)
    common.update(
        {
            "buffer_size": config.sac_buffer_size,
            "learning_starts": config.sac_learning_starts,
            "tau": config.sac_tau,
            "train_freq": (config.sac_train_freq, config.sac_train_freq_unit),
            "gradient_steps": config.sac_gradient_steps,
            "action_noise": action_noise,
            "optimize_memory_usage": config.sac_optimize_memory_usage,
            "n_steps": config.sac_n_steps,
            "ent_coef": config.sac_ent_coef,
            "target_update_interval": config.sac_target_update_interval,
            "target_entropy": config.sac_target_entropy,
            "use_sde": config.sac_use_sde,
            "sde_sample_freq": config.sac_sde_sample_freq,
            "use_sde_at_warmup": config.sac_use_sde_at_warmup,
        }
    )
    return common, action_noise


def _full_horizon_evaluation_config(
    env_config: PortfolioEnvConfig,
) -> PortfolioEnvConfig:
    """驗證期跑完整時間；碰到原硬回撤線後只保留近零曝險。"""
    if env_config.max_drawdown_limit >= 1.0:
        return replace(
            env_config,
            episode_length=None,
            random_start=False,
            initial_capital_randomization=0.0,
            slippage_randomization=0.0,
        )
    return replace(
        env_config,
        episode_length=None,
        random_start=False,
        initial_capital_randomization=0.0,
        slippage_randomization=0.0,
        soft_drawdown_limit=env_config.max_drawdown_limit,
        soft_drawdown_multiplier=0.0001,
        max_drawdown_limit=1.0,
    )


def evaluate_rl_model(
    model: Any,
    frame: pd.DataFrame,
    feature_columns: list[str],
    env_config: PortfolioEnvConfig,
    *,
    deterministic: bool,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """逐根執行模型並產生可直接檢查的資產與交易紀錄。"""
    evaluation_config = _full_horizon_evaluation_config(env_config)
    base_env = PortfolioTradingEnv(frame, feature_columns, evaluation_config)
    discrete_actions = hasattr(getattr(model, "action_space", None), "n")
    env = DiscretePortfolioActionWrapper(base_env) if discrete_actions else base_env
    observation, _ = env.reset(seed=seed)
    rows: list[dict[str, object]] = []
    while True:
        action, _ = model.predict(observation, deterministic=deterministic)
        action_value = float(np.asarray(action).reshape(-1)[0])
        environment_action: int | np.ndarray = (
            int(action_value) if discrete_actions else np.array([action_value], dtype=np.float32)
        )
        observation, reward, terminated, truncated, info = env.step(environment_action)
        rows.append({**info, "action": action_value, "reward": float(reward)})
        if terminated or truncated:
            break
    result = pd.DataFrame(rows)
    final_equity = float(result.iloc[-1]["equity"])
    step_returns = pd.to_numeric(result["log_return"], errors="coerce").dropna()
    annualization = infer_annualization_periods(frame)
    return_std = float(step_returns.std(ddof=0))
    downside = step_returns.loc[step_returns < 0]
    downside_std = float(downside.std(ddof=0)) if not downside.empty else 0.0
    mean_return = float(step_returns.mean())
    positive_returns = float(step_returns.clip(lower=0).sum())
    negative_returns = abs(float(step_returns.clip(upper=0).sum()))
    closed_trade_pnl = (
        pd.to_numeric(
            result.loc[result.get("trade_closed", False).astype(bool), "closed_trade_pnl"],
            errors="coerce",
        ).dropna()
        if "trade_closed" in result
        else pd.Series(dtype=float)
    )
    profitable_trades = closed_trade_pnl.loc[closed_trade_pnl > 0]
    losing_trades = closed_trade_pnl.loc[closed_trade_pnl < 0]
    trade_profit = float(profitable_trades.sum())
    trade_loss = abs(float(losing_trades.sum()))
    annual_return = float(np.expm1(np.clip(mean_return * annualization, -50, 50)))
    max_drawdown = abs(float(result["drawdown"].min()))
    timestamp_index = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
    monthly_equity = (
        pd.Series(
            pd.to_numeric(result["equity"], errors="coerce").to_numpy(),
            index=timestamp_index,
        )
        .dropna()
        .resample("ME")
        .last()
    )
    monthly_returns = monthly_equity.pct_change()
    if not monthly_equity.empty:
        monthly_returns.iloc[0] = monthly_equity.iloc[0] / evaluation_config.initial_capital - 1
    worst_month = float(monthly_returns.min()) if not monthly_returns.empty else 0.0
    buy_and_hold_return = float(frame.iloc[-1]["close"] / frame.iloc[0]["close"] - 1)
    actions = pd.to_numeric(result["action"], errors="coerce")
    positions = pd.to_numeric(result["position_fraction"], errors="coerce")
    action_intents = result.get("action_intent", pd.Series("", index=result.index)).astype(str)
    action_tolerance = max(
        (evaluation_config.max_position_fraction + evaluation_config.max_short_fraction) * 0.01,
        1e-4,
    )
    if discrete_actions:
        at_long_limit = positions >= (evaluation_config.max_position_fraction - action_tolerance)
        at_short_limit = (
            positions <= (-evaluation_config.max_short_fraction + action_tolerance)
            if evaluation_config.allow_short
            else pd.Series(False, index=actions.index)
        )
    else:
        normalized_limit = (
            1.0
            if evaluation_config.normalized_action_space
            else (evaluation_config.max_position_fraction)
        )
        normalized_short_limit = (
            -1.0
            if evaluation_config.normalized_action_space
            else -evaluation_config.max_short_fraction
        )
        at_long_limit = actions >= (normalized_limit - action_tolerance)
        at_short_limit = (
            actions <= (normalized_short_limit + action_tolerance)
            if evaluation_config.allow_short
            else pd.Series(False, index=actions.index)
        )
    metrics: dict[str, float | int] = {
        "steps": len(result),
        "final_equity": final_equity,
        "total_return": final_equity / evaluation_config.initial_capital - 1,
        "max_drawdown": max_drawdown,
        "total_reward": float(result["reward"].sum()),
        "trades": int((result["side"] != "HOLD").sum()),
        "total_fees": float(result["fee"].sum()),
        "total_slippage": float(result["slippage_cost"].sum()),
        "total_short_carry": float(result["short_carry_cost"].sum()),
        "total_funding": float(result.get("funding_cost", 0.0).sum()),
        "total_liquidation_fees": float(result.get("liquidation_fee", 0.0).sum()),
        "liquidation_count": int(result.get("liquidated", False).astype(bool).sum()),
        "fee_to_initial_capital": float(
            (
                result["fee"]
                + result["slippage_cost"]
                + result["short_carry_cost"]
                + result.get("funding_cost", 0.0).abs()
                + result.get("liquidation_fee", 0.0)
            ).sum()
        )
        / evaluation_config.initial_capital,
        "turnover_to_initial_capital": float(result["trade_notional"].sum())
        / evaluation_config.initial_capital,
        "positive_step_rate": float((result["log_return"] > 0).mean()),
        "annual_return": annual_return,
        "sharpe_ratio": (
            mean_return / return_std * np.sqrt(annualization) if return_std > 0 else 0.0
        ),
        "sortino_ratio": (
            mean_return / downside_std * np.sqrt(annualization) if downside_std > 0 else 0.0
        ),
        "profit_factor": (
            trade_profit / trade_loss
            if trade_loss > 0
            else positive_returns / negative_returns
            if negative_returns > 0
            else 0.0
        ),
        "closed_trades": len(closed_trade_pnl),
        "win_rate": (float((closed_trade_pnl > 0).mean()) if len(closed_trade_pnl) else 0.0),
        "expectancy": (float(closed_trade_pnl.mean()) if len(closed_trade_pnl) else 0.0),
        "payoff_ratio": (
            float(profitable_trades.mean() / abs(losing_trades.mean()))
            if len(profitable_trades) and len(losing_trades)
            else 0.0
        ),
        "calmar_ratio": annual_return / max_drawdown if max_drawdown > 0 else 0.0,
        "worst_month": worst_month,
        "buy_and_hold_return": buy_and_hold_return,
        "excess_return": final_equity / evaluation_config.initial_capital - 1 - buy_and_hold_return,
        # 用來辨識 PPO 是否退化成永遠輸出最大多單或最大空單。
        "action_std": float(actions.std(ddof=0)),
        "action_saturation_ratio": float((at_long_limit | at_short_limit).mean()),
        "long_exposure_ratio": float((positions > 1e-3).mean()),
        "short_exposure_ratio": float((positions < -1e-3).mean()),
        "flat_exposure_ratio": float((positions.abs() <= 1e-3).mean()),
        "maximum_effective_leverage": float(positions.abs().max()),
        "hold_intent_ratio": float(action_intents.eq("HOLD_POSITION").mean()),
        "wait_flat_intent_ratio": float(action_intents.eq("WAIT_FLAT").mean()),
        "close_intent_ratio": float(action_intents.eq("CLOSE_POSITION").mean()),
        "directional_intent_ratio": float(
            action_intents.isin({"TARGET_LONG", "TARGET_SHORT"}).mean()
        ),
        "risk_override_ratio": float(
            result.get("risk_reasons", pd.Series("", index=result.index))
            .astype(str)
            .str.len()
            .gt(0)
            .mean()
        ),
        "total_reward_turnover_penalty": float(
            pd.to_numeric(
                result.get("turnover_penalty", pd.Series(0.0, index=result.index)),
                errors="coerce",
            ).sum()
        ),
    }
    return result, metrics


def evaluate_rl_markets(
    model: Any,
    frames: FrameCollection,
    feature_columns: list[str],
    env_config: PortfolioEnvConfig,
    *,
    deterministic: bool,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float | int], dict[str, dict[str, float | int]]]:
    """評估單一或多市場資料，通用模型同時回傳逐市場與等權整體指標。"""
    if isinstance(frames, pd.DataFrame):
        rows, metrics = evaluate_rl_model(
            model,
            frames,
            feature_columns,
            env_config,
            deterministic=deterministic,
            seed=seed,
        )
        return rows, metrics, {}

    all_rows: list[pd.DataFrame] = []
    market_metrics: dict[str, dict[str, float | int]] = {}
    for index, (market, frame) in enumerate(frames.items()):
        rows, metrics = evaluate_rl_model(
            model,
            frame,
            feature_columns,
            env_config,
            deterministic=deterministic,
            seed=seed + index,
        )
        rows.insert(0, "market", market)
        all_rows.append(rows)
        market_metrics[market] = metrics

    metric_frame = pd.DataFrame.from_dict(market_metrics, orient="index")
    weights = metric_frame["steps"] / metric_frame["steps"].sum()
    aggregate: dict[str, float | int] = {
        "market_count": len(market_metrics),
        "steps": int(metric_frame["steps"].sum()),
        "final_equity": float(metric_frame["final_equity"].mean()),
        "total_return": float(metric_frame["total_return"].mean()),
        "max_drawdown": float(metric_frame["max_drawdown"].max()),
        "total_reward": float(metric_frame["total_reward"].sum()),
        "trades": int(metric_frame["trades"].sum()),
        "total_fees": float(metric_frame["total_fees"].sum()),
        "total_slippage": float(metric_frame["total_slippage"].sum()),
        "total_short_carry": float(metric_frame["total_short_carry"].sum()),
        "fee_to_initial_capital": float(metric_frame["fee_to_initial_capital"].mean()),
        "turnover_to_initial_capital": float(metric_frame["turnover_to_initial_capital"].mean()),
        "positive_step_rate": float((metric_frame["positive_step_rate"] * weights).sum()),
        "sharpe_ratio": float((metric_frame["sharpe_ratio"] * weights).sum()),
        "sortino_ratio": float((metric_frame["sortino_ratio"] * weights).sum()),
        "profit_factor": float((metric_frame["profit_factor"] * weights).sum()),
        "buy_and_hold_return": float(metric_frame["buy_and_hold_return"].mean()),
        "excess_return": float(metric_frame["excess_return"].mean()),
        "positive_market_ratio": float((metric_frame["total_return"] > 0).mean()),
        "worst_market_return": float(metric_frame["total_return"].min()),
        "action_std": float((metric_frame["action_std"] * weights).sum()),
        "action_saturation_ratio": float((metric_frame["action_saturation_ratio"] * weights).sum()),
        "long_exposure_ratio": float((metric_frame["long_exposure_ratio"] * weights).sum()),
        "short_exposure_ratio": float((metric_frame["short_exposure_ratio"] * weights).sum()),
        "flat_exposure_ratio": float((metric_frame["flat_exposure_ratio"] * weights).sum()),
        "hold_intent_ratio": float((metric_frame["hold_intent_ratio"] * weights).sum()),
        "wait_flat_intent_ratio": float(
            (metric_frame["wait_flat_intent_ratio"] * weights).sum()
        ),
        "close_intent_ratio": float((metric_frame["close_intent_ratio"] * weights).sum()),
        "directional_intent_ratio": float(
            (metric_frame["directional_intent_ratio"] * weights).sum()
        ),
        "risk_override_ratio": float(
            (metric_frame["risk_override_ratio"] * weights).sum()
        ),
        "total_reward_turnover_penalty": float(
            metric_frame["total_reward_turnover_penalty"].sum()
        ),
    }
    populated_rows = [frame for frame in all_rows if not frame.empty]
    if populated_rows:
        columns = list(
            dict.fromkeys(
                column for frame in populated_rows for column in frame.columns
            )
        )
        concat_inputs = [frame.dropna(axis=1, how="all") for frame in populated_rows]
        combined = pd.concat(concat_inputs, ignore_index=True).reindex(columns=columns)
    else:
        combined = pd.DataFrame()
    return combined, aggregate, market_metrics


def train_rl_agent(
    environment_dir: str | Path,
    config: RLTrainingConfig,
    *,
    progress_callback: ProgressCallback | None = None,
    resume_from: str | Path | None = None,
    backend_status: TrainingBackendStatus | None = None,
) -> RLTrainingResult:
    """訓練 PPO／SAC，保存 Checkpoint，並以驗證集選模後測試。"""
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        CallbackList,
        CheckpointCallback,
        EvalCallback,
    )
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    configure_torch_cpu_threads(config.cpu_threads)
    status = backend_status or inspect_training_backend()
    resolved_device = resolve_training_device(config.device, status)
    environment_path = Path(environment_dir)
    metadata, features, env_config, train_frame, validation_frame, test_frame = (
        _load_environment_artifact(environment_path)
    )
    preflight = assess_pretraining_readiness(
        metadata=metadata,
        feature_columns=features,
        env_config=env_config,
        train=train_frame,
        validation=validation_frame,
        test=test_frame,
    )
    ensure_pretraining_integrity(preflight)
    paths = _create_training_paths(environment_path, config.algorithm)
    started_at = _utc_now()
    started_clock = perf_counter()
    running_payload: dict[str, object] = {
        "status": "running",
        "created_at": started_at,
        "environment_dir": str(environment_path),
        "environment_source": metadata.get("source", {}),
        "environment_expert": metadata.get("expert", {"kind": env_config.expert_kind}),
        "environment_ai_context": metadata.get("ai_context", {}),
        "training_config": config.to_dict(),
        "requested_device": config.device,
        "resolved_device": resolved_device,
        "torch_version": status.torch_version,
        "stable_baselines_version": status.stable_baselines_version,
        "cuda_devices": list(status.cuda_devices),
        "resume_from": str(resume_from) if resume_from is not None else None,
        "pretraining_readiness": preflight.to_dict(),
    }
    _write_json(paths.metadata_json, running_payload)

    def make_train_env(rank: int) -> Callable[[], Any]:
        def factory() -> Any:
            if isinstance(train_frame, dict):
                env = UniversalPortfolioTradingEnv(
                    {name: frame.copy() for name, frame in train_frame.items()},
                    features,
                    env_config,
                    selection_mode="random",
                )
            else:
                env = PortfolioTradingEnv(train_frame.copy(), features, env_config)
            wrapped = DiscretePortfolioActionWrapper(env) if config.algorithm == "ppo" else env
            wrapped.action_space.seed(config.seed + rank)
            return Monitor(wrapped)

        return factory

    evaluation_config = _full_horizon_evaluation_config(env_config)

    def make_validation_env() -> Any:
        if isinstance(validation_frame, dict):
            env = UniversalPortfolioTradingEnv(
                {name: frame.copy() for name, frame in validation_frame.items()},
                features,
                evaluation_config,
                selection_mode="cycle",
            )
        else:
            env = PortfolioTradingEnv(validation_frame.copy(), features, evaluation_config)
        wrapped = DiscretePortfolioActionWrapper(env) if config.algorithm == "ppo" else env
        return Monitor(wrapped)

    train_vec = DummyVecEnv([make_train_env(rank) for rank in range(config.n_envs)])
    validation_vec = DummyVecEnv([make_validation_env])
    model_class = PPO if config.algorithm == "ppo" else SAC
    callbacks: list[Any] = []
    training_logger: Any | None = None

    class JsonProgressCallback(BaseCallback):
        """將訓練進度節流寫入 JSON，並通知 Dashboard。"""

        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.initial_steps = 0
            self.update_every = max(config.total_timesteps // 100, config.n_envs)
            self.last_reported = -1
            self.reward_window: deque[float] = deque(maxlen=1_000)
            self.last_payload: dict[str, object] = {}

        def _on_training_start(self) -> None:
            self.initial_steps = int(self.model.num_timesteps)

        def _logger_metrics(self) -> dict[str, float]:
            """擷取 PPO／SAC 共通的重要學習指標，忽略尚未產生的欄位。"""
            aliases = {
                "rollout/ep_rew_mean": "episode_reward_mean",
                "train/value_loss": "value_loss",
                "train/policy_gradient_loss": "policy_loss",
                "train/actor_loss": "actor_loss",
                "train/critic_loss": "critic_loss",
                "train/entropy_loss": "entropy_loss",
                "train/approx_kl": "approx_kl",
                "train/explained_variance": "explained_variance",
                "eval/mean_reward": "evaluation_reward",
            }
            values = getattr(self.logger, "name_to_value", {})
            result: dict[str, float] = {}
            for source, target in aliases.items():
                try:
                    value = float(values[source])
                except (KeyError, TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    result[target] = value
            return result

        def _latest_snapshot(self) -> dict[str, object]:
            """回傳最近一個平行環境的市場與投資組合狀態。"""
            infos = self.locals.get("infos", [])
            if not isinstance(infos, (list, tuple)):
                return {}
            fields = [
                "market",
                "timestamp",
                "side",
                "target_fraction",
                "position_fraction",
                "equity",
                "drawdown",
                "holding_bars",
            ]
            for info in reversed(infos):
                if not isinstance(info, dict):
                    continue
                snapshot: dict[str, object] = {}
                for field in fields:
                    value = info.get(field)
                    if isinstance(value, np.generic):
                        value = value.item()
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        snapshot[field] = value
                return snapshot
            return {}

        def _report(self, force: bool = False) -> None:
            completed = max(int(self.num_timesteps) - self.initial_steps, 0)
            completed = min(completed, config.total_timesteps)
            if not force and completed - self.last_reported < self.update_every:
                return
            elapsed = max(perf_counter() - started_clock, 1e-9)
            steps_per_second = completed / elapsed
            remaining = max(config.total_timesteps - completed, 0)
            metrics = {
                "steps_per_second": steps_per_second,
                "eta_seconds": remaining / steps_per_second if steps_per_second > 0 else None,
                **self._logger_metrics(),
            }
            if self.reward_window:
                metrics["recent_reward_mean"] = float(np.mean(self.reward_window))
            payload: dict[str, object] = {
                "status": "running",
                "completed_timesteps": completed,
                "total_timesteps": config.total_timesteps,
                "progress": completed / config.total_timesteps,
                "elapsed_seconds": elapsed,
                "device": resolved_device,
                "metrics": metrics,
                "snapshot": self._latest_snapshot(),
                "updated_at": _utc_now(),
            }
            _write_json(paths.progress_json, payload)
            if progress_callback is not None:
                progress_callback(payload)
            self.last_reported = completed
            self.last_payload = payload

        def _on_step(self) -> bool:
            rewards = np.asarray(self.locals.get("rewards", []), dtype=float).reshape(-1)
            self.reward_window.extend(float(value) for value in rewards if np.isfinite(value))
            self._report()
            return True

        def _on_training_end(self) -> None:
            self._report(force=True)

    progress_tracker = JsonProgressCallback()
    callbacks.append(progress_tracker)
    if config.checkpoint_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=max(config.checkpoint_freq // config.n_envs, 1),
                save_path=str(paths.checkpoints_dir),
                name_prefix=config.algorithm,
                save_replay_buffer=(config.algorithm == "sac" and config.save_replay_buffer),
                verbose=0,
            )
        )
    if config.evaluation_freq > 0:
        callbacks.append(
            EvalCallback(
                validation_vec,
                best_model_save_path=str(paths.best_model.parent),
                log_path=str(paths.run_dir / "evaluations"),
                eval_freq=max(config.evaluation_freq // config.n_envs, 1),
                n_eval_episodes=max(
                    config.n_eval_episodes,
                    len(validation_frame) if isinstance(validation_frame, dict) else 1,
                ),
                deterministic=config.deterministic_eval,
                render=False,
                verbose=0,
                warn=False,
            )
        )

    try:
        action_shape = getattr(train_vec.action_space, "shape", ())
        action_dimension = int(action_shape[-1]) if action_shape else 1
        parameters, _ = build_algorithm_parameters(config, action_dimension)
        if resume_from is None:
            model = model_class(
                "MlpPolicy",
                train_vec,
                device=resolved_device,
                **parameters,
            )
            reset_num_timesteps = True
        else:
            resume_path = Path(resume_from)
            if not resume_path.exists():
                raise FileNotFoundError(f"續跑模型不存在：{resume_path}")
            model = model_class.load(resume_path, env=train_vec, device=resolved_device)
            reset_num_timesteps = False
        training_logger = configure(str(paths.logs_dir), ["csv"])
        model.set_logger(training_logger)
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=CallbackList(callbacks),
            log_interval=1,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=False,
        )
        model.save(paths.final_model)
        if config.algorithm == "sac" and config.save_replay_buffer:
            model.save_replay_buffer(paths.run_dir / "final_replay_buffer.pkl")

        evaluation_payload = {
            **progress_tracker.last_payload,
            "status": "evaluating",
            "progress": 1.0,
            "updated_at": _utc_now(),
        }
        _write_json(paths.progress_json, evaluation_payload)
        if progress_callback is not None:
            progress_callback(evaluation_payload)

        selected_model = paths.best_model if paths.best_model.exists() else paths.final_model
        evaluation_model = model_class.load(selected_model, device=resolved_device)
        validation_rows, validation_metrics, validation_market_metrics = evaluate_rl_markets(
            evaluation_model,
            validation_frame,
            features,
            env_config,
            deterministic=config.deterministic_eval,
            seed=config.seed,
        )
        test_rows, test_metrics, test_market_metrics = evaluate_rl_markets(
            evaluation_model,
            test_frame,
            features,
            env_config,
            deterministic=config.deterministic_eval,
            seed=config.seed,
        )
        validation_rows.to_csv(paths.validation_csv, index=False, encoding="utf-8")
        test_rows.to_csv(paths.test_csv, index=False, encoding="utf-8")
        duration = perf_counter() - started_clock
        metrics = {
            "validation": validation_metrics,
            "test": test_metrics,
            "validation_markets": validation_market_metrics,
            "test_markets": test_market_metrics,
        }
        completed_payload = {
            **running_payload,
            "status": "complete",
            "completed_at": _utc_now(),
            "duration_seconds": duration,
            "selected_model": str(selected_model),
            "final_model": str(paths.final_model),
            "best_model": str(paths.best_model) if paths.best_model.exists() else None,
            "metrics": metrics,
        }
        _write_json(paths.metadata_json, completed_payload)
        _write_json(
            environment_path / "environment.json",
            {
                **metadata,
                "status": "trained",
                "training_started": True,
                "last_training": {
                    "run_dir": str(paths.run_dir),
                    "algorithm": config.algorithm,
                    "completed_at": completed_payload["completed_at"],
                    "selected_model": str(selected_model),
                },
            },
        )
        _write_json(
            paths.progress_json,
            {
                **progress_tracker.last_payload,
                "status": "complete",
                "completed_timesteps": config.total_timesteps,
                "total_timesteps": config.total_timesteps,
                "progress": 1.0,
                "elapsed_seconds": duration,
                "device": resolved_device,
                "updated_at": _utc_now(),
            },
        )
        from ai_quant_trading.operations.integrity import build_artifact_manifest

        build_artifact_manifest(paths.run_dir)
        return RLTrainingResult(
            paths,
            config,
            config.device,
            resolved_device,
            duration,
            metrics,
            selected_model,
        )
    except Exception as exc:
        _write_json(
            paths.progress_json,
            {
                **progress_tracker.last_payload,
                "status": "failed",
                "elapsed_seconds": perf_counter() - started_clock,
                "device": resolved_device,
                "error": str(exc),
                "updated_at": _utc_now(),
            },
        )
        _write_json(
            paths.metadata_json,
            {
                **running_payload,
                "status": "failed",
                "failed_at": _utc_now(),
                "error": str(exc),
            },
        )
        raise
    finally:
        if training_logger is not None:
            training_logger.close()
        train_vec.close()
        validation_vec.close()


def list_rl_training_runs(output_dir: str | Path) -> list[Path]:
    """列出所有已開始的訓練 Run，最新者優先。"""
    root = Path(output_dir)
    if not root.exists():
        return []
    runs = [path.parent for path in root.rglob("training.json")]
    return sorted(runs, key=lambda path: path.stat().st_mtime, reverse=True)


def list_rl_checkpoints(training_dir: str | Path) -> list[Path]:
    """列出可續跑的最佳、最終與定期 Checkpoint 模型。"""
    root = Path(training_dir)
    if not root.exists():
        return []
    candidates = list(root.rglob("*.zip"))
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)
