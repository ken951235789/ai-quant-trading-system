"""固定事件的資金配置消融；不提供實盤、槓桿或自由出場能力。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd

from ai_quant_trading.operations.integrity import build_artifact_manifest
from ai_quant_trading.persistence import write_json_atomic
from ai_quant_trading.transformer.economics import block_confidence_interval


@dataclass(frozen=True)
class SizingConfig:
    initial_capital: float = 1000.0
    fixed_fraction: float = 0.10
    maximum_fraction: float = 0.20
    reference_atr_fraction: float = 0.003
    max_settled_drawdown: float = 0.10
    total_timesteps: int = 100_000
    seeds: tuple[int, ...] = (42, 137, 2026, 2027, 2028)

    def __post_init__(self) -> None:
        values = (self.initial_capital, self.fixed_fraction, self.maximum_fraction,
                  self.reference_atr_fraction, self.max_settled_drawdown)
        if not np.isfinite(values).all() or self.initial_capital <= 0:
            raise ValueError("資金配置參數必須有限且本金為正")
        if not 0 < self.fixed_fraction <= self.maximum_fraction <= 1:
            raise ValueError("本研究禁止槓桿，部位比例必須介於 0 和 1")
        if self.reference_atr_fraction <= 0 or not 0 < self.max_settled_drawdown < 1:
            raise ValueError("波動基準或回撤限制不合法")
        if self.total_timesteps < 1 or len(set(self.seeds)) != len(self.seeds) or not self.seeds:
            raise ValueError("步數與種子設定不合法")


def observation_columns(horizon: int) -> list[str]:
    # 未來成交價、退出時間、實際收益只能交給環境結算，禁止進入 observation。
    return [f"predicted_return_{horizon}", f"tradeability_probability_{horizon}",
            f"predicted_return_q10_{horizon}", f"predicted_return_q50_{horizon}",
            f"predicted_return_q90_{horizon}", "causal_atr_fraction", "event_side"]


def fit_scaler(frame: pd.DataFrame, horizon: int) -> dict:
    values = frame[observation_columns(horizon)].to_numpy(dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("資金配置 Train 特徵不足或非有限值")
    return {"columns": observation_columns(horizon), "mean": values.mean(axis=0).tolist(),
            "std": np.maximum(values.std(axis=0), 1e-8).tolist(), "fit_on": "train_only"}


class EventSizingEnv(gym.Env):
    """一個 step 是一筆固定事件；空手亦保留同樣排程，三組不更換交易機會。

    這是結算權益層級的資金配置研究，不模擬棒內保證金、撮合容量或帳戶清算。
    方向、成交時間與價格皆由既有事件回放決定；SAC 只能決定入場配置比例。
    """

    def __init__(self, frame: pd.DataFrame, horizon: int, config: SizingConfig, scaler: dict):
        self.frame = frame.reset_index(drop=True).copy()
        self.horizon, self.config = horizon, config
        if self.frame.empty or scaler["columns"] != observation_columns(horizon) or scaler["fit_on"] != "train_only":
            raise ValueError("資金配置環境缺少資料或 observation 契約錯誤")
        columns = observation_columns(horizon)
        values = self.frame[columns].to_numpy(dtype=float)
        means, stds = np.asarray(scaler["mean"]), np.asarray(scaler["std"])
        if means.shape != (len(columns),) or stds.shape != means.shape or not np.isfinite([*means, *stds]).all() or (stds <= 0).any():
            raise ValueError("資金配置 scaler 不合法")
        required = ["endpoint", "event_entry_endpoint", "event_exit_endpoint", f"actual_return_{horizon}", "causal_atr_fraction"]
        if not np.isfinite(values).all() or not np.isfinite(self.frame[required].to_numpy(dtype=float)).all():
            raise ValueError("資金配置事件含非有限值")
        if (self.frame.causal_atr_fraction <= 0).any() or not self.frame.event_side.isin([-1, 1]).all():
            raise ValueError("事件波動或方向不合法")
        if (1 + config.maximum_fraction * self.frame[f"actual_return_{horizon}"] <= 0).any():
            raise ValueError("事件可能耗盡本金，需要完整清算模型，不適用本研究")
        if not (self.frame.event_entry_endpoint > self.frame.endpoint).all() or not (self.frame.event_exit_endpoint >= self.frame.event_entry_endpoint).all():
            raise ValueError("事件訊號與成交時間不合法")
        if len(self.frame) > 1 and not (self.frame.endpoint.to_numpy()[1:] > self.frame.event_exit_endpoint.to_numpy()[:-1]).all():
            raise ValueError("同訊號比較需要有序、不重疊事件")
        self.values = np.clip((values - means) / stds, -10, 10).astype(np.float32)
        self.action_space = gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-10.0, 10.0, shape=(len(columns) + 2,), dtype=np.float32)
        self.reset()

    def _observation(self) -> np.ndarray:
        if self.index >= len(self.frame):
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        account = [np.clip(self.equity / self.config.initial_capital - 1, -10, 10),
                   self.equity / self.peak - 1]
        return np.r_[self.values[self.index], account].astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.index = 0
        self.equity = self.peak = self.config.initial_capital
        self.halted = False
        return self._observation(), {}

    def step(self, action):
        if self.index >= len(self.frame):
            raise RuntimeError("事件序列已結束，必須 reset")
        raw = np.asarray(action, dtype=float)
        if raw.shape != (1,) or not np.isfinite(raw).all():
            raise ValueError("資金配置動作必須為單一有限值")
        allocation = float(np.clip(raw[0], 0, 1)) * self.config.maximum_fraction
        if self.halted:
            allocation = 0.0
        row = self.frame.iloc[self.index]
        net = float(row[f"actual_return_{self.horizon}"])
        before = self.equity
        self.equity = max(0.0, before * (1 + allocation * net))
        self.peak = max(self.peak, self.equity)
        self.halted = self.halted or 1 - self.equity / self.peak >= self.config.max_settled_drawdown
        self.index += 1
        bankrupt = self.equity <= 0
        terminated = bankrupt
        truncated = self.index >= len(self.frame) and not bankrupt
        # 淨收益已包含完整事件成本，reward 不重複扣費，也不獎勵開單。
        reward = float(np.log(max(self.equity, 1e-12) / max(before, 1e-12)))
        info = {"endpoint": int(row.endpoint), "exit_endpoint": int(row.event_exit_endpoint),
                "allocation": allocation, "net_unit_return": net, "pnl": self.equity - before,
                "equity": self.equity, "halted": self.halted}
        return self._observation(), reward, terminated, truncated, info


def evaluate_sizing(frame: pd.DataFrame, horizon: int, config: SizingConfig, scaler: dict, policy) -> tuple[pd.DataFrame, dict]:
    env = EventSizingEnv(frame, horizon, config, scaler)
    obs, _ = env.reset()
    rows = []
    while True:
        if isinstance(policy, str):
            fraction = config.fixed_fraction
            if policy == "volatility":
                fraction *= config.reference_atr_fraction / float(env.frame.iloc[env.index].causal_atr_fraction)
            elif policy != "fixed":
                raise ValueError("未登記的資金配置基準")
            action = np.array([min(fraction, config.maximum_fraction) / config.maximum_fraction])
        else:
            action, _ = policy.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        rows.append(info)
        if terminated or truncated:
            break
    result = pd.DataFrame(rows)
    equity = np.r_[config.initial_capital, result.equity]
    return result, {"total_return": float(equity[-1] / equity[0] - 1),
                    "settled_max_drawdown": float(-(equity / np.maximum.accumulate(equity) - 1).min()),
                    "events": len(result), "active_events": int(result.allocation.gt(1e-8).sum()),
                    "mean_allocation": float(result.allocation.mean()),
                    "scope": "固定事件的結算權益，不是逐棒完整風控或實盤資格"}


def run_sizing_comparison(splits: dict[str, pd.DataFrame], horizon: int, config: SizingConfig, output: Path, contract_hash: str) -> dict:
    """先凍結驗證／測試，所有 seed 完整揭露；不依 Test 選模型。"""
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import EvalCallback
    import torch

    if output.exists():
        raise FileExistsError("資金配置研究禁止覆寫")
    output.mkdir(parents=True)
    torch.set_num_threads(2)
    scaler = fit_scaler(splits["train"], horizon)
    write_json_atomic(output / "contract.json", {"contract_sha256": contract_hash, "horizon": horizon,
        "action": "entry_allocation_only_0_to_1", "config": asdict(config), "scaler": scaler,
        "same_event_schedule_even_when_flat": True, "live_eligible": False})
    results, evaluations = [], {}
    for split, frame in splits.items():
        frame.to_csv(output / f"{split}_events.csv", index=False)
    for split in ("validation", "test"):
        for name in ("fixed", "volatility"):
            evaluation, metrics = evaluate_sizing(splits[split], horizon, config, scaler, name)
            evaluation.to_csv(output / f"{split}_{name}.csv", index=False)
            evaluations[split, name] = evaluation
            results.append({"split": split, "policy": name, "metrics": metrics})
    for seed in config.seeds:
        folder = output / f"seed_{seed}"
        train = EventSizingEnv(splits["train"], horizon, config, scaler)
        validation = EventSizingEnv(splits["validation"], horizon, config, scaler)
        callback = EvalCallback(validation, best_model_save_path=str(folder),
            eval_freq=max(config.total_timesteps // 5, 1), n_eval_episodes=1, deterministic=True,
            verbose=0, warn=False)
        model = SAC("MlpPolicy", train, seed=seed, device="cpu", learning_rate=0.0001,
            buffer_size=100_000, learning_starts=min(2_000, max(config.total_timesteps // 10, 1)),
            batch_size=128, policy_kwargs={"net_arch": [64, 64]}, verbose=0)
        model.learn(total_timesteps=config.total_timesteps, callback=callback)
        model.save(folder / "final_model.zip")
        selected = SAC.load(folder / "best_model.zip", device="cpu")
        for split in ("validation", "test"):
            evaluation, metrics = evaluate_sizing(splits[split], horizon, config, scaler, selected)
            evaluation.to_csv(folder / f"{split}.csv", index=False)
            improvements = {}
            for name in ("fixed", "volatility"):
                baseline = evaluations[split, name]
                if not evaluation.endpoint.equals(baseline.endpoint):
                    raise ValueError("基準與 SAC 沒有使用同一事件排程")
                agent_log = np.log1p(evaluation.allocation * evaluation.net_unit_return)
                base_log = np.log1p(baseline.allocation * baseline.net_unit_return)
                low, high = block_confidence_interval((agent_log - base_log).to_numpy())
                improvements[name] = {"paired_log_return_ci_low": low, "paired_log_return_ci_high": high}
            results.append({"split": split, "policy": "sac", "seed": seed,
                            "metrics": metrics, "paired_improvement": improvements})
        print(f"SAC entry sizing seed={seed} complete", flush=True)
    evidence = [row for row in results if row["policy"] == "sac"]
    baseline_drawdowns = {split: min(row["metrics"]["settled_max_drawdown"] for row in results
        if row["split"] == split and row["policy"] in {"fixed", "volatility"}) for split in ("validation", "test")}
    passed = len(config.seeds) >= 5 and all(row["metrics"]["total_return"] > 0
        and row["metrics"]["settled_max_drawdown"] <= baseline_drawdowns[row["split"]]
        and all(pair["paired_log_return_ci_low"] > 0 for pair in row["paired_improvement"].values()) for row in evidence)
    summary = {"results": results, "incremental_value_passed": passed,
        "stage4": "eligible_for_separate_research_design" if passed else "locked",
        "free_exit_enabled": False, "live_eligible": False}
    write_json_atomic(output / "comparison.json", summary)
    build_artifact_manifest(output)
    return summary
