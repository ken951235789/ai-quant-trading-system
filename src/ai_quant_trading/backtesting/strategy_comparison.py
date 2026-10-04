"""在保存的 RL 樣本外資料上，以同一撮合與風控比較固定規則。"""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ai_quant_trading.features.intraday import _directional_movement, _rsi
from ai_quant_trading.operations.integrity import build_artifact_manifest, sha256_file
from ai_quant_trading.persistence import write_json_atomic
from ai_quant_trading.reinforcement_learning.config import PortfolioEnvConfig
from ai_quant_trading.reinforcement_learning.policy import load_rl_policy
from ai_quant_trading.reinforcement_learning.training import (
    _full_horizon_evaluation_config, evaluate_rl_model,
)
from ai_quant_trading.transformer.strategy_events import validate_event_bars


RULES = {
    "cash": "持續空手，收益基準，不是成功的交易模型。",
    "long_risk_managed": "空手時建立多單，同向續抱；仍有共同停損停利，不是純買入持有。",
    "ema_trend": "EMA20>EMA50 且 close>EMA200 做多，反向做空；否則平倉。",
    "donchian_breakout": "收盤突破前20根高低進場，跌破／突破前10根反側平倉。",
    "rsi_reversion": "RSI14<30 多、>70 空；回到50平倉。",
    "bollinger_reversion": "20根、母體標準差2倍的布林通道外反向進場，回到中線平倉。",
    "macd_trend": "MACD12/26/9 柱體與價格相對EMA200同向時交易，條件失效平倉。",
    "vwap_reversion": "ADX14<20且偏離當日UTC累積VWAP超過1ATR反向進場，回到VWAP或ADX>=25平倉。",
    "hourly_trend": "已收盤1h EMA20/50決定方向；高週期收盤前不更新。",
    "transformer_return_only": "保存的5根毛收益預測超過固定往返15bps+2bps安全邊際才順向進場，否則平倉。",
}


def indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """只用當時已知價格，暖機可以讀取測試前資料但不能讀未來標籤。"""
    validate_event_bars(frame)
    result = frame.copy().reset_index(drop=True)
    result["timestamp"] = pd.to_datetime(result.timestamp, utc=True)
    close = result.close
    for n in (12, 20, 26, 50, 200):
        result[f"ema{n}"] = close.ewm(span=n, adjust=False, min_periods=n).mean()
    result["rsi"] = _rsi(close)
    result["bb_z"] = (close - close.rolling(20).mean()) / close.rolling(20).std(ddof=0).replace(0, np.nan)
    macd = result.ema12 - result.ema26
    result["macd_hist"] = macd - macd.ewm(span=9, adjust=False, min_periods=9).mean()
    result["atr"], result["adx"], _, _ = _directional_movement(result.high, result.low, close)
    for n in (10, 20):
        result[f"prior_high{n}"] = result.high.shift(1).rolling(n).max()
        result[f"prior_low{n}"] = result.low.shift(1).rolling(n).min()
    day = result.timestamp.dt.floor("D")
    typical = (result.high + result.low + close) / 3
    result["vwap"] = (typical * result.volume).groupby(day).cumsum() / result.volume.groupby(day).cumsum().replace(0, np.nan)
    bars = result.set_index("timestamp")["close"]
    hourly = bars.resample("1h").last().loc[bars.resample("1h").count().eq(4)]
    trend = np.sign(hourly.ewm(span=20, adjust=False, min_periods=20).mean()
                    - hourly.ewm(span=50, adjust=False, min_periods=50).mean())
    trend.index += pd.Timedelta(hours=1)
    result["hourly_direction"] = trend.reindex(result.timestamp + pd.Timedelta(minutes=15), method="ffill").to_numpy()
    return result


def rule_direction(name: str, row: pd.Series, position: float) -> float:
    """回傳 -1／0／1 目標方向；續抱實際成交部位而非假想已成交訊號。"""
    current = float(np.sign(position)) if abs(position) > 1e-8 else 0.0
    if name == "cash":
        return 0.0
    if name == "long_risk_managed":
        return 1.0
    if name == "ema_trend":
        return 1.0 if row.ema20 > row.ema50 and row.close > row.ema200 else (
            -1.0 if row.ema20 < row.ema50 and row.close < row.ema200 else 0.0)
    if name == "macd_trend":
        return 1.0 if row.macd_hist > 0 and row.close > row.ema200 else (
            -1.0 if row.macd_hist < 0 and row.close < row.ema200 else 0.0)
    if name == "hourly_trend":
        return float(row.hourly_direction) if np.isfinite(row.hourly_direction) else 0.0
    if name == "transformer_return_only":
        value = float(row.expected_return)
        return float(np.sign(value)) if abs(value) > 0.0017 else 0.0
    if name == "donchian_breakout":
        if current > 0:
            return 0.0 if row.close < row.prior_low10 else current
        if current < 0:
            return 0.0 if row.close > row.prior_high10 else current
        return 1.0 if row.close > row.prior_high20 else (-1.0 if row.close < row.prior_low20 else 0.0)
    if name in {"rsi_reversion", "bollinger_reversion", "vwap_reversion"}:
        if name == "rsi_reversion":
            value, boundary = row.rsi - 50, 20.0
        elif name == "bollinger_reversion":
            value, boundary = row.bb_z, 2.0
        else:
            if row.adx >= 0.25 or (current == 0 and row.adx >= 0.20):
                return 0.0
            value, boundary = (row.close - row.vwap) / row.atr, 1.0
        if not np.isfinite(value):
            return 0.0
        if current:
            return 0.0 if current * value >= 0 else current
        return 1.0 if value < -boundary else (-1.0 if value > boundary else 0.0)
    raise ValueError(f"未登記的策略：{name}")


class RulePolicy:
    """將規則意圖轉成模型原有動作語意，不把 0 誤當成平倉。"""

    def __init__(self, name: str, frame: pd.DataFrame, features: int, config: PortfolioEnvConfig):
        if name not in RULES:
            raise ValueError("未登記的策略")
        self.name, self.frame, self.features, self.config = name, frame, features, config
        self.index = 0

    def predict(self, observation: np.ndarray, *, deterministic: bool) -> tuple[np.ndarray, None]:
        del deterministic
        position = float(observation[self.features + 1])
        direction = rule_direction(self.name, self.frame.iloc[self.index], position)
        self.index += 1
        config = self.config
        if config.action_semantics == "hold_close_target":
            if direction == 0:
                action = (config.hold_action_threshold + config.close_action_threshold) / 2
            elif direction * position > 0:
                action = 0.0
            else:
                action = direction if config.normalized_action_space else direction * (
                    config.max_position_fraction if direction > 0 else config.max_short_fraction)
        else:
            target = position if direction * position > 0 else direction * (
                config.max_position_fraction if direction > 0 else config.max_short_fraction)
            action = target / (config.max_position_fraction if target >= 0 else config.max_short_fraction) \
                if config.normalized_action_space else target
        return np.array([action], dtype=np.float32), None


def cost_configs(base: PortfolioEnvConfig) -> dict[str, PortfolioEnvConfig]:
    """相同風控與部位上限，只改成交成本；不改訓練或部署設定。"""
    return {
        "native": base,
        "base": replace(base, fee_rate=.0005, slippage_rate=.0002, spread_rate=.0001),
        "stress": replace(base, fee_rate=.0005, slippage_rate=.0005, spread_rate=.0001),
        "zero": replace(base, fee_rate=0, slippage_rate=0, spread_rate=0,
                        short_borrow_rate_annual=0, liquidation_fee_rate=0),
    }


def summarize(evaluation: pd.DataFrame, metrics: dict, initial_capital: float) -> dict:
    """保留原引擎指標，但空手不能宣稱勝率，日收益取代高度自相關的K棒推論。"""
    result = dict(metrics)
    closed = evaluation.loc[evaluation.trade_closed.astype(bool), "closed_trade_pnl"]
    if closed.empty:
        for key in ("win_rate", "expectancy", "payoff_ratio", "profit_factor"):
            result[key] = None
    else:
        losses, wins = closed[closed < 0], closed[closed > 0]
        result["profit_factor"] = float(wins.sum() / -losses.sum()) if len(losses) else None
    index = pd.to_datetime(evaluation.timestamp, utc=True) + pd.Timedelta(minutes=15)
    daily = pd.Series(evaluation.equity.to_numpy(), index=index).resample("D").last().dropna()
    returns = daily.pct_change()
    returns.iloc[0] = daily.iloc[0] / initial_capital - 1
    reasons = evaluation.risk_reasons.fillna("").astype(str).str.split(",").explode()
    result.update({
        "daily_return_std": float(returns.std(ddof=1)), "worst_day": float(returns.min()),
        "best_day": float(returns.max()), "daily_returns": {str(k): float(v) for k, v in returns.items()},
        "raw_action_min": float(evaluation.action.min()), "raw_action_max": float(evaluation.action.max()),
        "mean_absolute_exposure": float(evaluation.position_fraction.abs().mean()),
        "risk_reason_counts": {str(k): int(v) for k, v in reasons[reasons.ne("")].value_counts().items()},
        "ending_quantity": float(evaluation.quantity.iloc[-1]),
        "expectancy_units": "USDT per closed position, not per order or notional percentage",
        "risk_terminated": bool(evaluation.risk_terminated.any()),
    })
    return result


def run_comparison(training_dir: Path, output: Path) -> Path:
    if output.exists():
        raise FileExistsError("比較目錄已存在，禁止覆寫")
    import torch

    torch.set_num_threads(2)
    policy = load_rl_policy(training_dir, device="cpu")
    config = policy.env_config
    if not config.allow_short or config.execution_mode != "perpetual":
        raise ValueError("本研究只接受 BTC 永續多空模型")
    source = policy.environment_dir
    frames = {name: pd.read_csv(source / f"{name}.csv") for name in ("train", "validation", "test")}
    for name, frame in frames.items():
        validate_event_bars(frame)
        if len(frame) != policy.environment_metadata["split_rows"][name]:
            raise ValueError("保存的 split 筆數不符")
        if not np.isfinite(frame[policy.feature_columns].to_numpy(dtype=float)).all():
            raise ValueError("模型輸入含非有限值")
        if not np.isfinite(frame.expected_return.to_numpy(dtype=float)).all():
            raise ValueError("缺少保存的 Transformer 原始收益")
    joined = pd.concat(frames.values(), ignore_index=True)
    validate_event_bars(joined)
    cutoff = pd.Timestamp(policy.environment_metadata["ai_context"]["transformer_provenance"]["safe_rl_start_exclusive"])
    if pd.Timestamp(joined.timestamp.iloc[0]) <= cutoff:
        raise ValueError("RL 使用的 Transformer 輸入早於安全起點")
    if policy.environment_metadata["normalization"]["fit_on"] != "train_only":
        raise ValueError("標準化不符 train-only 契約")
    observed = policy.model.observation_space.shape[0]
    if observed != policy.environment_metadata["observation_size"]:
        raise ValueError("模型輸入維度與保存環境不同")
    prepared = indicators(joined)
    offsets = {"validation": len(frames["train"]), "test": len(frames["train"]) + len(frames["validation"])}
    configurations = cost_configs(config)
    model_path = policy.training_dir / "best_model" / "best_model.zip"
    if not model_path.exists():
        model_path = policy.training_dir / "final_model.zip"
    inputs = [source / f"{name}.csv" for name in frames] + [source / "environment.json",
              policy.training_dir / "training.json", model_path]
    project = Path(__file__).resolve().parents[3]
    code = [Path(__file__), project / "src/ai_quant_trading/reinforcement_learning/environment.py",
            project / "src/ai_quant_trading/reinforcement_learning/training.py",
            project / "src/ai_quant_trading/reinforcement_learning/actions.py",
            project / "src/ai_quant_trading/features/intraday.py"]
    plan = {"created_at": datetime.now(timezone.utc).isoformat(), "rules": RULES,
        "training_dir": str(policy.training_dir), "model_sha256": sha256_file(model_path),
        "model_training_config": policy.training_metadata["training_config"],
        "source_sha256": {str(p.resolve()): sha256_file(p) for p in inputs},
        "code_sha256": {str(p.relative_to(project)): sha256_file(p) for p in code},
        "windows": {name: {"rows": len(frame), "start": str(frame.timestamp.iloc[0]),
                            "end": str(frame.timestamp.iloc[-1])} for name, frame in frames.items()},
        "environment_configs": {k: asdict(_full_horizon_evaluation_config(v)) for k, v in configurations.items()},
        "selection": "全部固定策略呈現，不依 validation 或 test 選新 Champion；不重訓。",
        "policy_names": ["sac", *RULES], "seed": 11, "pristine_holdout": False,
        "limitations": ["單一SAC seed與約11天Test，不能推論長期勝負",
            "原生完整回測會在原硬回撤後近零曝險，非精確停止；保留此共同語意並揭露",
            "funding沿用保存資料按持有時間比例攤提，非精確結算時刻撮合",
            "期末按市值計價，未強制平倉，因此未扣剩餘部位未來出場成本",
            "native亦使用修正後保護單滑價及進場費損益，非舊引擎逐位元重現",
            "完整行情重採樣只供規則暖機，不改SAC保存的標準化輸入"],
        "live_eligible": False}
    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output / "plan.json", plan)
    rows = []
    try:
        for split in ("validation", "test"):
            raw = frames[split].copy()
            features = prepared.iloc[offsets[split]:offsets[split] + len(raw)].reset_index(drop=True)
            for scenario, current in configurations.items():
                frame = raw.copy()
                if scenario != "native":
                    frame["spread_bps"] = current.spread_rate * 10_000
                if scenario == "zero":
                    frame["funding_rate"] = 0.0
                for name in plan["policy_names"]:
                    agent = policy.model if name == "sac" else RulePolicy(name, features, len(policy.feature_columns), current)
                    evaluation, metrics = evaluate_rl_model(agent, frame, policy.feature_columns, current,
                                                            deterministic=True, seed=11)
                    if len(evaluation) != len(frame) - 1:
                        raise ValueError("提前終止，不能將不同期間直接比較；請先檢查清算風險")
                    path = f"{split}_{scenario}_{name}.csv"
                    evaluation.to_csv(output / path, index=False)
                    rows.append({"split": split, "scenario": scenario, "policy": name, "path": path,
                                 "metrics": summarize(evaluation, metrics, current.initial_capital)})
                    write_json_atomic(output / "progress.json", {"status": "running", "completed": len(rows),
                        "total": 88, "current": path})
                print(f"{split} {scenario}：11 組完成", flush=True)
        for path, expected in plan["source_sha256"].items():
            if sha256_file(Path(path)) != expected:
                raise ValueError("研究期間來源被修改")
        summary = {"status": "complete", "results": rows, "live_eligible": False,
                   "source_unchanged": True, "new_training": False}
        write_json_atomic(output / "comparison.json", summary)
        write_json_atomic(output / "progress.json", {"status": "complete", "completed": len(rows), "total": 88})
        build_artifact_manifest(output, [p for p in output.iterdir() if p.is_file()])
    except BaseException as error:
        write_json_atomic(output / "progress.json", {"status": "failed", "error_type": type(error).__name__})
        raise
    return output / "comparison.json"
