"""固定規則候選研究與模型拒單診斷，不搜尋測試門檻、不部署模型。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai_quant_trading.operations.integrity import build_artifact_manifest, sha256_file, verify_artifact_manifest
from ai_quant_trading.persistence import write_json_atomic
from ai_quant_trading.transformer.economics import block_confidence_interval
from ai_quant_trading.transformer.strategy_events import (
    StrategyEventConfig, prepare_event_frame, replay_event, select_event_trades,
)


@dataclass(frozen=True)
class CandidateRule:
    """研究開始前固定的規則，不根據績效回頭搜尋參數。"""

    name: str
    entry: str = "breakout"
    stop_atr: float = 2.0
    target_atr: float = 4.0
    holding_bars: int = 32
    regime_exit: str = "loss"
    cost_to_stop_cap: float | None = None


RULES = (
    CandidateRule("R0_original"),
    CandidateRule("R1_nearer_target", target_atr=2.0),
    CandidateRule("R2_longer_hold", holding_bars=64),
    CandidateRule("R3_exit_on_opposite", regime_exit="opposite"),
    CandidateRule("R4_cost_budget", cost_to_stop_cap=0.25),
    CandidateRule("R5_trend_pullback", entry="pullback"),
    CandidateRule("R6_range_sweep", entry="range_sweep", stop_atr=1.5, target_atr=1.5,
                  holding_bars=16, regime_exit="disabled"),
)
COSTS = {
    "zero": {"fee": 0.0, "slippage": 0.0, "spread": 0.0, "funding": 0.0},
    "base": {"fee": 5.0, "slippage": 2.0, "spread": 1.0, "funding": 1.0},
    "stress": {"fee": 5.0, "slippage": 5.0, "spread": 1.0, "funding": 1.0},
}


def rule_frame(prepared: pd.DataFrame, rule: CandidateRule) -> pd.DataFrame:
    """訊號只依當根收盤可知資訊，壓力成本情境也使用相同訊號集合。"""
    result = prepared.copy()
    close, opening = result["close"], result["open"]
    regime = result["event_regime"]
    if rule.entry == "breakout":
        side = result["candidate_side"].to_numpy(copy=True)
    elif rule.entry == "pullback":
        ema20 = close / (1 + result["mtf_15m_ema20_gap"])
        long = (regime == 1) & (close.shift(1) <= ema20.shift(1)) & (close > ema20) & (close > opening)
        short = (regime == -1) & (close.shift(1) >= ema20.shift(1)) & (close < ema20) & (close < opening)
        side = np.select([long, short], [1, -1], 0)
    elif rule.entry == "range_sweep":
        high = result["high"].shift(1).rolling(20).max()
        low = result["low"].shift(1).rolling(20).min()
        quiet = result["mtf_1h_adx_14"] < 0.20
        long = quiet & (result["low"] < low) & (close > low) & (close > opening)
        short = quiet & (result["high"] > high) & (close < high) & (close < opening)
        side = np.select([long & ~short, short & ~long], [1, -1], 0)
    else:
        raise ValueError("未知的候選進場規則")
    warm = result.timestamp + pd.Timedelta(minutes=15) >= result.timestamp.iloc[0] + pd.Timedelta(minutes=15, hours=1000)
    valid = warm & np.isfinite(result.event_atr) & (result.event_atr > 0)
    if rule.cost_to_stop_cap is not None:
        # 固定使用基本情境預算：往返 15 bps + 一次 1 bps funding；不是已知未來成本。
        stop_fraction = rule.stop_atr * result.event_atr / close
        valid &= 0.0016 / stop_fraction <= rule.cost_to_stop_cap
    result["candidate_side"] = np.where(valid, side, 0)
    return result


def replay_window(frame: pd.DataFrame, rule: CandidateRule, cost: dict,
                  start: int, stop: int) -> pd.DataFrame:
    """共用 65 棒尾端留白；逐筆回放，不用未來收益挑選事件。"""
    if not 0 <= start < stop <= len(frame):
        raise ValueError("研究時間窗口不合法")
    config = StrategyEventConfig(stop_atr=rule.stop_atr, target_atr=rule.target_atr,
        max_holding_bars=rule.holding_bars, spread_bps=cost["spread"],
        funding_reserve_bps_per_settlement=cost["funding"])
    points = np.flatnonzero(frame.candidate_side.to_numpy())
    points = points[(points >= start) & (points + 65 < stop)]
    next_signal = start
    results = []
    for point in points:
        if point < next_signal:
            continue
        trade = replay_event(frame, int(point), config, fee_bps_per_side=cost["fee"],
                             slippage_bps_per_side=cost["slippage"], regime_exit=rule.regime_exit)
        trade["signal_at"] = frame.iloc[point].timestamp.isoformat()
        trade["holding_bars"] = trade["exit_endpoint"] - trade["entry_endpoint"]
        last = frame.iloc[int(trade["exit_endpoint"])]
        trade["ambiguous_stop_target_bar"] = bool(
            trade["exit_reason"] == "stop" and last.low <= min(trade["stop_price"], trade["target_price"])
            and last.high >= max(trade["stop_price"], trade["target_price"]))
        side = trade["side"]
        friction = (cost["slippage"] + cost["spread"] / 2) / 10_000
        reference_entry = trade["entry_price"] / (1 + side * friction)
        reference_exit = trade["exit_price"] / (1 - side * friction)
        trade["gross_same_fills"] = side * (reference_exit / reference_entry - 1)
        trade["slippage_spread_return"] = (trade["gross_same_fills"] - trade["net_return"]
                                            - trade["fee_return"] - trade["funding_return"])
        results.append(trade)
        next_signal = int(trade["exit_endpoint"]) + config.cooldown_bars + 1
    return pd.DataFrame(results, columns=(list(results[0]) if results else [
        "endpoint", "entry_endpoint", "exit_endpoint", "net_return", "side", "exit_reason",
        "holding_bars", "gross_same_fills", "fee_return", "funding_return", "slippage_spread_return",
        "ambiguous_stop_target_bar", "signal_at"]))


def expectancy(trades: pd.DataFrame, column: str = "net_return") -> dict:
    values = trades[column].to_numpy(dtype=float)
    if not len(values):
        return {"trades": 0, "mean": None, "ci_low": None, "ci_high": None}
    if not np.isfinite(values).all():
        raise ValueError("交易收益含非有限值")
    low, high = block_confidence_interval(values, repetitions=2000)
    winners, losers = values[values > 0], values[values < 0]
    gain = float(winners.mean()) if len(winners) else 0.0
    loss = float(-losers.mean()) if len(losers) else 0.0
    return {"trades": len(values), "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else None,
        "worst": float(values.min()), "ci_low": low if len(values) > 1 else None,
        "ci_high": high if len(values) > 1 else None, "win_rate": float((values > 0).mean()),
        "mean_win": gain, "mean_loss_magnitude": loss,
        "realized_payoff_ratio": gain / loss if loss > 0 else None,
        "descriptive_breakeven_win_rate": loss / (gain + loss) if gain + loss else None,
        "profit_factor": float(winners.sum() / -losers.sum()) if len(losers) else None}


def trade_diagnostics(trades: pd.DataFrame) -> dict:
    result = expectancy(trades)
    if not len(trades):
        return result
    result.update({name: float(trades[name].mean()) for name in (
        "gross_same_fills", "fee_return", "funding_return", "slippage_spread_return", "holding_bars")})
    result["ambiguous_stop_target_bars"] = int(trades.ambiguous_stop_target_bar.sum())
    result["exits"] = {str(key): expectancy(group) for key, group in trades.groupby("exit_reason")}
    result["sides"] = {str(key): expectancy(group) for key, group in trades.groupby("side")}
    return result


def archived_model_diagnostics(study: Path) -> dict:
    """只讀凍結預測，區分機率閘門與收益閘門；不回頭調整任何門檻。"""
    verified = verify_artifact_manifest(study)
    records = json.loads((study / "records.json").read_text(encoding="utf-8"))
    audit = json.loads((study / "audit.json").read_text(encoding="utf-8"))
    output = []
    pooled: dict[tuple, list[pd.DataFrame]] = {}
    for record in records:
        path = study / record["run_dir"] / "test_predictions.csv"
        expected = audit["prediction_sha256"][path.relative_to(study).as_posix()]
        if sha256_file(path) != expected:
            raise ValueError("歷史模型預測被修改，不能用於本次診斷")
        summary = json.loads((path.parent / "training.json").read_text(encoding="utf-8"))
        frame = pd.read_csv(path)
        horizon = int(summary["model_config"]["primary_horizon"])
        reference = summary["event_reference"]
        actual, predicted = frame[f"actual_return_{horizon}"], frame[f"predicted_return_{horizon}"]
        probability = frame[f"tradeability_probability_{horizon}"]
        policies = {}
        for policy in ("original", "return_only", "probability_only"):
            changed = frame.copy()
            if policy == "return_only":
                changed[f"tradeability_probability_{horizon}"] = 1.0
            elif policy == "probability_only":
                changed[f"predicted_return_{horizon}"] = 1.0
            config = StrategyEventConfig(**summary["training_config"]["strategy_event_config"])
            trades, reasons = select_event_trades(changed, horizon, config, 2.0, filtered=True)
            policies[policy] = {**expectancy(trades, f"actual_return_{horizon}"), "rejections": reasons}
            if record["stage"] == "walk_forward":
                pooled.setdefault((record["variant"], record["seed"], policy), []).append(trades)
        output.append({"stage": record["stage"], "fold": record["fold"], "variant": record["variant"],
            "seed": record["seed"], "candidates": len(frame), "policies": policies,
            "train_mean": reference["return_mean"], "train_positive_rate": reference["positive_probability"],
            "actual_mean": float(actual.mean()), "predicted_mean": float(predicted.mean()),
            "actual_std": float(actual.std(ddof=0)), "predicted_std": float(predicted.std(ddof=0)),
            "actual_positive_rate": float(frame[f"actual_tradeability_{horizon}"].mean()),
            "probability_min": float(probability.min()), "probability_max": float(probability.max()),
            "probability_mean": float(probability.mean()), "prediction_above_2bps": int((predicted > 0.0002).sum()),
            "probability_above_55pct": int((probability >= 0.55).sum()),
            "return_skill": record["test"]["metrics"]["event_return_skill"],
            "probability_skill": record["test"]["metrics"]["event_probability_skill"]})
    combined = []
    for (variant, seed, policy), parts in pooled.items():
        trades = pd.concat(parts, ignore_index=True)
        if len(trades) and trades.timestamp_ns.duplicated().any():
            raise ValueError("模型診斷的向前交易重複")
        combined.append({"variant": variant, "seed": seed, "policy": policy,
                         **expectancy(trades, "actual_return_32")})
    return {"runs": output, "walk_forward_pooled": combined, "deployment_changed": False,
        "verified_archive_files": len(verified),
        "scope": "移除單一閘門只是凍結預測的反事實診斷，不是測試集最佳化或可部署的新策略。"}


def run_candidate_research(source: Path, model_study: Path, output: Path) -> Path:
    if output.exists():
        raise FileExistsError("研究輸出目錄已存在，禁止覆寫")
    source_hash = sha256_file(source)
    frame = pd.read_csv(source)
    rows = len(frame)
    if rows < 20_000:
        raise ValueError("研究資料太少")
    end_development = int(rows * 0.6)
    windows = [{"name": "development", "start": 0, "stop": end_development}]
    windows += [{"name": f"forward_{index}", "start": int(rows * start) + 32, "stop": int(rows * stop)}
                for index, (start, stop) in enumerate(((.6, .7), (.7, .8), (.8, .9)), 1)]
    root = Path(__file__).resolve().parents[3]
    output.mkdir(parents=True, exist_ok=False)
    plan = {"created_at": datetime.now(timezone.utc).isoformat(), "source_sha256": source_hash,
        "source_rows": rows, "source": str(source.resolve()), "rules": [asdict(rule) for rule in RULES],
        "model_study": str(model_study.resolve()),
        "model_records_sha256": sha256_file(model_study / "records.json"),
        "model_audit_sha256": sha256_file(model_study / "audit.json"),
        "costs_bps": COSTS, "windows": windows, "common_tail_purge_bars": 65,
        "seed_bootstrap": 20260923, "bootstrap_repetitions": 2000,
        "selection": "只在 development 基本情境 >=100 筆且均值 95% 下界 >0，並通過壓力成本平均 >0，才列候選；依基本均值排序。",
        "forward_policy": "全部預先登記規則皆呈現；不依 forward 排名挑部署策略。",
        "pristine_holdout": False, "last_10pct_unused_this_study": True, "live_eligible": False,
        "code_sha256": {str(path.relative_to(root)): sha256_file(path) for path in (
            Path(__file__), root / "src/ai_quant_trading/transformer/strategy_events.py",
            root / "src/ai_quant_trading/transformer/economics.py")},
        "references": ["https://www.aqr.com/insights/datasets/time-series-momentum-original-paper-data",
                       "https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf"],
        "limitations": ["規則假說，不宣稱文獻證明 BTC 15m 有效", "funding 是雙向準備金，稅後未驗證",
                        "全部歷史已在先前研究被觀察，屬回溯研究", "每筆名目收益，不是投資組合年化績效"]}
    write_json_atomic(output / "plan.json", plan)
    try:
        # 留白以外的高時間框由完整過去資訊因果重採樣，不用未來資料擬合任何參數。
        prepared = prepare_event_frame(frame.iloc[:windows[-1]["stop"]].copy(), StrategyEventConfig())
        if sha256_file(source) != source_hash:
            raise ValueError("來源資料在讀取過程中變動")
        frames = {rule.name: rule_frame(prepared, rule) for rule in RULES}
        records, pooled, selection = [], {}, None
        for window in windows:
            for rule in RULES:
                for scenario, cost in COSTS.items():
                    trades = replay_window(frames[rule.name], rule, cost, window["start"], window["stop"])
                    name = f"{window['name']}_{rule.name}_{scenario}.csv"
                    trades.to_csv(output / name, index=False)
                    records.append({"window": window["name"], "rule": rule.name, "scenario": scenario,
                                    "path": name, "metrics": trade_diagnostics(trades)})
                    if window["name"] != "development":
                        pooled.setdefault((rule.name, scenario), []).append(trades)
                    write_json_atomic(output / "progress.json", {"status": "running", "completed": len(records),
                        "total": len(windows) * len(RULES) * len(COSTS), "current": name})
                print(f"{window['name']} {rule.name} 完成", flush=True)
            if window["name"] == "development":
                eligible = []
                for rule in RULES:
                    base = next(row["metrics"] for row in records if row["rule"] == rule.name and row["scenario"] == "base")
                    stress = next(row["metrics"] for row in records if row["rule"] == rule.name and row["scenario"] == "stress")
                    if base["trades"] >= 100 and base["ci_low"] > 0 and stress["trades"] > 0 and stress["mean"] > 0:
                        eligible.append((rule.name, base["mean"]))
                selection = {"eligible": [name for name, _ in eligible],
                             "selected": max(eligible, key=lambda pair: pair[1])[0] if eligible else None,
                             "development_only": True, "live_eligible": False}
                write_json_atomic(output / "selection.json", selection)
        combined = []
        for (rule, scenario), parts in pooled.items():
            trades = pd.concat(parts, ignore_index=True)
            if len(trades) and trades.endpoint.duplicated().any():
                raise ValueError("向前重播出現重複決策")
            name = f"pooled_{rule}_{scenario}.csv"
            trades.to_csv(output / name, index=False)
            combined.append({"rule": rule, "scenario": scenario, "path": name, "metrics": trade_diagnostics(trades)})
        if sha256_file(source) != source_hash:
            raise ValueError("研究期間資料變動")
        summary = {"status": "complete", "selection": selection, "results": records, "pooled_forward": combined,
                   "model_diagnostics": archived_model_diagnostics(model_study), "live_eligible": False,
                   "source_unchanged": True, "scope": "固定規則研究，不訓練、不接入 SAC、不更動部署"}
        write_json_atomic(output / "research.json", summary)
        write_json_atomic(output / "progress.json", {"status": "complete", "completed": len(records), "total": 84})
        build_artifact_manifest(output, [path for path in output.rglob("*") if path.is_file()])
    except BaseException as error:
        write_json_atomic(output / "progress.json", {"status": "failed", "error_type": type(error).__name__})
        raise
    return output / "research.json"
