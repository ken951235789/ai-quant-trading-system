"""低換手候選研究：先凍結假說、再檢查成本後期望，不自動部署策略。"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai_quant_trading.backtesting.candidate_research import (
    COSTS, CandidateRule, expectancy, replay_window, rule_frame, trade_diagnostics,
)
from ai_quant_trading.operations.integrity import build_artifact_manifest, sha256_file, verify_artifact_manifest
from ai_quant_trading.persistence import write_json_atomic
from ai_quant_trading.transformer.strategy_events import StrategyEventConfig, prepare_event_frame


# 固定四個新假說，不做參數網格；原策略使用相同期間及成本重新回放。
RULES = (
    CandidateRule("C0_original"),
    CandidateRule("C1_hourly_breakout_loss", entry="hourly_breakout", holding_bars=64),
    CandidateRule("C2_hourly_breakout_opposite", entry="hourly_breakout", holding_bars=64, regime_exit="opposite"),
    CandidateRule("C3_hourly_pullback_loss", entry="hourly_pullback", holding_bars=64),
    CandidateRule("C4_hourly_pullback_opposite", entry="hourly_pullback", holding_bars=64, regime_exit="opposite"),
)


def hourly_entries(hourly: pd.DataFrame, entry: str) -> pd.Series:
    """只使用已收盤小時棒；突破區間不包含訊號棒本身。"""
    close = hourly.close
    if entry == "hourly_breakout":
        long = close > hourly.high.shift(1).rolling(20).max()
        short = close < hourly.low.shift(1).rolling(20).min()
    elif entry == "hourly_pullback":
        ema = close.ewm(span=20, adjust=False, min_periods=20).mean()
        long = (close.shift(1) <= ema.shift(1)) & (close > ema) & (close > hourly.open)
        short = (close.shift(1) >= ema.shift(1)) & (close < ema) & (close < hourly.open)
    else:
        raise ValueError("未知的小時進場規則")
    return pd.Series(np.select([long & ~short, short & ~long], [1, -1], 0), index=hourly.index)


def prepare_rule(prepared: pd.DataFrame, rule: CandidateRule) -> pd.DataFrame:
    if rule.entry == "breakout":
        return rule_frame(prepared, rule)
    result = prepared.copy()
    bars = result.set_index("timestamp")
    grouped = bars.resample("1h", closed="left", label="left")
    hourly = grouped.agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    hourly = hourly.loc[grouped.close.count().eq(4)]
    signals = hourly_entries(hourly, rule.entry)
    signals.index = signals.index + pd.Timedelta(hours=1)
    hourly.index = hourly.index + pd.Timedelta(hours=1)
    decisions = pd.DatetimeIndex(result.timestamp + pd.Timedelta(minutes=15))
    aligned_signal = signals.reindex(decisions, fill_value=0).to_numpy()
    # 比例乘回當時已收盤小時價格，不能乘變動中的 15m 價格而漂移 ATR。
    hourly_close = hourly.close.reindex(decisions, method="ffill").to_numpy()
    result["event_atr"] = result.mtf_1h_atr_pct * hourly_close
    warm = decisions >= decisions[0] + pd.Timedelta(hours=1000)
    valid = warm & np.isfinite(result.event_atr) & result.event_atr.gt(0)
    valid &= aligned_signal == result.event_regime
    result["candidate_side"] = np.where(valid, aligned_signal, 0)
    return result


def research_windows(rows: int) -> list[dict]:
    if rows < 20_000:
        raise ValueError("候選研究至少需要 20,000 棒")
    windows = [{"name": "development", "start": 0, "stop": int(rows * .6)}]
    windows += [{"name": f"forward_{i}", "start": int(rows * a) + 64, "stop": int(rows * b)}
                for i, (a, b) in enumerate(((.6, .7), (.7, .8), (.8, .9)), 1)]
    return windows


def enrich_trades(trades: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """補上可追溯時間與風險單位；不把未來結果帶回進場條件。"""
    result = trades.copy()
    for name, column in (("entry_at", "entry_endpoint"), ("exit_at", "exit_endpoint")):
        result[name] = [frame.timestamp.iloc[int(point)].isoformat() for point in result[column]]
    if result.empty:
        result["planned_stop_fraction"] = pd.Series(dtype=float)
        result["net_r"] = pd.Series(dtype=float)
    else:
        result["planned_stop_fraction"] = abs(result.entry_price - result.stop_price) / result.entry_price
        if not result.planned_stop_fraction.gt(0).all():
            raise ValueError("計畫停損距離必須為正")
        result["net_r"] = result.net_return / result.planned_stop_fraction
    return result


def measure(trades: pd.DataFrame, window: dict) -> dict:
    metrics = trade_diagnostics(trades)
    metrics["net_r"] = expectancy(trades, "net_r")
    metrics["trades_per_30_days"] = len(trades) / (window["stop"] - window["start"]) * 96 * 30
    metrics["calendar_days"] = (window["stop"] - window["start"]) / 96
    metrics["mean_cost_to_stop"] = None if trades.empty else float(
        ((trades.fee_return + trades.funding_return + trades.slippage_spread_return) / trades.planned_stop_fraction).mean())
    # 開發期三段穩定性只作開發選擇；按進場切分，出場越界的事件不計入該子段。
    edges = np.linspace(window["start"], window["stop"], 4, dtype=int)
    metrics["subperiods"] = [expectancy(trades.loc[trades.entry_endpoint.ge(a) & trades.exit_endpoint.lt(b)])
                              for a, b in zip(edges[:-1], edges[1:])]
    return metrics


def select_development(records: list[dict]) -> dict:
    development = [row for row in records if row["window"] == "development"]
    expected = {(rule.name, scenario) for rule in RULES for scenario in COSTS}
    actual = [(row["rule"], row["scenario"]) for row in development]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("開發區間情境缺漏或重複")
    verdicts = {}
    for rule in RULES:
        values = {r["scenario"]: r["metrics"] for r in development if r["rule"] == rule.name}
        base, stress = values["base"], values["stress"]
        checks = {
            "at_least_100_trades": base["trades"] >= 100,
            "positive_net_ci_lower": base["ci_low"] is not None and base["ci_low"] > 0,
            "positive_stress_mean": stress["mean"] is not None and stress["mean"] > 0,
            "two_of_three_development_periods_positive": sum(
                part["trades"] >= 20 and part["mean"] > 0 for part in base["subperiods"]) >= 2,
        }
        verdicts[rule.name] = {"checks": checks, "passed": all(checks.values()), "mean": base["mean"]}
    eligible = [name for name, value in verdicts.items() if value["passed"]]
    selected = max(eligible, key=lambda name: verdicts[name]["mean"]) if eligible else None
    return {"selected": selected, "eligible": eligible, "rules": verdicts,
            "selection_source": "development_only", "live_eligible": False}


def forward_gate(records: list[dict], pooled: list[dict], selection: dict) -> dict:
    selected = selection["selected"]
    if selected is None:
        return {"passed": False, "reason": "no_development_candidate", "research_candidate": None}
    base = next(r["metrics"] for r in pooled if r["rule"] == selected and r["scenario"] == "base")
    stress = next(r["metrics"] for r in pooled if r["rule"] == selected and r["scenario"] == "stress")
    folds = [r["metrics"] for r in records if r["rule"] == selected and r["scenario"] == "base"
             and r["window"] != "development"]
    checks = {"at_least_100_forward_trades": base["trades"] >= 100,
              "positive_forward_ci_lower": base["ci_low"] is not None and base["ci_low"] > 0,
              "positive_stress_mean": stress["mean"] is not None and stress["mean"] > 0,
              "all_three_forward_periods_positive": len(folds) == 3 and all(
                  value["trades"] >= 20 and value["mean"] > 0 for value in folds)}
    return {"passed": all(checks.values()), "checks": checks,
            "research_candidate": selected if all(checks.values()) else None}


def run_research(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("研究輸出已存在，禁止覆寫")
    source_hash = sha256_file(source)
    bars = pd.read_csv(source)
    if sha256_file(source) != source_hash:
        raise ValueError("讀取期間原始資料改變")
    windows = research_windows(len(bars))
    output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[3]
    code_paths = [Path(__file__), Path(__file__).with_name("candidate_research.py"),
                  root / "src/ai_quant_trading/transformer/strategy_events.py",
                  root / "src/ai_quant_trading/transformer/economics.py",
                  root / "src/ai_quant_trading/features/intraday.py"]
    plan = {"created_at": datetime.now(timezone.utc).isoformat(), "source": str(source),
        "source_sha256": source_hash, "source_rows": len(bars),
        "source_start": str(bars.timestamp.iloc[0]), "source_end": str(bars.timestamp.iloc[-1]),
        "rules": [asdict(rule) for rule in RULES], "costs_bps": COSTS, "windows": windows,
        "hypothesis": "已收盤1h訊號與1h ATR可降低成本相對停損距離的負擔，是否提升淨期望仍待驗證",
        "factorial": "新規則2x2：突破/回調進場 x 趨勢失效/反向退出；其餘設定相同",
        "control_caveat": "相對原15m規則同時改變訊號頻率、ATR尺度、最長持有時間，不能只歸因單項",
        "entry": "next_15m_open", "protective_checks": "every_15m_stop_first",
        "new_rule_atr": "closed_1h", "common_tail_purge_bars": 65, "forward_embargo_bars": 64,
        "selection": "Development >=100 trades, mean CI low>0, stress mean>0, >=2/3 subperiods with >=20 trades and mean>0",
        "forward_gate": "Selected only: >=100 pooled trades, CI low>0, stress mean>0, all3 folds >=20 trades and mean>0",
        "bootstrap_repetitions": 2000, "bootstrap_seed": 20260923,
        "pristine_holdout": False, "tail_evaluated": False, "live_eligible": False,
        "code_sha256": {str(p.relative_to(root)): sha256_file(p) for p in code_paths},
        "references": ["https://www.nber.org/papers/w24877", "https://www.aqr.com/Insights/Research/Journal-Article/Time-Series-Momentum"],
        "limitations": ["回溯研究，不是全新封存集", "Funding是準備金；費率假設、稅後未驗證",
                        "淨期望與R值不是帳戶年化或棒內風險", "未建限價成交、容量或即時下單模型",
                        "文獻不是BTC小時規則獲利證明", "研究規則不自動改變Transformer標籤或Live"]}
    write_json_atomic(output / "plan.json", plan)
    records, pooled, combined = [], {}, []
    total = len(windows) * len(RULES) * len(COSTS)
    try:
        prepared = prepare_event_frame(bars.iloc[:windows[-1]["stop"]], StrategyEventConfig())
        frames = {rule.name: prepare_rule(prepared, rule) for rule in RULES}
        for window in windows:
            for rule in RULES:
                for scenario, cost in COSTS.items():
                    frame = frames[rule.name]
                    trades = enrich_trades(replay_window(frame, rule, cost, window["start"], window["stop"]), frame)
                    name = f"{window['name']}_{rule.name}_{scenario}.csv"
                    trades.to_csv(output / name, index=False)
                    records.append({"window": window["name"], "rule": rule.name, "scenario": scenario,
                                    "path": name, "metrics": measure(trades, window)})
                    if window["name"] != "development":
                        pooled.setdefault((rule.name, scenario), []).append(trades)
                    write_json_atomic(output / "progress.json", {"status": "running", "completed": len(records),
                        "total": total, "current": name})
                print(f"{window['name']} {rule.name} complete", flush=True)
            if window["name"] == "development":
                selection = select_development(records)
                write_json_atomic(output / "selection.json", selection)
        forward_window = {"start": windows[1]["start"], "stop": windows[-1]["stop"]}
        for (name, scenario), parts in pooled.items():
            trades = pd.concat(parts, ignore_index=True)
            if trades.endpoint.duplicated().any():
                raise ValueError("向前交易重複")
            path = f"pooled_{name}_{scenario}.csv"
            trades.to_csv(output / path, index=False)
            combined.append({"rule": name, "scenario": scenario, "path": path,
                             "metrics": measure(trades, forward_window)})
        verdict = forward_gate(records, combined, selection)
        result = {"status": "complete", "selection": selection, "results": records,
                  "pooled_forward": combined, "forward_gate": verdict, "live_eligible": False,
                  "next_stage": "new_label_contract_required" if verdict["passed"] else "blocked_by_rule_quality"}
        if sha256_file(source) != source_hash:
            raise ValueError("研究期間原始資料改變")
        for name, expected in plan["code_sha256"].items():
            if sha256_file(root / name) != expected:
                raise ValueError(f"研究期間程式改變：{name}")
        write_json_atomic(output / "research.json", result)
        write_json_atomic(output / "progress.json", {"status": "complete", "completed": len(records), "total": total})
        build_artifact_manifest(output, [p for p in output.rglob("*") if p.is_file()])
        return result
    except BaseException as error:
        write_json_atomic(output / "progress.json", {"status": "failed", "completed": len(records),
                          "total": total, "error_type": type(error).__name__})
        raise


def audit_research(output: Path) -> dict:
    """完整重播加獨立成本算式核對；不只相信摘要檔內寫著 PASS。"""
    output = output.resolve()
    verified = verify_artifact_manifest(output)
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    result = json.loads((output / "research.json").read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[3]
    if sha256_file(Path(plan["source"])) != plan["source_sha256"]:
        raise ValueError("行情來源雜湊不符")
    for name, expected in plan["code_sha256"].items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"研究程式版本改變：{name}")
    if plan["rules"] != [asdict(rule) for rule in RULES] or plan["costs_bps"] != COSTS:
        raise ValueError("研究規則或成本不符合凍結版本")
    if plan["windows"] != research_windows(plan["source_rows"]):
        raise ValueError("時間窗口不符合預登記")
    windows = {w["name"]: w for w in plan["windows"]}
    expected_keys = {(w, r.name, c) for w in windows for r in RULES for c in COSTS}
    keys = [(r["window"], r["rule"], r["scenario"]) for r in result["results"]]
    if len(keys) != len(expected_keys) or set(keys) != expected_keys:
        raise ValueError("研究情境缺漏或重複")
    bars = pd.read_csv(plan["source"])
    if len(bars) != plan["source_rows"]:
        raise ValueError("行情筆數不符")
    prepared = prepare_event_frame(bars.iloc[:plan["windows"][-1]["stop"]], StrategyEventConfig())
    frames = {rule.name: prepare_rule(prepared, rule) for rule in RULES}
    rules = {rule.name: rule for rule in RULES}
    pooled, checked_records, checked_pooled = {}, [], []
    for row in result["results"]:
        rule, window, cost = rules[row["rule"]], windows[row["window"]], COSTS[row["scenario"]]
        path = f"{row['window']}_{row['rule']}_{row['scenario']}.csv"
        if row["path"] != path:
            raise ValueError("交易檔案名稱不符合情境")
        frame = frames[rule.name]
        expected = enrich_trades(replay_window(frame, rule, cost, window["start"], window["stop"]), frame)
        actual = pd.read_csv(output / path)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=False, atol=1e-14, rtol=1e-12)
        if len(actual):
            ratio = actual.exit_price / actual.entry_price
            fees = cost["fee"] / 10_000 * (1 + ratio)
            start = pd.to_datetime(actual.entry_at, utc=True).dt.floor("8h")
            end = pd.to_datetime(actual.exit_at, utc=True).dt.floor("8h")
            funding = (end - start) / pd.Timedelta(hours=8) * cost["funding"] / 10_000
            np.testing.assert_allclose(actual.fee_return, fees, atol=1e-14)
            np.testing.assert_allclose(actual.funding_return, funding, atol=1e-14)
            np.testing.assert_allclose(actual.net_return, actual.side * (ratio - 1) - fees - funding, atol=1e-14)
            np.testing.assert_allclose(actual.net_return, actual.gross_same_fills - actual.fee_return
                                       - actual.funding_return - actual.slippage_spread_return, atol=1e-14)
        metrics = measure(expected, window)
        if metrics != row["metrics"]:
            raise ValueError("分段統計無法重現")
        checked_records.append({**row, "metrics": metrics})
        if row["window"] != "development":
            pooled.setdefault((rule.name, row["scenario"]), []).append(expected)
    pooled_keys = [(r["rule"], r["scenario"]) for r in result["pooled_forward"]]
    if len(pooled_keys) != len(pooled) or set(pooled_keys) != set(pooled):
        raise ValueError("合併情境缺漏或重複")
    forward_window = {"start": plan["windows"][1]["start"], "stop": plan["windows"][-1]["stop"]}
    for row in result["pooled_forward"]:
        path = f"pooled_{row['rule']}_{row['scenario']}.csv"
        if row["path"] != path:
            raise ValueError("合併檔案名稱不符")
        expected = pd.concat(pooled[row["rule"], row["scenario"]], ignore_index=True)
        pd.testing.assert_frame_equal(pd.read_csv(output / path), expected, check_dtype=False,
                                      check_exact=False, atol=1e-14, rtol=1e-12)
        metrics = measure(expected, forward_window)
        if metrics != row["metrics"]:
            raise ValueError("合併統計無法重現")
        checked_pooled.append({**row, "metrics": metrics})
    selection = select_development(checked_records)
    if selection != result["selection"] or selection != json.loads((output / "selection.json").read_text(encoding="utf-8")):
        raise ValueError("開發選擇無法重現")
    gate = forward_gate(checked_records, checked_pooled, selection)
    expected_next = "new_label_contract_required" if gate["passed"] else "blocked_by_rule_quality"
    if gate != result["forward_gate"] or result["next_stage"] != expected_next or result["live_eligible"]:
        raise ValueError("向前閘門或研究資格錯誤")
    return {"status": "PASS", "scenarios": len(keys), "trade_csv_count": len(keys) + len(pooled),
            "verified_artifacts": len(verified), "source_unchanged": True, "code_unchanged": True,
            "all_trades_replayed": True, "independent_fee_funding_formula_verified": True,
            "development_selection_reproduced": True, "forward_gate_reproduced": True,
            "selected": selection["selected"], "live_eligible": False}
