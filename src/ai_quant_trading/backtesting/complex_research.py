"""固定範圍複合策略比較；重用既有成交、統計及選擇契約，不自動部署。"""

from dataclasses import asdict, replace
from datetime import datetime, timezone
from itertools import groupby
import json
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd

from ai_quant_trading.backtesting.candidate_research import COSTS, expectancy
from ai_quant_trading.backtesting.complex_strategies import (
    FAMILIES, RISK_LEVELS, complex_frame, complex_grid, prepare_complex_cache,
)
from ai_quant_trading.backtesting.grid_execution import REASONS, batch_events, holding_limit, reference_check, scheduled_indices
from ai_quant_trading.backtesting.parameter_sweep import (
    ALLOCATION, _months, joint_max_statistic, search_shortlist, summarize, validation_selection, windows_for,
)
from ai_quant_trading.operations.integrity import build_artifact_manifest, sha256_file, verify_artifact_manifest
from ai_quant_trading.persistence import write_json_atomic


def complex_neighbors(grid, positive):
    lookup = {p: i for i, p in enumerate(grid)}
    levels = {"strict": (False, True), **{k: v for k, v in RISK_LEVELS.items() if k != "regime_exit"}}
    fractions = np.zeros(len(grid))
    for i, p in enumerate(grid[1:], start=1):
        neighbors = []
        for name, choices in levels.items():
            at = choices.index(getattr(p, name))
            for offset in (-1, 1):
                if 0 <= at + offset < len(choices):
                    other = replace(p, **{name: choices[at + offset]})
                    neighbors.append(bool(positive[lookup[other]]))
        fractions[i] = np.mean(neighbors)
    return fractions


def export_trades(output, cache, grid, ids, windows):
    folder = output / "trades"
    folder.mkdir()
    pooled, checked = [], 0
    for i in ids:
        p = grid[i]
        frame = complex_frame(cache, p)
        for scenario, cost in COSTS.items():
            events = holding_limit(batch_events(frame, p.execution(), cost), frame, p.holding_bars, cost)
            parts = []
            for w in windows:
                chosen = scheduled_indices(events, w["start"], w["stop"], p.cooldown_bars)
                for index in chosen:
                    reference_check(events, frame, p.execution(), cost, int(index))
                    checked += 1
                trades = pd.DataFrame({key: values[chosen] for key, values in events.items()})
                for which in ("entry", "exit"):
                    trades[f"{which}_at"] = frame.timestamp.iloc[trades[f"{which}_endpoint"].to_numpy(dtype=int)].to_numpy()
                trades["exit_reason"] = trades.reason_code.map(REASONS)
                trades["net_r"] = trades.net_return / (abs(trades.entry_price - trades.stop_price) / trades.entry_price)
                trades["equity"] = 1000 * (1 + ALLOCATION * trades.net_return).cumprod()
                trades.to_csv(folder / f"C{i:04d}_{w['name']}_{scenario}.csv", index=False)
                if w["name"].startswith("forward"):
                    parts.append(trades)
            joined = pd.concat(parts, ignore_index=True)
            joined["equity"] = 1000 * (1 + ALLOCATION * joined.net_return).cumprod()
            joined.to_csv(folder / f"C{i:04d}_forward_{scenario}.csv", index=False)
            curve = np.r_[1000., joined.equity.to_numpy()]
            pooled.append({"parameter_id": i, "scenario": scenario, **expectancy(joined),
                "net_r": expectancy(joined, "net_r"), "settled_return": float(curve[-1] / 1000 - 1),
                "settled_drawdown": float((1 - curve / np.maximum.accumulate(curve)).max())})
    return {"results": pooled, "reference_checked_trades_excluding_pooled_duplicates": checked}


def run_complex_research(source: Path, previous: Path, output: Path):
    source, previous, output = source.resolve(), previous.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("複合研究禁止覆寫既有結果，請使用新輸出目錄")
    verify_artifact_manifest(previous)
    old_plan = json.loads((previous / "plan.json").read_text(encoding="utf-8"))["fingerprint"]
    source_hash = sha256_file(source)
    if source_hash != old_plan["source_sha256"]:
        raise ValueError("與原始參數比較的來源行情不同")
    bars = pd.read_csv(source)
    windows, grid = windows_for(len(bars)), complex_grid()
    if windows != old_plan["windows"] or COSTS != old_plan["costs_bps"]:
        raise ValueError("兩次研究的期間或成本不一致")
    root = Path(__file__).resolve().parents[3]
    paths = list(old_plan["code_sha256"]) + [str(Path(__file__).relative_to(root)),
        str(Path(__file__).with_name("complex_strategies.py").relative_to(root))]
    code_hashes = {name: sha256_file(root / name) for name in paths}
    params = pd.DataFrame([{"parameter_id": i, **asdict(p), "entry": p.family} for i, p in enumerate(grid)])
    output.mkdir(parents=True)
    write_json_atomic(output / "plan.json", {"created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source), "source_sha256": source_hash, "source_rows": len(bars),
        "source_start": str(bars.timestamp.iloc[0]), "source_end": str(bars.timestamp.iloc[-1]),
        "code_sha256": code_hashes, "previous": str(previous), "windows": windows,
        "costs_bps": COSTS, "allocation": ALLOCATION, "families": FAMILIES, "parameters": len(grid),
        "risk_grid": RISK_LEVELS, "timeframes": ["15m", "1h"], "strict": [False, True],
        "selection": "每家族/週期 Search 至少150筆、stress月均log前5；Validation至少50筆再排名；Forward不得參與挑選",
        "quality": "同前輪：Search/Validation正base與stress、max-stat p<=.05、Search鄰域>=60%、結算DD<=10%",
        "forward_quality": "合格選中者>=100筆、CI下界>0、正stress、三段各>=20筆且均值>0、結算DD<=10%",
        "bootstrap": {"repetitions": 2000, "monthly_block": 3, "seed": 20261002},
        "pristine_holdout": False, "tail_evaluated": False, "live_eligible": False,
        "account_caveat": "固定10%名目配置、1000 USDT結算權益、無槓桿；無盤中清算/風控、非稅後實盤",
        "liquidity_caveat": "假突破收回只是OHLCV代理，不是訂單簿、掃停損或真實主力行為證據"})
    params.to_csv(output / "parameters.csv", index=False)
    cache = prepare_complex_cache(bars.iloc[:windows[-1]["stop"]])
    codes, months = _months(cache["raw"], windows)
    if min(len(months[name]) for name in ("search", "validation")) < 6:
        raise ValueError("Search 與 Validation 各至少需要六個月份")
    monthly = {f"{w['name']}_{c}": np.zeros((len(grid), len(months[w["name"]]))) for w in windows for c in COSTS}
    groups = [list(values) for _, values in groupby(enumerate(grid), key=lambda value: value[1].signal_key)]
    folder = output / "groups"
    folder.mkdir()
    started, probes, rows = monotonic(), 0, []
    try:
        for group_id, values in enumerate(groups):
            frame = complex_frame(cache, values[0][1])
            local_rows = []
            # 同一訊號、保護價和退出方式共用64棒批次，再切持有上限與冷卻。
            full_cache = {}
            for i, p in values:
                for scenario, cost in COSTS.items():
                    key = p.stop_atr, p.reward_r, p.regime_exit, scenario
                    if key not in full_cache:
                        full_cache[key] = batch_events(frame, p.execution(), cost)
                    events = holding_limit(full_cache[key], frame, p.holding_bars, cost)
                    if len(events["endpoint"]):
                        reference_check(events, frame, p.execution(), cost, (i * 37 + list(COSTS).index(scenario) * 101) % len(events["endpoint"]))
                        probes += 1
                    for w in windows:
                        chosen = scheduled_indices(events, w["start"], w["stop"], p.cooldown_bars)
                        stats, month_returns = summarize(events, chosen, codes, months[w["name"]], w)
                        local_rows.append({"parameter_id": i, "window": w["name"], "scenario": scenario, **stats})
                        monthly[f"{w['name']}_{scenario}"][i] = month_returns
            pd.DataFrame(local_rows).to_csv(folder / f"{group_id:02d}.csv.gz", index=False)
            rows.extend(local_rows)
            write_json_atomic(output / "progress.json", {"status": "sweeping", "groups_completed": group_id+1,
                "groups_total": len(groups), "parameters_completed": values[-1][0] + 1, "elapsed_seconds": monotonic()-started})
            print(f"complex {group_id+1}/{len(groups)}: {values[-1][0]+1}/{len(grid)} parameters", flush=True)
        results = pd.DataFrame(rows)
        results.to_csv(output / "results.csv.gz", index=False)
        np.savez_compressed(output / "monthly.npz", **monthly)
        # 原策略在所有窗口/成本必須與前輪相同，避免引擎或資料差異冒充改進。
        old_results = pd.read_csv(previous / "results.csv.gz")
        old_id = json.loads((previous / "research.json").read_text(encoding="utf-8"))["baseline_id"]
        columns = ["trades", "mean_net", "sum_net", "log_growth", "settled_drawdown"]
        order = ["window", "scenario"]
        np.testing.assert_allclose(results.loc[results.parameter_id.eq(0)].sort_values(order)[columns],
            old_results.loc[old_results.parameter_id.eq(old_id)].sort_values(order)[columns], rtol=1e-10, atol=1e-12)
        search = results.loc[results.window.eq("search")]
        base = search.loc[search.scenario.eq("base")].set_index("parameter_id").sort_index()
        stress = search.loc[search.scenario.eq("stress")].set_index("parameter_id").sort_index()
        test = joint_max_statistic(monthly["search_base"][1:])
        p_adjusted, p_raw = np.r_[1., test["p_adjusted"]], np.r_[1., test["p_unadjusted"]]
        positive = base.trades.ge(150) & base.monthly_mean_log_return.gt(0) & stress.monthly_mean_log_return.gt(0)
        neighbors = complex_neighbors(grid, positive.to_numpy())
        checks = positive.to_numpy() & (p_adjusted <= .05) & (neighbors >= .6) & base.settled_drawdown.le(.1).to_numpy()
        shortlist = search_shortlist(params.loc[params.parameter_id.ne(0)], search.loc[search.parameter_id.ne(0)])
        write_json_atomic(output / "search_selection.json", {"shortlist": shortlist, "source": "search_only"})
        validation = results.loc[results.window.eq("validation")]
        vp = joint_max_statistic(monthly["validation_base"][shortlist])["p_adjusted"] if shortlist else []
        selection = validation_selection(shortlist, validation, dict(enumerate(checks)), dict(zip(shortlist, vp)))
        write_json_atomic(output / "selection.json", selection)
        diagnostics = params.assign(search_p_raw=p_raw, search_p_adjusted=p_adjusted,
            search_positive_neighbor_fraction=neighbors, search_quality_passed=checks)
        diagnostics["validation_p_adjusted"] = diagnostics.parameter_id.map(dict(zip(shortlist, vp)))
        diagnostics.to_csv(output / "parameter_diagnostics.csv", index=False)
        # 每一家族只匯出驗證排名最前者，避免前五全是幾乎相同的參數。
        leaders = {}
        for i in selection["validation_ranked"]:
            leaders.setdefault(grid[i].family, i)
        ids = sorted(set([0, *leaders.values(), *([] if selection["selected"] is None else [selection["selected"]])]))
        write_json_atomic(output / "progress.json", {"status": "exporting_and_replaying", "parameters_completed": len(grid)})
        exported = export_trades(output, cache, grid, ids, windows)
        write_json_atomic(output / "exported_candidates.json", exported)
        selected = selection["selected"]
        passed = False
        if selected is not None:
            b = next(x for x in exported["results"] if x["parameter_id"] == selected and x["scenario"] == "base")
            s = next(x for x in exported["results"] if x["parameter_id"] == selected and x["scenario"] == "stress")
            folds = results.loc[results.parameter_id.eq(selected) & results.scenario.eq("base") & results.window.str.startswith("forward")]
            passed = bool(b["trades"] >= 100 and b["ci_low"] > 0 and s["mean"] > 0 and len(folds) == 3
                and folds.trades.ge(20).all() and folds.mean_net.gt(0).all() and b["settled_drawdown"] <= .1)
        if sha256_file(source) != source_hash or any(sha256_file(root / name) != value for name, value in code_hashes.items()):
            raise ValueError("研究期間行情或程式被修改")
        outcome = {"status": "complete", "parameters": len(grid), "complex_parameters": len(grid)-1,
            "scenarios": len(results), "reference_checks": probes, "baseline_matches_previous_all_scenarios": True,
            "all_exported_trades_reference_checked": True, "exported_ids": ids, "family_validation_leaders": leaders,
            "selection": selection, "search_positive_base": int(base.monthly_mean_log_return.iloc[1:].gt(0).sum()),
            "search_quality_passed": int(checks.sum()), "search_max_stat_passed": int((p_adjusted[1:] <= .05).sum()),
            "forward_quality_passed": passed, "live_eligible": False, "pristine_holdout": False,
            "elapsed_seconds": monotonic()-started}
        write_json_atomic(output / "research.json", outcome)
        write_json_atomic(output / "progress.json", outcome)
        build_artifact_manifest(output, [p for p in output.rglob("*") if p.is_file()])
        return outcome
    except BaseException as error:
        write_json_atomic(output / "progress.json", {"status": "failed", "error_type": type(error).__name__})
        raise
