"""可續跑的完整離散網格研究；搜尋、驗證、後續回放分開，不自動部署。"""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
from itertools import groupby, product
import json
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd

from ai_quant_trading.backtesting.candidate_research import COSTS, expectancy
from ai_quant_trading.backtesting.grid_execution import (
    LEVELS, REASONS, GridParams, batch_events, holding_limit, parameter_grid,
    prepare_cache, reference_check, scheduled_indices, signal_frame,
)
from ai_quant_trading.operations.integrity import build_artifact_manifest, sha256_file
from ai_quant_trading.persistence import write_json_atomic


ALLOCATION = .10


def windows_for(rows: int) -> list[dict]:
    if rows < 20_000:
        raise ValueError("大型研究至少需要 20,000 根連續行情")
    return [{"name": name, "start": int(rows * a) + (64 if a else 0), "stop": int(rows * b)}
            for name, a, b in (("search", 0, .4), ("validation", .4, .6),
                               ("forward_1", .6, .7), ("forward_2", .7, .8), ("forward_3", .8, .9))]


def summarize(events: dict, selected: np.ndarray, month_codes: np.ndarray, months: np.ndarray,
              window: dict) -> tuple[dict, np.ndarray]:
    values = events["net_return"][selected]
    log_steps = np.log1p(ALLOCATION * values)
    if not np.isfinite(log_steps).all():
        raise ValueError("事件耗盡研究資金，需完整清算模型，不能繼續")
    monthly = np.zeros(len(months))
    positions = np.searchsorted(months, month_codes[events["exit_endpoint"][selected]])
    np.add.at(monthly, positions, log_steps)
    equity = np.exp(np.r_[0., np.cumsum(log_steps)])
    entry = events["entry_price"][selected]
    risk = abs(entry - events["stop_price"][selected]) / entry
    count = len(values)
    gains, losses = values[values > 0].sum(), -values[values < 0].sum()
    metrics = {"trades": count, "mean_net": float(values.mean()) if count else np.nan,
        "sum_net": float(values.sum()), "sum_squared_net": float(np.square(values).sum()),
        "std_net": float(values.std(ddof=1)) if count > 1 else np.nan,
        "worst_trade": float(values.min()) if count else np.nan,
        "wins": int((values > 0).sum()), "win_rate": float((values > 0).mean()) if count else np.nan,
        "gains": float(gains), "losses": float(losses), "profit_factor": float(gains / losses) if losses > 0 else np.nan,
        "mean_net_r": float((values / risk).mean()) if count else np.nan,
        "sum_net_r": float((values / risk).sum()), "log_growth": float(log_steps.sum()),
        "settled_return": float(equity[-1] - 1),
        "settled_drawdown": float((1 - equity / np.maximum.accumulate(equity)).max()),
        "monthly_mean_log_return": float(monthly.mean()),
        "trades_per_30_days": count * 96 * 30 / (window["stop"] - window["start"]),
        "mean_holding_bars": float((events["exit_endpoint"][selected] - events["entry_endpoint"][selected]).mean()) if count else np.nan,
        "mean_fee": float(events["fee_return"][selected].mean()) if count else np.nan,
        "mean_funding": float(events["funding_return"][selected].mean()) if count else np.nan}
    return metrics, monthly


def joint_max_statistic(monthly: np.ndarray, *, repetitions: int = 2000, block: int = 3,
                        seed: int = 20261002) -> dict:
    """共享月份區塊重抽樣的最大統計量診斷；不是 PBO 或嚴格有限樣本保證。"""
    if monthly.ndim != 2 or monthly.shape[1] < 6 or not np.isfinite(monthly).all():
        raise ValueError("月份矩陣不足或含非有限值")
    if repetitions < 20 or not 1 <= block <= monthly.shape[1]:
        raise ValueError("區塊 bootstrap 設定不合法")
    n = monthly.shape[1]
    mean = monthly.mean(axis=1)
    se = monthly.std(axis=1, ddof=1) / np.sqrt(n)
    valid = se > 1e-12
    denominator = np.where(valid, se, 1.)
    observed = np.where(valid, mean / denominator, 0.)
    centered = (monthly - mean[:, None]) / denominator[:, None]
    random = np.random.default_rng(seed)
    maxima, individual = [], np.zeros(len(monthly), dtype=int)
    for start in range(0, repetitions, 128):
        size = min(128, repetitions - start)
        starts = random.integers(0, n, (size, int(np.ceil(n / block))))
        indices = ((starts[..., None] + np.arange(block)) % n).reshape(size, -1)[:, :n]
        weights = np.zeros((size, n))
        np.add.at(weights, (np.arange(size)[:, None], indices), 1 / n)
        sampled = weights @ centered.T
        maxima.extend(sampled.max(axis=1))
        individual += (sampled >= observed[None, :]).sum(axis=0)
    maximum = np.sort(maxima)
    adjusted = (repetitions - np.searchsorted(maximum, observed, side="left") + 1) / (repetitions + 1)
    raw = (individual + 1) / (repetitions + 1)
    adjusted[~valid], raw[~valid] = 1., 1.
    return {"p_adjusted": adjusted, "p_unadjusted": raw, "statistic": observed,
            "max_distribution": maximum}


def neighborhood_fraction(grid: list[GridParams], positive: np.ndarray) -> np.ndarray:
    index = {params: i for i, params in enumerate(grid)}
    axes = ("adx", "confirmation", "lookback", "stop_atr", "reward_r", "holding_bars", "cooldown_bars")
    result = []
    for params in grid:
        neighbors = []
        for name in axes:
            values = LEVELS[name]
            at = values.index(getattr(params, name))
            for offset in (-1, 1):
                if 0 <= at + offset < len(values):
                    neighbor = replace(params, **{name: values[at + offset]})
                    if neighbor in index:
                        neighbors.append(bool(positive[index[neighbor]]))
        result.append(float(np.mean(neighbors)) if neighbors else 0.)
    return np.asarray(result)


def search_shortlist(parameters: pd.DataFrame, search: pd.DataFrame) -> list[int]:
    """每個進場家族/週期最多五組；禁止傳入其他期間選擇。"""
    if not search.window.eq("search").all():
        raise ValueError("搜尋選擇只能讀 search 區間")
    base = search.loc[search.scenario.eq("base")].set_index("parameter_id")
    stress = search.loc[search.scenario.eq("stress")].set_index("parameter_id")
    ranking = parameters.set_index("parameter_id").join(base[["trades"]])
    ranking["rank_score"] = stress.monthly_mean_log_return
    ranking = ranking.loc[ranking.trades.ge(150)]
    ranking = ranking.sort_values(["rank_score", "parameter_id"], ascending=[False, True])
    return ranking.groupby(["timeframe", "entry"], sort=True).head(5).index.astype(int).tolist()


def validation_selection(shortlist: list[int], validation: pd.DataFrame,
                         search_checks: dict[int, bool], validation_p: dict[int, float]) -> dict:
    if not validation.window.eq("validation").all():
        raise ValueError("驗證選擇不可讀 Forward")
    base = validation.loc[validation.scenario.eq("base")].set_index("parameter_id")
    stress = validation.loc[validation.scenario.eq("stress")].set_index("parameter_id")
    ranked = sorted(shortlist, key=lambda i: (-float(stress.loc[i, "monthly_mean_log_return"]), i))
    checks = {}
    for i in ranked:
        checks[i] = {"search_quality": bool(search_checks[i]), "validation_trades_50": bool(base.loc[i, "trades"] >= 50),
            "validation_base_positive": bool(base.loc[i, "monthly_mean_log_return"] > 0),
            "validation_stress_positive": bool(stress.loc[i, "monthly_mean_log_return"] > 0),
            "validation_max_stat_p_le_005": bool(validation_p[i] <= .05),
            "validation_settled_drawdown_le_10pct": bool(base.loc[i, "settled_drawdown"] <= .10)}
    qualified = [i for i in ranked if all(checks[i].values())]
    return {"shortlist": shortlist, "validation_ranked": ranked,
        "diagnostic_leader": ranked[0] if ranked else None,
        "selected": qualified[0] if qualified else None, "checks": checks,
        "selection_source": "search_then_validation_only", "live_eligible": False}


def _months(frame: pd.DataFrame, windows: list[dict]) -> tuple[np.ndarray, dict]:
    codes = (frame.timestamp.dt.year * 12 + frame.timestamp.dt.month).to_numpy()
    return codes, {w["name"]: np.unique(codes[w["start"]:w["stop"]]) for w in windows}


def _export_candidates(output: Path, cache: dict, grid: list[GridParams], ids: list[int],
                       windows: list[dict], source_rows: pd.DataFrame) -> dict:
    folder = output / "trades"
    folder.mkdir(exist_ok=True)
    pooled = []
    for i in ids:
        params = grid[i]
        frame = signal_frame(cache, params)
        for scenario, cost in COSTS.items():
            events = holding_limit(batch_events(frame, params, cost), frame, params.holding_bars, cost)
            parts = []
            for window in windows:
                selected = scheduled_indices(events, window["start"], window["stop"], params.cooldown_bars)
                for index in selected:
                    reference_check(events, frame, params, cost, int(index))
                trades = pd.DataFrame({name: values[selected] for name, values in events.items()})
                trades["entry_at"] = source_rows.timestamp.iloc[trades.entry_endpoint.to_numpy(dtype=int)].to_numpy()
                trades["exit_at"] = source_rows.timestamp.iloc[trades.exit_endpoint.to_numpy(dtype=int)].to_numpy()
                trades["exit_reason"] = trades.reason_code.map(REASONS)
                trades["net_r"] = trades.net_return / (abs(trades.entry_price - trades.stop_price) / trades.entry_price)
                trades["equity"] = 1000 * (1 + ALLOCATION * trades.net_return).cumprod()
                trades.to_csv(folder / f"G{i:05d}_{window['name']}_{scenario}.csv", index=False)
                if window["name"].startswith("forward"):
                    parts.append(trades)
            combined = pd.concat(parts, ignore_index=True)
            combined["equity"] = 1000 * (1 + ALLOCATION * combined.net_return).cumprod()
            combined.to_csv(folder / f"G{i:05d}_forward_{scenario}.csv", index=False)
            curve = np.r_[1000., combined.equity.to_numpy()]
            pooled.append({"parameter_id": i, "scenario": scenario,
                **expectancy(combined), "net_r": expectancy(combined, "net_r"),
                "settled_return": float(curve[-1] / 1000 - 1),
                "settled_drawdown": float((1 - curve / np.maximum.accumulate(curve)).max())})
    return {"results": pooled, "all_exported_trades_reference_checked": True}


def run_sweep(source: Path, output: Path, *, resume: bool = False) -> dict:
    source, output = source.resolve(), output.resolve()
    if output.exists() and not resume:
        raise FileExistsError("輸出已存在；只能使用明確的 --resume 核對後續跑")
    source_hash = sha256_file(source)
    bars = pd.read_csv(source)
    windows = windows_for(len(bars))
    grid = parameter_grid()
    root = Path(__file__).resolve().parents[3]
    code_paths = [Path(__file__), Path(__file__).with_name("grid_execution.py"),
        root / "src/ai_quant_trading/transformer/strategy_events.py",
        root / "src/ai_quant_trading/transformer/economics.py",
        root / "src/ai_quant_trading/backtesting/candidate_research.py",
        root / "src/ai_quant_trading/features/intraday.py"]
    fingerprint = {"source": str(source), "source_sha256": source_hash,
        "code_sha256": {str(p.relative_to(root)): sha256_file(p) for p in code_paths},
        "grid": {k: list(v) for k, v in LEVELS.items()}, "parameters": len(grid),
        "source_rows": len(bars), "windows": windows, "costs_bps": COSTS,
        "allocation": ALLOCATION, "bootstrap_repetitions": 2000, "bootstrap_month_block": 3,
        "bootstrap_seed": 20261002}
    output.mkdir(parents=True, exist_ok=True)
    if resume:
        old = json.loads((output / "plan.json").read_text(encoding="utf-8"))
        if old["fingerprint"] != fingerprint:
            raise ValueError("行情、程式或搜尋設定改變，禁止混合續跑")
    else:
        write_json_atomic(output / "plan.json", {"created_at": datetime.now(timezone.utc).isoformat(),
            "fingerprint": fingerprint, "source_start": str(bars.timestamp.iloc[0]), "source_end": str(bars.timestamp.iloc[-1]),
            "selection": "Search每家族/週期最多5組，依stress月均log收益排序，至少150筆；只用Validation再選擇",
            "quality": "Search max-stat p<=.05、正stress、鄰域>=60%、結算DD<=10%；Validation>=50筆、正base/stress、max-stat p<=.05、結算DD<=10%",
            "forward_quality": "選中者合併>=100筆、淨均值CI下界>0、壓力均值>0、三段各>=20筆且均值>0、結算DD<=10%",
            "max_stat_caveat": "共享中心化循環月份區塊的最大統計量敏感度診斷；非PBO、非嚴格FWER保證，需近似平穩假設",
            "pristine_holdout": False, "tail_evaluated": False, "live_eligible": False,
            "account_caveat": "1000 USDT、固定10%名目配置、無槓桿；只記結算權益，無棒內清算及每日風控，非使用者真實本金",
            "references": ["https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf"]})
    prepared = bars.iloc[:windows[-1]["stop"]].copy()
    cache = prepare_cache(prepared)
    codes, months = _months(cache["raw"], windows)
    lookup = {p: i for i, p in enumerate(grid)}
    groups = [(key, list(values)) for key, values in groupby(grid, key=lambda p: p.signal_key)]
    folder = output / "groups"
    folder.mkdir(exist_ok=True)
    started = monotonic()
    try:
        for group_id, (_, variants) in enumerate(groups):
            csv_path, npz_path = folder / f"{group_id:03d}.csv.gz", folder / f"{group_id:03d}.npz"
            receipt = folder / f"{group_id:03d}.json"
            if receipt.exists():
                saved = json.loads(receipt.read_text(encoding="utf-8"))
                if sha256_file(csv_path) != saved["csv_sha256"] or sha256_file(npz_path) != saved["npz_sha256"]:
                    raise ValueError("續跑區塊成品被修改")
                continue
            frame = signal_frame(cache, variants[0])
            rows, audits = [], 0
            ids = [lookup[p] for p in variants]
            local = {i: j for j, i in enumerate(ids)}
            matrices = {f"{w['name']}_{c}": np.zeros((len(ids), len(months[w["name"]]))) for w in windows for c in COSTS}
            for stop_atr, reward_r, exit_mode in product(LEVELS["stop_atr"], LEVELS["reward_r"], LEVELS["regime_exit"]):
                prototype = replace(variants[0], stop_atr=stop_atr, reward_r=reward_r, regime_exit=exit_mode)
                for scenario, cost in COSTS.items():
                    full_events = batch_events(frame, prototype, cost)
                    for holding in LEVELS["holding_bars"]:
                        events = holding_limit(full_events, frame, holding, cost)
                        for cooldown in LEVELS["cooldown_bars"]:
                            params = replace(prototype, holding_bars=holding, cooldown_bars=cooldown)
                            i = lookup[params]
                            if len(events["endpoint"]):
                                reference_check(events, frame, params, cost, (i * 37 + list(COSTS).index(scenario) * 101) % len(events["endpoint"]))
                                audits += 1
                            for window in windows:
                                selected = scheduled_indices(events, window["start"], window["stop"], cooldown)
                                metrics, monthly = summarize(events, selected, codes, months[window["name"]], window)
                                rows.append({"parameter_id": i, "window": window["name"], "scenario": scenario, **metrics})
                                matrices[f"{window['name']}_{scenario}"][local[i]] = monthly
            pd.DataFrame(rows).to_csv(csv_path, index=False, compression="gzip")
            np.savez_compressed(npz_path, ids=ids, **matrices)
            write_json_atomic(receipt, {"csv_sha256": sha256_file(csv_path), "npz_sha256": sha256_file(npz_path),
                "parameters": len(ids), "scenarios": len(rows), "reference_checks": audits})
            write_json_atomic(output / "progress.json", {"status": "sweeping", "groups_completed": group_id + 1,
                "groups_total": len(groups), "parameters_completed": sum(len(v) for _, v in groups[:group_id + 1]),
                "elapsed_this_session_seconds": monotonic() - started})
            print(f"grid {group_id + 1}/{len(groups)}: {len(rows)} scenarios; elapsed={monotonic()-started:.1f}s", flush=True)
        results = pd.concat([pd.read_csv(folder / f"{i:03d}.csv.gz") for i in range(len(groups))], ignore_index=True)
        results.to_csv(output / "results.csv.gz", index=False, compression="gzip")
        parameters = pd.DataFrame([{"parameter_id": i, **asdict(p)} for i, p in enumerate(grid)])
        parameters.to_csv(output / "parameters.csv", index=False)
        monthly = {f"{w['name']}_{c}": np.zeros((len(grid), len(months[w["name"]]))) for w in windows for c in COSTS}
        reference_checks = 0
        for i in range(len(groups)):
            with np.load(folder / f"{i:03d}.npz", allow_pickle=False) as stored:
                for name in monthly:
                    monthly[name][stored["ids"]] = stored[name]
            reference_checks += json.loads((folder / f"{i:03d}.json").read_text(encoding="utf-8"))["reference_checks"]
        write_json_atomic(output / "progress.json", {"status": "selection", "parameters_completed": len(grid)})
        search = results.loc[results.window.eq("search")]
        search_base = search.loc[search.scenario.eq("base")].set_index("parameter_id").sort_index()
        search_stress = search.loc[search.scenario.eq("stress")].set_index("parameter_id").sort_index()
        search_test = joint_max_statistic(monthly["search_base"])
        positive = search_base.trades.ge(150) & search_base.monthly_mean_log_return.gt(0) & search_stress.monthly_mean_log_return.gt(0)
        neighbors = neighborhood_fraction(grid, positive.to_numpy())
        checks = positive.to_numpy() & (search_test["p_adjusted"] <= .05) & (neighbors >= .6) & search_base.settled_drawdown.le(.1).to_numpy()
        shortlist = search_shortlist(parameters, search)
        write_json_atomic(output / "search_selection.json", {"shortlist": shortlist, "selection_source": "search_only"})
        validation = results.loc[results.window.eq("validation")]
        validation_test = joint_max_statistic(monthly["validation_base"][shortlist]) if shortlist else None
        p_validation = dict(zip(shortlist, validation_test["p_adjusted"])) if shortlist else {}
        selection = validation_selection(shortlist, validation, dict(enumerate(checks)), p_validation)
        write_json_atomic(output / "selection.json", selection)
        diagnostics = parameters.copy()
        diagnostics["search_p_raw"] = search_test["p_unadjusted"]
        diagnostics["search_p_adjusted"] = search_test["p_adjusted"]
        diagnostics["search_positive_neighbor_fraction"] = neighbors
        diagnostics["search_quality_passed"] = checks
        diagnostics["validation_p_adjusted"] = diagnostics.parameter_id.map(p_validation)
        diagnostics.to_csv(output / "parameter_diagnostics.csv", index=False)
        baseline = lookup[GridParams("15m", "breakout", 25, 2, 20, 2., 2., 32, 4, "loss")]
        export_ids = sorted(set([baseline] + selection["validation_ranked"][:5]
                                + ([selection["selected"]] if selection["selected"] is not None else [])))
        exported = _export_candidates(output, cache, grid, export_ids, windows, cache["raw"])
        write_json_atomic(output / "exported_candidates.json", exported)
        passed = False
        selected_id = selection["selected"]
        if selected_id is not None:
            base = next(r for r in exported["results"] if r["parameter_id"] == selected_id and r["scenario"] == "base")
            stress = next(r for r in exported["results"] if r["parameter_id"] == selected_id and r["scenario"] == "stress")
            folds = results.loc[results.parameter_id.eq(selected_id) & results.scenario.eq("base") & results.window.str.startswith("forward")]
            passed = bool(base["trades"] >= 100 and base["ci_low"] > 0 and stress["mean"] > 0
                and len(folds) == 3 and folds.trades.ge(20).all() and folds.mean_net.gt(0).all() and base["settled_drawdown"] <= .1)
        if sha256_file(source) != source_hash or any(sha256_file(root / name) != value for name, value in fingerprint["code_sha256"].items()):
            raise ValueError("研究期間行情或程式變更")
        result = {"status": "complete", "parameters": len(grid), "scenarios": len(results),
            "reference_checks": reference_checks, "all_exported_trades_reference_checked": True,
            "baseline_id": baseline, "exported_ids": export_ids, "selection": selection,
            "forward_quality_passed": passed, "live_eligible": False, "pristine_holdout": False,
            "search_positive_base": int(search_base.monthly_mean_log_return.gt(0).sum()),
            "search_max_stat_passed": int((search_test["p_adjusted"] <= .05).sum()),
            "search_quality_passed": int(checks.sum()), "elapsed_this_session_seconds": monotonic() - started}
        write_json_atomic(output / "research.json", result)
        write_json_atomic(output / "progress.json", result)
        build_artifact_manifest(output, [p for p in output.rglob("*") if p.is_file()])
        return result
    except BaseException as error:
        write_json_atomic(output / "progress.json", {"status": "failed", "error_type": type(error).__name__,
                          "elapsed_this_session_seconds": monotonic() - started})
        raise
