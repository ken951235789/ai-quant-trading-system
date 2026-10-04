"""核對複合策略的版本、情境完整性、月收益、候選選擇及成交帳務。"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai_quant_trading.backtesting.candidate_research import COSTS
from ai_quant_trading.backtesting.complex_strategies import complex_grid
from ai_quant_trading.backtesting.parameter_sweep import ALLOCATION, search_shortlist, validation_selection
from ai_quant_trading.operations.integrity import sha256_file, verify_artifact_manifest
from ai_quant_trading.persistence import write_json_atomic


def audit(output):
    verified = verify_artifact_manifest(output)
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    research = json.loads((output / "research.json").read_text(encoding="utf-8"))
    assert sha256_file(Path(plan["source"])) == plan["source_sha256"]
    for name, digest in plan["code_sha256"].items():
        assert sha256_file(root / name) == digest
    results = pd.read_csv(output / "results.csv.gz")
    params = pd.read_csv(output / "parameters.csv")
    diagnostics = pd.read_csv(output / "parameter_diagnostics.csv").set_index("parameter_id")
    assert len(params) == len(complex_grid()) == 1153
    assert len(results) == 17295 and not results.duplicated(["parameter_id", "window", "scenario"]).any()
    assert results.groupby("parameter_id").size().eq(15).all()
    assert set(results.scenario) == set(COSTS)
    assert set(results.window) == {w["name"] for w in plan["windows"]}
    np.testing.assert_allclose(np.expm1(results.log_growth), results.settled_return, atol=1e-12)
    monthly_rows = 0
    with np.load(output / "monthly.npz", allow_pickle=False) as monthly:
        for w in plan["windows"]:
            for c in COSTS:
                part = results.loc[results.window.eq(w["name"]) & results.scenario.eq(c)].set_index("parameter_id").sort_index()
                np.testing.assert_allclose(monthly[f"{w['name']}_{c}"].sum(axis=1), part.log_growth, atol=1e-12)
                monthly_rows += len(part)
    search = results.loc[results.window.eq("search") & results.parameter_id.ne(0)]
    shortlist = search_shortlist(params.loc[params.parameter_id.ne(0)], search)
    selection = validation_selection(shortlist, results.loc[results.window.eq("validation")],
        diagnostics.search_quality_passed.to_dict(), diagnostics.validation_p_adjusted.to_dict())
    assert json.loads(json.dumps(selection)) == research["selection"]
    trade_rows, files = 0, 0
    for path in (output / "trades").glob("*.csv"):
        f = pd.read_csv(path)
        tokens = path.stem.split("_")
        i, window, scenario = int(tokens[0][1:]), "_".join(tokens[1:-1]), tokens[-1]
        cost = COSTS[scenario]
        net = f.side * (f.exit_price / f.entry_price - 1) - f.fee_return - f.funding_return
        np.testing.assert_allclose(net, f.net_return, atol=1e-12)
        np.testing.assert_allclose(f.fee_return, cost["fee"] / 10000 * (1 + f.exit_price / f.entry_price), atol=1e-12)
        bucket = 8 * 3600 * 1_000_000_000
        entry_ns = pd.to_datetime(f.entry_at, utc=True).dt.as_unit("ns").astype("int64")
        exit_ns = pd.to_datetime(f.exit_at, utc=True).dt.as_unit("ns").astype("int64")
        np.testing.assert_allclose(f.funding_return, (exit_ns // bucket - entry_ns // bucket) * cost["funding"] / 10000, atol=1e-12)
        np.testing.assert_allclose(f.equity, 1000 * (1 + ALLOCATION * f.net_return).cumprod(), atol=1e-9)
        assert not f.endpoint.duplicated().any()
        assert (f.entry_endpoint == f.endpoint + 1).all() and (f.exit_endpoint >= f.entry_endpoint).all()
        if len(f) > 1:
            assert (f.endpoint.to_numpy()[1:] > f.exit_endpoint.to_numpy()[:-1]).all()
        mask = results.window.str.startswith("forward") if window == "forward" else results.window.eq(window)
        expected = results.loc[results.parameter_id.eq(i) & results.scenario.eq(scenario) & mask]
        assert len(f) == expected.trades.sum()
        np.testing.assert_allclose(f.net_return.sum(), expected.sum_net.sum(), atol=1e-11)
        trade_rows += len(f)
        files += 1
    assert research["baseline_matches_previous_all_scenarios"] and research["all_exported_trades_reference_checked"]
    return {"status": "PASS", "manifest_files_verified": len(verified), "scenario_rows": len(results),
        "monthly_rows_checked": monthly_rows, "source_and_code_unchanged": True,
        "selection_reproduced": True, "exported_csv_files": files,
        "trade_rows_including_pooled_duplicates": trade_rows,
        "reference_checks": research["reference_checks"], "baseline_matches_previous": True,
        "live_eligible": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit(args.output.resolve())
    write_json_atomic(args.output / "audit.json", result)
    print(json.dumps(result, ensure_ascii=False))
