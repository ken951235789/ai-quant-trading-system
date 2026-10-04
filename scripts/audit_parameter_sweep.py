"""獨立核對網格完整性、檔案版本、月收益加總與已匯出交易。"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai_quant_trading.backtesting.candidate_research import COSTS
from ai_quant_trading.backtesting.grid_execution import parameter_grid
from ai_quant_trading.backtesting.parameter_sweep import ALLOCATION, search_shortlist, validation_selection
from ai_quant_trading.operations.integrity import sha256_file, verify_artifact_manifest
from ai_quant_trading.persistence import write_json_atomic


def audit(output: Path) -> dict:
    verified = verify_artifact_manifest(output)
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))["fingerprint"]
    root = Path(__file__).resolve().parents[1]
    assert sha256_file(Path(plan["source"])) == plan["source_sha256"]
    for name, digest in plan["code_sha256"].items():
        assert sha256_file(root / name) == digest, name
    result = pd.read_csv(output / "results.csv.gz")
    params = pd.read_csv(output / "parameters.csv")
    diagnostics = pd.read_csv(output / "parameter_diagnostics.csv").set_index("parameter_id")
    assert len(params) == len(parameter_grid()) == 11664
    assert len(result) == 174960 and not result.duplicated(["parameter_id", "window", "scenario"]).any()
    assert result.groupby("parameter_id").size().eq(15).all()
    assert set(result.window) == {w["name"] for w in plan["windows"]}
    assert set(result.scenario) == set(COSTS)
    assert np.isfinite(result[["log_growth", "settled_return", "settled_drawdown"]]).all().all()
    np.testing.assert_allclose(np.expm1(result.log_growth), result.settled_return, atol=1e-12)
    monthly_rows, reference_checks = 0, 0
    for receipt_path in sorted((output / "groups").glob("*.json")):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        stem = receipt_path.stem
        assert sha256_file(receipt_path.with_name(stem + ".csv.gz")) == receipt["csv_sha256"]
        assert sha256_file(receipt_path.with_name(stem + ".npz")) == receipt["npz_sha256"]
        part = pd.read_csv(receipt_path.with_name(stem + ".csv.gz"))
        with np.load(receipt_path.with_name(stem + ".npz"), allow_pickle=False) as stored:
            for w in plan["windows"]:
                for c in COSTS:
                    selected = part.loc[part.window.eq(w["name"]) & part.scenario.eq(c)].set_index("parameter_id")
                    np.testing.assert_allclose(stored[f"{w['name']}_{c}"].sum(axis=1),
                        selected.loc[stored["ids"], "log_growth"], atol=1e-12)
                    monthly_rows += len(selected)
        reference_checks += receipt["reference_checks"]
    assert monthly_rows == len(result) and reference_checks == 34992
    shortlist = search_shortlist(params, result.loc[result.window.eq("search")])
    selected = validation_selection(shortlist, result.loc[result.window.eq("validation")],
        diagnostics.search_quality_passed.to_dict(), diagnostics.validation_p_adjusted.to_dict())
    # JSON 會把整數鍵轉成字串，先正規化再比較。
    assert json.loads(json.dumps(selected)) == json.loads((output / "selection.json").read_text(encoding="utf-8"))
    trades_checked = files_checked = 0
    for path in (output / "trades").glob("*.csv"):
        frame = pd.read_csv(path)
        scenario = path.stem.rsplit("_", 1)[1]
        cost = COSTS[scenario]
        net = frame.side * (frame.exit_price / frame.entry_price - 1) - frame.fee_return - frame.funding_return
        np.testing.assert_allclose(net, frame.net_return, atol=1e-12)
        np.testing.assert_allclose(frame.fee_return, cost["fee"] / 10000 * (1 + frame.exit_price / frame.entry_price), atol=1e-12)
        entry_ns = pd.to_datetime(frame.entry_at, utc=True).dt.as_unit("ns").astype("int64")
        exit_ns = pd.to_datetime(frame.exit_at, utc=True).dt.as_unit("ns").astype("int64")
        bucket = 8 * 3600 * 1_000_000_000
        np.testing.assert_allclose(frame.funding_return, (exit_ns // bucket - entry_ns // bucket) * cost["funding"] / 10000, atol=1e-12)
        np.testing.assert_allclose(frame.equity, 1000 * (1 + ALLOCATION * frame.net_return).cumprod(), atol=1e-9)
        assert not frame.endpoint.duplicated().any()
        assert (frame.entry_endpoint == frame.endpoint + 1).all()
        assert (frame.exit_endpoint >= frame.entry_endpoint).all()
        if len(frame) > 1:
            assert (frame.endpoint.to_numpy()[1:] > frame.exit_endpoint.to_numpy()[:-1]).all()
        tokens = path.stem.split("_")
        parameter_id, window = int(tokens[0][1:]), "_".join(tokens[1:-1])
        expected = result.loc[result.parameter_id.eq(parameter_id) & result.scenario.eq(scenario)
            & (result.window.str.startswith("forward") if window == "forward" else result.window.eq(window))]
        assert len(frame) == expected.trades.sum()
        np.testing.assert_allclose(frame.net_return.sum(), expected.sum_net.sum(), atol=1e-11)
        np.testing.assert_allclose(np.log1p(ALLOCATION * frame.net_return).sum(), expected.log_growth.sum(), atol=1e-11)
        trades_checked += len(frame)
        files_checked += 1
    return {"status": "PASS", "manifest_files_verified": len(verified), "scenario_rows": len(result),
        "monthly_rows_checked": monthly_rows, "reference_checks": reference_checks,
        "exported_csv_files": files_checked, "exported_csv_trade_rows_including_pooled_duplicates": trades_checked,
        "selection_reproduced": True, "source_and_code_unchanged": True, "live_eligible": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline-csv", type=Path)
    args = parser.parse_args()
    result = audit(args.output.resolve())
    if args.baseline_csv is not None:
        baseline = json.loads((args.output / "research.json").read_text(encoding="utf-8"))["baseline_id"]
        old = pd.read_csv(args.baseline_csv)
        current = pd.read_csv(args.output / "trades" / f"G{baseline:05d}_forward_base.csv")
        columns = ["endpoint", "entry_endpoint", "exit_endpoint", "side", "entry_price", "exit_price",
                   "fee_return", "funding_return", "net_return"]
        np.testing.assert_allclose(old[columns], current[columns], rtol=1e-12, atol=1e-12)
        result["previous_baseline"] = {"matched": True, "trade_count": len(current),
            "source_sha256": sha256_file(args.baseline_csv), "source": str(args.baseline_csv)}
    write_json_atomic(args.output / "audit.json", result)
    print(json.dumps(result, ensure_ascii=False))
