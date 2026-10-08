"""對已完成的多 R 模型產生合法 OOS 停利研究提案，不改部署。"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_quant_trading.operations.integrity import build_artifact_manifest, sha256_file, verify_artifact_manifest  # noqa: E402
from ai_quant_trading.persistence import write_json_atomic  # noqa: E402
from ai_quant_trading.research.candidate_contract import CandidateContract  # noqa: E402
from ai_quant_trading.research.exit_selection import ExitForecast, choose_exit  # noqa: E402


def audit(runs, source, output, *, diagnostic_only=False):
    import pandas as pd
    source_sha = sha256_file(source)
    bars = pd.read_csv(source)
    loaded = []
    for run in runs:
        verify_artifact_manifest(run, required=True)
        meta = json.loads((run / "training.json").read_text(encoding="utf-8"))
        contract_meta = meta["data_diagnostics"]["candidate_contract"]
        if source_sha != contract_meta["source_sha256"]:
            raise ValueError("退出研究來源與模型不同")
        contract = CandidateContract(**contract_meta["parameters"])
        selection = pd.read_csv(run / "validation_predictions.csv")
        test = pd.read_csv(run / "test_predictions.csv").set_index("endpoint", drop=False)
        if not test.index.is_unique or selection.empty:
            raise ValueError("退出研究缺少可核對的選擇區間")
        known_index = int(selection.event_exit_endpoint.max()) + 1
        if not 0 <= known_index < len(bars):
            raise ValueError("模型資訊結束點不合法")
        known = pd.Timestamp(bars.timestamp.iloc[known_index]).isoformat()
        loaded.append((run, contract, test, known, sha256_file(run / meta["artifacts"]["model"]),
                       bool(meta.get("calibration"))))
    if len(loaded) < 2:
        raise ValueError("需要至少兩種獨立 R 模型")
    points = sorted(set.intersection(*(set(item[2].index) for item in loaded)))
    rows = []
    for point in points:
        forecasts = []
        for run, contract, frame, known, checkpoint, calibrated in loaded:
            row = frame.loc[point]
            h = contract.holding_bars
            decision = (pd.Timestamp(bars.timestamp.iloc[point]) + pd.Timedelta(minutes=15)).isoformat()
            forecasts.append(ExitForecast(contract, f"{source_sha}:{point}", int(row.event_side),
                decision, known, checkpoint, float(row[f"predicted_return_{h}"]),
                float(row[f"tradeability_probability_{h}"]), calibrated, False))
        proposal = choose_exit(forecasts, require_quality=not diagnostic_only)
        rows.append({"endpoint": point, **proposal})
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(output / "proposals.csv", index=False)
    summary = {"engineering": "PASS", "research_quality": "INSUFFICIENT_EVIDENCE", "live_eligible": False,
        "diagnostic_only": diagnostic_only, "candidates": len(points),
        "exit_proposals": sum(row["action"] == "research_exit_proposal" for row in rows),
        "source_sha256": source_sha, "independent_reward_r": [item[1].reward_r for item in loaded],
        "excluded_not_common_to_all_models": {str(item[1].reward_r): len(item[2])-len(points) for item in loaded},
        "automatic_training": False, "actual_returns_used_to_choose": False,
        "limits": ["不同退出標籤導致的跨界清除取交集，不冒充全部事件", "Test 已多次研究，不是全新封存集", "提案不更改停損與槓桿，也不解鎖交易"]}
    write_json_atomic(output / "report.json", summary)
    build_artifact_manifest(output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(args.runs, args.source, args.output, diagnostic_only=args.diagnostic_only), ensure_ascii=False))


if __name__ == "__main__":
    main()
