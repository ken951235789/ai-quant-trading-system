"""使用已保存模型預測做四組閘門消融，不重訓、不選 Test 最佳門檻。"""

# ruff: noqa: E402
import argparse
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from audit_candidate_training_research import portable_path, verify_trade_table
from ai_quant_trading.operations.integrity import sha256_file, verify_artifact_manifest, build_artifact_manifest
from ai_quant_trading.research.candidate_contract import CandidateContract, prepare_candidate_frame, candidate_outcomes
from ai_quant_trading.research.candidate_evaluation import score_diagnostics
from ai_quant_trading.research.event_attribution import attribute_events
from ai_quant_trading.research.probability_sweep import trade_metrics, assert_trade_parity
from ai_quant_trading.transformer.strategy_events import select_event_trades


def dump(path, payload):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, allow_nan=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.study.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("請使用新輸出目錄")
    verified = set(verify_artifact_manifest(root))
    def checked(path):
        if path.resolve().relative_to(root).as_posix() not in verified:
            raise ValueError("來源未包含在研究雜湊清單")
        return path
    study = json.loads(checked(root / "study.json").read_text(encoding="utf-8"))
    source_plan = json.loads(checked(root / "plan.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True)
    dump(output / "plan.json", {"schema": "gate_ablation_v1", "research_only": True,
        "modes": ["none", "return", "probability", "both"], "scenarios": ["base", "stress"],
        "threshold_selection": "none; original_contract_threshold", "live_eligible": False,
        "study_sha256": sha256_file(root / "study.json"), "source_manifest_sha256": sha256_file(root / "artifact_manifest.json"),
        "source_plan": source_plan.get("version"), "runner_sha256": sha256_file(Path(__file__)),
        "command": [sys.executable, *sys.argv], "history_status": "previously_researched_not_fresh_holdout"})
    cache, rows, diagnostics = {}, [], []
    for record in study["records"]:
        run = portable_path(record["run_dir"], root / "runs")
        metadata = json.loads(checked(run / "training.json").read_text(encoding="utf-8"))
        config = metadata["training_config"]
        contract = CandidateContract(**config["candidate_contract"])
        source = checked(portable_path(metadata["sources"][0]["path"], root))
        h, edge = contract.holding_bars, config["economic_minimum_edge_bps"]
        source_hash = sha256_file(source)
        predictions = {split: pd.read_csv(checked(run / f"{split}_predictions.csv")) for split in ("validation", "test")}
        boundaries = np.unique(np.quantile(predictions["validation"][f"predicted_return_{h}"], [.2, .4, .6, .8])).tolist()
        cache_key = (contract.digest, source_hash)
        if cache_key not in cache:
            frame = prepare_candidate_frame(pd.read_csv(source), contract)
            cache[cache_key] = {}
            for scenario, c in (("base", contract), ("stress", replace(contract, slippage_bps=contract.slippage_bps + 3))):
                outcomes = candidate_outcomes(frame, c)
                cache[cache_key][scenario] = (c, attribute_events(frame, outcomes, c, source_hash).set_index("endpoint", drop=False))
        for split, p in predictions.items():
            diagnostics.append({"name": record["name"], "split": split, **score_diagnostics(p, h, boundaries)})
            for scenario, (c, all_trades) in cache[cache_key].items():
                current = p.copy()
                available = all_trades.loc[current.endpoint.astype(int)]
                window = next(w for w in source_plan["windows"] if w["fold"] == record["fold"])
                boundary = window["validation_end"] if split == "validation" else window["source_end"]
                if (available.label_end_endpoint >= boundary).any():
                    raise ValueError("成本情境跨越允許標籤邊界")
                current["event_exit_endpoint"] = available.exit_endpoint.to_numpy()
                current[f"actual_return_{h}"] = available.net_return.to_numpy()
                for mode in ("none", "return", "probability", "both"):
                    selected, rejections = select_event_trades(current, h, c.event_config(), edge, filtered=True, filter_mode=mode)
                    trades = all_trades.loc[selected.endpoint.astype(int)]
                    verify_trade_table(trades, c.cooldown_bars)
                    if mode in {"none", "both"}:
                        policy = "no_ai" if mode == "none" else "transformer"
                        assert_trade_parity(trades, pd.read_csv(checked(run / f"{split}_policies/{policy}_{scenario}_trades.csv")))
                    filename = f"{record['name']}_{split}_{scenario}_{mode}.csv"
                    trades.to_csv(output / filename, index=False)
                    rows.append({**{k: record[k] for k in ("name", "family", "variant", "seed", "fold")},
                        "split": split, "scenario": scenario, "filter_mode": mode,
                        "probability_threshold": c.event_config().minimum_probability, "minimum_net_edge_bps": edge,
                        "candidate_count": len(p), "rejected": json.dumps(rejections),
                        "schedule_file": filename, **trade_metrics(trades)})
        print(f"完成 {record['name']}", flush=True)
    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
    dump(output / "score_diagnostics.json", diagnostics)
    helper = ROOT / "tools/research_reports/new_report.py"
    spec = importlib.util.spec_from_file_location("report_helper", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.build_report(report_type="model_research", project="AIQuantTradingSystem",
        revision="working_tree", scope="既有預測四組篩選與成本壓力對照")
    report.update(overall_status="WARN", summary="工程回放完成，未依 Test 挑選新門檻，研究仍需跨期驗證。",
        checks=[{"id": "GATE-001", "name": "旧無 AI 與雙閘門逐筆一致、成本及時間對帳", "status": "PASS", "evidence": ["comparison.csv"]}],
        changes=["新增單一機率、單一收益的獨立作用比較與分數分組診斷。"],
        metrics={"engineering": "PASS", "research_quality": "INSUFFICIENT_EVIDENCE", "live_eligible": False,
                 "models": len(study["records"]), "comparison_rows": len(rows), "sac_started": False},
        residual_risks=["單一期已被研究；分數診斷候選有重疊", "成本為假設，funding 為準備金，稅未驗證",
                        "僅結算績效，未驗證完整帳戶浮虧與清算"],
        next_actions=["先在允許的 Calibration/Selection 凍結方案，再做多折向前驗證；未達標不啟動 SAC。"])
    lines = ["# 四種篩選條件的独立比較", "", "工程 PASS；研究證據不足；SAC / Live 不解鎖。",
        "既有模型不重訓，機率與收益門檻沿用契約；Test 不選參。每個成本情境重新撮合。",
        "", "| 策略 | 特徵 | Seed | 篩選 | Test 基本成本成交 | 平均淨報酬 |", "|---|---|---:|---|---:|---:|"]
    for r in rows:
        if r["split"] == "test" and r["scenario"] == "base":
            value = "無成交" if r["mean_net"] is None else f"{r['mean_net']:+.4%}"
            lines.append(f"| {r['family']} | {r['variant']} | {r['seed']} | {r['filter_mode']} | {r['trades']} | {value} |")
    lines += ["", "none=無 AI；return=只看預測淨收益；probability=只看成功機率；both=原雙条件。",
              "校準分組與分位數覆蓋率見 score_diagnostics.json。收益為名目部位，非帳戶報酬。",
              "相同種子共享行情，不能合併當獨立樣本；8 筆僅是描述性 CI 最低計算量。"]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    report["artifacts"] = [{"path": p.name, "role": "research_evidence", "sha256": sha256_file(p)}
                           for p in output.iterdir() if p.is_file()]
    dump(output / "report.json", report)
    build_artifact_manifest(output, files=list(output.iterdir()))
    print(json.dumps(report["metrics"]))


if __name__ == "__main__":
    main()
