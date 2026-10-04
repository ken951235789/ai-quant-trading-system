"""只匯出允許的彙總欄位；不複製本機報告、路徑、帳戶或逐筆交易。"""

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_COLUMNS = ["family", "variant", "split", "scenario", "threshold_pct", "seeds", "active_seeds",
    "trades_mean", "trades_min", "trades_max", "mean_net_mean", "mean_net_min", "mean_net_max",
    "actual_win_rate_mean", "settled_return_10pct_mean"]


def sanitize_summary(frame):
    """白名單欄位仍限制分類值，避免自由文字透過欄位名稱合規的內容洩漏。"""
    frame = frame[SUMMARY_COLUMNS].copy()
    allowed = {"family": {"original", "trend_pullback", "vwap_reversion"},
        "variant": {"F_existing", "F_compact_combined"}, "split": {"selection", "test"},
        "scenario": {"zero", "base", "stress"}}
    for name, values in allowed.items():
        if not set(frame[name]).issubset(values):
            raise ValueError(f"不允許公開的分類欄位：{name}")
    for name in SUMMARY_COLUMNS[4:]:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    return {"columns": SUMMARY_COLUMNS,
            "rows": frame.astype(object).where(frame.notna(), None).values.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("禁止覆蓋既有公開摘要")
    evidence = json.loads((args.results / "report.json").read_text(encoding="utf-8"))["metrics"]
    if evidence["scenario_rows"] != 10908 or evidence["models"] != 18 or evidence["live_eligible"]:
        raise ValueError("本次公開匯出只接受指定的研究矩陣及研究隔離狀態")
    tests_passed = int(evidence["related_tests_passed"])
    if tests_passed <= 0 or evidence["related_tests_failed"]:
        raise ValueError("沒有通過的研究測試證據")
    frame = pd.read_csv(args.results / "seed_summary.csv")
    if (len(frame) != 3636 or frame.duplicated(SUMMARY_COLUMNS[:5]).any()
            or not (frame.seeds == 3).all() or not frame.active_seeds.between(0, 3).all()
            or not frame.threshold_pct.between(0, 100).all()
            or not (frame.threshold_pct % 1 == 0).all()):
        raise ValueError("公開彙總矩陣不完整或不符合固定三seed設計")
    table = sanitize_summary(frame)
    spec = importlib.util.spec_from_file_location("new_report", ROOT / "tools/research_reports/new_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.build_report(report_type="model_research", project="AIQuantTradingSystem",
        revision="threshold_sensitivity_v1", scope="既有候選模型之101個機率門檻研究，僅公開聚合統計")
    report.update(overall_status="WARN", summary="放寬機率門檻恢復部分成交，但未建立可部署的成本後優勢。",
        changes=["公開研究限定掃描工具、回歸測試與聚合結果；不修改部署設定。"],
        checks=[{"id": "EXEC-001", "name": "本機相關模組測試", "status": "PASS", "evidence": [f"{tests_passed} passed"]},
                {"id": "EXEC-002", "name": "原策略逐筆一致性", "status": "PASS", "evidence": ["216組55%及無AI核對"]},
                {"id": "MODEL-001", "name": "研究品質及實盤資格", "status": "BLOCKED", "evidence": ["成本與跨期穩健性證據不足"]}],
        metrics={"engineering": "PASS", "research_quality": "INSUFFICIENT_EVIDENCE", "live_eligible": False,
            "models": 18, "thresholds": 101, "threshold_scenarios": 10908, "baseline_scenarios": 108,
            "test_start_utc": "2025-09-17T15:00:00Z", "test_end_utc": "2026-03-19T09:30:00Z",
            "account_equity_validated": False, "aggregated_table": table},
        artifacts=[{"path": "docs/PROBABILITY_THRESHOLD_RESULTS_20261005.md", "role": "public_research_summary"}],
        residual_risks=["Test已反覆研究，不能挑最高數字當正式參數。", "seeds與101門檻非獨立行情樣本。",
            "假設費率及funding準備金，非稅後收益。", "只計結算權益，不包含持倉浮虧、保證金與清算。"],
        next_actions=["先行Selection選參、多時段向前驗證；未通過前不啟動SAC或實盤。"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
