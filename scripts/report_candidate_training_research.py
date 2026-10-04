"""填寫已建立的專案報告骨架，附測試、唯讀稽核及煙霧結果。"""

import argparse
import json
from pathlib import Path
import platform
import subprocess
import xml.etree.ElementTree as ET

from audit_candidate_training_research import audit
from run_candidate_training_research import manifest_all, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--report-root", required=True, type=Path)
    parser.add_argument("--tests", type=Path, help="本次最終 pytest XML；省略時使用報告目錄內預設檔")
    args = parser.parse_args()
    root = args.report_root.resolve()
    test_path = args.tests.resolve() if args.tests else root / "pytest_results.xml"
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    if report["overall_status"] != "DRAFT":
        raise ValueError("報告已完成，禁止覆寫；請另外建立新報告")
    study = json.loads((args.study / "study.json").read_text(encoding="utf-8"))
    plan = json.loads((args.study / "plan.json").read_text(encoding="utf-8"))
    checked = audit(args.study)
    write_json(root / "audit.json", checked)
    suites = ET.parse(test_path).getroot().findall("testsuite")
    tests = {key: sum(int(s.get(key, 0)) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
    if tests["failures"] or tests["errors"] or not tests["tests"]:
        raise ValueError("必要測試未通過")
    revision = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    changes = ["新增版本化候選契約，複用現有複合訊號與成交引擎，舊設定不變",
        "統一新研究相對量能定義，保存成本、特徵、來源及標籤結束契約",
        "加入固定路徑成本對帳、MFE/MAE 界線及分離的停損後診斷",
        "Train-only 重疊權重、候選模型隔離、三方濾網及成本重新撮合",
        "完整四格配對向前矩陣加成本特徵消融；本次只執行三種候選各一輪煙霧",
        "新增單元測試、唯讀成品稽核、CSV 在內的雜湊清單及中文教學"]
    residuals = ["一輪煙霧只證明流程，正式 3-seed/135-run 矩陣尚未執行",
        "歷史已被反覆研究；封存新行情前需盤點全專案已觀察日期，不能把本輪尾端當新 holdout",
        "手續費、滑價、價差及雙向 funding 準備金是假設，沒有真實撮合或個人稅後驗證",
        "退出棒路徑未知，MFE/MAE 是保守界線；Maker 掛單、流動性容量未模擬",
        "僅研究不重疊單策略名目收益，未計完整持倉浮虧、組合曝險、保證金與清算",
        "成本情境凍結模型分數後重新排程；未宣稱模型學過零成本或壓力成本分布",
        "重疊權重、區塊 CI、有限 seeds 都不能消除多重搜尋及市場變化風險",
        "研究模型仍禁止部署，SAC/Paper/Live、GitHub、EXE 皆未更動"]
    actions = ["先確認本報告及契約，再批准一個預先登記的最小配對研究；不根據已看 Test 挑策略",
        "完整研究前量測完整時間窗 3-5 epochs 的成本，再確定是否採完整 135-run 或縮小預登記矩陣",
        "若不能跨期勝過 Train 常數及統計濾網，停止擴大模型；若樣本不足不放寬門檻硬湊",
        "通過成本壓力、統計與新行情觀察後，才評估 SAC 的固定/波動率/學習部位对照"]
    feature_study = plan["version"] == "candidate_feature_study_v1"
    title = "候選交易契約與訓練流程驗收"
    summary_text = "P0/P1 工程與三策略煙霧完成；正式模型品質證據不足，實盤與 P2 仍封鎖。"
    document = "docs/CANDIDATE_TRAINING_CONTRACT_20261003.md"
    if feature_study:
        title = "候選特徵精簡、成交資訊與 SAC 前置評估"
        summary_text = "五組特徵與三種候選的本機工程流程完成；煙霧不能證明正期望值，SAC 不啟動。"
        document = "docs/CANDIDATE_FEATURE_STUDY_20261003.md"
        changes = ["保留既有契約雜湊與模型相容，新增版本化精簡/成交/交易空間特徵",
            "沿用本機成交欄位，逐根核對 OHLCV；研究來源不覆寫、不下載、不含未來外部欄位",
            "主動成交不平衡只使用已收盤的 15m/1h/4h；活動分母為前20棒",
            "加入已知關鍵價距離、回調幅度與有上限的訊號/趨勢年齡，無未來 pivot",
            "五組固定 MSE 公平比較，沿用相同候選、成本、55%門檻、標籤與小型V3",
            "新增特徵因果、資料失敗封鎖、跨週期收盤、舊契約及 Train-only 測試",
            "產生明確的 SAC 延後評估，不更改既有 require_sizing_gate 或部署模型"]
        residuals[0] = "三策略乘五特徵組各一輪、單seed只驗證工程；三seed跨期正式135次研究未執行"
        residuals += ["成交特徵不是委託簿、真實價差或持倉量；Funding/OI/深度未冒充完整五年資料",
            "價格空間是事前已知價位的幾何距離，不是真實掛單或保證能達到的獲利空間",
            "新增特徵有效性未被證實；同一測試歷史已反覆觀察，不能依煙霧最佳分数挑贏家"]
        actions = ["保留五組預登記假說，先批准完整時間窗3至5輪資源測量，再決定正式矩陣",
            "使用相同3 seeds與向前切分比較均值/分散/最差成本情境，不把單seed煙霧當結果",
            "若沒有跨期樣本外成本後改善，停止擴模與SAC訓練，回頭檢查候選機制",
            "SAC需本候選契約合法跨擬合OOS訊號及既有品質閘門；3seed研究不自動解除原五seed要求"]
    report.update(project={"name": "AIQuantTradingSystem", "revision": revision + " + preserved working changes",
        "environment": f"Windows; Python {platform.python_version()}; CPU threads=2"},
        overall_status="WARN", summary=summary_text,
        changes=changes, findings=[
            {"id": "DATA-001", "severity": "MEDIUM", "status": "RESOLVED", "category": "contract",
             "component": "candidate_contract", "title": "策略與指標語意以版本隔離",
             "evidence": ["src/ai_quant_trading/research/candidate_contract.py", "tests/research/test_candidate_contract.py"],
             "impact": "避免複合回測與原突破訓練標籤混用", "recommendation": "保留舊結果；新研究明確指定契約"},
            {"id": "MODEL-001", "severity": "MEDIUM", "status": "OPEN", "category": "evidence",
             "component": "Transformer", "title": "僅完成煙霧，無正式獲利證據",
             "evidence": [str(args.study / "study.json")], "impact": "不能升級 Champion 或訓練部署 SAC",
             "recommendation": "先批准最小配對向前研究，再收集新的封存行情"}],
        checks=[{"id": "TEST-001", "name": "完整 pytest 回歸", "status": "PASS", "evidence": [str(test_path), str(tests)]},
            {"id": "AUDIT-001", "name": "來源、模型隔離、成本帳務與參考引擎", "status": "PASS", "evidence": [str(root / "audit.json")]},
            {"id": "MODEL-001", "name": "正式模型品質", "status": "BLOCKED", "evidence": ["只完成一輪煙霧，正式配對矩陣未獲授權執行"]},
            {"id": "LIVE-001", "name": "實盤與 SAC 升級", "status": "BLOCKED", "evidence": ["live_eligible=false; p2_sac=BLOCKED"]}],
        metrics={"tests": tests, "audit": checked, "smoke_seconds": study["elapsed_seconds"],
                 "formal_plan_runs": plan["formal_runs"], "records": study["records"],
                 "paired_comparison": study["paired_comparison"], "sac_assessment": study.get("sac_assessment"),
                 "flow_source": plan.get("flow_source")},
        artifacts=[{"path": str(args.study.resolve()), "role": "三候選煙霧與所有模型/交易產物"},
                   {"path": str(test_path), "role": "完整測試"},
                   {"path": document, "role": "中文操作及限制"}],
        residual_risks=residuals, next_actions=actions)
    write_json(root / "report.json", report)
    lines = [f"# {title}", "", f"Report ID：{report['report_id']}",
        f"版本：{report['project']['revision']}", "", "## 結論", report["summary"],
        f"完整測試：{tests['tests']} 個測試計數（包含 subtests），失敗 {tests['failures']}，錯誤 {tests['errors']}。",
        f"共 {len(study['records'])} 個模型各一輪，CPU 2 執行緒，共 {study['elapsed_seconds']:.1f} 秒。", "",
        "## 煙霧結果", "候選數不等於成交數；一輪結果不能作為選模或獲利結論。",
        "| 策略/版本 | 特徵 | Train 候選 | Calibration | Selection | Test 候選 | 無 AI 交易 | 統計交易 | AI 交易 | 無 AI 每筆均值 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in study["records"]:
        s, p = r["samples"], r["test_policies"]
        mean = p["no_ai_base"]["mean"]
        label = r['family'] + (f"/{r['variant']}" if feature_study else "")
        lines.append(f"| {label} | {r.get('feature_count', '-')} | {s['train']} | {s['calibration']} | {s['selection']} | {s['test']} | "
            f"{p['no_ai_base']['trades']} | {p['statistical_base']['trades']} | {p['transformer_base']['trades']} | "
            + (f"{mean:.3%} |" if mean is not None else "無樣本 |"))
    lines += ["", "AI 零交易的期望值為 null，不視為 0% 收益或風控成功。",
        "三策略完整分布、最差交易與成本壓力見 study.json；本表不是帳戶報酬。", "",
        "## 變更"] + [f"- {x}" for x in changes]
    lines += ["", "## 稽核", f"完整性檔案 {checked['manifest_files']}、模型 {checked['models']}、"
        f"預測 {checked['prediction_rows']} 列、政策交易 {checked['nonoverlap_trade_rows']} 列、"
        f"慢速參考探針 {checked['slow_reference_probes']}。",
        "政策交易列包含不同成本及濾網重播，不能當成獨立行情樣本。",
        "", "## 操作", f"完整指令與契約：`{document}`。",
        "入口：`scripts/run_candidate_training_research.py`，預設只產生計畫；新五組需 --feature-study，--smoke 僅一輪。",
        "完整 --run-research --seeds 42 137 2026 為 135 次模型訓練，尚未開始；成交來源與完整指令見文件。",
        "煙霧含準備與歸因，不按比例推估完整135次訓練時間；先完整窗 benchmark 再排資源。",
        "", "## 未驗證"] + [f"- {x}" for x in residuals]
    lines += ["", "## 下一步"] + [f"{i}. {x}" for i, x in enumerate(actions, 1)]
    if feature_study:
        lines += ["", "## SAC 評估", study["sac_assessment"]["recommendation"]]
        lines += [f"- {reason}" for reason in study["sac_assessment"]["reasons"]]
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest_all(root)
    print(root / "report.md")


if __name__ == "__main__":
    main()
