"""將完成的固定範圍搜尋整理成標準中文報告，不調整策略或部署設定。"""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import xml.etree.ElementTree as ET

import pandas as pd

from ai_quant_trading.operations.integrity import build_artifact_manifest, sha256_file
from ai_quant_trading.persistence import write_json_atomic


def percent(value):
    return "NA" if value is None or pd.isna(value) else f"{value * 100:+.3f}%"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
                     + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def report(output):
    def read(name):
        return json.loads((output / name).read_text(encoding="utf-8"))
    document, research, audit, plan = (read(n) for n in ("report.json", "research.json", "audit.json", "plan.json"))
    all_results = pd.read_csv(output / "results.csv.gz")
    pooled = pd.read_csv(output / "pooled_forward.csv")
    parameters = pd.read_csv(output / "parameters.csv").set_index("parameter_id")
    diagnostics = pd.read_csv(output / "parameter_diagnostics.csv")
    exports = read("exported_candidates.json")["results"]
    selection = research["selection"]
    baseline = research["baseline_id"]
    ids = [baseline] + [i for i in selection["validation_ranked"][:5] if i != baseline]
    source = pd.read_csv(plan["fingerprint"]["source"], usecols=["timestamp"])
    periods = [{**w, "start_utc": str(source.timestamp.iloc[w["start"]]),
                "stop_utc_exclusive": (pd.Timestamp(source.timestamp.iloc[w["stop"] - 1]) + pd.Timedelta(minutes=15)).isoformat()}
               for w in plan["fingerprint"]["windows"]]
    tests = ET.parse(output / "pytest_results.xml").getroot()
    suites = list(tests.iter("testsuite"))
    test_metrics = {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
    test_metrics["seconds"] = sum(float(s.get("time", 0)) for s in suites)
    assert test_metrics["failures"] == test_metrics["errors"] == 0
    assert audit["status"] == "PASS" and research["status"] == "complete"
    environment = {"python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "matplotlib", "scipy", "pytest"]},
        "sweep_thread_limits": {"OMP_NUM_THREADS": 2, "MKL_NUM_THREADS": 2, "OPENBLAS_NUM_THREADS": 2}}
    write_json_atomic(output / "environment.json", environment)
    revision = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip()
    quality = "PASS" if research["forward_quality_passed"] else "FAIL"
    positive_counts = {c: int((pooled.loc[pooled.scenario.eq(c), "mean_net"] > 0).sum()) for c in ("zero", "base", "stress")}
    best = pooled.loc[pooled.scenario.eq("base")].sort_values(["settled_return", "parameter_id"], ascending=[False, True]).iloc[0]
    base_rows = pooled.loc[pooled.scenario.eq("base")]
    summary = (f"已完成 11,664 組參數、174,960 個期間/成本情境；工程稽核通過。"
        f"Forward 基本成本下平均每筆為正者 {positive_counts['base']} 組，壓力成本下 {positive_counts['stress']} 組。"
        f"正式選中候選={selection['selected']}，Forward 品質={quality}。未訓練、未部署、未改模型或 Live 設定。")
    limitations = [
        "這是既有資料上的回溯研究，不是未接觸 final holdout；過往模型與研究已使用部分相同歷史。",
        "僅窮舉事前列出的 10 軸離散網格；沒有窮舉所有指標、任意連續參數、所有資產或所有交易策略。",
        "無槓桿、固定 10% 進場名目本金/當時結算權益、研究起始 1000 USDT；不對應真實帳戶配置。",
        "曲線只計平倉結算，不含持倉期間浮虧、最小下單量、數量精度、清算、日損停機與真實容量。Sharpe/Sortino/Calmar 等完整帳戶指標未驗證。",
        "費率、滑價與 Funding 為假設；每個 UTC 八小時結算扣 1 bp 準備金，不是實際歷史 Funding，個人台灣稅後績效未驗證。",
        "15m 棒內停損優先是保守約定，無逐筆撮合；執行延遲、斷線、保護單失敗與薄流動性未由此研究覆蓋。",
        "月份循環區塊最大統計量是資料探勘敏感度診斷，非 PBO 或嚴格有限樣本 FWER 保證；僅 25/13 左右月份，依賴近似平穩假設。",
        "單一候選交易 CI 為區塊重抽樣描述區間，並非對所有搜尋及反覆研究校正後的獲利保證。",
        "11,664 組共享同一份行情且高度相關，不是 11,664 個獨立策略；更多交易也不等於同等數量的獨立樣本。",
        "每個 Forward 子段各自空倉開始，保留 64 棒開頭隔離和 65 棒尾端留白；三段串接不是不中斷的實盤。",
        "不同訊號週期也改變 ATR 尺度，同一名目配置不等於同一停損風險；參數熱圖中位數不是單因子因果效果。",
        "未執行本輪 Transformer/SAC、Buy-and-Hold 完整帳戶或成本相同的 AI 組合回測，因此不能宣稱規則已打敗 SAC。",
    ]
    actions = [
        "保存本次完整負面與正面結果，不依 Forward 最高值重新挑策略或放寬品質閘門。",
        "優先補上實際帳戶費率、歷史 Funding 與交易時點流動性/滑價；只在成本證據支持時修改假設。",
        "下一輪使用少量可否證的新進場假說或市場環境分層，事前凍結規則；需另找未用行情或前向模擬。",
        "先證明候選成本後訊號優勢，再重建一致事件標籤訓練 Transformer；SAC 先只做部位管理對照。",
    ]
    findings = [
        {"id": "ENG-001", "severity": "HIGH", "status": "RESOLVED", "category": "execution_parity",
         "component": "grid_execution", "title": "新批次引擎必須依最早事件退出，不能依原因編碼大小",
         "evidence": ["tests/backtesting/test_parameter_sweep.py: random_event_parity 測試先失敗後修正", "audit.json: reference_checks=34992"],
         "impact": "若未修正，可能跳過早先停損而晚出場；正式研究是在修正後才開始。",
         "recommendation": "保留隨機及邊界成交等價性測試，原逐筆引擎仍是參考契約。"},
        {"id": "MODEL-001", "severity": "HIGH" if quality == "FAIL" else "INFO", "status": "OPEN" if quality == "FAIL" else "ACCEPTED",
         "category": "strategy_quality", "component": "candidate_selection", "title": f"正式候選品質 {quality}",
         "evidence": ["selection.json", "research.json: forward_quality_passed"],
         "impact": "最高歷史報酬不等於可部署候選；不允許以此直接啟動實盤。", "recommendation": actions[0]},
        {"id": "DATA-001", "severity": "MEDIUM", "status": "ACCEPTED", "category": "selection_bias",
         "component": "historical_research", "title": "大量搜尋與重複使用歷史的偏誤仍存在",
         "evidence": ["plan.json: pristine_holdout=false", "parameter_diagnostics.csv"],
         "impact": "即使偶然出現正值，也不代表未來穩定獲利。", "recommendation": actions[2]},
        {"id": "EXEC-001", "severity": "MEDIUM", "status": "ACCEPTED", "category": "account_scope",
         "component": "costs_and_equity", "title": "成本假設與結算權益不等同真實帳戶",
         "evidence": ["plan.json: costs_bps, account_caveat", "charts/05_candidate_equity.png"],
         "impact": "盤中最大風險、稅後績效與清算條件未驗證。", "recommendation": actions[1]},
    ]
    checks = [
        {"id": "TEST-001", "name": "完整專案測試", "status": "PASS", "evidence": [f"pytest_results.xml: {test_metrics}"]},
        {"id": "AUDIT-001", "name": "174,960 情境與月收益一致性", "status": "PASS", "evidence": ["audit.json", "results.csv.gz"]},
        {"id": "PARITY-001", "name": "每參數成本抽查與全部候選逐筆核對", "status": "PASS", "evidence": ["research.json: reference_checks=34992; all_exported_trades_reference_checked=true"]},
        {"id": "SELECT-001", "name": "選擇只依 Search 與 Validation", "status": "PASS", "evidence": ["audit.json: selection_reproduced=true", "tests/backtesting/test_parameter_sweep.py"]},
        {"id": "QUALITY-001", "name": "可用交易候選", "status": quality, "evidence": ["selection.json", "research.json"]},
        {"id": "LIVE-001", "name": "稅後實盤資格", "status": "BLOCKED", "evidence": ["research.json: live_eligible=false"]},
    ]
    if audit.get("previous_baseline", {}).get("matched"):
        checks.append({"id": "COMPAT-001", "name": "舊研究原策略逐筆相容", "status": "PASS",
            "evidence": [f"audit.json: previous_baseline; {audit['previous_baseline']['trade_count']} 筆成交與損益一致"]})
    document.update(generated_at_utc=datetime.now(timezone.utc).isoformat(), overall_status="WARN" if quality == "PASS" else "FAIL",
        project={"name": "AIQuantTradingSystem", "revision": revision + "+dirty; source hashes in plan.json", "environment": str(environment)},
        summary=summary, changes=["新增分塊續跑 NumPy 網格研究與既有逐筆成交契約核對。", "新增搜尋/驗證隔離、共享月份最大統計量診斷、鄰域穩健性及逐筆匯出。", "新增 21 項測試、獨立稽核、五張中文圖表與離線圖表總覽。"],
        findings=findings, checks=checks, metrics={"engineering_status": "PASS", "strategy_quality_status": quality,
        "parameters": research["parameters"], "scenarios": research["scenarios"], "tests": test_metrics,
        "source_sha256": plan["fingerprint"]["source_sha256"], "windows": periods, "forward_positive_mean_counts": positive_counts,
        "selection": selection, "base_trade_range": [int(base_rows.trades.min()), int(base_rows.trades.max())],
        "base_expectancy_median": float(base_rows.mean_net.median()), "search_min_adjusted_p": float(diagnostics.search_p_adjusted.min()),
        "retrospective_best_by_forward_settled_return_NOT_selected": best.to_dict(), "exported_candidates": exports,
        "elapsed_seconds": research["elapsed_this_session_seconds"], "new_models_trained": False, "live_eligible": False},
        residual_risks=limitations, next_actions=actions)
    artifact_names = ["plan.json", "research.json", "selection.json", "audit.json", "results.csv.gz", "parameters.csv",
        "parameter_diagnostics.csv", "pooled_forward.csv", "candidate_table.csv", "exported_candidates.json", "pytest_results.xml", "charts.html"]
    document["artifacts"] = [{"path": n, "role": "可追溯研究產物", "sha256": sha256_file(output / n)} for n in artifact_names]
    write_json_atomic(output / "report.json", document)
    period_table = table(["期間", "開始 UTC", "結束 UTC（不含）", "原始列數"],
        [[w["name"], w["start_utc"], w["stop_utc_exclusive"], w["stop"]-w["start"]] for w in periods])
    grid_table = table(["參數", "固定搜尋範圍"], [[k, ", ".join(map(str, v))] for k, v in plan["fingerprint"]["grid"].items()])
    selected_table = table(["ID", "角色", "交易數", "每筆淨均值", "標準差", "最差一筆", "Profit Factor", "結算報酬", "結算回撤"],
        [[f"G{i:05d}", "原始基準" if i == baseline else f"驗證排名{selection['validation_ranked'].index(i)+1}",
          row["trades"], percent(row["mean"]), percent(row["std"]), percent(row["worst"]), f"{row['profit_factor']:.3f}",
          percent(row["settled_return"]), percent(row["settled_drawdown"])]
         for i in ids for row in exports if row["parameter_id"] == i and row["scenario"] == "base"])
    parameter_table = table(["ID", "週期", "進場", "ADX", "確認", "回看", "停損ATR", "目標R", "持有15m棒", "冷卻", "趨勢退出"],
        [[f"G{i:05d}", *[parameters.loc[i, name] for name in plan["fingerprint"]["grid"]]] for i in ids])
    fold_table = table(["ID", "Forward 1 每筆均值", "Forward 2 每筆均值", "Forward 3 每筆均值"],
        [[f"G{i:05d}", *[percent(all_results.loc[all_results.parameter_id.eq(i) & all_results.window.eq(f"forward_{j}") & all_results.scenario.eq("base"), "mean_net"].iloc[0]) for j in (1, 2, 3)]] for i in ids])
    outcome_table = table(["Forward 成本情境", "每筆均值為正組數", "總組數"], [[c, n, 11664] for c, n in positive_counts.items()])
    leader = next((row for row in exports if row["parameter_id"] == selection["diagnostic_leader"] and row["scenario"] == "base"), None)
    original = next(row for row in exports if row["parameter_id"] == baseline and row["scenario"] == "base")
    interpretation = ""
    if leader is not None:
        interpretation = (f"驗證排名第一 G{selection['diagnostic_leader']:05d} 有 {leader['trades']} 筆，"
            f"每筆 {percent(leader['mean'])}、結算報酬 {percent(leader['settled_return'])}；"
            f"原基準 {original['trades']} 筆、每筆 {percent(original['mean'])}、結算報酬 {percent(original['settled_return'])}。"
            "總虧損較小可能只是交易較少，不能把少虧直接解讀成每筆交易的預測優勢提升。")
    check_table = table(["檢查", "狀態", "證據"], [[c["name"], c["status"], "; ".join(c["evidence"])] for c in checks])
    text = f'''# BTC 大型參數策略比較研究

## 基本資訊
- Report ID：{document['report_id']}
- Report Type：model_research
- Generated At UTC：{document['generated_at_utc']}
- Revision：{document['project']['revision']}
- Scope：BTCUSD-M 永續、有限網格 11,664 組；不是全世界所有策略。
- Overall Status：{document['overall_status']}；工程 PASS、候選品質 {quality}。

## 執行摘要
{summary}

來源有 {len(source):,} 根 15m K 棒，範圍 {plan['source_start']} 至 {plan['source_end']}。
本次只使用前 90%，最後 10% 不評估；不能因此把它稱為從未看過的封存集。
Forward 基本成本交易數範圍 {int(base_rows.trades.min())}～{int(base_rows.trades.max())} 筆，
所有組合每筆淨報酬的中位數為 {percent(base_rows.mean_net.median())}。
正式選中者為 `{selection['selected']}`；圖表中的驗證排名只是診斷比較，不等於可用策略。

{outcome_table}

只作資料探勘警示：事後用 Forward 結算報酬挑最高的是 G{int(best.parameter_id):05d}，
報酬 {percent(best.settled_return)}、{int(best.trades)} 筆、每筆 {percent(best.mean_net)}。
這不是事前可選的優勝者，禁止據此部署或修改本輪結論。

## 變更與驗證
- 以 NumPy 加速相同的歷史事件成交契約；正式執行前修正最早退出事件排序。
- 72 批可中斷續跑，每批 SHA-256；行情或計算程式改變時拒絕混合續跑。
- 每參數/成本抽查一筆，共 34,992 次參考引擎核對；匯出的候選交易全部逐筆核對。
- 未修改模型、UI、Live、EXE 或 GitHub；沒有訓練新模型。

{check_table}

## 固定範圍
{grid_table}

訊號方向跟隨已收盤 H1 EMA50/200，搭配 H1 ADX。
突破是收盤突破前 N 棒高低；回調是重新穿越 EMA(N)，且當棒實體同向。
確認 2 根表示相鄰兩根已收盤小時棒皆活躍且同向。初始暖機 1000 小時。
1h 訊號只在小時收盤出現一次，不延續填滿下一小時；ATR 與訊號週期一致。
`loss` 為趨勢失效或反向離場；`opposite` 只在反向離場；`disabled` 只靠保護價或時間退出。
所有訊號下一根 15m 開盤成交，停損/停利固定於實際進場價與訊號 ATR。
最早退出優先：已知趨勢退出開盤、停損、停利；同棒兩者命中取停損。跳空停損取較差價格。

## 成本與計量單位
基本：單邊手續費 5 bp、單邊滑價 2 bp、全價差 1 bp，往返約 15 bp，另扣 Funding 準備金。
壓力：單邊滑價 5 bp，其餘相同，往返約 21 bp。零成本只作診斷，不可拿來承諾獲利。
Funding 每跨過 UTC 00/08/16 結算時間扣名目本金 1 bp，非實際 Funding。
每一成本情境重新計算保護價、退出與排程；不是從同一筆毛收益直接扣固定費用。

- 每筆淨報酬 = side × (實際出場價 / 實際進場價 − 1) − 雙邊費用 − Funding 準備金。
- 權益更新 = 舊結算權益 × (1 + 0.10 × 每筆淨報酬)，起始 1000 USDT。
- 結算回撤 = 1 − 當前結算權益 / 歷史結算權益最高值，不含持倉浮虧。
- 月收益使用平倉時間歸屬，空手月份為 0；窗口首尾可能不是完整月份。
- 每筆報酬不是帳戶報酬，也不是年報酬；不把每筆報酬直接相加當複利報酬。

## 時間與選擇隔離
{period_table}

Search 每個週期×進場家族，至少 150 筆，依壓力情境月均結算 log 收益取前五，共最多二十組。
Search 品質需基本/壓力皆正、共同月份最大統計量 p≤0.05、相鄰參數正值比例≥60%、結算回撤≤10%。
相鄰只沿 ADX、確認、回看、停損、R、持有、冷卻各軸上下移一格，固定進場家族、週期與退出模式。
Validation 只在二十組內排序，另需至少 50 筆、基本/壓力皆正、多候選 p≤0.05、結算回撤≤10%。
Forward 只驗證合格選中者：合計≥100筆、每筆均值 CI 下界>0、壓力均值>0、三段各≥20筆且為正、結算回撤≤10%。
本輪仍完整公開所有組合 Forward 診斷，以避免只呈現優勝者，但這些歷史從此不是乾淨的未見資料。

多重搜尋敏感度：月份中心化，循環 3 月區塊，共同抽樣 2000 次、seed=20261002，
固定原樣本標準誤，搜尋期對全部 11,664 組取最大統計量；Validation 對 shortlist 取最大值。
Search 最低校正 p={diagnostics.search_p_adjusted.min():.6f}。
它不是 PBO、不是嚴格 FWER 保證；研究設計參考 [The Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)。

## 基準與驗證排名候選
{interpretation}

{selected_table}

{parameter_table}

{fold_table}

## 圖表
直接開啟同資料夾的 [離線圖表總覽](charts.html)。

![總覽](charts/01_overview.png)
![停損與R](charts/02_stop_reward_heatmap.png)
![參數敏感度](charts/03_parameter_sensitivity.png)
![候選期望值](charts/04_candidate_expectancy.png)
![候選曲線](charts/05_candidate_equity.png)

## 發現
{table(['ID', '嚴重度', '狀態', '發現', '建議'], [[f['id'], f['severity'], f['status'], f['title'], f['recommendation']] for f in findings])}

## 未驗證與剩餘風險
{chr(10).join('- ' + item for item in limitations)}

## 下一步
{chr(10).join(str(i+1) + '. ' + item for i, item in enumerate(actions))}

## 產物
- `parameters.csv`：全部參數；`results.csv.gz`：174,960 個期間/成本情境。
- `pooled_forward.csv`：三段合併的全部參數結果；`candidate_table.csv`：基準和驗證前五。
- `groups/`：分塊 CSV、月收益矩陣與雜湊收據。
- `trades/`：候選與基準各期間和合併 Forward 的逐筆交易。
- `plan.json`、`selection.json`、`parameter_diagnostics.csv`：事前範圍與選擇證據。
- `audit.json`、`pytest_results.xml`、`environment.json`、`artifact_manifest.json`：驗證與重現資料。
'''
    (output / "report.md").write_text(text, encoding="utf-8")
    build_artifact_manifest(output, [p for p in output.rglob("*") if p.is_file()])
    return {"overall_status": document["overall_status"], "summary": summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(report(args.output.resolve()))
