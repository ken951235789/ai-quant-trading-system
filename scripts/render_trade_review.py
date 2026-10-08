"""從已稽核的機率門檻研究建立離線逐筆 K 線圖，不改動任何原始成品。"""

# 數值套件載入前限制執行緒。
# ruff: noqa: E402
import argparse
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import sys

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from audit_candidate_training_research import portable_path, verify_trade_table
from ai_quant_trading.operations.integrity import sha256_file, verify_artifact_manifest, build_artifact_manifest
from ai_quant_trading.research.candidate_contract import CandidateContract
from ai_quant_trading.research.probability_sweep import trade_metrics
from ai_quant_trading.research.trade_review import contained_file, review_trades, render_review_html, render_example_png
from ai_quant_trading.research.trade_review_bundle import export_review_bundle


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", required=True, type=Path)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sweep, study, output = args.sweep.resolve(), args.study.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("輸出目錄已存在；請指定新名稱，避免覆寫研究與圖表")
    verified = set(verify_artifact_manifest(sweep))
    plan = read_json(sweep / "plan.json")
    if sha256_file(study / "study.json") != plan["input_study_sha256"]:
        raise ValueError("模型研究來源與門檻掃描不同")
    study_manifest = read_json(study / "artifact_manifest.json")
    if sha256_file(study / "artifact_manifest.json") != plan["input_manifest_sha256"]:
        raise ValueError("模型研究清單已變動")

    def checked_study_file(relative):
        path = contained_file(study, relative)
        if sha256_file(path) != study_manifest["artifacts"][relative]["sha256"]:
            raise ValueError(f"研究成品雜湊不一致：{relative}")
        return path

    source = checked_study_file("frozen_fold_3.csv")
    source_hash = sha256_file(source)
    bars = pd.read_csv(source, usecols=["timestamp", "open", "high", "low", "close", "volume"])
    bars["timestamp"] = pd.to_datetime(bars.timestamp, utc=True)
    prices = bars[["open", "high", "low", "close"]].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("K 棒价格含非有限值或非正數")
    if ((bars.high < bars[["open", "close", "low"]].max(axis=1))
            | (bars.low > bars[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("OHLC 價格順序不合法")
    contracts = {}
    model_contracts = {}
    for record in read_json(study / "study.json")["records"]:
        run = portable_path(record["run_dir"], study / "runs")
        path = checked_study_file((run / "training.json").relative_to(study).as_posix())
        contract = CandidateContract(**read_json(path)["training_config"]["candidate_contract"])
        scenarios = {"base": contract, "zero": replace(contract, fee_bps=0., slippage_bps=0.,
                     spread_bps=0., funding_reserve_bps=0.),
                     "stress": replace(contract, slippage_bps=contract.slippage_bps + 3)}
        for scenario, item in scenarios.items():
            contracts[item.digest] = item.to_dict()
            model_contracts[record["name"], scenario] = item.digest
    all_rows = pd.read_csv(sweep / "all_thresholds.csv")
    all_rows["policy"] = "transformer"
    base = pd.read_csv(sweep / "baselines.csv")
    # 無 AI 不受種子或特徵版本影響；先核對相同排程，才去除重複顯示。
    if (base.groupby(["family", "split", "scenario"]).schedule_id.nunique() != 1).any():
        raise ValueError("無 AI 基準的種子 / 特徵排程並不一致")
    base = base.drop_duplicates(["family", "split", "scenario"]).copy()
    base["threshold_pct"] = -1
    rows = pd.concat([base, all_rows], ignore_index=True)
    summaries, schedules, cache, checks = [], [], {}, []
    fields = ["family", "policy", "variant", "seed", "split", "scenario", "threshold_pct",
              "contract_sha256", "trades", "actual_win_rate", "average_win", "average_loss",
              "payoff_ratio", "profit_factor", "mean_net"]
    for row in rows.to_dict("records"):
        digest = model_contracts[row["name"], row["scenario"]]
        if digest != row["contract_sha256"]:
            raise ValueError("CSV 成交契約不符合訓練設定")
        filename = row["schedule_file"]
        if filename not in verified:
            raise ValueError("排程未納入既有完整性清單")
        if filename not in cache:
            trades = pd.read_csv(contained_file(sweep, filename))
            if len(trades) and set(trades.contract_sha256) != {digest}:
                raise ValueError("逐筆成交契約不同")
            verify_trade_table(trades, contracts[digest]["cooldown_bars"])
            enriched = review_trades(bars, trades, source_hash)
            np.testing.assert_allclose([t["planned_rr"] for t in enriched],
                                       contracts[digest]["reward_r"], rtol=1e-9, atol=1e-9)
            metrics = trade_metrics(trades)
            cache[filename] = (len(schedules), metrics)
            schedules.append(enriched)
            checks.append({"path": filename, "sha256": sha256_file(sweep / filename),
                           "trades": len(trades), "exit_reasons": trades.exit_reason.value_counts().to_dict()})
        number, metrics = cache[filename]
        for field in ("trades", "actual_win_rate", "average_win", "average_loss", "payoff_ratio", "profit_factor", "mean_net"):
            expected = row[field]
            if pd.isna(expected):
                if metrics[field] is not None:
                    raise ValueError(f"{field} 空值不符")
            else:
                np.testing.assert_allclose(expected, metrics[field], atol=1e-11, rtol=1e-9)
        record = {k: None if pd.isna(row[k]) else row[k] for k in fields}
        record["schedule"] = number
        summaries.append(record)
    trades = [t for schedule in schedules for t in schedule]
    if not trades:
        raise ValueError("此研究完全沒有成交，不能建立逐筆示例")
    lo = max(0, min(t["entry"] for t in trades) - 128)
    hi = min(len(bars), max(t["exit"] for t in trades) + 129)
    window = bars.iloc[lo:hi]
    times = pd.to_datetime(window.timestamp, utc=True)
    # 只嵌入研究窗口附近 K 線，不把五年資料全部重複打包。
    packed = [[int(t.timestamp() * 1000), float(open_), float(high), float(low), float(close)]
              for t, open_, high, low, close in zip(times, window.open, window.high, window.low, window.close, strict=True)]
    windows = {}
    for item in plan["split_windows"]:
        family = next(r["family"] for r in summaries if item["name"].startswith("f3_" + r["family"] + "_"))
        split = "selection" if item["split"] == "validation" else item["split"]
        windows[family + "|" + split] = item["candidate_start_utc"].replace("+00:00", "") + " 至 " + item["candidate_end_utc"].replace("+00:00", "")
    payload = {"rows": summaries, "schedules": schedules, "offset": lo, "bars": packed,
               "contracts": contracts, "windows": windows}
    output.mkdir(parents=True)
    render_review_html(payload, output / "trades.html")
    export_review_bundle(payload, output / "desktop_bundle", source_sha256=source_hash,
                         label="BTC 15m 候選策略機率門檻研究")
    example = next(r for r in summaries if r["policy"] == "no_ai" and r["family"] == "original"
                   and r["scenario"] == "base" and r["split"] == "test")
    render_example_png(bars, schedules[example["schedule"]][0], output / "first_trade.png")
    save_json(output / "verification.json", {"input_manifest_files": len(verified),
        "summary_rows_checked": len(rows), "unique_schedules": len(schedules),
        "source_sha256": source_hash, "source_rows": len(bars), "embedded_bars": len(window),
        "schedules": checks, "account_equity_validated": False})
    selected = base[(base.split == "test") & (base.scenario == "base")]
    default_trades = sum(r["trades"] for r in summaries if r["policy"] == "transformer"
                         and r["threshold_pct"] == 55 and r["split"] == "test" and r["scenario"] == "base")
    lines = ["# BTC 15m 進出場與盈虧比", "", "[開啟離線互動 K 線](trades.html)", "", "![原突破基準第一筆](first_trade.png)", "",
        "## 實際結果", "", "以下是無 AI 基準，基本成本、Test 約 2025-09 至 2026-03；不是五年樣本外回測。",
        f"18 個模型在現行 55% 機率門檻的 Test 合計 {default_trades} 筆，不能計算模型的實際盈虧比。", "",
        "| 策略 | 成交 | 平均獲利 | 平均虧損幅度 | 淨盈虧比 | 實際勝率 | 每筆平均淨報酬 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    names = {"original": "原突破", "trend_pullback": "趨勢回調", "vwap_reversion": "VWAP 回歸"}
    for r in selected.itertuples():
        lines.append(f"| {names[r.family]} | {r.trades} | {r.average_win:.3%} | {r.average_loss:.3%} | {r.payoff_ratio:.3f}:1 | {r.actual_win_rate:.2%} | {r.mean_net:.3%} |")
    lines += ["", "## 出場規則", "", "此批候選事件採固定規則，Transformer 篩選是否進場；不是 SAC 自由選擇出場。",
        "1. 訊號棒收盤後，下一根 15m K 棒開盤，以含假設滑價、半價差的市價進場。",
        "2. 進場時固定停損 2 ATR、停利 4 ATR，預定報酬:風險為 2:1；不會隨後續 ATR 移動。",
        "3. 原突破在方向狀態消失（含中性）時退出；趨勢回調只在反向狀態退出；VWAP 回歸停用狀態退出。狀態只用上一根已收盤資料，於下一棒開盤退出。",
        "4. 最長持有 32 根完整 15m K 棒（8 小時），期限到的棒開盤退出，不再計入該棒後續觸價。",
        "5. 同棒同時觸及停損與停利，先停損。跳空越過停損採較差開盤价，再加入成本；不假設一定成交在停損價。",
        "6. 出場後冷卻 4 根訊號棒，下一合法訊號索引需大於出場索引 +4，仍於訊號下一棒成交。",
        "", "## 怎麼看圖", "", "雙擊 trades.html 即可，無須開交易主程式或連網。預設原突破、無 AI、基本成本、Test 第一筆，並未挑選獲利交易。",
        "上方可選三策略、無 AI / Transformer、兩組特徵、三 seeds、0-100% 門檻、Selection / Test、三種成本。箭頭、交易選單或下方成交列切換逐筆交易。",
        "圖上有進場 / 出場點、固定停利 / 停損線及本筆實現 R；時區預設台北 UTC+8，可改 UTC。提供深色與淺色。",
        "無 AI 沒有種子 / 特徵差異，因此相關選項停用。Transformer 的 0% 仍保留預測淨收益 >2 bps 濾網，不等於無 AI。",
        "停損 / 停利僅知道發生於哪根 K 棒，不知道棒內精確成交時刻；標記在該棒時間座標，非宣稱開盤成交。",
        "", "## 公式與成本", "", "- 實際盈虧比 = 獲利交易的平均淨報酬 / 虧損交易的平均淨損失絕對值。不是 Profit Factor。",
        "- 單筆實現 R = 淨報酬 / (初始停損價格距離 / 進場價)。停損可能虧超過 -1R，停利扣成本後通常少於 +2R。",
        "- 價差與滑價已反映在成交價；成交價損益只再扣手續費與 funding 準備金，不能重複扣价差和滑價。",
        "- 基本成本假設：單邊費率 5 bps、單邊滑價 2 bps、全價差 1 bps、每跨 UTC 00/08/16 結算正準備金 1 bps。壓力成本提高單邊滑價至 5 bps。",
        "- 各成本情境重新撮合，交易筆數與路徑可能改變。所有百分比以進場名目部位為分母，非帳戶本金回報。",
        "", "## 限制", "", "工程帳務核對 PASS；研究品質仍為證據不足，實盤資格 BLOCKED。",
        "Test 已反覆研究，不能從本圖事後挑最佳門檻。零成交不是低風險；此圖不計完整逐棒帳戶浮虧、保證金、清算或稅。",
        "Funding 是準備金而非實際歷史費率。出場後價格是歷史背景，不是任何模型的未来預測。",
        "本次沒有訓練、重跑撮合、更換模型、修改部署門檻或上傳 GitHub。"]
    with (output / "report.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    helper = ROOT / "tools/research_reports/new_report.py"
    spec = importlib.util.spec_from_file_location("skill_report", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.build_report(report_type="model_research", project="AIQuantTradingSystem",
        revision=plan["revision"] + " + working tree", scope="既有門檻研究之逐筆成交與 K 線對齊視覺稽核")
    report.update(overall_status="WARN", summary="新增離線逐筆圖表；未修改任何交易規則或授予實盤資格。",
        changes=["核對資料雜湊、成交時間、方向及成本帳務，計算單筆 R。", "全部已保存門檻與基準可切換，零成交不捏造資料。"],
        checks=[{"id": "CHART-001", "name": "來源、契約、帳務、時間與統計一致性", "status": "PASS", "evidence": ["verification.json"]},
                {"id": "LIVE-001", "name": "實盤資格", "status": "BLOCKED", "evidence": ["純歷史研究，不含完整帳戶風險驗證"]}],
        metrics={"engineering": "PASS", "research_quality": "INSUFFICIENT_EVIDENCE", "live_eligible": False,
                 "unique_schedules": len(schedules), "summary_rows_checked": len(rows),
                 "default_55_test_trades": default_trades,
                 "no_ai_base_test": selected[["family", "trades", "average_win", "average_loss", "payoff_ratio", "actual_win_rate", "mean_net"]].to_dict("records")},
        artifacts=[{"path": name, "role": role, "sha256": sha256_file(output / name)} for name, role in
                   [("trades.html", "offline_trade_chart"), ("first_trade.png", "static_first_trade"), ("report.md", "human_report"), ("verification.json", "validation_evidence")]],
        residual_risks=["歷史 Test 重複研究，不能當作全新 holdout。", "零成交與少量成交不提供可交易證據。",
                        "棒內價格先後、實際 funding、稅務、帳戶浮虧與清算未驗證。"],
        next_actions=["依退出原因研究成本與提前退出影響，再預先固定假說做跨期驗證，不從圖表挑選贏家。"])
    save_json(output / "report.json", report)
    build_artifact_manifest(output, files=[p for p in output.rglob("*") if p.is_file()])
    print(json.dumps({"output": str(output), "schedules": len(schedules), "rows": len(rows),
                      "html_mb": (output / "trades.html").stat().st_size / 1e6}, ensure_ascii=False))


if __name__ == "__main__":
    main()
