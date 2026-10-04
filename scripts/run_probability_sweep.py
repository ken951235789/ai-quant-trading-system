"""對已保存的候選模型預測掃描 101 個機率門檻；不訓練、不部署、不挑選正式門檻。"""

# 數值套件載入前先限制執行緒，並接入專案 src 路徑。
# ruff: noqa: E402

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import torch

from audit_candidate_training_research import audit, portable_path, verify_trade_table
from ai_quant_trading.operations.integrity import build_artifact_manifest, sha256_file, verify_artifact_manifest
from ai_quant_trading.research.candidate_contract import CandidateContract, candidate_outcomes, prepare_candidate_frame
from ai_quant_trading.research.event_attribution import attribute_events
from ai_quant_trading.research.probability_sweep import ThresholdReplay, aggregate_seeds, assert_trade_parity, schedule_id, trade_metrics


LABELS = {"original": "原突破", "trend_pullback": "趨勢回調", "vwap_reversion": "VWAP 回歸",
          "F_existing": "舊特徵", "F_compact_combined": "新精簡特徵"}


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def plot_results(rows, baselines, output):
    """每條細線是一個 seed；不把多個模型視為額外獨立行情。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    font = Path("C:/Windows/Fonts/msjh.ttc")
    if font.exists():
        plt.rcParams["font.family"] = FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    colors = {"zero": "#808080", "base": "#006b9a", "stress": "#c54242"}
    scenarios = {"zero": "零成本", "base": "基本成本", "stress": "壓力成本"}
    metrics = [("mean_net", 10000, "每筆淨期望值 (bps)", "expectancy.png"),
               ("trades", 1, "實際不重疊成交數", "trade_count.png"),
               ("settled_return_10pct", 100, "10% 配置結算報酬 (%)，非完整帳戶回測", "settled_return.png")]
    test = rows[rows.split == "test"]
    for metric, scale, title, filename in metrics:
        fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True)
        for i, family in enumerate(("original", "trend_pullback", "vwap_reversion")):
            for j, variant in enumerate(("F_existing", "F_compact_combined")):
                ax = axes[i, j]
                subset = test[(test.family == family) & (test.variant == variant)]
                for scenario, color in colors.items():
                    group = subset[subset.scenario == scenario]
                    for _, seed in group.groupby("seed"):
                        ax.plot(seed.threshold_pct, seed[metric] * scale, color=color, alpha=.23, lw=.8)
                    mean = group.groupby("threshold_pct")[metric].mean()
                    ax.plot(mean.index, mean.to_numpy() * scale, color=color, label=scenarios[scenario], lw=1.7)
                baseline = baselines[(baselines.family == family) & (baselines.variant == variant)
                                     & (baselines.split == "test") & (baselines.scenario == "base")]
                ax.axhline(baseline[metric].mean() * scale, color="#278347", ls=":", label="無 AI / 基本成本")
                ax.axvline(55, color="#242424", ls="--", lw=.9, label="現行 55%")
                ax.axhline(0, color="#999999", lw=.6)
                ax.set_title(f"{LABELS[family]} | {LABELS[variant]}", fontsize=11)
                ax.set_xlim(0, 100)
                ax.grid(alpha=.14)
                if i == 2:
                    ax.set_xlabel("預測成功機率門檻 (%)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(.5, .953), fontsize=9)
        fig.suptitle(title, fontsize=15, y=.99)
        note = ("Test 已用於研究。細線：3 seeds；粗線：全部 seeds 的成交數平均（含零成交）。"
                if metric == "trades" else
                "Test 已用於研究。細線：3 seeds；粗線：有成交 seeds 的平均。零成交報酬留白，不代表零風險。")
        fig.text(.5, .012, note, ha="center", fontsize=10)
        fig.tight_layout(rect=(0, .035, 1, .925))
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)


def write_report(output, plan, rows, baselines, aggregate, verification, elapsed):
    spec = importlib.util.spec_from_file_location("skill_report", ROOT / "tools/research_reports/new_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.build_report(report_type="model_research", project="AIQuantTradingSystem",
        revision=plan["revision"], scope="18 個既有模型，101 機率門檻，Selection/Test 與三種成本之研究敏感度")
    report["project"]["environment"] = f"{platform.platform()}; Python {platform.python_version()}; CPU threads=2"
    test = rows[(rows.split == "test") & (rows.scenario == "base")]
    test_55 = test[test.threshold_pct == 55]
    risks = ["Test 已反覆研究；101 門檻與 seeds 互相依賴，不能挑最高結果宣稱新的 holdout 或 Champion。",
        "95% block CI 為逐排程描述性區間，未校正多重比較；少於8筆不計算，避免極小樣本區塊抽樣退化。8筆並不代表統計充分。",
        "基礎市場約五年，但本次 Test 只有單一約六個月窗口，不是五年樣本外績效。",
        "各成本情境重新撮合與排程；模型仍預測基本成本標籤，沒有為零成本或壓力成本重訓。",
        "手續費、滑價、價差為假設；funding 是正準備金而非歷史實際費率，稅務未驗證。",
        "10% 配置結算權益未包含持倉浮虧、棒內保證金/清算、日損停機、最小下單額及真實資金容量。",
        "盈虧比與損益兩平勝率為事後描述；成功機率標籤淨收益 >2bps，實際勝率則是 >0。",
        "沒有 SAC 訓練、實盤、部署更換、GitHub 上傳或 EXE 打包。"]
    actions = ["先觀察相鄰門檻、三 seeds、Selection/Test 與壓力成本是否一致，不追逐單一最高數字。",
        "最小下一輪：只在先行校準/選擇窗口預先固定一個規則，對相同候選跑至少三個非重疊向前測試窗口。",
        "若成本後優勢無法跨期重現、依賴少數贏家、或壓力測試轉負，停止升級 SAC/實盤；不要用槓桿補償。",
        "若已有跨期改善，再完成逐棒 MTM、真實費率/funding、前瞻封存與紙上交易驗證。"]
    report.update(overall_status="WARN", summary="已完成全門檻敏感度；工程驗證與研究獲利資格分離，未選擇或部署新門檻。",
        changes=["新增研究限定的 ThresholdReplay，沿用既有不重疊排程及撮合。",
                 "新增可重現命令、逐排程成交保存、成本對帳、全門檻 CSV、圖表及標準報告。"],
        findings=[{"id": "MODEL-001", "severity": "MEDIUM", "status": "ACCEPTED", "category": "research_validity",
            "component": "threshold_sensitivity", "title": "歷史門檻掃描不能作為正式選參證據",
            "evidence": ["plan.json", "all_thresholds.csv"], "impact": "大量事後比較容易高估最佳門檻。",
            "recommendation": "跨期驗證與前瞻封存；本次不授予任何部署資格。"}],
        checks=[{"id": "DATA-001", "name": "來源、既有模型與帳務唯讀稽核", "status": "PASS", "evidence": ["input_audit.json"]},
            {"id": "EXEC-001", "name": "55% 及無 AI 逐筆重現舊結果", "status": "PASS", "evidence": [f"{verification['policy_parity_checks']} 組逐筆核對", "verification.json"]},
            {"id": "EXEC-002", "name": "不重疊、冷卻、成本對帳及完整掃描", "status": "PASS", "evidence": ["verification.json", "all_thresholds.csv", "schedules/"]},
            {"id": "MODEL-001", "name": "可交易品質與新 holdout", "status": "BLOCKED", "evidence": ["研究敏感度不足以授予資格"]}],
        metrics={"engineering": "PASS", "research_quality": "INSUFFICIENT_EVIDENCE", "live_eligible": False,
            "p2_sac": "BLOCKED", "models": 18, "thresholds": 101, "scenario_rows": len(rows),
            "baseline_rows": len(baselines), "test55_total_trades_across_models": int(test_55.trades.sum()),
            "elapsed_seconds": elapsed, "input_audit": verification["input_audit"],
            "test_candidate_windows": plan["split_windows"], "account_equity_validated": False},
        residual_risks=risks, next_actions=actions)
    for name in ("plan.json", "all_thresholds.csv", "baselines.csv", "seed_summary.csv", "verification.json",
                 "expectancy.png", "trade_count.png", "settled_return.png"):
        report["artifacts"].append({"path": name, "role": "research_evidence", "sha256": sha256_file(output / name)})
    dump(output / "report.json", report)
    display = aggregate[(aggregate.split == "test") & (aggregate.scenario == "base")
        & aggregate.threshold_pct.isin([0, 20, 25, 30, 35, 40, 45, 50, 55, 100])].copy()
    table = ["|策略|特徵|門檻|有成交 seeds|成交數平均|淨期望值平均 (bps)|實際勝率平均|10%配置結算報酬平均|",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    def fmt(value, scale=1, suffix=""):
        return "無成交/未定義" if pd.isna(value) else f"{value * scale:.2f}{suffix}"
    for r in display.itertuples():
        table.append(f"|{LABELS[r.family]}|{LABELS[r.variant]}|{r.threshold_pct}%|{r.active_seeds}/3|{r.trades_mean:.1f}|"
            f"{fmt(r.mean_net_mean, 10000)}|{fmt(r.actual_win_rate_mean, 100, '%')}|{fmt(r.settled_return_10pct_mean, 100, '%')}|")
    body = ["# Transformer 機率門檻完整敏感度研究", "", f"Report ID：{report['report_id']}",
        f"UTC：{report['generated_at_utc']}；Revision：{plan['revision']}（既有工作樹含修改，另保存程式 SHA256）。",
        "", "## 結論範圍", "工程 PASS；研究品質證據不足；Live/SAC BLOCKED。沒有更改原 55% 設定。",
        "本研究改的是預測成功機率門檻，不是指定實際勝率。期望值 = 勝率 × 平均淨獲利 - 敗率 × 平均淨虧損。",
        "實際勝率低於 50% 仍可能有正期望，但止損、資金控制不會自動創造預測優勢。",
        "", "## 方法", "18 模型 × 101 門檻 × 3 成本 × 2 區間 = 10,908 列，另列 108 個無 AI 對照。",
        "0% 仍要求模型預測淨收益 >0.02%；100% 只接受機率等於 1。門檻包含邊界。",
        "每個成本重新撮合，再依持倉與冷卻挑選；保留相同 stop=2ATR、target=4ATR、最長32棒、冷卻4棒。",
        "基本成本：手續費5bps/邊、滑價2bps/邊、完整價差1bp、funding準備金1bp/結算；壓力滑價再加3bps/邊。",
        "validation_predictions.csv 是原流程的 Selection；本次只描述，不使用 Test 選參。",
        "收益與勝率是有成交 seeds 的平均；成交數平均含全部3個 seeds（含零成交）。細線為各 seed，active_seeds另列。",
        "", "## 圖表", "![每筆淨期望值](expectancy.png)", "![成交數](trade_count.png)",
        "![固定10%配置結算報酬](settled_return.png)", "", "## 部分門檻摘要", *table,
        "", "完整 0% 至100% 每1%明細見 all_thresholds.csv；含 win rate、盈虧比、PF、CI、最佳5筆敏感度與成本歸因。",
        "schedules/ 保存每條唯一成交排程，CSV 的 schedule_file 可追查逐筆出入場、MFE/MAE 與成本。",
        "", "## 驗證", f"已核對 {verification['policy_parity_checks']} 組原55%/無AI結果；逐筆冷卻、成本與標籤邊界檢查通過。",
        "來源稽核見 input_audit.json；程式測試另見測試紀錄。", "", "## 限制", *[f"- {r}" for r in risks],
        "", "## 下一步與停止條件", *[f"- {a}" for a in actions], "", "## 重現命令", "```powershell", plan["command"], "```", ""]
    (output / "report.md").write_text("\n".join(body), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True, help="含 study.json 的下載結果目錄")
    parser.add_argument("--bundle-root", type=Path, required=True, help="原始白名單 staging 封包")
    parser.add_argument("--output", type=Path, required=True, help="新目錄，不覆寫已存在研究")
    args = parser.parse_args()
    torch.set_num_threads(2)
    started = time.perf_counter()
    output, root = args.output.resolve(), args.study.resolve()
    output.mkdir(parents=True, exist_ok=False)
    print("核對輸入模型、資料與既有帳務...", flush=True)
    checked = audit(root, args.bundle_root)
    dump(output / "input_audit.json", checked)
    study = json.loads((root / "study.json").read_text(encoding="utf-8"))
    original_plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    records = study["records"]
    if len(records) != 18 or len({r["fold"] for r in records}) != 1:
        raise ValueError("此研究固定為18個既有配對模型、單一窗口；其他設計須先另定實驗計畫")
    prepared, split_windows, point_sets = [], [], {}
    for record in records:
        run = portable_path(record["run_dir"], root / "runs")
        training = json.loads((run / "training.json").read_text(encoding="utf-8"))
        contract = CandidateContract(**training["training_config"]["candidate_contract"])
        if training["training_config"]["economic_minimum_edge_bps"] != 2:
            raise ValueError("本研究要求凍結原本2bps收益閘門")
        source = portable_path(training["sources"][0]["path"], root)
        key = (contract.digest, sha256_file(source))
        predictions = {s: pd.read_csv(run / f"{s}_predictions.csv") for s in ("validation", "test")}
        for split, prediction in predictions.items():
            if prediction.series_index.nunique() != 1 or prediction.empty:
                raise ValueError("本研究只接受非空單一BTC序列")
            point_sets.setdefault(key, set()).update(prediction.endpoint.astype(int))
            dates = pd.to_datetime(prediction.timestamp_ns, unit="ns", utc=True)
            split_windows.append({"name": record["name"], "split": split, "candidate_start_utc": str(dates.min()),
                                  "candidate_end_utc": str(dates.max()), "candidate_rows": len(prediction)})
        prepared.append((record, run, contract, source, key, predictions))
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    code_paths = [Path(__file__), ROOT / "src/ai_quant_trading/research/probability_sweep.py",
                  ROOT / "tests/research/test_probability_sweep.py"]
    plan = {"version": "threshold_sensitivity_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "降低預測成功機率門檻可能增加成交；成本後期望值不保證改善。",
        "revision": revision, "research_only": True, "live_eligible": False, "cpu_threads": 2,
        "thresholds_pct": list(range(101)), "minimum_net_edge_bps": 2, "deployed_threshold_unchanged": .55,
        "input_study_sha256": sha256_file(root / "study.json"), "input_plan_sha256": sha256_file(root / "plan.json"),
        "input_manifest_sha256": sha256_file(root / "artifact_manifest.json"),
        "code_sha256": {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in code_paths},
        "input_execution_code_sha256": original_plan["code_sha256"], "split_windows": split_windows,
        "threshold_selection": "none; retrospective_diagnostic_only", "cost_comparison": "rematch_each_scenario",
        "ci": "descriptive_block_bootstrap_200_unadjusted_for_search_min8", "settled_allocation": .1,
        "command": subprocess.list2cmdline([sys.executable, "-X", "utf8", str(Path(__file__).resolve()),
            "--study", str(root), "--bundle-root", str(args.bundle_root.resolve()),
            "--output", str(output.with_name(output.name + "_repeat"))])}
    dump(output / "plan.json", plan)
    cache, summaries, rows, baselines = {}, {}, [], []
    verification = {"input_audit": checked, "policy_parity_checks": 0, "unique_schedules": 0}
    (output / "schedules").mkdir()
    for number, (record, run, contract, source, key, predictions) in enumerate(prepared, 1):
        if key not in cache:
            frame = prepare_candidate_frame(pd.read_csv(source), contract)
            scenarios = {"zero": replace(contract, fee_bps=0., slippage_bps=0., spread_bps=0., funding_reserve_bps=0.),
                         "base": contract, "stress": replace(contract, slippage_bps=contract.slippage_bps + 3.)}
            cache[key] = {}
            for scenario_name, scenario in scenarios.items():
                outcomes = candidate_outcomes(frame, scenario)
                subset = outcomes[outcomes.endpoint.isin(point_sets[key])]
                attributed = attribute_events(frame, subset, scenario, key[1]).set_index("endpoint", drop=False)
                cache[key][scenario_name] = (scenario, attributed)
        for split, prediction in predictions.items():
            for scenario_name, (scenario, attributed) in cache[key].items():
                available = attributed.loc[prediction.endpoint.astype(int)]
                window = next(w for w in original_plan["windows"] if w["fold"] == record["fold"])
                boundary = window["validation_end"] if split == "validation" else window["source_end"]
                if (available.label_end_endpoint > boundary).any():
                    raise ValueError("成本情境標籤跨越允許區間")
                current = prediction.copy()
                current["event_exit_endpoint"] = available.exit_endpoint.to_numpy()
                current[f"actual_return_{contract.holding_bars}"] = available.net_return.to_numpy()
                replay = ThresholdReplay(current, contract.holding_bars, scenario.event_config(), 2)
                identity = {k: record[k] for k in ("name", "family", "variant", "seed", "fold")}
                identity.update(split="selection" if split == "validation" else "test", scenario=scenario_name,
                                candidate_rows=len(prediction), contract_sha256=scenario.digest,
                                max_probability=float(prediction[f"tradeability_probability_{contract.holding_bars}"].max()),
                                return_passing_candidates=int((prediction[f"predicted_return_{contract.holding_bars}"] > .0002).sum()))

                def summarize(selected):
                    trades = attributed.loc[selected.endpoint.astype(int)].copy()
                    verify_trade_table(trades, contract.cooldown_bars)
                    digest = schedule_id(trades)
                    cache_id = (scenario.digest, digest)
                    if cache_id not in summaries:
                        filename = f"schedules/{scenario.digest[:12]}_{digest}.csv"
                        trades.to_csv(output / filename, index=False)
                        summaries[cache_id] = {**trade_metrics(trades), "schedule_id": digest, "schedule_file": filename}
                    return summaries[cache_id]

                for policy, selected in (("no_ai", replay.baseline), ("transformer", replay.select(.55))):
                    expected = pd.read_csv(run / f"{split}_policies/{policy}_{scenario_name}_trades.csv")
                    actual = attributed.loc[selected.endpoint.astype(int)]
                    assert_trade_parity(actual, expected)
                    verification["policy_parity_checks"] += 1
                baselines.append({**identity, "policy": "no_ai", **summarize(replay.baseline)})
                for percent in range(101):
                    probability_pass = prediction[f"tradeability_probability_{contract.holding_bars}"] >= percent / 100
                    joint_pass = probability_pass & (prediction[f"predicted_return_{contract.holding_bars}"] > .0002)
                    rows.append({**identity, "threshold_pct": percent,
                        "probability_passing_candidates": int(probability_pass.sum()),
                        "joint_passing_candidates": int(joint_pass.sum()), **summarize(replay.select(percent / 100))})
        print(f"完成 {number}/18: {record['name']}", flush=True)
        dump(output / "progress.json", {"completed_models": number, "rows": len(rows), "status": "running"})
    frame, base = pd.DataFrame(rows), pd.DataFrame(baselines)
    if len(frame) != 10908 or len(base) != 108 or frame.duplicated(["name", "split", "scenario", "threshold_pct"]).any():
        raise ValueError("門檻掃描未完整或含重複組合")
    aggregate = aggregate_seeds(rows)
    frame.to_csv(output / "all_thresholds.csv", index=False)
    base.to_csv(output / "baselines.csv", index=False)
    aggregate.to_csv(output / "seed_summary.csv", index=False)
    verification.update(unique_schedules=len(summaries), threshold_rows=len(frame), status="PASS")
    dump(output / "verification.json", verification)
    plot_results(frame, base, output)
    write_report(output, plan, frame, base, aggregate, verification, time.perf_counter() - started)
    subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "tools/research_reports/validate_report.py"),
                    str(output / "report.json")], check=True)
    dump(output / "progress.json", {"completed_models": 18, "rows": len(rows), "status": "complete"})
    build_artifact_manifest(output, [p for p in output.rglob("*") if p.is_file()])
    verified = verify_artifact_manifest(output)
    print(json.dumps({"output": str(output), "engineering": "PASS", "research": "INSUFFICIENT_EVIDENCE",
        "manifest_files": len(verified), "unique_schedules": len(summaries), "live_eligible": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
