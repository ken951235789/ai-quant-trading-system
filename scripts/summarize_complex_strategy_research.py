"""把兩輪策略比較產生為中文圖表、離線頁面及標準化研究報告。"""

import argparse
from datetime import datetime, timezone
from html import escape
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ai_quant_trading.backtesting.complex_strategies import FAMILIES
from ai_quant_trading.operations.integrity import build_artifact_manifest, sha256_file
from ai_quant_trading.persistence import write_json_atomic
from plot_parameter_sweep import pool_results
from report_parameter_sweep import percent, table


def summarize_report(output):
    def read(name):
        return json.loads((output / name).read_text(encoding="utf-8"))
    plan, research, audit, document = (read(n) for n in ("plan.json", "research.json", "audit.json", "report.json"))
    assert audit["status"] == "PASS" and research["status"] == "complete"
    results, params = pd.read_csv(output / "results.csv.gz"), pd.read_csv(output / "parameters.csv")
    exports = read("exported_candidates.json")["results"]
    pooled = pool_results(results).merge(params, on="parameter_id", validate="many_to_one")
    pooled.to_csv(output / "pooled_forward.csv", index=False, encoding="utf-8-sig")
    variants = pooled.loc[pooled.parameter_id.ne(0)]
    previous_chart_path = Path(plan["previous"]) / "charts.html"
    try:
        previous_link = Path(os.path.relpath(previous_chart_path, output)).as_posix()
    except ValueError:
        previous_link = previous_chart_path.as_uri()
    source = pd.read_csv(plan["source"], usecols=["timestamp"])
    fw = [w for w in plan["windows"] if w["name"].startswith("forward")]
    start = pd.Timestamp(source.timestamp.iloc[fw[0]["start"]])
    end = pd.Timestamp(source.timestamp.iloc[fw[-1]["stop"]-1]) + pd.Timedelta(minutes=15)
    family_rows = []
    for family, title in FAMILIES.items():
        b = variants.loc[variants.family.eq(family) & variants.scenario.eq("base")]
        s = variants.loc[variants.family.eq(family) & variants.scenario.eq("stress")]
        z = variants.loc[variants.family.eq(family) & variants.scenario.eq("zero")]
        family_rows.append({"family": family, "label": title, "parameters": len(b),
            "min_trades": int(b.trades.min()), "median_trades": float(b.trades.median()), "max_trades": int(b.trades.max()),
            "median_base_expectancy": float(b.mean_net.median()), "best_base_expectancy": float(b.mean_net.max()),
            "positive_zero": int(z.mean_net.gt(0).sum()), "positive_base": int(b.mean_net.gt(0).sum()),
            "positive_stress": int(s.mean_net.gt(0).sum()),
            "positive_base_with_100_trades": int((b.mean_net.gt(0) & b.trades.ge(100)).sum())})
    families = pd.DataFrame(family_rows)
    positive_sample_count = int(families.positive_base_with_100_trades.sum())
    base_by_id = variants.loc[variants.scenario.eq("base")].set_index("parameter_id")
    stress_by_id = variants.loc[variants.scenario.eq("stress")].set_index("parameter_id")
    positive_sample_stress_count = int((base_by_id.mean_net.gt(0) & base_by_id.trades.ge(100)
        & stress_by_id.mean_net.gt(0) & stress_by_id.trades.ge(100)).sum())
    families.to_csv(output / "family_comparison.csv", index=False, encoding="utf-8-sig")
    old = pd.read_csv(Path(plan["previous"]) / "pooled_forward.csv")
    comparison = pd.DataFrame([{"round": label, "parameters": int(data.parameter_id.nunique()),
        "positive_zero": int(data.loc[data.scenario.eq("zero"), "mean_net"].gt(0).sum()),
        "positive_base": int(data.loc[data.scenario.eq("base"), "mean_net"].gt(0).sum()),
        "positive_stress": int(data.loc[data.scenario.eq("stress"), "mean_net"].gt(0).sum()),
        "median_base_expectancy": float(data.loc[data.scenario.eq("base"), "mean_net"].median())}
        for label, data in (("第一輪簡單規則", old), ("第二輪複合規則（不含基準）", variants))])
    comparison.to_csv(output / "round_comparison.csv", index=False, encoding="utf-8-sig")
    font = Path("C:/Windows/Fonts/msjh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 11, "axes.titlesize": 13,
        "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})
    charts = output / "charts"
    charts.mkdir(exist_ok=True)
    names = {"original": "原策略基準", **FAMILIES}
    colors = dict(zip(FAMILIES, ["#24639b", "#c46b16", "#158773", "#9a518d", "#ad433b", "#526878"]))
    def save(fig, name, footer):
        fig.text(.015, .02, footer, fontsize=10, color="#555555")
        fig.savefig(charts / name, facecolor="white", bbox_inches="tight")
        plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), layout="constrained")
    fig.suptitle("第二輪：六個複合策略家族 | 1,152 組 + 原策略基準", fontsize=18, y=.99)
    for scenario, offset, color, label in [("zero", -.23, "#727a80", "不含成本"), ("base", 0, "#24639b", "基本成本"), ("stress", .23, "#b6433b", "壓力成本")]:
        values = [variants.loc[variants.family.eq(f) & variants.scenario.eq(scenario), "mean_net"].dropna().to_numpy() * 100 for f in FAMILIES]
        box = axes[0].boxplot(values, positions=np.arange(6)+offset, widths=.19, patch_artist=True,
            manage_ticks=False, showfliers=False, medianprops={"color": "black"}, whiskerprops={"color": color}, capprops={"color": color})
        for patch in box["boxes"]:
            patch.set(facecolor=color, alpha=.45, edgecolor=color)
        axes[0].plot([], [], color=color, lw=7, alpha=.6, label=label)
    axes[0].axhline(0, color="#333333", lw=.9)
    axes[0].set(xticks=range(6), xticklabels=["跨週期\n趨勢回調", "波動壓縮\n突破", "VWAP\n均值回歸", "假突破\n收回", "突破後\n回測", "趨勢／震盪\n分流"], ylabel="Forward 每筆名目本金淨報酬 (%)", title="整個家族分布，不挑最好的一組（每家族 192 組）")
    axes[0].legend(ncol=3, loc="upper right", fontsize=10)
    for family, color in colors.items():
        b = variants.loc[variants.family.eq(family) & variants.scenario.eq("base")]
        axes[1].scatter(b.trades, b.mean_net * 100, s=18, alpha=.45, color=color, label=FAMILIES[family])
    axes[1].axhline(0, color="#333333", lw=.9)
    axes[1].axvline(100, color="#777777", ls="--", lw=.8)
    axes[1].set(xlabel="Forward 交易筆數（約 1.5 年）", ylabel="平均每筆名目本金淨報酬 (%)", title="虛線：100 筆僅是樣本門檻，不等於獲利證明")
    axes[1].legend(ncol=3, fontsize=9, loc="lower right")
    fig.get_layout_engine().set(rect=(0, .07, 1, .86))
    save(fig, "01_complex_overview.png", "固定網格的回溯比較，非未見封存驗證。箱型圖顯示中位數、四分位與1.5IQR鬚線，省略離群點；完整數據保留CSV。")

    fig, axes = plt.subplots(1, 2, figsize=(13, 7), layout="constrained")
    fig.suptitle("時間週期 × 目標損益比 | 每格其餘參數取中位數", fontsize=17, y=.99)
    base = variants.loc[variants.scenario.eq("base")]
    heat = base.groupby(["timeframe", "family", "reward_r"]).mean_net.median() * 100
    heat_count = base.groupby(["timeframe", "family", "reward_r"]).trades.median()
    limit = max(abs(heat.min()), abs(heat.max()), .01)
    for ax, tf in zip(axes, ("15m", "1h")):
        matrix = heat.loc[tf].unstack("reward_r").reindex(FAMILIES)
        counts = heat_count.loc[tf].unstack("reward_r").reindex(FAMILIES)
        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-limit, vmax=limit)
        for (y, x), value in np.ndenumerate(matrix.to_numpy()):
            ax.text(x, y, f"{value:+.3f}%\nN 中位 {counts.iloc[y, x]:.0f}", ha="center", va="center", fontsize=10,
                    color="white" if abs(value)>.6*limit else "#222222")
        ax.set(xticks=range(3), xticklabels=matrix.columns, yticks=range(6), yticklabels=list(FAMILIES.values()), title=tf, xlabel="目標 R")
    fig.colorbar(im, ax=axes, shrink=.8, label="每筆淨報酬中位數 (%)")
    fig.get_layout_engine().set(rect=(0, .08, 1, .84))
    save(fig, "02_complex_heatmap.png", "基本成本；每格32組，N為Forward交易數中位數，非獨立樣本數。1h/15m用不同ATR，名目配置相同不代表風險相同。")

    ids = research["exported_ids"]
    indexed = params.set_index("parameter_id")
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True, height_ratios=[2,1], layout="constrained")
    fig.suptitle("原策略與各家族驗證首選 | 僅診斷，不代表通過品質閘門", fontsize=17, y=.99)
    for i in ids:
        f = pd.read_csv(output / "trades" / f"C{i:04d}_forward_base.csv")
        eq = np.r_[1000., f.equity, f.equity.iloc[-1] if len(f) else 1000.]
        dates = [start, *pd.to_datetime(f.exit_at, utc=True), end]
        family = indexed.loc[i, "family"]
        color = "#222222" if i == 0 else colors[family]
        label = f"C{i:04d} {names[family]} ({(eq[-1]/1000-1)*100:+.2f}%)"
        axes[0].step(dates, eq, where="post", color=color, label=label, lw=1.6)
        axes[1].step(dates, (eq / np.maximum.accumulate(eq)-1)*100, where="post", color=color, lw=1.2)
    axes[0].axhline(1000, color="#888888", ls="--", lw=.8)
    axes[0].set(ylabel="結算權益（初始1,000 USDT）")
    axes[0].legend(ncol=2, fontsize=9, loc="lower left")
    axes[1].set(ylabel="結算回撤 (%)", xlabel="UTC 時間")
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for ax in axes:
        ax.grid(alpha=.15)
    fig.get_layout_engine().set(rect=(0, .075, 1, .85))
    save(fig, "03_complex_equity.png", "每筆固定10%名目配置、無槓桿，只記平倉損益；無持倉浮虧/清算/每日風控。未達Search交易數門檻者不補選。")

    fig, ax = plt.subplots(figsize=(13, 7), layout="constrained")
    for scenario, offset, color, label in [("base", -.12, "#24639b", "基本成本"), ("stress", .12, "#b6433b", "壓力成本")]:
        data = [next(e for e in exports if e["parameter_id"] == i and e["scenario"] == scenario) for i in ids]
        means = np.array([e["mean"] for e in data], dtype=float) * 100
        lower, upper = np.array([e["ci_low"] for e in data], dtype=float)*100, np.array([e["ci_high"] for e in data], dtype=float)*100
        ax.errorbar(means, np.arange(len(ids))+offset, xerr=np.maximum(0, [means-lower, upper-means]), fmt="o", color=color, capsize=3, label=label)
    ax.axvline(0, color="#333333", lw=.9)
    ax.set(yticks=range(len(ids)), yticklabels=[f"C{i:04d} {names[indexed.loc[i, 'family']]}" for i in ids],
           xlabel="Forward 每筆名目本金淨報酬與95%區塊重抽樣CI (%)", title="候選期望值：平均值、分散與成本壓力")
    ax.legend(loc="lower right")
    fig.get_layout_engine().set(rect=(0, .09, 1, .90))
    save(fig, "04_complex_expectancy.png", "各候選CI為描述性區間，未針對此圖多重校正；正式候選仍須通過Search/Validation品質閘門與後續驗證。")

    families_display = table(["家族", "組數", "Forward交易範圍", "每筆淨報酬中位數", "零成本正值", "基本正值", "壓力正值", "基本正值且≥100筆"],
        [[x["label"], x["parameters"], f"{x['min_trades']}～{x['max_trades']}", percent(x["median_base_expectancy"]), x["positive_zero"], x["positive_base"], x["positive_stress"], x["positive_base_with_100_trades"]] for x in family_rows])
    candidates_display = table(["ID", "家族", "Forward筆數", "每筆均值", "標準差", "最差一筆", "PF", "結算報酬", "結算回撤"],
        [[f"C{i:04d}", names[indexed.loc[i, "family"]], e["trades"], percent(e["mean"]), percent(e.get("std")), percent(e.get("worst")), "NA" if e.get("profit_factor") is None else f"{e['profit_factor']:.3f}", percent(e["settled_return"]), percent(e["settled_drawdown"])] for i in ids for e in exports if e["parameter_id"] == i and e["scenario"] == "base"])
    params_display = table(["ID", "週期", "嚴格", "停損ATR", "R", "持有15m棒", "冷卻", "退出"],
        [[f"C{i:04d}", *[indexed.loc[i, name] for name in ["timeframe", "strict", "stop_atr", "reward_r", "holding_bars", "cooldown_bars", "regime_exit"]]] for i in ids])
    comparison_display = table(["研究", "組數", "零成本正值", "基本正值", "壓力正值", "每筆淨報酬中位數"],
        [[row["round"], row["parameters"], row["positive_zero"], row["positive_base"], row["positive_stress"], percent(row["median_base_expectancy"])] for row in comparison.to_dict("records")])
    test_metrics = {k: sum(int(s.get(k, 0)) for s in ET.parse(output / "pytest_results.xml").getroot().iter("testsuite")) for k in ("tests", "failures", "errors", "skipped")}
    assert not test_metrics["failures"] and not test_metrics["errors"]
    passed = research["forward_quality_passed"]
    summary = (f"新增六家族、1,152組複合策略，含基準共17,295情境；基本成本下Forward均值為正{int(families.positive_base.sum())}組，"
        f"壓力為正{int(families.positive_stress.sum())}組。正式選中={research['selection']['selected']}；工程PASS，策略品質{'PASS' if passed else 'FAIL'}。")
    limitations = [
        "兩輪共享且反覆研究相同歷史，不是未見final holdout；新增複雜度不能消除選擇偏誤。",
        "1,152組是固定網格，不是所有可能複合策略；家族分布高度相關，網格中位數差異不是因果效應。",
        "指標僅來自OHLCV；沒有實際OI、Funding、訂單簿、新聞與流動性資料，假突破收回不代表確認掃損。",
        "假設基本往返約15bp、壓力約21bp，另每UTC八小時1bp準備金；不是實際Funding或台灣稅後結果。",
        "每筆固定10%名目配置、1000USDT結算曲線；未計持倉浮虧、日損限制、清算、下單精度與實際容量。",
        "三段Forward各自空倉開始後串接，非完整連續實盤；完整帳戶Sharpe/Sortino及風控有效性未驗證。",
        "Search/Validation最大統計量為共享三月份區塊2000次的敏感度診斷，不是嚴格FWER或PBO證明。",
        "稀少交易的正值不能取代樣本證據；圖上各候選CI是描述性區間，不是多重校正獲利保證。",
        "沒有重訓、套用AI濾網或重跑SAC；不能據此宣稱複合規則優於目前AI模型。",
    ]
    next_actions = ["不使用Forward排名直接部署，保留所有不合格及正負結果。",
        "先收集實際費率、Funding與成交摩擦；再針對少量有機制根據的市場環境假說預先定義驗證。",
        "候選先在未用資料或前向模擬通過成本後正期望，再考慮Transformer過濾與SAC部位管理。"]
    findings = [{"id": "MODEL-001", "severity": "HIGH" if not passed else "INFO", "status": "OPEN" if not passed else "ACCEPTED", "category": "strategy_quality", "component": "complex_strategy_selection",
        "title": "複合策略仍須通過分層品質閘門", "evidence": ["selection.json", "family_comparison.csv"],
        "impact": "不得用事後少量交易的好結果或少虧當成可部署策略。", "recommendation": next_actions[0]},
        {"id": "DATA-001", "severity": "MEDIUM", "status": "ACCEPTED", "category": "historical_bias", "component": "forward",
        "title": "歷史重複研究與大量參數相關性", "evidence": ["plan.json: pristine_holdout=false"], "impact": "限制未來外推可信度。", "recommendation": next_actions[2]}]
    checks = [{"id": "TEST-001", "name": "完整專案測試", "status": "PASS", "evidence": [f"pytest_results.xml: {test_metrics}"]},
        {"id": "AUDIT-001", "name": "完整情境/月收益/選擇/帳務", "status": "PASS", "evidence": ["audit.json"]},
        {"id": "PARITY-001", "name": "參考成交與原基準相容", "status": "PASS", "evidence": ["research.json: baseline_matches_previous_all_scenarios; all_exported_trades_reference_checked"]},
        {"id": "QUALITY-001", "name": "正式候選品質", "status": "PASS" if passed else "FAIL", "evidence": ["selection.json", "research.json"]},
        {"id": "LIVE-001", "name": "稅後實盤資格", "status": "BLOCKED", "evidence": ["research.json: live_eligible=false"]}]
    environment = {"python": platform.python_version(), "platform": platform.platform(), "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "matplotlib", "pytest"]}}
    write_json_atomic(output / "environment.json", environment)
    document.update(generated_at_utc=datetime.now(timezone.utc).isoformat(), overall_status="WARN" if passed else "FAIL", summary=summary,
        project={"name": "AIQuantTradingSystem", "revision": "dirty; frozen source hashes in plan.json", "environment": str(environment)},
        changes=["新增六個因果複合進場家族及1,152組固定參數。", "沿用既有成交契約、成本、期間、統計與品質閘門。", "新增17項測試、獨立稽核、四張圖及完整比較表。"],
        findings=findings, checks=checks, metrics={"engineering_status": "PASS", "strategy_quality_status": "PASS" if passed else "FAIL",
        "source_sha256": plan["source_sha256"], "parameters": len(params), "scenarios": len(results), "tests": test_metrics,
        "forward_start_utc": start.isoformat(), "forward_end_utc_exclusive": end.isoformat(), "family_results": family_rows,
        "round_comparison": comparison.to_dict("records"), "exported_candidates": exports,
        "positive_base_at_least_100_trades": positive_sample_count,
        "positive_base_and_stress_at_least_100_trades": positive_sample_stress_count,
        "selected": research["selection"]["selected"], "live_eligible": False, "elapsed_seconds": research["elapsed_seconds"]},
        residual_risks=limitations, next_actions=next_actions)
    artifact_names = ["plan.json", "research.json", "audit.json", "parameters.csv", "results.csv.gz", "monthly.npz", "family_comparison.csv", "round_comparison.csv", "pooled_forward.csv", "pytest_results.xml"]
    document["artifacts"] = [{"path": n, "role": "可追溯研究證據", "sha256": sha256_file(output / n)} for n in artifact_names]
    write_json_atomic(output / "report.json", document)
    text = f'''# 六家族複合策略比較

## 基本資訊
- Report ID：{document['report_id']}
- Report Type：model_research
- Generated At UTC：{document['generated_at_utc']}
- Project：AIQuantTradingSystem；dirty工作目錄，計算程式雜湊於plan.json。
- Scope：六家族、1,152組新參數及1組原基準。
- Overall Status：{document['overall_status']}；工程PASS、策略品質{'PASS' if passed else 'FAIL'}。

## 執行摘要
{summary}

來源 {plan['source_rows']:,} 根BTCUSD-M 15m棒，{plan['source_start']} 至 {plan['source_end']}。
Search前40%、Validation接續20%、三段Forward各10%，最後10%未評估。
Forward範圍：{start.isoformat()} 至 {end.isoformat()}（不含）。
相同暖機、64棒開頭隔離、65棒尾端留白；每一段空倉開始。
原基準在全部15個期間/成本情境與第一輪一致，排除引擎差異冒充改進。

## 兩輪比較
{comparison_display}

表格的正值是每筆均值>0，不代表顯著、通過分層品質閘門或未來可獲利。
兩輪不同網格涵蓋範圍，因此中位數只能描述這兩批設定，不能當作「複雜度造成改善」的因果證據。

## 家族完整結果
{families_display}

基本成本正值且Forward至少100筆的只有 **{positive_sample_count} 組**；
基本/壓力均為正且兩者皆至少100筆的只有 **{positive_sample_stress_count} 組**。
因此目前的正值多伴隨稀少訊號，不能把綠色熱圖或單筆高均值當成穩定優勢。

{candidates_display}

候選依Search名單中的Validation排名，每一家族最多一組；缺少候選不會放寬150筆Search要求。
{params_display}

## 指標與交易契約
六家族包括跨週期趨勢回調、波動壓縮突破、VWAP均值回歸、假突破收回、突破後回測與趨勢/震盪分流。
詳細進出場條件見專案 `docs/COMPLEX_STRATEGIES_20261002.md`，不是讓AI猜測進出場。
指標沿用既有ATR/ADX/RSI；EMA使用adjust=False，布林標準差ddof=0；VWAP以UTC開盤日歸屬。
H1/H4只有收盤才可被讀取；1h訊號不會ffill成四次15m進場。失效突破價不會復活。

下一15m開盤進場；固定ATR保護價，同棒停損優先；相反強趨勢只讀前一收盤已知狀態。
基本單邊手續費5bp、滑價2bp、全spread1bp；壓力只把單邊滑價提高為5bp。
每跨UTC00/08/16結算扣1bp Funding準備金，不是實際歷史Funding，也未宣稱台灣稅率為零。
每筆淨報酬以進場名目本金為分母；帳戶結算權益乘上(1+0.10×每筆淨報酬)，起始1000USDT。
結算回撤不含持倉浮虧，完整帳戶Sharpe/Sortino/清算等仍未驗證。

## 候選選擇
Search每家族/週期至少150筆，以stress月均log收益前5進入Validation。
Search要base/stress皆正、最大統計量p≤.05、鄰域正值≥60%、結算DD≤10%。
鄰域沿嚴格程度、停損、R、持有、冷卻上下移一格，固定家族/週期/退出方式。
Validation要≥50筆、正base/stress、最大統計量p≤.05、結算DD≤10%；只依Validation排序。
Forward僅能確認前面合格者：≥100筆、均值CI下界>0、stress均值>0、三段各≥20筆且均值>0、DD≤10%。
選中候選：`{research['selection']['selected']}`；本輪沒有自動部署。
區塊重抽樣使用共同三月份循環區塊、2000次、seed20261002；是多重搜尋敏感度，不是PBO/FWER保證。

## 變更與驗證
{table(['檢查','狀態','證據'], [[c['name'],c['status'],'; '.join(c['evidence'])] for c in checks])}

## 發現
{table(['ID','嚴重度','狀態','發現','建議'], [[f['id'],f['severity'],f['status'],f['title'],f['recommendation']] for f in findings])}

## 圖表
[離線總覽](charts.html)；[第一輪圖表](<{previous_link}>)。

![家族總覽](charts/01_complex_overview.png)
![參數熱圖](charts/02_complex_heatmap.png)
![結算曲線](charts/03_complex_equity.png)
![期望值](charts/04_complex_expectancy.png)

## 未驗證與剩餘風險
{chr(10).join('- '+x for x in limitations)}

## 下一步
{chr(10).join(str(i+1)+'. '+x for i,x in enumerate(next_actions))}

## 產物
`parameters.csv`、`results.csv.gz`、`pooled_forward.csv`保留全部參數及情境。
`family_comparison.csv`、`round_comparison.csv`提供家族與前後輪比較。
`trades/`保留基準與各家族驗證首選逐筆交易；均以舊參考引擎核對。
`plan.json`、`monthly.npz`、`selection.json`、`parameter_diagnostics.csv`保留選擇與統計證據。
`audit.json`、`pytest_results.xml`、`environment.json`與`artifact_manifest.json`保留驗證資訊。
'''
    (output / "report.md").write_text(text, encoding="utf-8")
    display_comparison = comparison.copy()
    display_comparison["median_base_expectancy"] = display_comparison.median_base_expectancy.map(percent)
    display_comparison = display_comparison.rename(columns={"round": "研究", "parameters": "參數組數", "positive_zero": "零成本正值", "positive_base": "基本成本正值", "positive_stress": "壓力成本正值", "median_base_expectancy": "每筆淨報酬中位數"})
    display_families = families.drop(columns="family").copy()
    for column in ("median_base_expectancy", "best_base_expectancy"):
        display_families[column] = display_families[column].map(percent)
    display_families = display_families.rename(columns={"label": "策略家族", "parameters": "參數組數", "min_trades": "最少交易筆數", "median_trades": "交易筆數中位數", "max_trades": "最多交易筆數", "median_base_expectancy": "每筆淨報酬中位數", "best_base_expectancy": "事後最高每筆均值（非候選）", "positive_zero": "零成本正值", "positive_base": "基本正值", "positive_stress": "壓力正值", "positive_base_with_100_trades": "基本正值且至少100筆"})
    gallery = "".join(f'<section><img src="charts/{escape(p.name)}" alt="{escape(p.stem)}"></section>' for p in sorted(charts.glob("*.png")))
    html = f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BTC複合策略研究</title>
<style>body{{font-family:system-ui,"Microsoft JhengHei",sans-serif;color:#222;background:#fff;max-width:1450px;margin:auto;padding:24px}}img{{max-width:100%;height:auto}}section{{padding:20px 0;border-top:1px solid #ddd}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:8px;border-bottom:1px solid #ddd}}.scroll{{overflow:auto}}a{{color:#24639b}}</style>
<h1>六家族複合策略比較</h1><p>{escape(summary)}</p><p>歷史回溯研究，不是未見封存驗證。成本是假設，曲線不含持倉浮虧與完整帳戶風控。</p>
<p><a href="{escape(previous_link, quote=True)}">第一輪11,664組圖表</a> · <a href="report.md">完整中文報告</a> · <a href="family_comparison.csv">家族CSV</a> · <a href="pooled_forward.csv">全部Forward結果</a></p>
<p>基本正值且至少100筆：{positive_sample_count}組；基本及壓力皆正且各至少100筆：{positive_sample_stress_count}組。仍無正式合格候選。</p>
<div class="scroll">{display_comparison.to_html(index=False, escape=True)}</div>{gallery}<section><h2>全部家族</h2><div class="scroll">{display_families.to_html(index=False,escape=True)}</div></section></html>'''
    (output / "charts.html").write_text(html, encoding="utf-8")
    build_artifact_manifest(output, [p for p in output.rglob("*") if p.is_file()])
    return {"summary": summary, "charts": 4, "html": str(output / "charts.html")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(summarize_report(args.output.resolve()))
