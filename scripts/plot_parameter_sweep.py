"""產生離線研究圖表；不得用 Forward 排名冒充事前可選的策略。"""

import argparse
from html import escape
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"15m 突破": "#2563a6", "15m 回調": "#c26215", "1h 突破": "#138273", "1h 回調": "#92549e"}


def pool_results(results):
    f = results.loc[results.window.str.startswith("forward")]
    pooled = f.groupby(["parameter_id", "scenario"]).agg(
        trades=("trades", "sum"), sum_net=("sum_net", "sum"), gains=("gains", "sum"),
        losses=("losses", "sum"), wins=("wins", "sum"), log_growth=("log_growth", "sum"),
        worst_trade=("worst_trade", "min"), sum_squared_net=("sum_squared_net", "sum"))
    pooled["mean_net"] = pooled.sum_net / pooled.trades.replace(0, np.nan)
    pooled["win_rate"] = pooled.wins / pooled.trades.replace(0, np.nan)
    pooled["profit_factor"] = pooled.gains / pooled.losses.replace(0, np.nan)
    pooled["settled_return"] = np.expm1(pooled.log_growth)
    return pooled.reset_index()


def make_plots(output):
    font = Path("C:/Windows/Fonts/msjh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 11, "axes.titlesize": 13,
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})
    r = pd.read_csv(output / "results.csv.gz")
    p = pd.read_csv(output / "parameters.csv")
    research = json.loads((output / "research.json").read_text(encoding="utf-8"))
    exported = json.loads((output / "exported_candidates.json").read_text(encoding="utf-8"))["results"]
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    pooled = pool_results(r).merge(p, on="parameter_id", validate="many_to_one")
    pooled["family"] = pooled.timeframe + " " + pooled.entry.map({"breakout": "突破", "pullback": "回調"})
    pooled.to_csv(output / "pooled_forward.csv", index=False, encoding="utf-8-sig")
    base = pooled.loc[pooled.scenario.eq("base")].set_index("parameter_id")
    search = r.loc[r.window.eq("search") & r.scenario.eq("base")].set_index("parameter_id")
    charts = output / "charts"
    charts.mkdir(exist_ok=True)
    def save(fig, name, footer):
        fig.text(.015, .018, footer, fontsize=10, color="#555555")
        fig.savefig(charts / name, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    fig, ax = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    fig.suptitle("BTC 大型策略參數比較 | 11,664 組 × 5 時段 × 3 成本", fontsize=18, y=.99)
    for family, color in COLORS.items():
        rows = base.loc[base.family.eq(family)]
        ax[0, 0].scatter(search.loc[rows.index, "settled_return"] * 100, rows.settled_return * 100,
                         s=7, alpha=.22, color=color, label=family, rasterized=True)
        ax[1, 0].scatter(rows.trades, rows.mean_net * 100, s=7, alpha=.23, color=color)
    ax[0, 0].axhline(0, color="#555555", lw=.8)
    ax[0, 0].axvline(0, color="#555555", lw=.8)
    ax[0, 0].set(xlabel="Search 結算報酬 (%)", ylabel="Forward 結算報酬 (%)", title="搜尋好看，後續是否仍好看？")
    ax[0, 0].legend(markerscale=3, fontsize=10)
    bins = np.linspace(pooled.mean_net.min() * 100, pooled.mean_net.max() * 100, 65)
    for scenario, label, color in [("zero", "不含成本", "#707980"), ("base", "基本成本", "#2563a6"), ("stress", "壓力成本", "#bd463b")]:
        rows = pooled.loc[pooled.scenario.eq(scenario)]
        ax[0, 1].hist(rows.mean_net * 100, bins=bins, histtype="step", lw=1.8, color=color, label=f"{label}：正值 {(rows.mean_net > 0).sum():,} 組")
    ax[0, 1].axvline(0, color="#333333", lw=.8)
    ax[0, 1].set(xlabel="Forward 平均每筆名目本金淨報酬 (%)", ylabel="參數組數", title="成本後期望值分布")
    ax[0, 1].legend(fontsize=10)
    ax[1, 0].axhline(0, color="#555555", lw=.8)
    ax[1, 0].set(xlabel="Forward 交易筆數（約 1.5 年）", ylabel="平均每筆名目本金淨報酬 (%)", title="交易更多不等於期望值更高")
    counts = [11664, research["search_positive_base"], research["search_max_stat_passed"], research["search_quality_passed"], int(research["selection"]["selected"] is not None), int(research["forward_quality_passed"])]
    labels = ["搜尋組合", "Search 成本後正值", "Search 多重搜尋診斷通過", "Search 全部品質條件通過", "Validation 選中候選", "Forward 品質通過"]
    ax[1, 1].barh(labels[::-1], counts[::-1], color=["#138273"] * 2 + ["#707980"] * 4)
    for i, value in enumerate(counts[::-1]):
        ax[1, 1].text(value + 150, i, f"{value:,}", va="center")
    ax[1, 1].set(xlim=(0, 14000), title="品質閘門：不得事後拿 Forward 冠軍升級")
    fig.get_layout_engine().set(rect=(0, .065, 1, .88))
    save(fig, "01_overview.png", "研究性歷史回放，非未接觸封存集。結算報酬：固定 10% 名目配置、無槓桿；不是完整帳戶風控結果。")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    fig.suptitle("停損 × 損益比敏感度 | 同一格其餘參數取中位數", fontsize=17, y=.99)
    grouped = base.groupby(["family", "stop_atr", "reward_r"]).mean_net.median() * 100
    limit = max(abs(grouped.min()), abs(grouped.max()), .01)
    for ax, family in zip(axes.flat, COLORS):
        matrix = grouped.loc[family].unstack("reward_r")
        im = ax.imshow(matrix.to_numpy(), cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
        for (y, x), value in np.ndenumerate(matrix.to_numpy()):
            ax.text(x, y, f"{value:+.3f}%", ha="center", va="center", fontsize=14,
                    color="white" if abs(value) > .6 * limit else "#222222")
        ax.set(xticks=range(len(matrix.columns)), xticklabels=matrix.columns,
               yticks=range(len(matrix.index)), yticklabels=matrix.index,
               xlabel="目標 R（停利距離 / 停損距離）", ylabel="停損 ATR 倍數", title=family)
    fig.colorbar(im, ax=axes, label="Forward 每筆名目本金淨報酬中位數 (%)", shrink=.8)
    fig.get_layout_engine().set(rect=(0, .065, 1, .88))
    save(fig, "02_stop_reward_heatmap.png", "基本成本；每格彙整 324 組，不是挑出格內最高績效。ATR 使用訊號週期；所有成交仍以 15m K 棒重播。")

    fig, axes = plt.subplots(1, 4, figsize=(15, 5), layout="constrained")
    fig.suptitle("進場與持有條件敏感度 | Forward 基本成本", fontsize=17, y=.99)
    for ax, (column, title) in zip(axes, [("adx", "ADX 門檻"), ("lookback", "突破 / EMA 回看棒數"), ("holding_bars", "持有上限（15m 棒）"), ("cooldown_bars", "冷卻期（15m 棒）")]):
        for family, color in COLORS.items():
            median = base.loc[base.family.eq(family)].groupby(column).mean_net.median() * 100
            ax.plot(median.index, median, "o-", color=color, label=family)
        ax.axhline(0, color="#555555", lw=.8)
        ax.set(xlabel=title, ylabel="每筆淨報酬中位數 (%)", xticks=sorted(base[column].unique()))
    axes[0].legend(fontsize=9)
    fig.get_layout_engine().set(rect=(0, .11, 1, .81))
    save(fig, "03_parameter_sensitivity.png", "不同參數會改變持倉排程，交易集合不是包含關係；中位數比較不是因果消融，也不是新策略推薦。")

    ids = [research["baseline_id"]] + [i for i in research["selection"]["validation_ranked"][:5] if i != research["baseline_id"]]
    labels = [f"G{i:05d}" + (" 原始基準" if i == research["baseline_id"] else f" 驗證排名 {research['selection']['validation_ranked'].index(i)+1}") for i in ids]
    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    for scenario, offset, color, label in [("zero", -.22, "#707980", "不含成本"), ("base", 0, "#2563a6", "基本成本"), ("stress", .22, "#bd463b", "壓力成本")]:
        rows = [next(e for e in exported if e["parameter_id"] == i and e["scenario"] == scenario) for i in ids]
        means = np.array([row["mean"] for row in rows]) * 100
        bounds = np.array([[row["ci_low"], row["ci_high"]] for row in rows]) * 100
        ax.errorbar(means, np.arange(len(ids)) + offset, xerr=np.maximum(0, np.array([means - bounds[:, 0], bounds[:, 1] - means])),
                    fmt="o", capsize=3, color=color, label=label)
    ax.axvline(0, color="#333333", lw=1)
    ax.set(yticks=range(len(ids)), yticklabels=labels, xlabel="Forward 每筆名目本金淨報酬與區塊重抽樣 95% CI (%)",
           title="原始基準與驗證排名前五 | 僅診斷，不代表通過品質閘門")
    ax.legend(loc="lower right")
    fig.get_layout_engine().set(rect=(0, .09, 1, .90))
    save(fig, "04_candidate_expectancy.png", "區間是各策略的描述性信賴區間，未對此圖多候選再次校正；正式篩選另用共同月份最大統計量診斷。")

    source = pd.read_csv(plan["fingerprint"]["source"], usecols=["timestamp"])
    forward = [w for w in plan["fingerprint"]["windows"] if w["name"].startswith("forward")]
    start = pd.Timestamp(source.timestamp.iloc[forward[0]["start"]])
    end = pd.Timestamp(source.timestamp.iloc[forward[-1]["stop"] - 1]) + pd.Timedelta(minutes=15)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, height_ratios=[2, 1], layout="constrained")
    fig.suptitle("候選策略後續歷史曲線 | 基本成本、10% 名目配置", fontsize=17, y=.99)
    for j, i in enumerate(ids):
        trades = pd.read_csv(output / "trades" / f"G{i:05d}_forward_base.csv")
        stamps = [start, *pd.to_datetime(trades.exit_at, utc=True), end]
        equity = np.r_[1000., trades.equity, trades.equity.iloc[-1] if len(trades) else 1000.]
        color = "#222222" if j == 0 else plt.get_cmap("tab10")(j)
        axes[0].step(stamps, equity, where="post", label=f"{labels[j]}  ({(equity[-1]/1000-1)*100:+.2f}%)", color=color, lw=1.7)
        axes[1].step(stamps, (equity / np.maximum.accumulate(equity) - 1) * 100, where="post", color=color, lw=1.2)
    axes[0].axhline(1000, color="#777777", ls="--", lw=.8)
    axes[0].set(ylabel="結算權益（初始 1,000 USDT）")
    axes[0].legend(fontsize=9, loc="lower left", ncol=2)
    axes[1].set(ylabel="結算權益回撤 (%)", xlabel="UTC 時間")
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for ax in axes:
        ax.grid(alpha=.15)
    fig.get_layout_engine().set(rect=(0, .075, 1, .86))
    save(fig, "05_candidate_equity.png", "只在平倉時計算損益，未包含持倉盤中浮虧、清算、稅負及實盤異常；Forward 三段各自空倉起算後串接。")

    selected_rows = pooled.loc[pooled.parameter_id.isin(ids)].copy()
    selected_rows.to_csv(output / "candidate_table.csv", index=False, encoding="utf-8-sig")
    table = selected_rows.loc[selected_rows.scenario.eq("base"), ["parameter_id", "timeframe", "entry", "adx", "confirmation", "lookback", "stop_atr", "reward_r", "holding_bars", "cooldown_bars", "regime_exit", "trades", "mean_net", "settled_return"]]
    for column in ("mean_net", "settled_return"):
        table[column] = table[column].map(lambda x: f"{100*x:+.3f}%")
    gallery = "".join(f'<section><img src="charts/{escape(path.name)}" alt="{escape(path.stem)}"></section>' for path in sorted(charts.glob("*.png")))
    html = f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BTC 11,664 組策略比較</title><style>body{{font-family:system-ui,"Microsoft JhengHei",sans-serif;max-width:1440px;margin:auto;padding:24px;color:#222;background:#fff}}img{{max-width:100%;height:auto}}section{{border-top:1px solid #ddd;padding:20px 0}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:8px;border-bottom:1px solid #ddd}}.scroll{{overflow:auto}}a{{color:#2563a6}}</style>
<h1>BTC 大型策略參數比較</h1><p>11,664 組、174,960 個期間／成本情境。正式候選：{escape(str(research['selection']['selected']))}；Live 仍停用。</p>
<p>這是既有歷史資料上的回溯研究，不是全市場所有策略，也不是未接觸的最終封存驗證。</p>
<p><a href="report.md">完整研究報告</a> · <a href="candidate_table.csv">候選表格 CSV</a> · <a href="pooled_forward.csv">全部組合後續結果 CSV</a></p>
{gallery}<section><h2>基準與驗證排名前五：基本成本</h2><div class="scroll">{table.to_html(index=False, escape=True)}</div></section></html>'''
    (output / "charts.html").write_text(html, encoding="utf-8")
    return {"charts": len(list(charts.glob("*.png"))), "html": str(output / "charts.html")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(make_plots(args.output.resolve()))
