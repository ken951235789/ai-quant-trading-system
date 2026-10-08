"""將已保存成交轉為唯讀圖表資料，不重新撮合或挑選最佳交易。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def contained_file(root: Path, relative: str) -> Path:
    """圖表只讀取指定研究目錄內的檔案。"""
    base = root.resolve()
    path = (base / relative).resolve()
    path.relative_to(base)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def script_json(value: object) -> str:
    """防止資料文字提前結束 HTML script 區塊。"""
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            .replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026"))


def review_trades(bars: pd.DataFrame, trades: pd.DataFrame, source_hash: str) -> list[dict]:
    """核對 K 棒位置與帳務，再計算幾何 R 與扣成本後 R。"""
    times = pd.to_datetime(bars.timestamp, utc=True)
    if times.isna().any() or not times.is_monotonic_increasing or times.duplicated().any():
        raise ValueError("K 棒時間戳必須唯一且遞增")
    if trades.empty:
        return []
    fields = ["endpoint", "entry_endpoint", "exit_endpoint", "side", "entry_price",
              "exit_price", "stop_price", "target_price", "gross_same_path_return",
              "spread_return", "slippage_return", "fee_return", "funding_return", "net_return"]
    if not np.isfinite(trades[fields].to_numpy(dtype=float)).all():
        raise ValueError("交易價格、位置或成本含非有限值")
    if set(trades.source_sha256) != {source_hash}:
        raise ValueError("成交與 K 線來源雜湊不一致")
    answer = []
    for row in trades.to_dict("records"):
        indices = [row[c] for c in ("endpoint", "entry_endpoint", "exit_endpoint")]
        if any(int(x) != x for x in indices):
            raise ValueError("成交位置不是整數")
        signal, entry, exit_ = map(int, indices)
        if not 0 <= signal < entry <= exit_ < len(bars) or entry != signal + 1:
            raise ValueError("成交位置超界或不是下一棒進場")
        side = row["side"]
        start, end = float(row["entry_price"]), float(row["exit_price"])
        stop, target = float(row["stop_price"]), float(row["target_price"])
        if side not in (-1, 1) or min(start, end, stop, target) <= 0:
            raise ValueError("方向或價格不合法")
        risk, reward = side * (start - stop), side * (target - start)
        if risk <= 0 or reward <= 0:
            raise ValueError("停損停利與交易方向不一致")
        if row["exit_reason"] not in {"stop", "target", "regime", "time"}:
            raise ValueError("未支援的出場原因")
        expected_times = {"entry_at": times.iloc[entry],
                          "signal_at": times.iloc[signal] + pd.Timedelta(minutes=15),
                          "label_end_at": times.iloc[exit_] + pd.Timedelta(minutes=15)}
        for field, expected in expected_times.items():
            if pd.Timestamp(row[field]) != expected:
                raise ValueError(f"{field} 與來源 K 棒不符")
        price_return = side * (end / start - 1)
        net = float(row["net_return"])
        np.testing.assert_allclose(price_return - row["fee_return"] - row["funding_return"],
                                   net, atol=1e-12, rtol=1e-10)
        # 價差與滑價已在成交價內，只在固定路徑歸因表拆解，不能再扣一次。
        np.testing.assert_allclose(row["gross_same_path_return"] - sum(row[f] for f in
            ("spread_return", "slippage_return", "fee_return", "funding_return")), net,
            atol=1e-12, rtol=1e-10)
        intrabar = row["exit_reason"] in {"stop", "target"}
        answer.append({
            "signal": signal, "entry": entry, "exit": exit_, "side": int(side),
            "entry_price": start, "exit_price": end, "stop": stop, "target": target,
            "reason": row["exit_reason"], "intrabar": intrabar,
            "planned_rr": reward / risk, "risk_fraction": risk / start,
            "net_r": net / (risk / start), "price_return": price_return,
            "net_return": net, "gross": float(row["gross_same_path_return"]),
            "spread": float(row["spread_return"]), "slippage": float(row["slippage_return"]),
            "fee": float(row["fee_return"]), "funding": float(row["funding_return"]),
            "holding_min": (times.iloc[exit_] - times.iloc[entry]).total_seconds() / 60,
            "holding_max": (times.iloc[exit_] - times.iloc[entry]).total_seconds() / 60
                + (15 if intrabar else 0),
        })
    return answer


def render_review_html(payload: dict, target: Path) -> None:
    """Plotly 隨 HTML 內嵌，雙擊即可離線開啟，不呼叫外部服務。"""
    from plotly.offline import get_plotlyjs

    template = Path(__file__).with_name("trade_review.html").read_text(encoding="utf-8")
    content = template.replace("/*__PAYLOAD__*/", script_json(payload))
    content = content.replace("/*__PLOTLY__*/", get_plotlyjs())
    with target.open("x", encoding="utf-8") as stream:
        stream.write(content)


def render_example_png(
    bars: pd.DataFrame, trade: dict, target: Path, *,
    context_label: str = "原突破・無 AI 基準・基本成本",
    trade_label: str = "Test 第一筆（依時間，非挑選贏家）",
) -> None:
    """直接以行情繪製靜態範例，供無法開啟互動 HTML 的環境閱讀。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import Rectangle

    if target.exists():
        raise FileExistsError("不覆寫舊圖表")
    font = Path("C:/Windows/Fonts/msjh.ttc")
    if font.is_file():
        plt.rcParams["font.family"] = FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    entry, exit_ = trade["entry"], trade["exit"]
    lo, hi = max(0, entry - 48), min(len(bars), exit_ + 33)
    window = bars.iloc[lo:hi]
    times = pd.to_datetime(bars.timestamp, utc=True).dt.tz_convert("Asia/Taipei")
    fig, (ax, details) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [5, 1]})
    green, red = "#14795c", "#b53747"
    for i, row in enumerate(window.itertuples(), start=lo):
        color = green if row.close >= row.open else red
        ax.vlines(i, row.low, row.high, color=color, linewidth=.9)
        ax.add_patch(Rectangle((i - .32, min(row.open, row.close)), .64,
                     max(abs(row.close-row.open), .1), facecolor=color, edgecolor=color))
    start, end = trade["entry_price"], trade["exit_price"]
    for price, color, name in [(start, "#344352", "進場"), (trade["stop"], red, "停損"),
                               (trade["target"], green, "停利")]:
        ax.hlines(price, entry, hi + 3, colors=color, linestyles="--", linewidth=1)
        ax.text(hi + 3, price, f" {name} {price:,.2f}", color=color, fontsize=10, va="center")
    for price, color in [(trade["stop"], red), (trade["target"], green)]:
        ax.add_patch(Rectangle((entry, min(start, price)), max(1, exit_-entry+1), abs(price-start),
                     facecolor=color, alpha=.08, edgecolor="none"))
    ax.plot([entry, exit_], [start, end], color="#52616d", linestyle=":", linewidth=1)
    ax.scatter([entry], [start], marker="^" if trade["side"] == 1 else "v", s=110, color="#167bb3", zorder=5)
    ax.scatter([exit_], [end], marker="x", s=90, color=red if trade["net_return"] < 0 else green, zorder=5)
    ax.annotate("做多進場" if trade["side"] == 1 else "做空進場", (entry, start),
                xytext=(-55, 35), textcoords="offset points", arrowprops={"arrowstyle": "->"}, fontsize=11)
    reasons = {"stop": "停損出場", "target": "停利出場", "regime": "狀態退出", "time": "到期出場"}
    ax.annotate(f"{reasons[trade['reason']]} {end:,.2f}\n淨報酬 {trade['net_return']:.3%} / {trade['net_r']:+.3f}R",
                (exit_, end), xytext=(50, -55), textcoords="offset points",
                arrowprops={"arrowstyle": "->"}, fontsize=11,
                bbox={"boxstyle": "square,pad=0.3", "facecolor": "white", "edgecolor": "#d7dce0", "alpha": .95})
    ticks = list(range(lo, hi, 12))
    ax.set_xticks(ticks, [times.iloc[i].strftime("%m/%d\n%H:%M") for i in ticks])
    ax.set_xlim(lo - 1, hi + 17)
    ax.set_ylabel("BTC 價格 / USDT")
    ax.set_xlabel("台北時間 UTC+8；每根 15 分鐘")
    ax.grid(alpha=.16)
    fig.suptitle(f"BTCUSDT 逐筆交易 | {context_label}", x=.08, ha="left", fontsize=16)
    ax.set_title(f"{trade_label}  |  預定報酬:風險 {trade['planned_rr']:.2f}:1", loc="left", fontsize=11)
    details.axis("off")
    details.text(0, .80, f"進場 {times.iloc[entry]:%Y-%m-%d %H:%M}    出場棒 {times.iloc[exit_]:%Y-%m-%d %H:%M}    "
                 f"初始停損距離 {trade['risk_fraction']:.3%}    預定停利距離 {trade['risk_fraction']*trade['planned_rr']:.3%}", fontsize=11)
    details.text(0, .38, f"固定路徑毛收益 {trade['gross']:.3%} - 價差 {trade['spread']:.3%} - 滑價 {trade['slippage']:.3%} "
                 f"- 費用 {trade['fee']:.3%} - funding 準備金 {trade['funding']:.3%} = 淨報酬 {trade['net_return']:.3%}", fontsize=10)
    details.text(0, -.06, "停損 / 停利只知道出場棒，棒內時間未知。出場後 K 線是歷史，不是預測。報酬為名目部位報酬，非帳戶或槓桿報酬。", fontsize=10, color="#59646f")
    fig.tight_layout(rect=(0, .025, 1, .95))
    fig.savefig(target, dpi=140)
    plt.close(fig)
