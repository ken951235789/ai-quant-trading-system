"""在桌面 App 中檢視已保存研究成交，不啟動推論、回測或下單。"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ai_quant_trading.dashboard.charts import make_candlestick_figure
from ai_quant_trading.dashboard.ui import themed_dataframe, themed_widget_key
from ai_quant_trading.research.trade_review_bundle import FILENAME, load_review_bundle

FAMILIES = {"original": "原突破", "trend_pullback": "趨勢回調", "vwap_reversion": "VWAP 回歸"}
REASONS = {"stop": "停損", "target": "停利", "regime": "狀態退出", "time": "持有到期"}


@st.cache_data(ttl=60, max_entries=3, show_spinner=False)
def _load(path: str, modified: int, manifest_modified: int):
    return load_review_bundle(Path(path))


def _number(value, *, percent=False):
    if value is None or pd.isna(value):
        return "無法計算"
    return f"{value:.3%}" if percent else f"{value:.2f}"


def _bar_time(payload, index, timezone):
    return pd.Timestamp(payload["bars"][index-payload["offset"]][0], unit="ms", tz="UTC").tz_convert(timezone)


def make_trade_review_figure(payload, trade, *, dark_mode=False, context=64, timezone="Asia/Taipei"):
    """以同一份來源 K 棒標記模擬成交，棒內退出只對齊出場棒。"""
    offset, bars = payload["offset"], payload["bars"]
    lo = max(offset, trade["entry"]-context)
    hi = min(offset+len(bars), trade["exit"]+context+1)
    frame = pd.DataFrame(bars[lo-offset:hi-offset], columns=["timestamp", "open", "high", "low", "close"])
    # 先轉指定時區再移除 tz，避免 Plotly 隨電腦時區再次轉換。
    frame["timestamp"] = pd.to_datetime(frame.timestamp, unit="ms", utc=True).dt.tz_convert(timezone).dt.tz_localize(None)
    entry_time = _bar_time(payload, trade["entry"], timezone).tz_localize(None)
    exit_time = _bar_time(payload, trade["exit"], timezone).tz_localize(None)
    figure = make_candlestick_figure(frame, show_volume=False, dark_mode=dark_mode)
    ink, green, red = ("#e6edf3", "#4ed0a5", "#ff8391") if dark_mode else ("#263442", "#14795c", "#b53747")
    figure.add_trace(go.Scatter(x=[entry_time, exit_time], y=[trade["entry_price"], trade["exit_price"]],
        mode="lines", line={"color": ink, "dash": "dot", "width": 1}, showlegend=False, hoverinfo="skip"))
    for timestamp, price, text, symbol, color in [
        (entry_time, trade["entry_price"], "做多進場" if trade["side"] == 1 else "做空進場",
         "triangle-up" if trade["side"] == 1 else "triangle-down", "#359fe1"),
        (exit_time, trade["exit_price"], f"{REASONS[trade['reason']]} / {trade['net_r']:+.3f}R", "x", green if trade["net_return"] > 0 else red),
    ]:
        figure.add_trace(go.Scatter(x=[timestamp], y=[price], name=text, mode="markers",
            marker={"symbol": symbol, "size": 13, "color": color}, hovertemplate=text+"<br>%{y:,.2f}<extra></extra>"))
    for price, name, color in [(trade["entry_price"], "進場", ink), (trade["stop"], "停損", red), (trade["target"], "停利", green)]:
        figure.add_shape(type="line", x0=entry_time, x1=frame.timestamp.iloc[-1], y0=price, y1=price,
                         line={"color": color, "dash": "dash", "width": 1})
        figure.add_annotation(xref="paper", x=1, y=price, text=f"{name} {price:,.2f}", showarrow=False,
            xanchor="right", font={"color": color}, bgcolor="#1b2022" if dark_mode else "#fff")
    for price, color in [(trade["stop"], "rgba(210,63,82,.10)"), (trade["target"], "rgba(18,148,103,.10)")]:
        figure.add_shape(type="rect", x0=entry_time, x1=exit_time+pd.Timedelta(minutes=15),
            y0=trade["entry_price"], y1=price, fillcolor=color, line={"width": 0}, layer="below")
    low = min(frame.low.min(), trade["stop"], trade["entry_price"], trade["exit_price"])
    high = max(frame.high.max(), trade["target"], trade["entry_price"], trade["exit_price"])
    padding = max((high-low)*.1, .01)
    figure.update_layout(height=540, margin={"l": 10, "r": 10, "t": 45, "b": 10})
    figure.update_yaxes(range=[low-padding, high+padding], title="USDT")
    figure.update_xaxes(title="台北 UTC+8" if timezone == "Asia/Taipei" else "UTC")
    return figure


def render_trade_review_page(project_root: Path, dark_mode: bool):
    root = project_root / "data" / "processed" / "research_reviews"
    bundles = sorted((p.parent for p in root.glob(f"*/{FILENAME}")), reverse=True)
    if not bundles:
        st.info("尚未匯入逐筆研究資料。")
        return
    selected = st.selectbox("研究批次", bundles, format_func=lambda p: p.name, key="review_bundle")
    try:
        body = _load(str(selected), (selected/FILENAME).stat().st_mtime_ns,
                     (selected/"artifact_manifest.json").stat().st_mtime_ns)
        payload = body["payload"]
    except (OSError, KeyError, TypeError, ValueError) as exc:
        st.error(f"研究資料無法讀取：{exc}")
        return
    st.caption(f"{body['label']} · 唯讀研究 · 非實盤資格")
    columns = st.columns(4)
    family = columns[0].selectbox("策略", list(FAMILIES), format_func=FAMILIES.get, key="review_family")
    policy = columns[1].selectbox("訊號篩選", ["no_ai", "transformer"],
        format_func=lambda x: "無 AI 基準" if x == "no_ai" else "Transformer", key="review_policy")
    split = columns[2].selectbox("研究區間", ["test", "selection"],
        format_func=lambda x: "Test（已研究）" if x == "test" else "Selection", key="review_split")
    scenario = columns[3].selectbox("成本", ["base", "stress", "zero"],
        format_func={"base": "基本假設", "stress": "壓力假設", "zero": "零成本"}.get, key="review_cost")
    cols = st.columns(3)
    variant = cols[0].selectbox("特徵版本", ["F_compact_combined", "F_existing"],
        format_func=lambda x: "新精簡特徵" if x == "F_compact_combined" else "舊特徵", disabled=policy=="no_ai", key="review_variant")
    seed = cols[1].selectbox("Seed", [42, 137, 2026], disabled=policy=="no_ai", key="review_seed")
    threshold = cols[2].number_input("預測成功機率門檻 %", 0, 100, 55, 1, disabled=policy=="no_ai", key="review_threshold")
    rows = [r for r in payload["rows"] if (r["family"], r["policy"], r["split"], r["scenario"]) == (family, policy, split, scenario)
            and (policy=="no_ai" or (r["variant"], r["seed"], r["threshold_pct"]) == (variant, seed, threshold))]
    if len(rows) != 1:
        st.error("缺少或重複研究條件，不以其他結果代替。")
        return
    row = rows[0]
    trades = payload["schedules"][row["schedule"]]
    columns = st.columns(3)
    for i, (name, value) in enumerate([("成交數", str(row["trades"])), ("實际勝率", _number(row["actual_win_rate"], percent=True)),
        ("實際淨盈虧比", _number(row["payoff_ratio"])), ("平均獲利", _number(row["average_win"], percent=True)),
        ("平均虧損幅度", _number(row["average_loss"], percent=True)), ("每筆平均淨報酬", _number(row["mean_net"], percent=True))]):
        columns[i%3].metric(name, value)
    st.caption(f"訊號範圍（UTC）：{payload['windows'][family+'|'+split]}")
    if policy == "no_ai":
        st.caption("無 AI 規則基準，不是 Transformer 或 SAC 成交。")
    else:
        st.caption("預測淨收益仍须 >2 bps；0% 機率門檻不等於無 AI。不變更正式門檻。")
    contract = payload["contracts"][row["contract_sha256"]]
    with st.expander("出場規則與成本"):
        regimes = {"loss": "方向狀態消失（含中性）退出", "opposite": "反向狀態退出", "disabled": "不使用狀態退出"}
        st.write(f"固定停損 {contract['stop_atr']} ATR、停利 {contract['stop_atr']*contract['reward_r']} ATR；最長 {contract['holding_bars']} 棒。{regimes[contract['regime_exit']]}。")
        st.write(f"單邊手續費 {contract['fee_bps']} bps、滑價 {contract['slippage_bps']} bps、全價差 {contract['spread_bps']} bps；每次結算 funding 準備金 {contract['funding_reserve_bps']} bps。")
        st.caption("同棒觸及停損停利採停損優先；跳空可能超過停損。各成本情境重新撮合。資金費為準備金而非實際歷史費率，稅務未驗證。")
    if not trades:
        st.warning("此條件沒有成交，盈虧比無法計算；不代表零風險或獲利。")
        return
    cols = st.columns([3, 1, 1])
    timezone = cols[2].selectbox("時區", ["Asia/Taipei", "UTC"], format_func=lambda x: "台北 UTC+8" if x=="Asia/Taipei" else "UTC", key="review_timezone")
    context = cols[1].selectbox("前後 K 棒", [32, 64, 128], index=1, key="review_context")
    # 排程改變時重設交易索引；同一排程切主題或調整圖表不丟失選取。
    state_key = "review_trade_index"
    if st.session_state.get("review_schedule") != (str(selected), row["schedule"]):
        st.session_state["review_schedule"] = (str(selected), row["schedule"])
        st.session_state[state_key] = 0
    index = cols[0].selectbox("逐筆交易", list(range(len(trades))), key=state_key,
        format_func=lambda i: f"{i+1}/{len(trades)} · {_bar_time(payload,trades[i]['entry'],timezone):%Y-%m-%d %H:%M} · {'多' if trades[i]['side']==1 else '空'} · {trades[i]['net_return']:+.3%}")
    trade = trades[index]
    cols = st.columns(4)
    for col, label, value in zip(cols, ["預定報酬／風險", "實現淨 R", "本筆淨報酬", "退出原因"],
        [f"{trade['planned_rr']:.2f}:1", f"{trade['net_r']:+.3f}R", f"{trade['net_return']:+.3%}", REASONS[trade["reason"]]], strict=True):
        col.metric(label, value)
    st.plotly_chart(make_trade_review_figure(payload, trade, dark_mode=dark_mode, context=context, timezone=timezone),
        width="stretch", key=themed_widget_key("review_candles"), config={"displaylogo": False, "scrollZoom": True})
    st.caption("進出場標記為含摩擦的模擬成交價；停損／停利只知道出場棒，無棒內精確時間。出場後價格是歷史，不是預測。")
    detail = pd.DataFrame([{"項目": label, "報酬": trade[key]} for label, key in
        [("固定路徑毛收益", "gross"), ("價差", "spread"), ("滑價", "slippage"), ("手續費", "fee"), ("Funding 準備金", "funding"), ("淨報酬", "net_return")]])
    with st.expander("本筆成本帳務"):
        themed_dataframe(detail, hide_index=True, width="stretch", key=themed_widget_key("review_costs"))
        st.caption("滑價與價差已反映在成交價，不能再次扣除。淨報酬以進場名目部位為分母，非帳戶本金回報。")
    with st.expander("所有成交"):
        table = pd.DataFrame([{"序號": i+1, "進場時間": str(_bar_time(payload,t['entry'],timezone)),
            "方向": "多" if t['side']==1 else "空", "進場價": t['entry_price'], "出場價": t['exit_price'],
            "出場原因": REASONS[t['reason']], "淨報酬": t['net_return'], "實現R": t['net_r']} for i,t in enumerate(trades)])
        themed_dataframe(table, hide_index=True, width="stretch", key=themed_widget_key("review_trades"))
    st.caption("研究品質仍為證據不足；本頁不含完整帳戶浮虧、保證金或清算驗證，不會開啟下單。")
