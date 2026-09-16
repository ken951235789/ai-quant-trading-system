"""模擬與實盤交易機器人的即時監控頁。"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ai_quant_trading.automation import (
    automation_paths,
    read_automation_status,
)
from ai_quant_trading.dashboard.robot_monitoring import (
    PositionExposure,
    ProfitMetrics,
    assess_robot_health,
    calculate_position_exposure,
    calculate_profit_metrics,
    finite_or_zero,
    signed_money,
)
from ai_quant_trading.dashboard.charts import make_model_trading_figure
from ai_quant_trading.dashboard.realtime_status import load_realtime_status
from ai_quant_trading.dashboard.trading_activity import (
    load_paper_market_frame,
    realtime_stream_is_fresh,
)
from ai_quant_trading.dashboard.transformer_risk_preview import (
    TransformerRiskPreview,
    run_transformer_risk_preview,
)
from ai_quant_trading.dashboard.ui import themed_dataframe, themed_widget_key
from ai_quant_trading.live_trading.storage import (
    CYCLE_COLUMNS,
    LivePosition,
    ORDER_COLUMNS as LIVE_ORDER_COLUMNS,
    SNAPSHOT_COLUMNS,
    emergency_halt_reason,
    live_trading_paths,
    load_positions,
    read_live_csv,
)
from ai_quant_trading.paper_trading import (
    list_paper_accounts,
    load_account_state,
)
from ai_quant_trading.paper_trading.storage import (
    ORDER_COLUMNS,
    PERFORMANCE_COLUMNS,
    POSITION_COLUMNS,
    PREDICTION_COLUMNS,
    TRADE_COLUMNS,
    read_account_csv,
)


@st.cache_resource
def _transformer_preview_executor() -> ThreadPoolExecutor:
    """共用單一背景執行緒，避免 Transformer 推論阻塞監控頁。"""
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="transformer-preview")


def _model_path(value: str, root: Path) -> Path:
    """解析新舊帳戶的模型參照，並支援搬移後的桌面版資料夾。"""
    path = Path(value)
    if path.is_absolute() and path.exists():
        return path

    if not path.is_absolute():
        direct = (root / path).resolve()
        if direct.is_relative_to(root.resolve()) and direct.exists():
            return direct
    else:
        direct = path

    # 舊版帳戶只保存訓練資料夾名稱；依目前資料結構尋找同名成品。
    name = path.name
    candidates = [
        *(root / "data" / "processed" / "rl" / "environments").glob(f"*/training/{name}"),
        root / "data" / "processed" / "models" / name,
    ]
    existing = [candidate.resolve() for candidate in candidates if candidate.exists()]
    if existing:
        return max(existing, key=lambda candidate: candidate.stat().st_mtime)
    return direct


def _profit_figure(
    curve: pd.DataFrame,
    *,
    view: str,
    range_label: str,
    initial_equity: float,
    dark_mode: bool,
) -> go.Figure:
    """建立穩定尺寸的資產、損益或回撤線圖。"""
    chart = curve.copy()
    if not chart.empty and range_label != "全部":
        days = {"7 天": 7, "30 天": 30, "90 天": 90}[range_label]
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
        filtered = chart[chart["timestamp"] >= cutoff]
        if not filtered.empty:
            chart = filtered
    settings = {
        "資產曲線": ("equity", "資產", "#2f9e78", initial_equity),
        "累計損益": ("cumulative_pnl", "累計損益", "#2f6fad", 0.0),
        "回撤": ("drawdown", "回撤", "#c84f4f", 0.0),
    }
    column, label, color, reference = settings[view]
    figure = go.Figure()
    if not chart.empty:
        figure.add_trace(
            go.Scatter(
                x=chart["timestamp"],
                y=chart[column],
                mode="lines",
                name=label,
                line={"color": color, "width": 2},
                hovertemplate=(
                    "%{x|%Y-%m-%d %H:%M}<br>%{y:.2%}<extra></extra>"
                    if column == "drawdown"
                    else "%{x|%Y-%m-%d %H:%M}<br>%{y:,.2f}<extra></extra>"
                ),
            )
        )
    figure.add_hline(
        y=reference,
        line_width=1,
        line_dash="dot",
        line_color="#78838c",
    )
    figure.update_layout(
        height=370,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        showlegend=False,
        hovermode="x unified",
        paper_bgcolor="#121516" if dark_mode else "#ffffff",
        plot_bgcolor="#121516" if dark_mode else "#ffffff",
        font={"color": "#eef2f1" if dark_mode else "#263442"},
        xaxis={"gridcolor": "#343b3d" if dark_mode else "#e6ebef"},
        yaxis={
            "gridcolor": "#343b3d" if dark_mode else "#e6ebef",
            "tickformat": ".1%" if column == "drawdown" else ",.2f",
        },
    )
    return figure


def _metric_row(
    *,
    status: str,
    metrics: ProfitMetrics,
    currency: str,
) -> None:
    cards = st.columns(5)
    cards[0].metric("機器人狀態", status)
    cards[1].metric(f"目前資產 · {currency}", f"{metrics.current_equity:,.2f}")
    cards[2].metric(
        "今日獲利",
        signed_money(metrics.today_pnl),
        delta=f"{metrics.today_return:+.2%}",
    )
    cards[3].metric(
        "本週獲利",
        signed_money(metrics.week_pnl),
        delta=f"{metrics.week_return:+.2%}",
    )
    cards[4].metric(
        "累計獲利",
        signed_money(metrics.cumulative_pnl),
        delta=f"{metrics.cumulative_return:+.2%}",
    )


def _performance_controls(prefix: str) -> tuple[str, str]:
    columns = st.columns([3, 1])
    with columns[0]:
        view = st.segmented_control(
            "圖表",
            ["資產曲線", "累計損益", "回撤"],
            default="資產曲線",
            width="stretch",
            key=f"{prefix}_chart_view",
        )
    with columns[1]:
        range_label = st.selectbox(
            "範圍",
            ["7 天", "30 天", "90 天", "全部"],
            index=1,
            key=f"{prefix}_chart_range",
        )
    return str(view), str(range_label)


def _render_latest_model_decision(predictions: pd.DataFrame) -> None:
    """顯示 RL 提案、Transformer 確認與最終風控結果。"""
    if predictions.empty:
        st.info("模型尚未完成第一個 15m 判斷。")
        return
    latest = predictions.iloc[-1]
    cards = st.columns(5)
    cards[0].metric(
        "RL 原始目標",
        f"{finite_or_zero(latest.get('model_target_fraction')):.1%}",
    )
    cards[1].metric(
        "Transformer 確認後",
        f"{finite_or_zero(latest.get('transformer_target_fraction')):.1%}",
    )
    cards[2].metric(
        "最終目標部位",
        f"{finite_or_zero(latest.get('approved_target_fraction')):.1%}",
    )
    cards[3].metric(
        "Transformer 趨勢",
        str(latest.get("transformer_trend", "-")),
    )
    selected_leverage = int(max(finite_or_zero(latest.get("selected_leverage")), 1))
    cards[4].metric("風控槓桿", f"{selected_leverage}x")
    st.caption(
        f"預期報酬：1 根 {finite_or_zero(latest.get('transformer_return_1')):+.3%}｜"
        f"5 根 {finite_or_zero(latest.get('transformer_return_5')):+.3%}｜"
        f"20 根 {finite_or_zero(latest.get('transformer_return_20')):+.3%}　"
        f"偏多 {finite_or_zero(latest.get('transformer_bull_probability')):.1%}／"
        f"偏空 {finite_or_zero(latest.get('transformer_bear_probability')):.1%}／"
        f"不確定性 {finite_or_zero(latest.get('transformer_uncertainty')):.1%}"
    )
    st.caption(
        f"機率校準："
        f"{'已校準' if finite_or_zero(latest.get('transformer_probability_calibrated')) >= 0.5 else '未校準'}｜"
        f"淨損益比 {finite_or_zero(latest.get('leverage_risk_reward')):.2f}｜"
        f"單位本金淨期望 {finite_or_zero(latest.get('leverage_net_expectancy')):+.3%}｜"
        f"證據上限 {int(max(finite_or_zero(latest.get('leverage_evidence_cap')), 1))}x"
    )
    st.caption(
        f"Regime：{latest.get('market_regime', '-')}｜"
        f"資金倍率 {finite_or_zero(latest.get('capital_multiplier')):.1%}｜"
        f"單帳戶風控倍率 {finite_or_zero(latest.get('target_risk_multiplier')):.1%}｜"
        f"模型漂移："
        f"{'嚴重' if finite_or_zero(latest.get('model_input_severe_drift')) >= 0.5 else '正常'}"
    )
    st.caption(f"判斷原因：{latest.get('decision_guard_reason', '-')}")
    st.caption(f"槓桿原因：{latest.get('leverage_reason', '-')}")


def _runtime_path(root: Path, value: object) -> Path:
    """解析狀態檔內的絕對或相對成品路徑。"""
    path = Path(str(value or ""))
    return path if path.is_absolute() else root / path


def _render_transformer_risk_preview(preview: TransformerRiskPreview) -> None:
    """顯示單次推論結果；這個區塊不會觸發任何下單。"""
    st.markdown("**Transformer 單次風控預測**")
    cards = st.columns(4)
    cards[0].metric("市場判斷", preview.trend)
    cards[1].metric("偏多機率", f"{preview.bull_probability:.1%}")
    cards[2].metric("偏空機率", f"{preview.bear_probability:.1%}")
    cards[3].metric("不確定性", f"{preview.uncertainty:.1%}")
    st.caption(
        f"預測基準：{preview.timestamp}｜收盤價 {preview.close:,.2f}｜"
        f"1／5／20 根預期報酬 {preview.expected_return_1:+.3%}／"
        f"{preview.expected_return_5:+.3%}／{preview.expected_return_20:+.3%}｜"
        f"預期波動 {preview.expected_volatility:.3%}"
    )
    st.info(f"風控結論：{preview.risk_conclusion}")
    st.caption(f"模型：{preview.checkpoint_name}。本次只做分析，不會改變模擬倉或送出訂單。")


def _render_paper_model_chart(
    *,
    root: Path,
    state: object,
    paths: object,
    account_equity: float,
    currency: str,
    dark_mode: bool,
    plotly_config: dict[str, object],
) -> None:
    """顯示 WebSocket K 線、RL 成交與 Transformer 預測端點。"""
    controls = st.columns([2.5, 1.35, 1.35])
    controls[0].subheader("RL + Transformer 即時交易圖")
    bars = controls[1].selectbox(
        "顯示 K 線",
        [60, 120, 240, 500, None],
        index=1,
        format_func=lambda value: "全部（本機保留）" if value is None else f"{value} 根",
        key="robot_monitor_market_bars",
    )
    price_range = controls[2].selectbox(
        "價格範圍",
        ["K 線細節", "完整停損停利"],
        key="robot_monitor_price_range",
    )
    status_path = paths.account_dir / "automation" / "realtime_status.json"
    realtime_status = load_realtime_status(status_path)
    market = load_paper_market_frame(
        root / "data" / "raw",
        exchange=str(state.exchange),
        symbol=str(state.symbol),
        interval=str(state.interval),
        realtime_status=realtime_status,
        max_bars=None if bars is None else int(bars),
    )
    orders = read_account_csv(paths.orders_csv, ORDER_COLUMNS)
    predictions = read_account_csv(paths.predictions_csv, PREDICTION_COLUMNS)
    connected = realtime_stream_is_fresh(realtime_status)
    has_live_candle = bool(not market.empty and market.get("_is_live", pd.Series(dtype=bool)).any())
    if connected and has_live_candle:
        st.caption("Binance WebSocket 已連線；未收盤 K 線與報價每 5 秒更新。")
    elif connected:
        st.caption("Binance WebSocket 已連線，正在等待第一筆即時 K 線。")
    else:
        st.caption("機器人未連線時顯示最後保存行情；啟用模擬機器人後會加入即時未收盤 K 線。")
    if market.empty:
        st.warning("找不到這個帳戶的 BTC K 線，請先在市場資料完成更新。")
    else:
        st.plotly_chart(
            make_model_trading_figure(
                market,
                orders=orders,
                predictions=predictions,
                position={
                    "quantity": state.quantity,
                    "entry_time": state.entry_time,
                    "entry_price": state.entry_price,
                    "stop_loss": state.stop_loss,
                    "take_profit": state.take_profit,
                    "risk_budget": state.risk_budget,
                    "account_equity": account_equity,
                    "leverage": float(getattr(state, "position_leverage", 1.0)),
                    "currency": currency,
                },
                interval=str(state.interval),
                dark_mode=dark_mode,
                fit_position_levels=price_range == "完整停損停利",
            ),
            width="stretch",
            config=plotly_config,
            key="paper_robot_model_trading_chart",
        )
        st.caption(
            "紅綠區是目前持倉的停損／停利範圍，三角形是 RL 實際進場／加碼，"
            "叉號是減碼／出場；藍色虛線是 Transformer 依最後收盤價換算的 "
            "1、5、20 根預測端點，不代表保證價格路徑。"
        )
    _render_latest_model_decision(predictions)

    preview_key = f"transformer_risk_preview_{state.account_id}"
    error_key = f"transformer_risk_preview_error_{state.account_id}"
    job_key = f"transformer_risk_preview_job_{state.account_id}"
    job = st.session_state.get(job_key)
    job_running = isinstance(job, Future) and not job.done()
    preview_actions = st.columns([3, 1])
    preview_actions[0].caption(
        "Transformer 正在背景分析，圖表仍會持續更新。"
        if job_running
        else "單次風控預測會使用最新已收盤的 5m、15m、1h、4h、1d K 線，只分析、不下單。"
    )
    if preview_actions[1].button(
        "預測中" if job_running else "預測一次",
        icon=":material/query_stats:",
        width="stretch",
        key=f"run_transformer_risk_preview_{state.account_id}",
        disabled=job_running,
    ):
        checkpoint = _runtime_path(root, realtime_status.get("transformer_checkpoint"))
        st.session_state[job_key] = _transformer_preview_executor().submit(
            run_transformer_risk_preview,
            raw_dir=root / "data" / "raw",
            checkpoint_path=checkpoint,
            symbol=str(state.symbol),
            decision_interval=str(state.interval),
            device="cpu",
        )
        st.session_state.pop(error_key, None)

    job = st.session_state.get(job_key)
    if isinstance(job, Future) and job.done():
        try:
            st.session_state[preview_key] = job.result()
            st.session_state.pop(error_key, None)
        except Exception as exc:  # pragma: no cover - Streamlit 顯示層
            st.session_state[error_key] = f"{type(exc).__name__}: {exc}"
        finally:
            st.session_state.pop(job_key, None)
    preview = st.session_state.get(preview_key)
    if isinstance(preview, TransformerRiskPreview):
        _render_transformer_risk_preview(preview)
    error = st.session_state.get(error_key)
    if error:
        st.error(f"Transformer 單次預測失敗：{error}")


def _paper_robot_summary(
    paper_dir: Path,
    root: Path,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    rows: list[dict[str, object]] = []
    bundles: dict[str, dict[str, object]] = {}
    for paths in list_paper_accounts(paper_dir):
        try:
            state = load_account_state(paths)
            performance = read_account_csv(paths.performance_csv, PERFORMANCE_COLUMNS)
            trades = read_account_csv(paths.trades_csv, TRADE_COLUMNS)
            metrics, curve = calculate_profit_metrics(
                performance,
                initial_equity=state.config.initial_capital,
                trades=trades,
            )
            automation = read_automation_status(automation_paths(paths.account_dir / "automation"))
            realtime_status = load_realtime_status(
                paths.account_dir / "automation" / "realtime_status.json"
            )
            stream_status = dict(realtime_status.get("stream", {}))
            activity_at = (
                stream_status.get("last_message_at") if stream_status.get("connected") else None
            )
            model_path = _model_path(state.model_dir, root)
            health = assess_robot_health(
                automation,
                risk_halted=state.risk_halted,
                model_available=model_path.exists(),
                activity_at=activity_at,
            )
        except (OSError, ValueError, TypeError):
            continue
        position = "多單" if state.quantity > 0 else "空單" if state.quantity < 0 else "空手"
        latest_market_price = (
            finite_or_zero(performance.iloc[-1].get("close"))
            if not performance.empty
            else finite_or_zero(state.entry_price)
        )
        exposure = calculate_position_exposure(
            quantity=float(state.quantity),
            entry_price=state.entry_price,
            market_price=latest_market_price,
            equity=metrics.current_equity,
        )
        heartbeat = (
            "-"
            if health.heartbeat_age_minutes is None
            else f"{health.heartbeat_age_minutes:.1f} 分鐘前"
        )
        rows.append(
            {
                "帳戶": state.account_id,
                "狀態": health.status,
                "標的": state.symbol,
                "週期": state.interval,
                "專家": state.expert_kind,
                "持倉": position,
                "使用本金": exposure.capital_used,
                "目前部位": exposure.position_notional,
                "帳戶曝險": exposure.account_exposure,
                "槓桿": exposure.leverage,
                "目前資產": metrics.current_equity,
                "今日損益": metrics.today_pnl,
                "本週損益": metrics.week_pnl,
                "累計損益": metrics.cumulative_pnl,
                "心跳": heartbeat,
            }
        )
        bundles[state.account_id] = {
            "paths": paths,
            "state": state,
            "performance": performance,
            "trades": trades,
            "metrics": metrics,
            "curve": curve,
            "automation": automation,
            "health": health,
            "exposure": exposure,
            "model_path": model_path,
        }
    return pd.DataFrame(rows), bundles


def _render_paper_monitor(
    *,
    root: Path,
    paper_dir: Path,
    dark_mode: bool,
    plotly_config: dict[str, object],
) -> None:
    summary, bundles = _paper_robot_summary(paper_dir, root)
    if summary.empty:
        st.info("目前沒有模擬交易機器人。")
        return
    account_ids = list(bundles)
    selected = st.selectbox(
        "監控帳戶",
        account_ids,
        format_func=lambda value: (
            f"{value} · {bundles[value]['state'].symbol} · {bundles[value]['state'].interval}"
        ),
        key="robot_monitor_paper_account",
    )
    bundle = bundles[str(selected)]
    state = bundle["state"]
    metrics = bundle["metrics"]
    curve = bundle["curve"]
    health = bundle["health"]
    automation = bundle["automation"]
    paths = bundle["paths"]
    trades = bundle["trades"]
    exposure = bundle["exposure"]
    if not isinstance(metrics, ProfitMetrics) or not isinstance(curve, pd.DataFrame):
        st.error("模擬帳戶績效資料格式錯誤，請檢查帳戶紀錄。")
        return
    if not isinstance(exposure, PositionExposure):
        st.error("模擬帳戶曝險資料格式錯誤，請檢查持倉紀錄。")
        return

    exchange = str(state.exchange).strip().lower()
    currency = "USDT" if exchange in {"binance", "binance_futures", "bybit"} else "USD"
    _metric_row(status=health.status, metrics=metrics, currency=currency)
    details = st.columns(5)
    details[0].metric("市場", state.symbol)
    details[1].metric("K 線", state.interval)
    details[2].metric("持倉方向", exposure.direction)
    details[3].metric(
        "使用本金",
        f"{exposure.capital_used:,.2f} {currency}",
    )
    details[4].metric(
        "槓桿倍數",
        f"{exposure.leverage:.2f}x" if exposure.leverage > 0 else "未開倉",
    )
    performance_details = st.columns(5)
    performance_details[0].metric(
        "名目部位",
        f"{exposure.position_notional:,.2f} {currency}",
    )
    performance_details[1].metric("帳戶曝險", f"{exposure.account_exposure:.1%}")
    performance_details[2].metric("完成交易", f"{metrics.trade_count:,}")
    performance_details[3].metric(
        "勝率",
        f"{metrics.win_rate:.1%}" if metrics.trade_count else "-",
    )
    performance_details[4].metric("最大回撤", f"{metrics.maximum_drawdown:.2%}")

    model_chart_tab, profit_chart_tab = st.tabs(["即時模型交易圖", "資產績效"])
    with model_chart_tab:
        _render_paper_model_chart(
            root=root,
            state=state,
            paths=paths,
            account_equity=metrics.current_equity,
            currency=currency,
            dark_mode=dark_mode,
            plotly_config=plotly_config,
        )
    with profit_chart_tab:
        view, range_label = _performance_controls("paper_robot")
        if curve.empty:
            st.info("帳戶尚未完成第一根 K 線，暫時沒有資產曲線。")
        else:
            st.plotly_chart(
                _profit_figure(
                    curve,
                    view=view,
                    range_label=range_label,
                    initial_equity=state.config.initial_capital,
                    dark_mode=dark_mode,
                ),
                width="stretch",
                config=plotly_config,
                key="paper_robot_profit_chart",
            )

    position_tab, activity_tab, robot_tab, all_tab = st.tabs(
        ["目前持倉", "最近交易", "機器人狀況", "全部機器人"]
    )
    with position_tab:
        positions = read_account_csv(paths.positions_csv, POSITION_COLUMNS)
        if positions.empty:
            st.info("目前沒有持倉快照。")
        else:
            themed_dataframe(
                positions.tail(100).iloc[::-1],
                width="stretch",
                hide_index=True,
                key=themed_widget_key("robot_monitor_positions"),
            )
    with activity_tab:
        orders = read_account_csv(paths.orders_csv, ORDER_COLUMNS)
        activity_kind = st.segmented_control(
            "紀錄",
            ["成交", "訂單"],
            default="成交",
            key="robot_monitor_activity_kind",
        )
        activity = trades if activity_kind == "成交" else orders
        if activity.empty:
            st.info("目前沒有交易紀錄。")
        else:
            themed_dataframe(
                activity.tail(100).iloc[::-1],
                width="stretch",
                hide_index=True,
                key=themed_widget_key("robot_monitor_activity"),
            )
    with robot_tab:
        status_rows = pd.DataFrame(
            [
                {"項目": "執行狀態", "內容": health.status},
                {"項目": "最後訊息", "內容": health.message},
                {
                    "項目": "PID",
                    "內容": str(automation.pid) if automation.pid else "-",
                },
                {"項目": "啟動時間", "內容": automation.started_at or "-"},
                {"項目": "最後心跳", "內容": automation.heartbeat_at or "-"},
                {
                    "項目": "成功輪數",
                    "內容": f"{automation.successful_cycles}/{automation.attempted_cycles}",
                },
                {"項目": "連續錯誤", "內容": str(automation.consecutive_errors)},
                {"項目": "最後錯誤", "內容": automation.last_error or "-"},
                {"項目": "模型", "內容": str(bundle["model_path"])},
                {"項目": "最後 K 線", "內容": state.last_processed_timestamp or "-"},
                {"項目": "待執行原因", "內容": state.pending_reason},
                {
                    "項目": "待執行目標持倉",
                    "內容": (
                        "-"
                        if state.pending_target_fraction is None
                        else f"{state.pending_target_fraction:.2%}"
                    ),
                },
            ]
        )
        themed_dataframe(
            status_rows,
            width="stretch",
            hide_index=True,
            key=themed_widget_key("robot_monitor_health"),
        )
    with all_tab:
        display = summary.copy()
        for column in [
            "目前資產",
            "使用本金",
            "目前部位",
            "今日損益",
            "本週損益",
            "累計損益",
        ]:
            display[column] = display[column].map(
                lambda value: (
                    f"{finite_or_zero(value):+,.2f}"
                    if column in {"今日損益", "本週損益", "累計損益"}
                    else f"{finite_or_zero(value):,.2f}"
                )
            )
        display["帳戶曝險"] = display["帳戶曝險"].map(lambda value: f"{finite_or_zero(value):.1%}")
        display["槓桿"] = display["槓桿"].map(
            lambda value: (
                "未開倉" if finite_or_zero(value) <= 0 else f"{finite_or_zero(value):.2f}x"
            )
        )
        themed_dataframe(
            display,
            width="stretch",
            hide_index=True,
            key=themed_widget_key("robot_monitor_all"),
        )


def _live_environments(live_dir: Path) -> list[str]:
    environments: list[str] = []
    for environment in ("testnet", "demo", "live"):
        paths = live_trading_paths(live_dir, environment)
        if paths.environment_dir.exists():
            environments.append(environment)
    return environments


def _live_position_rows(
    positions: dict[str, LivePosition],
    *,
    market_price: float,
    equity: float,
    currency: str,
) -> list[dict[str, object]]:
    """將現貨持倉補上使用本金、名目部位、帳戶曝險與槓桿。"""
    rows: list[dict[str, object]] = []
    for item in positions.values():
        exposure = calculate_position_exposure(
            quantity=item.quantity,
            entry_price=item.entry_price,
            market_price=market_price,
            equity=equity,
        )
        rows.append(
            {
                "標的": item.symbol,
                "方向": exposure.direction,
                "數量": item.quantity,
                "進場價": item.entry_price,
                f"實際使用本金 · {currency}": exposure.capital_used,
                f"目前名目部位 · {currency}": exposure.position_notional,
                "帳戶曝險": f"{exposure.account_exposure:.1%}",
                "槓桿": "1.00x（現貨）",
                "停損": item.stop_loss,
                "停利": item.take_profit,
                "保護狀態": item.protection_status,
                "更新時間": item.updated_at,
            }
        )
    return rows


def _render_live_monitor(
    *,
    live_dir: Path,
    dark_mode: bool,
    plotly_config: dict[str, object],
) -> None:
    environments = _live_environments(live_dir)
    if not environments:
        st.info("目前沒有實盤或測試網機器人紀錄。")
        return
    environment = st.selectbox(
        "交易環境",
        environments,
        format_func=lambda value: {
            "testnet": "Binance Testnet",
            "demo": "Binance Demo",
            "live": "Binance Live",
        }[value],
        key="robot_monitor_live_environment",
    )
    paths = live_trading_paths(live_dir, environment)
    snapshots = read_live_csv(paths.account_snapshots_csv, SNAPSHOT_COLUMNS)
    if snapshots.empty:
        st.info("此環境尚未產生資產快照。")
        return
    equity = pd.to_numeric(snapshots["estimated_equity"], errors="coerce").dropna()
    if equity.empty or float(equity.iloc[0]) <= 0:
        st.warning("資產快照沒有有效 estimated_equity。")
        return
    initial_equity = float(equity.iloc[0])
    metrics, curve = calculate_profit_metrics(
        snapshots,
        initial_equity=initial_equity,
        equity_column="estimated_equity",
    )
    automation = read_automation_status(automation_paths(paths.environment_dir / "automation"))
    halt_reason = emergency_halt_reason(paths)
    health = assess_robot_health(
        automation,
        risk_halted=halt_reason is not None,
        model_available=True,
    )
    quote_asset = str(snapshots.iloc[-1].get("quote_asset", "USDT"))
    latest_price = finite_or_zero(snapshots.iloc[-1].get("price"))
    managed_positions = load_positions(paths)
    live_exposures = [
        calculate_position_exposure(
            quantity=item.quantity,
            entry_price=item.entry_price,
            market_price=latest_price,
            equity=metrics.current_equity,
        )
        for item in managed_positions.values()
    ]
    live_capital_used = sum(item.capital_used for item in live_exposures)
    live_notional = sum(item.position_notional for item in live_exposures)
    live_account_exposure = (
        live_notional / metrics.current_equity if metrics.current_equity > 0 else 0.0
    )
    _metric_row(status=health.status, metrics=metrics, currency=quote_asset)
    details = st.columns(5)
    details[0].metric("環境", environment.upper())
    details[1].metric("標的", str(snapshots.iloc[-1].get("symbol", "-")))
    details[2].metric("最大回撤", f"{metrics.maximum_drawdown:.2%}")
    details[3].metric("成功輪數", f"{automation.successful_cycles:,}")
    details[4].metric("緊急停機", halt_reason or "未啟用")
    position_details = st.columns(5)
    position_details[0].metric(
        "持倉方向",
        "多單" if managed_positions else "空手",
    )
    position_details[1].metric(
        "使用本金",
        f"{live_capital_used:,.2f} {quote_asset}",
    )
    position_details[2].metric(
        "名目部位",
        f"{live_notional:,.2f} {quote_asset}",
    )
    position_details[3].metric("帳戶曝險", f"{live_account_exposure:.1%}")
    position_details[4].metric(
        "槓桿倍數",
        "1.00x（現貨）" if managed_positions else "未開倉",
    )

    view, range_label = _performance_controls("live_robot")
    st.plotly_chart(
        _profit_figure(
            curve,
            view=view,
            range_label=range_label,
            initial_equity=initial_equity,
            dark_mode=dark_mode,
        ),
        width="stretch",
        config=plotly_config,
        key="live_robot_profit_chart",
    )
    positions_tab, orders_tab, cycles_tab, status_tab = st.tabs(
        ["目前持倉", "訂單", "執行週期", "機器人狀況"]
    )
    with positions_tab:
        if not managed_positions:
            st.info("目前沒有由系統管理的實盤持倉。")
        else:
            themed_dataframe(
                pd.DataFrame(
                    _live_position_rows(
                        managed_positions,
                        market_price=latest_price,
                        equity=metrics.current_equity,
                        currency=quote_asset,
                    )
                ),
                width="stretch",
                hide_index=True,
                key=themed_widget_key("live_robot_positions"),
            )
    with orders_tab:
        orders = read_live_csv(paths.orders_csv, LIVE_ORDER_COLUMNS, limit=100)
        if orders.empty:
            st.info("目前沒有實盤訂單。")
        else:
            themed_dataframe(
                orders.tail(100).iloc[::-1],
                width="stretch",
                hide_index=True,
                key=themed_widget_key("live_robot_orders"),
            )
    with cycles_tab:
        cycles = read_live_csv(paths.cycles_csv, CYCLE_COLUMNS, limit=100)
        if cycles.empty:
            st.info("目前沒有執行週期紀錄。")
        else:
            themed_dataframe(
                cycles.tail(100).iloc[::-1],
                width="stretch",
                hide_index=True,
                key=themed_widget_key("live_robot_cycles"),
            )
    with status_tab:
        themed_dataframe(
            pd.DataFrame(
                [
                    {"項目": "執行狀態", "內容": health.status},
                    {"項目": "最後訊息", "內容": health.message},
                    {"項目": "最後心跳", "內容": automation.heartbeat_at or "-"},
                    {
                        "項目": "連續錯誤",
                        "內容": str(automation.consecutive_errors),
                    },
                    {"項目": "最後錯誤", "內容": automation.last_error or "-"},
                    {"項目": "緊急停機", "內容": halt_reason or "未啟用"},
                ]
            ),
            width="stretch",
            hide_index=True,
            key=themed_widget_key("live_robot_health"),
        )


@st.fragment(run_every=5)
def _render_monitor_fragment(
    *,
    root: Path,
    paper_dir: Path,
    live_dir: Path,
    dark_mode: bool,
    plotly_config: dict[str, object],
) -> None:
    """每五秒更新交易紀錄與心跳，不重載整個 Dashboard。"""
    header = st.columns([4, 1])
    mode = header[0].segmented_control(
        "帳戶類型",
        ["模擬交易", "實盤／測試網"],
        default="模擬交易",
        width="stretch",
        key="robot_monitor_mode",
    )
    if header[1].button(
        "重新整理",
        icon=":material/refresh:",
        width="stretch",
        key="robot_monitor_refresh",
    ):
        st.rerun(scope="fragment")
    if mode == "實盤／測試網":
        _render_live_monitor(
            live_dir=live_dir,
            dark_mode=dark_mode,
            plotly_config=plotly_config,
        )
    else:
        _render_paper_monitor(
            root=root,
            paper_dir=paper_dir,
            dark_mode=dark_mode,
            plotly_config=plotly_config,
        )


def render_robot_monitoring_page(
    *,
    project_root: str | Path,
    paper_dir: str | Path,
    live_dir: str | Path,
    dark_mode: bool,
    plotly_config: dict[str, object],
) -> None:
    """顯示可持續自動更新的機器人績效與健康狀況。"""
    st.title("機器人監控")
    st.caption("損益與持倉 · 資產曲線 · 心跳與風控 · 每 5 秒更新")
    _render_monitor_fragment(
        root=Path(project_root),
        paper_dir=Path(paper_dir),
        live_dir=Path(live_dir),
        dark_mode=dark_mode,
        plotly_config=plotly_config,
    )
