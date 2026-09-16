"""Dashboard Plotly 圖表。"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ai_quant_trading.market_clock import interval_duration


GREEN = "#15876f"
RED = "#cf4b4b"
BLUE = "#2f6fad"
AMBER = "#c88719"
INK = "#263442"
MUTED = "#6b7885"
GRID = "#e3e8ed"
DARK_INK = "#e6edf3"
DARK_GRID = "#3a4240"
DARK_SURFACE = "#1b2022"

OVERLAY_COLORS = {
    "sma_5": BLUE,
    "sma_20": AMBER,
    "sma_60": "#7b61a8",
    "sma_200": RED,
    "ema_5": "#37a6a6",
    "ema_20": "#d16f3f",
    "ema_60": "#65788c",
    "ema_200": "#222f3e",
}


def _base_layout(figure: go.Figure, height: int, dark_mode: bool = False) -> None:
    """套用所有交易圖表共用的安靜視覺設定。"""
    ink = DARK_INK if dark_mode else INK
    grid = DARK_GRID if dark_mode else GRID
    surface = DARK_SURFACE if dark_mode else "#ffffff"
    figure.update_layout(
        template="plotly_dark" if dark_mode else "plotly_white",
        height=height,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor=surface,
        plot_bgcolor=surface,
        font=dict(family="Arial, sans-serif", color=ink, size=12),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            font=dict(color=ink),
            bgcolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(
            bgcolor=surface,
            bordercolor=grid,
            font=dict(color=ink),
        ),
    )
    axis_style = dict(
        showgrid=True,
        gridcolor=grid,
        zeroline=False,
        color=ink,
        tickfont=dict(color=ink),
        title_font=dict(color=ink),
    )
    figure.update_xaxes(**axis_style)
    figure.update_yaxes(**axis_style)


def make_candlestick_figure(
    frame: pd.DataFrame,
    overlays: list[str] | None = None,
    show_volume: bool = True,
    dark_mode: bool = False,
) -> go.Figure:
    """建立 OHLC K 線圖，可疊加移動平均與成交量。"""
    if frame.empty:
        raise ValueError("無法繪製空的 K 線資料")

    overlays = overlays or []
    rows = 2 if show_volume else 1
    row_heights = [0.76, 0.24] if show_volume else [1.0]
    figure = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=row_heights,
    )
    figure.add_trace(
        go.Candlestick(
            x=frame["timestamp"],
            open=frame["open"],
            high=frame["high"],
            low=frame["low"],
            close=frame["close"],
            name="OHLC",
            increasing_line_color=GREEN,
            decreasing_line_color=RED,
            increasing_fillcolor=GREEN,
            decreasing_fillcolor=RED,
        ),
        row=1,
        col=1,
    )

    for overlay in overlays:
        if overlay not in frame.columns:
            continue
        figure.add_trace(
            go.Scatter(
                x=frame["timestamp"],
                y=frame[overlay],
                mode="lines",
                name=overlay.upper(),
                line=dict(
                    color=(
                        DARK_INK
                        if dark_mode and overlay == "ema_200"
                        else OVERLAY_COLORS.get(overlay, MUTED)
                    ),
                    width=1.4,
                ),
            ),
            row=1,
            col=1,
        )

    if show_volume:
        volume_colors = [
            GREEN if close >= open_price else RED
            for open_price, close in zip(frame["open"], frame["close"], strict=False)
        ]
        figure.add_trace(
            go.Bar(
                x=frame["timestamp"],
                y=frame["volume"],
                name="成交量",
                marker_color=volume_colors,
                opacity=0.55,
            ),
            row=2,
            col=1,
        )
        figure.update_yaxes(title_text="Volume", row=2, col=1)

    figure.update_xaxes(rangeslider_visible=False)
    figure.update_yaxes(title_text="Price", row=1, col=1)
    _base_layout(figure, 650, dark_mode)
    return figure


def make_indicator_figure(
    frame: pd.DataFrame,
    mode: str,
    dark_mode: bool = False,
) -> go.Figure:
    """依選擇建立 RSI、MACD、波動率或成交量指標圖。"""
    if frame.empty:
        raise ValueError("無法繪製空的指標資料")

    figure = make_subplots(specs=[[{"secondary_y": mode in {"波動率", "成交量"}}]])
    timestamp = frame["timestamp"]

    if mode == "RSI":
        figure.add_trace(
            go.Scatter(x=timestamp, y=frame["rsi_14"], name="RSI 14", line=dict(color=BLUE))
        )
        figure.add_hline(y=70, line_color=RED, line_dash="dot")
        figure.add_hline(y=30, line_color=GREEN, line_dash="dot")
        figure.update_yaxes(range=[0, 100], title_text="RSI")
    elif mode == "MACD":
        histogram_colors = [GREEN if value >= 0 else RED for value in frame["macd_histogram"]]
        figure.add_trace(
            go.Bar(
                x=timestamp,
                y=frame["macd_histogram"],
                name="Histogram",
                marker_color=histogram_colors,
                opacity=0.55,
            )
        )
        figure.add_trace(
            go.Scatter(x=timestamp, y=frame["macd"], name="MACD", line=dict(color=BLUE))
        )
        figure.add_trace(
            go.Scatter(
                x=timestamp,
                y=frame["macd_signal"],
                name="Signal",
                line=dict(color=AMBER),
            )
        )
        figure.add_hline(y=0, line_color=MUTED, line_width=1)
    elif mode == "波動率":
        figure.add_trace(
            go.Scatter(x=timestamp, y=frame["atr_14"], name="ATR 14", line=dict(color=AMBER)),
            secondary_y=False,
        )
        figure.add_trace(
            go.Scatter(
                x=timestamp,
                y=frame["historical_volatility_20"] * 100,
                name="歷史波動率 %",
                line=dict(color=BLUE),
            ),
            secondary_y=True,
        )
        figure.update_yaxes(title_text="ATR", secondary_y=False)
        figure.update_yaxes(title_text="Volatility %", secondary_y=True)
    elif mode == "成交量":
        figure.add_trace(
            go.Bar(
                x=timestamp,
                y=frame["volume_change"] * 100,
                name="成交量變化 %",
                marker_color=BLUE,
                opacity=0.5,
            ),
            secondary_y=False,
        )
        figure.add_trace(
            go.Scatter(x=timestamp, y=frame["obv"], name="OBV", line=dict(color=GREEN)),
            secondary_y=True,
        )
        figure.update_yaxes(title_text="Volume change %", secondary_y=False)
        figure.update_yaxes(title_text="OBV", secondary_y=True)
    else:
        raise ValueError(f"不支援的指標圖：{mode}")

    _base_layout(figure, 430, dark_mode)
    return figure


def make_equity_figure(
    performance: pd.DataFrame,
    dark_mode: bool = False,
) -> go.Figure:
    """建立模擬帳戶資產曲線並套用目前主題。"""
    if performance.empty:
        raise ValueError("無法繪製空的模擬資產資料")

    figure = go.Figure(
        go.Scatter(
            x=performance["timestamp"],
            y=performance["equity"],
            name="總資產",
            mode="lines",
            line=dict(color=BLUE, width=2),
        )
    )
    figure.update_yaxes(title_text="Equity")
    _base_layout(figure, 320, dark_mode)
    return figure


def _order_marker_style(reason: str, side: str) -> tuple[str, str, str]:
    """把實際模擬訂單分類成容易辨識的進出場標記。"""
    normalized = reason.strip().lower()
    if normalized == "rl_increase_long":
        return "RL 多單進場／加碼", "triangle-up", GREEN
    if normalized == "rl_reduce_long":
        return "RL 多單減碼／出場", "x", AMBER
    if normalized == "rl_increase_short":
        return "RL 空單進場／加碼", "triangle-down", RED
    if normalized == "rl_reduce_short":
        return "RL 空單減碼／出場", "x", BLUE
    return f"風控／其他成交 · {side}", "diamond", MUTED


def _add_order_markers(
    figure: go.Figure,
    orders: pd.DataFrame,
    *,
    minimum_time: pd.Timestamp,
) -> None:
    """在 K 線上加入真正成交的 RL／風控訂單。"""
    if orders.empty or not {"timestamp", "fill_price"}.issubset(orders):
        return
    normalized = orders.copy()
    normalized["timestamp"] = pd.to_datetime(
        normalized["timestamp"], utc=True, errors="coerce", format="mixed"
    )
    normalized["fill_price"] = pd.to_numeric(
        normalized["fill_price"], errors="coerce"
    )
    normalized["notional"] = pd.to_numeric(
        normalized.get("notional", 0.0), errors="coerce"
    ).fillna(0.0)
    normalized = normalized.dropna(subset=["timestamp", "fill_price"])
    normalized = normalized[
        (normalized["timestamp"] >= minimum_time) & (normalized["notional"] >= 0.01)
    ]
    if normalized.empty:
        return
    normalized["reason"] = normalized.get("reason", "").fillna("").astype(str)
    normalized["side"] = normalized.get("side", "").fillna("").astype(str)
    normalized["_marker"] = normalized.apply(
        lambda row: _order_marker_style(row["reason"], row["side"]), axis=1
    )
    for (label, symbol, color), group in normalized.groupby("_marker", sort=False):
        target = pd.to_numeric(
            group.get("target_fraction", 0.0), errors="coerce"
        ).fillna(0.0)
        quantity = pd.to_numeric(group.get("quantity", 0.0), errors="coerce").fillna(0.0)
        custom = pd.DataFrame(
            {
                "reason": group["reason"],
                "target": target,
                "quantity": quantity,
            }
        ).to_numpy()
        figure.add_trace(
            go.Scatter(
                x=group["timestamp"],
                y=group["fill_price"],
                mode="markers",
                name=label,
                marker={
                    "symbol": symbol,
                    "size": 12,
                    "color": color,
                    "line": {"width": 1, "color": "#ffffff"},
                },
                customdata=custom,
                hovertemplate=(
                    "%{x|%Y-%m-%d %H:%M}<br>成交價 %{y:,.2f}<br>"
                    "目標部位 %{customdata[1]:.1%}<br>數量 %{customdata[2]:.6f}"
                    "<br>%{customdata[0]}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )


def _add_transformer_forecast(
    figure: go.Figure,
    predictions: pd.DataFrame,
    interval: str,
) -> None:
    """將 Transformer 的多個未來報酬端點換算成可視化價格。"""
    if predictions.empty or not {"timestamp", "close"}.issubset(predictions):
        return
    duration = interval_duration(interval)
    if duration is None:
        return
    prepared = predictions.copy()
    prepared["timestamp"] = pd.to_datetime(
        prepared["timestamp"], utc=True, errors="coerce", format="mixed"
    )
    prepared["close"] = pd.to_numeric(prepared["close"], errors="coerce")
    prepared = prepared.dropna(subset=["timestamp", "close"]).sort_values("timestamp")
    if prepared.empty:
        return
    latest = prepared.iloc[-1]
    anchor_time = pd.Timestamp(latest["timestamp"])
    anchor_price = float(latest["close"])
    forecast_times = [anchor_time]
    forecast_prices = [anchor_price]
    hover_labels = ["模型判斷起點"]
    for horizon in (1, 5, 20):
        value = pd.to_numeric(latest.get(f"transformer_return_{horizon}"), errors="coerce")
        if pd.isna(value):
            continue
        expected_return = float(value)
        forecast_times.append(anchor_time + duration * horizon)
        forecast_prices.append(anchor_price * (1.0 + expected_return))
        hover_labels.append(f"未來 {horizon} 根 · 預期報酬 {expected_return:+.3%}")
    if len(forecast_times) <= 1:
        return
    figure.add_trace(
        go.Scatter(
            x=forecast_times,
            y=forecast_prices,
            mode="lines+markers",
            name="Transformer 預測端點",
            line={"color": BLUE, "width": 2, "dash": "dash"},
            marker={"size": 8, "symbol": "circle-open"},
            text=hover_labels,
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:,.2f}<br>%{text}<extra></extra>",
        ),
        row=1,
        col=1,
    )


def _optional_number(value: object) -> float | None:
    """把帳戶 JSON 數值轉成可繪圖的有限浮點數。"""
    converted = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(converted) else float(converted)


def _add_position_risk_reward(
    figure: go.Figure,
    market: pd.DataFrame,
    position: dict[str, object],
    interval: str,
    *,
    fit_position_levels: bool,
) -> None:
    """繪製 TradingView 風格的進場、停損與停利區。"""
    quantity = _optional_number(position.get("quantity")) or 0.0
    entry_price = _optional_number(position.get("entry_price"))
    entry_time = pd.to_datetime(
        position.get("entry_time"), utc=True, errors="coerce"
    )
    if abs(quantity) <= 1e-12 or entry_price is None or entry_price <= 0:
        return
    if pd.isna(entry_time):
        entry_time = market["timestamp"].iloc[0]
    duration = interval_duration(interval)
    if duration is None:
        return

    stop_loss = _optional_number(position.get("stop_loss"))
    take_profit = _optional_number(position.get("take_profit"))
    current_price = float(market.iloc[-1]["close"])
    risk_budget = _optional_number(position.get("risk_budget"))
    account_equity = _optional_number(position.get("account_equity"))
    leverage = _optional_number(position.get("leverage")) or 1.0
    currency = str(position.get("currency") or "USDT")
    chart_start = pd.Timestamp(market["timestamp"].min())
    start = max(pd.Timestamp(entry_time), chart_start)
    end = max(
        pd.Timestamp(market["timestamp"].max()) + duration * 20,
        start + duration,
    )
    long_position = quantity > 0
    direction = "多單" if long_position else "空單"
    open_pnl = (
        (current_price - entry_price) * abs(quantity)
        if long_position
        else (entry_price - current_price) * abs(quantity)
    )
    capital_used = abs(quantity) * entry_price / leverage
    position_notional = abs(quantity) * current_price
    account_exposure = (
        position_notional / account_equity
        if account_equity is not None and account_equity > 0
        else None
    )

    if take_profit is not None:
        profit_low, profit_high = sorted((entry_price, take_profit))
        figure.add_shape(
            type="rect",
            x0=start,
            x1=end,
            y0=profit_low,
            y1=profit_high,
            fillcolor="rgba(21, 135, 111, 0.20)",
            line={"width": 0},
            layer="below",
            row=1,
            col=1,
        )
        target_return = (
            (take_profit - entry_price) / entry_price
            if long_position
            else (entry_price - take_profit) / entry_price
        )
        figure.add_shape(
            type="line",
            x0=start,
            x1=end,
            y0=take_profit,
            y1=take_profit,
            line={"color": GREEN, "width": 2},
            row=1,
            col=1,
        )
        figure.add_annotation(
            x=end,
            y=take_profit,
            text=f"停利 {take_profit:,.2f} · {target_return:+.2%}",
            showarrow=False,
            xanchor="right",
            yshift=12,
            bgcolor=GREEN,
            bordercolor=GREEN,
            font={"color": "#ffffff", "size": 11},
            row=1,
            col=1,
        )

    if stop_loss is not None:
        risk_low, risk_high = sorted((entry_price, stop_loss))
        figure.add_shape(
            type="rect",
            x0=start,
            x1=end,
            y0=risk_low,
            y1=risk_high,
            fillcolor="rgba(207, 75, 75, 0.20)",
            line={"width": 0},
            layer="below",
            row=1,
            col=1,
        )
        stop_return = (
            (entry_price - stop_loss) / entry_price
            if long_position
            else (stop_loss - entry_price) / entry_price
        )
        figure.add_shape(
            type="line",
            x0=start,
            x1=end,
            y0=stop_loss,
            y1=stop_loss,
            line={"color": RED, "width": 2},
            row=1,
            col=1,
        )
        figure.add_annotation(
            x=end,
            y=stop_loss,
            text=f"停損 {stop_loss:,.2f} · {-abs(stop_return):.2%}",
            showarrow=False,
            xanchor="right",
            yshift=-12,
            bgcolor=RED,
            bordercolor=RED,
            font={"color": "#ffffff", "size": 11},
            row=1,
            col=1,
        )

    figure.add_shape(
        type="line",
        x0=start,
        x1=end,
        y0=entry_price,
        y1=entry_price,
        line={"color": BLUE, "width": 2},
        row=1,
        col=1,
    )
    risk_distance = abs(entry_price - stop_loss) if stop_loss is not None else 0.0
    reward_distance = (
        abs(take_profit - entry_price) if take_profit is not None else 0.0
    )
    risk_reward = reward_distance / risk_distance if risk_distance > 0 else None
    details = [
        f"{direction}進場 {entry_price:,.2f}",
        f"數量 {abs(quantity):.6f}",
        f"本金 {capital_used:,.2f} {currency} · 部位 {position_notional:,.2f}",
        (
            f"帳戶曝險 {account_exposure:.1%} · 槓桿 {leverage:.2f}x"
            if account_exposure is not None
            else f"槓桿 {leverage:.2f}x"
        ),
        f"浮動損益 {open_pnl:+,.2f}",
    ]
    if risk_reward is not None:
        details.append(f"R/R {risk_reward:.2f}")
    if risk_budget is not None:
        details.append(f"風險 {risk_budget:,.2f} USDT")
    if stop_loss is not None:
        details.append(f"停損 {stop_loss:,.2f}")
    if take_profit is not None:
        details.append(f"停利 {take_profit:,.2f}")
    figure.add_annotation(
        x=end,
        y=entry_price,
        text="<br>".join(details),
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        yshift=10,
        align="left",
        bgcolor="rgba(47, 111, 173, 0.92)",
        bordercolor=BLUE,
        font={"color": "#ffffff", "size": 11},
        row=1,
        col=1,
    )

    levels = [
        value
        for value in (stop_loss, take_profit)
        if value is not None
    ]
    if levels and fit_position_levels:
        low = min(float(market["low"].min()), *levels)
        high = max(float(market["high"].max()), *levels)
        padding = max((high - low) * 0.04, entry_price * 0.001)
        figure.update_yaxes(range=[low - padding, high + padding], row=1, col=1)
    elif not fit_position_levels:
        # 短線停損停利可能離現價很遠；預設聚焦 K 線，避免蠟燭被壓成細線。
        low = float(market["low"].min())
        high = float(market["high"].max())
        padding = max((high - low) * 0.10, entry_price * 0.0005)
        figure.update_yaxes(range=[low - padding, high + padding], row=1, col=1)


def make_model_trading_figure(
    market: pd.DataFrame,
    *,
    orders: pd.DataFrame | None = None,
    predictions: pd.DataFrame | None = None,
    position: dict[str, object] | None = None,
    interval: str = "5m",
    dark_mode: bool = False,
    fit_position_levels: bool = False,
) -> go.Figure:
    """建立即時 K 線、RL 成交與 Transformer 判斷的整合圖。"""
    if market.empty:
        raise ValueError("沒有可繪製的即時 K 線")
    frame = market.copy()
    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"], utc=True, errors="coerce", format="mixed"
    )
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(
        subset=["timestamp", "open", "high", "low", "close", "volume"]
    ).sort_values("timestamp")
    if frame.empty:
        raise ValueError("即時 K 線沒有有效 OHLCV")
    if "_is_live" not in frame:
        frame["_is_live"] = False
    frame["_is_live"] = frame["_is_live"].fillna(False).astype(bool)

    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=[0.68, 0.19, 0.13],
    )
    closed = frame[~frame["_is_live"]]
    live = frame[frame["_is_live"]]
    if not closed.empty:
        figure.add_trace(
            go.Candlestick(
                x=closed["timestamp"],
                open=closed["open"],
                high=closed["high"],
                low=closed["low"],
                close=closed["close"],
                name="已收盤 K 線",
                increasing_line_color=GREEN,
                decreasing_line_color=RED,
                increasing_fillcolor=GREEN,
                decreasing_fillcolor=RED,
            ),
            row=1,
            col=1,
        )
    if not live.empty:
        figure.add_trace(
            go.Candlestick(
                x=live["timestamp"],
                open=live["open"],
                high=live["high"],
                low=live["low"],
                close=live["close"],
                name="即時未收盤 K 線",
                increasing_line_color="#2bbf9b",
                decreasing_line_color="#f06a6a",
                increasing_fillcolor="#2bbf9b",
                decreasing_fillcolor="#f06a6a",
                opacity=0.75,
            ),
            row=1,
            col=1,
        )

    order_frame = orders if orders is not None else pd.DataFrame()
    prediction_frame = predictions if predictions is not None else pd.DataFrame()
    _add_order_markers(figure, order_frame, minimum_time=frame["timestamp"].min())
    _add_transformer_forecast(figure, prediction_frame, interval)
    if position:
        _add_position_risk_reward(
            figure,
            frame,
            position,
            interval,
            fit_position_levels=fit_position_levels,
        )

    if not prediction_frame.empty and "timestamp" in prediction_frame:
        decision = prediction_frame.copy()
        decision["timestamp"] = pd.to_datetime(
            decision["timestamp"], utc=True, errors="coerce", format="mixed"
        )
        decision = decision.dropna(subset=["timestamp"])
        decision = decision[decision["timestamp"] >= frame["timestamp"].min()]
        for column, name, color, dash in (
            ("model_target_fraction", "RL 原始目標", "#7b61a8", "dot"),
            ("approved_target_fraction", "風控後最終目標", BLUE, "solid"),
        ):
            if column not in decision:
                continue
            values = pd.to_numeric(decision[column], errors="coerce") * 100
            figure.add_trace(
                go.Scatter(
                    x=decision["timestamp"],
                    y=values,
                    mode="lines",
                    name=name,
                    line={"color": color, "width": 1.8, "dash": dash},
                    hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.1f}%<extra></extra>",
                ),
                row=2,
                col=1,
            )
    figure.add_hline(y=0, line_width=1, line_dash="dot", line_color=MUTED, row=2, col=1)

    volume_colors = [
        GREEN if close >= open_price else RED
        for open_price, close in zip(frame["open"], frame["close"], strict=False)
    ]
    figure.add_trace(
        go.Bar(
            x=frame["timestamp"],
            y=frame["volume"],
            name="成交量",
            marker_color=volume_colors,
            opacity=0.5,
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:,.4f}<extra></extra>",
        ),
        row=3,
        col=1,
    )
    figure.update_xaxes(rangeslider_visible=False)
    figure.update_yaxes(title_text="BTC/USDT", row=1, col=1)
    figure.update_yaxes(title_text="目標 %", ticksuffix="%", row=2, col=1)
    figure.update_yaxes(title_text="量", row=3, col=1)
    _base_layout(figure, 760, dark_mode)
    return figure
