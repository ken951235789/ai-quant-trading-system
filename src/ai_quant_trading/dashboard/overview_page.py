"""交易總覽、Watchlist、模擬持倉與系統狀態頁。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ai_quant_trading.dashboard.charts import make_candlestick_figure
from ai_quant_trading.dashboard.services import (
    ensure_chart_features,
    list_ohlcv_files,
    load_market_frame,
    relative_file_label,
)
from ai_quant_trading.dashboard.ui import themed_dataframe, themed_widget_key
from ai_quant_trading.paper_trading import list_paper_accounts, load_account_state
from ai_quant_trading.reinforcement_learning import list_rl_training_runs


@st.cache_data(show_spinner=False)
def build_watchlist(
    path_values: tuple[tuple[str, float], ...],
    maximum_markets: int = 40,
) -> pd.DataFrame:
    """從每個市場最新固定檔建立可掃描的行情清單。"""
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    for path_text, _ in path_values:
        path = Path(path_text)
        try:
            frame = load_market_frame(path)
        except (OSError, ValueError):
            continue
        if frame.empty:
            continue
        latest = frame.iloc[-1]
        key = (
            str(latest.get("exchange", "unknown")),
            str(latest.get("symbol", path.stem)),
            str(latest.get("interval", "unknown")),
        )
        if key in seen:
            continue
        seen.add(key)
        previous_close = float(frame.iloc[-2]["close"]) if len(frame) > 1 else float(latest["close"])
        close = float(latest["close"])
        change = close / previous_close - 1 if previous_close else 0.0
        timestamp = pd.Timestamp(latest["timestamp"])
        rows.append(
            {
                "市場": (
                    "加密貨幣"
                    if key[0].lower() in {"binance", "binance_futures", "bybit"}
                    else "美股"
                ),
                "標的": key[1],
                "週期": key[2],
                "最新價": close,
                "漲跌": change,
                "成交量": float(latest.get("volume", 0)),
                "資料時間": timestamp,
                "檔案": str(path),
            }
        )
        if len(rows) >= maximum_markets:
            break
    return pd.DataFrame(rows)


def _paper_positions(paper_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for paths in list_paper_accounts(paper_dir):
        try:
            state = load_account_state(paths)
        except (OSError, ValueError, TypeError):
            continue
        if state.model_kind != "rl":
            continue
        direction = "多" if state.quantity > 0 else "空" if state.quantity < 0 else "空手"
        rows.append(
            {
                "帳戶": state.account_id,
                "專家": state.expert_kind,
                "標的": state.symbol,
                "方向": direction,
                "數量": state.quantity,
                "進場價": state.entry_price,
                "待執行目標": state.pending_target_fraction,
                "風控停機": state.risk_halted,
                "最後 K 線": state.last_processed_timestamp,
            }
        )
    return pd.DataFrame(rows)


def render_trading_overview_page(
    *,
    project_root: str | Path,
    raw_dir: str | Path,
    rl_dir: str | Path,
    paper_dir: str | Path,
    dark_mode: bool,
    plotly_config: dict[str, object],
) -> None:
    """顯示本機最新市場資料與交易系統執行狀態。"""
    root = Path(project_root)
    raw_root = Path(raw_dir)
    files = list_ohlcv_files(raw_root)
    path_values = tuple((str(path), path.stat().st_mtime) for path in files)
    watchlist = build_watchlist(path_values)
    training_runs = list_rl_training_runs(rl_dir)
    positions = _paper_positions(Path(paper_dir))

    st.title("交易總覽")
    latest_time = (
        pd.to_datetime(watchlist["資料時間"], utc=True).max()
        if not watchlist.empty
        else None
    )
    summary = st.columns(4)
    summary[0].metric("市場", f"{len(watchlist):,}")
    summary[1].metric("PPO／SAC 成品", f"{len(training_runs):,}")
    summary[2].metric("模擬帳戶", f"{len(positions):,}")
    summary[3].metric(
        "最新資料 UTC",
        latest_time.strftime("%m-%d %H:%M") if latest_time is not None else "-",
    )

    market_tab, positions_tab, system_tab = st.tabs(["市場", "模擬持倉", "系統"])
    with market_tab:
        if watchlist.empty:
            st.warning("目前沒有本機 OHLCV，請先到「市場資料」下載。")
        else:
            table = watchlist.drop(columns=["檔案"]).copy()
            table["漲跌"] = table["漲跌"].map(lambda value: f"{float(value):+.2%}")
            table["最新價"] = table["最新價"].map(lambda value: f"{float(value):,.4f}")
            table["成交量"] = table["成交量"].map(lambda value: f"{float(value):,.0f}")
            table["資料時間"] = pd.to_datetime(
                table["資料時間"],
                utc=True,
            ).dt.strftime("%Y-%m-%d %H:%M")
            themed_dataframe(
                table,
                width="stretch",
                hide_index=True,
                height=280,
                key=themed_widget_key("overview_watchlist"),
            )

            labels = [
                f"{row['標的']} · {row['週期']} · {row['市場']}"
                for _, row in watchlist.iterrows()
            ]
            selected = st.selectbox(
                "圖表標的",
                range(len(labels)),
                format_func=lambda index: labels[index],
            )
            selected_path = Path(str(watchlist.iloc[int(selected)]["檔案"]))
            try:
                frame = ensure_chart_features(load_market_frame(selected_path))
                maximum = min(len(frame), 1_000)
                visible = st.slider(
                    "顯示 K 線",
                    min_value=min(20, maximum),
                    max_value=maximum,
                    value=min(200, maximum),
                    step=10 if maximum >= 20 else 1,
                )
                chart_frame = frame.tail(int(visible))
                figure = make_candlestick_figure(
                    chart_frame,
                    overlays=["sma_20", "sma_60"],
                    show_volume=True,
                    dark_mode=dark_mode,
                )
                st.plotly_chart(
                    figure,
                    width="stretch",
                    config=plotly_config,
                    key="overview_candlestick",
                )
                st.caption(relative_file_label(selected_path, root))
            except (OSError, ValueError) as exc:
                st.error(str(exc))

    with positions_tab:
        if positions.empty:
            st.info("目前沒有 PPO 模擬帳戶。")
        else:
            themed_dataframe(
                positions,
                width="stretch",
                hide_index=True,
                key=themed_widget_key("overview_positions"),
            )

    with system_tab:
        config_path = root / "data" / "config" / "ai_pipeline.json"
        environment_count = len(
            [path for path in Path(rl_dir).glob("*") if path.is_dir()]
        )
        status = pd.DataFrame(
            [
                {
                    "元件": "市場資料",
                    "狀態": "就緒" if files else "無資料",
                    "數量": len(files),
                },
                {
                    "元件": "AI 管線設定",
                    "狀態": "已保存" if config_path.exists() else "使用預設值",
                    "數量": 1 if config_path.exists() else 0,
                },
                {
                    "元件": "RL 環境",
                    "狀態": "就緒" if environment_count else "尚未建立",
                    "數量": environment_count,
                },
                {
                    "元件": "RL 訓練成品",
                    "狀態": "可選擇" if training_runs else "尚無成品",
                    "數量": len(training_runs),
                },
            ]
        )
        themed_dataframe(
            status,
            width="stretch",
            hide_index=True,
            key=themed_widget_key("overview_system_status"),
        )
