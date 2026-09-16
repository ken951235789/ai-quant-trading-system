"""BTC 15m WebSocket 模型狀態元件。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


def load_realtime_status(path: str | Path) -> dict[str, object]:
    """讀取背景 Worker 原子寫入的即時狀態。"""
    source = Path(path)
    if not source.is_file():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        return dict(payload) if isinstance(payload, dict) else {}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {"state": "error", "last_error": "即時狀態檔無法讀取"}


def _local_time(value: object) -> str:
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return "-"
    return timestamp.tz_convert("Asia/Taipei").strftime("%m/%d %H:%M:%S")


@st.fragment(run_every=2)
def render_realtime_status(path: str | Path) -> None:
    """每兩秒更新 WebSocket、Transformer 與 RL 最新狀態。"""
    status = load_realtime_status(path)
    if not status:
        st.caption("尚未建立 WebSocket 即時狀態。")
        return
    stream = dict(status.get("stream", {}))
    decision = dict(status.get("latest_decision", {}))
    connected = bool(stream.get("connected", False))
    columns = st.columns(5)
    columns[0].metric("行情連線", "已連線" if connected else str(stream.get("state", "-")))
    columns[1].metric(
        "延遲",
        f"{float(stream.get('last_event_latency_ms', 0.0)):.0f} ms"
        if stream.get("last_event_latency_ms") is not None
        else "-",
    )
    columns[2].metric(
        "Spread",
        f"{float(stream.get('spread_bps', 0.0)):.2f} bps"
        if stream.get("spread_bps") is not None
        else "-",
    )
    columns[3].metric("Transformer 趨勢", str(decision.get("transformer_trend", "-")))
    columns[4].metric(
        "最終目標部位",
        f"{float(decision.get('approved_target_fraction', 0.0)):.1%}"
        if decision
        else "-",
    )
    closed = dict(stream.get("last_closed_by_interval", {}))
    st.caption(
        f"最新 1m：{_local_time(closed.get('1m'))}｜"
        f"最新 5m：{_local_time(closed.get('5m'))}｜"
        f"模型資料：{_local_time(status.get('last_market_timestamp'))}｜"
        f"處理耗時：{float(status.get('decision_seconds', 0.0)):.2f} 秒"
    )
    if decision:
        st.caption(
            f"RL 原始 {float(decision.get('model_target_fraction', 0.0)):.1%}｜"
            f"趨勢確認後 {float(decision.get('transformer_target_fraction', 0.0)):.1%}｜"
            f"牛 {float(decision.get('transformer_bull_probability', 0.0)):.1%}／"
            f"熊 {float(decision.get('transformer_bear_probability', 0.0)):.1%}｜"
            f"{decision.get('decision_guard_reason', '-') }"
        )
    error = status.get("last_error") or stream.get("last_error")
    if error:
        st.error(str(error))
