"""機器人監控圖所需的行情與模型紀錄整理。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ai_quant_trading.data_collection.csv_storage import canonical_ohlcv_path
from ai_quant_trading.persistence import read_csv_snapshot


PRICE_COLUMNS = ("open", "high", "low", "close", "volume")


def realtime_stream_is_fresh(
    realtime_status: dict[str, object] | None,
    *,
    now: pd.Timestamp | None = None,
    max_age_seconds: float = 15.0,
) -> bool:
    """確認 WebSocket 不只標示已連線，而且近期確實收到行情。"""
    if max_age_seconds <= 0:
        raise ValueError("max_age_seconds 必須大於 0")
    status = dict(realtime_status or {})
    stream = dict(status.get("stream", {}))
    if not bool(stream.get("connected", False)):
        return False
    last_message = pd.to_datetime(stream.get("last_message_at"), utc=True, errors="coerce")
    if pd.isna(last_message):
        return False
    current = now if now is not None else pd.Timestamp.now(tz="UTC")
    if current.tzinfo is None:
        current = current.tz_localize("UTC")
    else:
        current = current.tz_convert("UTC")
    age_seconds = (current - last_message).total_seconds()
    return -2.0 <= age_seconds <= max_age_seconds


def merge_live_kline(
    stored: pd.DataFrame,
    realtime_status: dict[str, object] | None,
    interval: str,
    *,
    max_bars: int | None = 240,
) -> pd.DataFrame:
    """將 WebSocket 尚未收盤 K 線合併到本機已收盤行情。"""
    if max_bars is not None and max_bars <= 0:
        raise ValueError("max_bars 必須大於 0")
    frame = stored.copy()
    frame["_is_live"] = False
    status = dict(realtime_status or {})
    stream = dict(status.get("stream", {}))
    latest_by_interval = dict(stream.get("latest_klines_by_interval", {}))
    live_payload = latest_by_interval.get(interval)
    payload_closed = bool(isinstance(live_payload, dict) and live_payload.get("closed", False))
    should_merge = payload_closed or realtime_stream_is_fresh(status)
    if should_merge and isinstance(live_payload, dict) and live_payload.get("timestamp"):
        live = pd.DataFrame([live_payload])
        live["_is_live"] = not payload_closed
        frame = pd.concat([frame, live], ignore_index=True, sort=False)

    if frame.empty or "timestamp" not in frame:
        return pd.DataFrame(columns=["timestamp", *PRICE_COLUMNS, "_is_live"])
    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"], utc=True, errors="coerce", format="mixed"
    )
    for column in PRICE_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["_is_live"] = frame["_is_live"].fillna(False).astype(bool)
    frame = frame.dropna(subset=["timestamp", *PRICE_COLUMNS])
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    if max_bars is not None:
        frame = frame.tail(max_bars)
    return frame.reset_index(drop=True)


def load_paper_market_frame(
    raw_dir: str | Path,
    *,
    exchange: str,
    symbol: str,
    interval: str,
    realtime_status: dict[str, object] | None = None,
    max_bars: int | None = 240,
) -> pd.DataFrame:
    """讀取固定 OHLCV，並在機器人運行時補上 WebSocket 即時 K 線。"""
    path = canonical_ohlcv_path(raw_dir, exchange, symbol, interval)
    try:
        stored = read_csv_snapshot(path) if path.is_file() else pd.DataFrame()
    except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError):
        stored = pd.DataFrame()
    return merge_live_kline(
        stored,
        realtime_status,
        interval,
        max_bars=max_bars,
    )
