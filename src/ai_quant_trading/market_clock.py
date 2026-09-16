"""回測、模擬與實盤共用的 K 線收盤判斷。"""

from __future__ import annotations

from datetime import datetime, timezone
import re

import pandas as pd


# 既有交易預設維持五週期；完整歷史下載可選擇下面的九週期集合。
BTC_MULTITIMEFRAME_INTERVALS = (
    "5m",
    "15m",
    "1h",
    "4h",
    "1d",
)

BTC_SUPPORTED_MULTITIMEFRAME_INTERVALS = (
    "1m", "3m", "5m", "15m", "30m", "1h", "4h", "12h", "1d",
)


def interval_duration(interval: str) -> pd.Timedelta | None:
    """將常見 Binance／Yahoo 週期轉成保守的 K 線長度。"""
    normalized = interval.strip().lower()
    fixed = {
        "1d": pd.Timedelta(days=1),
        "5d": pd.Timedelta(days=5),
        "1wk": pd.Timedelta(days=7),
        "1w": pd.Timedelta(days=7),
        "1mo": pd.Timedelta(days=31),
        "3mo": pd.Timedelta(days=93),
    }
    if normalized in fixed:
        return fixed[normalized]
    matched = re.fullmatch(r"(\d+)(m|h)", normalized)
    if not matched:
        return None
    value = int(matched.group(1))
    return pd.Timedelta(minutes=value) if matched.group(2) == "m" else pd.Timedelta(hours=value)


def completed_bars_only(frame: pd.DataFrame) -> pd.DataFrame:
    """排除交易所仍在形成中的最後一根 K 線。"""
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
    now = pd.Timestamp(datetime.now(timezone.utc))
    if "close_time_ms" in result.columns:
        close_times = pd.to_datetime(
            result["close_time_ms"],
            unit="ms",
            utc=True,
            errors="coerce",
        )
        known_close = close_times.notna()
        completed = ~known_close | (close_times <= now)
        return result.loc[completed].copy()

    if "interval" not in result.columns:
        return result
    duration = interval_duration(str(result.iloc[-1]["interval"]))
    if duration is None:
        return result
    return result.loc[result["timestamp"] + duration <= now].copy()
