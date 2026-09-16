"""市場資料品質檢查。"""

from __future__ import annotations

import pandas as pd

from ai_quant_trading.data_collection.schemas import (
    REQUIRED_MARKET_DATA_COLUMNS,
    REQUIRED_OHLCV_COLUMNS,
)


def _ensure_columns(frame: pd.DataFrame, required_columns: list[str], data_name: str) -> None:
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"{data_name} 缺少必要欄位：{missing_columns}")


def validate_ohlcv_dataframe(frame: pd.DataFrame) -> None:
    """檢查 OHLCV 是否符合後續特徵工程與回測的基本需求。"""
    _ensure_columns(frame, REQUIRED_OHLCV_COLUMNS, "OHLCV")
    if frame.empty:
        raise ValueError("OHLCV 資料是空的")

    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("OHLCV timestamp 含有無法解析的值")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("OHLCV timestamp 必須由舊到新排序")
    if timestamps.duplicated().any():
        raise ValueError("OHLCV timestamp 不可重複")

    numeric_columns = ["open", "high", "low", "close", "volume"]
    numeric_frame = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric_frame.isna().any().any():
        raise ValueError("OHLCV 價格或成交量欄位含有非數字")
    if (numeric_frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("OHLCV 價格必須大於 0")
    if (numeric_frame["volume"] < 0).any():
        raise ValueError("OHLCV volume 不可小於 0")

    highest_required = numeric_frame[["open", "close", "low"]].max(axis=1)
    lowest_required = numeric_frame[["open", "close", "high"]].min(axis=1)
    if (numeric_frame["high"] < highest_required).any():
        raise ValueError("OHLCV high 不可低於 open、close 或 low")
    if (numeric_frame["low"] > lowest_required).any():
        raise ValueError("OHLCV low 不可高於 open、close 或 high")


def validate_market_data_dataframe(frame: pd.DataFrame) -> None:
    """檢查 24 小時 market data 快照是否有基本欄位與合理數值。"""
    _ensure_columns(frame, REQUIRED_MARKET_DATA_COLUMNS, "Market Data")
    if frame.empty:
        raise ValueError("Market Data 資料是空的")

    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("Market Data timestamp 含有無法解析的值")

    numeric_columns = ["price_change", "price_change_percent", "volume", "quote_volume"]
    numeric_frame = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric_frame.isna().any().any():
        raise ValueError("Market Data 重要數值欄位含有非數字")
    if (numeric_frame[["volume", "quote_volume"]] < 0).any().any():
        raise ValueError("Market Data 成交量不可小於 0")

    trade_count = pd.to_numeric(frame["trade_count"], errors="coerce")
    if trade_count.isna().any() or (trade_count < 0).any():
        raise ValueError("Market Data trade_count 必須是非負整數")

