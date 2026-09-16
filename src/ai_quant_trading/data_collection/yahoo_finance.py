"""Yahoo Finance 美股歷史資料客戶端。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests

from ai_quant_trading.data_collection.http_client import create_secure_session, require_session
from ai_quant_trading.data_collection.schemas import EQUITY_OHLCV_COLUMNS
from ai_quant_trading.data_collection.time_utils import utc_now_iso


class YahooFinanceDataError(RuntimeError):
    """Yahoo Finance 資料下載或解析失敗時使用的例外。"""


VALID_YAHOO_INTERVALS = {
    "1m",
    "2m",
    "5m",
    "15m",
    "30m",
    "60m",
    "90m",
    "1h",
    "1d",
    "5d",
    "1wk",
    "1mo",
    "3mo",
}

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 AIQuantTradingSystem/0.1"}


def normalize_us_symbol(symbol: str) -> str:
    """標準化美股代號。"""
    return symbol.strip().upper()


def _default_period_for_interval(interval: str) -> str:
    """沒有指定日期範圍時，保留足夠資料供 MA200 與模型特徵暖機。"""
    if interval == "1m":
        return "5d"
    if interval in {"2m", "5m", "15m", "30m"}:
        return "1mo"
    if interval in {"60m", "90m", "1h"}:
        return "6mo"
    if interval in {"1d", "5d"}:
        return "5y"
    return "10y"


def _date_to_unix(value: str) -> int:
    """把日期或 ISO 時間轉成 UTC Unix 秒數。"""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return int(timestamp.timestamp())


def _float_or_zero(value: Any) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return float(value)


def _event_values(result: dict[str, Any]) -> tuple[dict[int, float], dict[int, float]]:
    """把股利與拆股事件整理成 timestamp 對數值的查詢表。"""
    events = result.get("events") or {}
    dividends = {
        int(item["date"]): float(item.get("amount", 0.0))
        for item in (events.get("dividends") or {}).values()
        if item.get("date") is not None
    }
    splits: dict[int, float] = {}
    for item in (events.get("splits") or {}).values():
        if item.get("date") is None:
            continue
        numerator = item.get("numerator")
        denominator = item.get("denominator")
        if numerator is not None and denominator:
            splits[int(item["date"])] = float(numerator) / float(denominator)
            continue
        ratio = str(item.get("splitRatio", "0:1")).split(":", maxsplit=1)
        splits[int(item["date"])] = float(ratio[0]) / float(ratio[1])
    return dividends, splits


@dataclass(slots=True)
class YahooFinanceClient:
    """使用 Yahoo Chart API 下載歷史 OHLCV，不依賴瀏覽器或 API Key。"""

    session: requests.Session | None = None
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.session is None:
            self.session = create_secure_session()

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "1d",
        start: str | None = None,
        end: str | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """下載單一美股代號的歷史 OHLCV。"""
        if interval not in VALID_YAHOO_INTERVALS:
            supported = ", ".join(sorted(VALID_YAHOO_INTERVALS))
            raise ValueError(f"目前不支援 interval={interval}，可用值：{supported}")
        if limit is not None and limit < 1:
            raise ValueError("limit 必須大於 0")

        symbol = normalize_us_symbol(symbol)
        params: dict[str, Any] = {
            "interval": "60m" if interval == "1h" else interval,
            "events": "div,splits",
            "includeAdjustedClose": "true",
        }
        if start or end:
            params["period1"] = _date_to_unix(start) if start else 0
            default_end = datetime.now(timezone.utc) + timedelta(days=1)
            params["period2"] = _date_to_unix(end) if end else int(default_end.timestamp())
        else:
            params["range"] = _default_period_for_interval(interval)

        session = require_session(self.session)
        try:
            response = session.get(
                YAHOO_CHART_URL.format(symbol=symbol),
                params=params,
                headers=REQUEST_HEADERS,
                timeout=self.timeout_seconds,
            )
            if response.status_code == 429:
                raise YahooFinanceDataError(
                    "Yahoo Finance 暫時回傳 Too Many Requests。請稍後再試，"
                    "或改用需要 API Key 的正式資料源，例如 Polygon、Alpaca 或 Alpha Vantage。"
                )
            response.raise_for_status()
            payload = response.json()
        except YahooFinanceDataError:
            raise
        except (requests.RequestException, ValueError) as exc:
            raise YahooFinanceDataError(f"Yahoo Finance 下載失敗：{exc}") from exc

        history = self._chart_to_history(payload, symbol)
        frame = self._history_to_dataframe(history, symbol=symbol, interval=interval)
        if limit is not None:
            frame = frame.tail(limit).reset_index(drop=True)
        return frame

    @staticmethod
    def _chart_to_history(payload: dict[str, Any], symbol: str) -> pd.DataFrame:
        """把 Yahoo Chart JSON 轉成以 UTC 時間為索引的表格。"""
        chart = payload.get("chart") or {}
        if chart.get("error"):
            description = chart["error"].get("description", "未知錯誤")
            raise YahooFinanceDataError(f"Yahoo Finance 回傳錯誤：{description}")
        results = chart.get("result") or []
        if not results:
            raise YahooFinanceDataError(f"{symbol} 沒有下載到任何資料")

        result = results[0]
        timestamps = result.get("timestamp") or []
        indicators = result.get("indicators") or {}
        quotes = indicators.get("quote") or []
        if not timestamps or not quotes:
            raise YahooFinanceDataError(f"{symbol} 沒有下載到任何資料")

        quote = quotes[0]
        adjusted_groups = indicators.get("adjclose") or []
        adjusted = adjusted_groups[0].get("adjclose", []) if adjusted_groups else []
        dividends, splits = _event_values(result)
        rows: list[dict[str, Any]] = []
        for index, unix_time in enumerate(timestamps):
            values = {
                name: (quote.get(name) or [None] * len(timestamps))[index]
                for name in ["open", "high", "low", "close", "volume"]
            }
            if any(values[name] is None for name in ["open", "high", "low", "close"]):
                continue
            close = float(values["close"])
            adj_close = adjusted[index] if index < len(adjusted) else close
            rows.append(
                {
                    "timestamp": pd.to_datetime(unix_time, unit="s", utc=True),
                    "Open": values["open"],
                    "High": values["high"],
                    "Low": values["low"],
                    "Close": close,
                    "Adj Close": close if adj_close is None else adj_close,
                    "Volume": values["volume"] or 0,
                    "Dividends": dividends.get(int(unix_time), 0.0),
                    "Stock Splits": splits.get(int(unix_time), 0.0),
                }
            )
        if not rows:
            raise YahooFinanceDataError(f"{symbol} 沒有可用的 OHLCV 資料")
        return pd.DataFrame(rows).set_index("timestamp")

    @staticmethod
    def _history_to_dataframe(history: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
        """把歷史表格轉成系統標準 OHLCV 欄位。"""
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        missing_columns = [column for column in required_columns if column not in history.columns]
        if missing_columns:
            raise YahooFinanceDataError(f"Yahoo Finance 回傳資料缺少欄位：{missing_columns}")

        normalized = history.copy()
        normalized.index = pd.to_datetime(normalized.index, utc=True)
        normalized = normalized.sort_index()

        records = []
        collected_at = utc_now_iso()
        for timestamp, row in normalized.iterrows():
            close = float(row["Close"])
            adj_close = row.get("Adj Close", close)
            records.append(
                {
                    "timestamp": timestamp.isoformat(timespec="milliseconds").replace(
                        "+00:00", "Z"
                    ),
                    "symbol": symbol,
                    "exchange": "yahoo_finance",
                    "interval": interval,
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": close,
                    "adj_close": float(adj_close) if pd.notna(adj_close) else close,
                    "volume": int(row["Volume"]),
                    "dividends": _float_or_zero(row.get("Dividends", 0.0)),
                    "stock_splits": _float_or_zero(row.get("Stock Splits", 0.0)),
                    "collected_at": collected_at,
                }
            )

        return pd.DataFrame.from_records(records, columns=EQUITY_OHLCV_COLUMNS)
