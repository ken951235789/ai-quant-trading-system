"""Binance Spot 公開市場資料 API 客戶端。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests

from ai_quant_trading.data_collection.http_client import create_secure_session, require_session
from ai_quant_trading.data_collection.schemas import (
    MARKET_DATA_COLUMNS,
    OHLCV_COLUMNS,
    ORDER_BOOK_CONTEXT_COLUMNS,
)
from ai_quant_trading.data_collection.time_utils import (
    parse_datetime_to_utc_ms,
    utc_ms_to_iso,
    utc_now_iso,
)


class BinanceApiError(RuntimeError):
    """Binance API 請求失敗時使用的例外。"""


INTERVAL_TO_MS = {
    "1s": 1_000,
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "6h": 6 * 60 * 60_000,
    "8h": 8 * 60 * 60_000,
    "12h": 12 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
    "3d": 3 * 24 * 60 * 60_000,
    "1w": 7 * 24 * 60 * 60_000,
}


def to_binance_symbol(symbol: str) -> str:
    """把研究系統使用的 BTC/USDT 格式轉成 Binance 使用的 BTCUSDT。"""
    return symbol.replace("/", "").replace("-", "").upper()


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


@dataclass(slots=True)
class BinanceSpotClient:
    """封裝 Binance Spot 公開市場資料 REST API。"""

    base_url: str = "https://api.binance.com"
    timeout: int = 30
    session: requests.Session | None = None

    def __post_init__(self) -> None:
        if self.session is None:
            self.session = create_secure_session()

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        """發送 GET 請求，並把 API 或網路錯誤轉成專案內部例外。"""
        clean_params = {key: value for key, value in params.items() if value is not None}
        url = f"{self.base_url}{path}"
        session = require_session(self.session)

        try:
            response = session.get(url, params=clean_params, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise BinanceApiError(f"Binance API 請求失敗：{exc}") from exc

        payload = response.json()
        if isinstance(payload, dict) and payload.get("code", 0) not in (0, None):
            raise BinanceApiError(f"Binance API 回傳錯誤：{payload}")
        return payload

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "1d",
        start: str | None = None,
        end: str | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """下載 OHLCV K 線資料，並轉成標準 DataFrame。"""
        if interval not in INTERVAL_TO_MS:
            supported = ", ".join(sorted(INTERVAL_TO_MS))
            raise ValueError(f"目前不支援 interval={interval}，可用值：{supported}")
        if limit < 1 or limit > 1000:
            raise ValueError("Binance K 線 limit 必須介於 1 到 1000")

        binance_symbol = to_binance_symbol(symbol)
        start_ms = parse_datetime_to_utc_ms(start)
        end_ms = parse_datetime_to_utc_ms(end)
        interval_ms = INTERVAL_TO_MS[interval]

        rows: list[list[Any]] = []
        next_start_ms = start_ms

        while True:
            payload = self._get(
                "/api/v3/klines",
                {
                    "symbol": binance_symbol,
                    "interval": interval,
                    "startTime": next_start_ms,
                    "endTime": end_ms,
                    "limit": limit,
                },
            )
            if not payload:
                break

            rows.extend(payload)

            # 沒有指定 start 時，只抓最近一批資料。
            if start_ms is None:
                break

            last_open_time = int(payload[-1][0])
            next_start_ms = last_open_time + interval_ms
            if end_ms is not None and next_start_ms > end_ms:
                break
            if len(payload) < limit:
                break

        return self._klines_to_dataframe(rows, symbol, interval)

    def fetch_24h_market_data(self, symbol: str) -> pd.DataFrame:
        """下載單一交易對的 24 小時市場統計快照。"""
        binance_symbol = to_binance_symbol(symbol)
        payload = self._get("/api/v3/ticker/24hr", {"symbol": binance_symbol})
        return self._ticker_24h_to_dataframe(payload, symbol)

    def fetch_spot_symbols(self, quote_asset: str = "USDT") -> list[str]:
        """取得目前可交易的現貨交易對，供介面建立完整搜尋清單。"""
        payload = self._get(
            "/api/v3/exchangeInfo",
            {
                "permissions": "SPOT",
                "symbolStatus": "TRADING",
                "showPermissionSets": "false",
            },
        )
        quote = quote_asset.upper()
        symbols: list[str] = []
        for item in payload.get("symbols", []):
            if item.get("status") != "TRADING" or item.get("quoteAsset") != quote:
                continue
            if item.get("isSpotTradingAllowed") is False:
                continue
            base = str(item.get("baseAsset", "")).upper()
            if base:
                symbols.append(f"{base}/{quote}")
        return sorted(set(symbols))

    def fetch_order_book_snapshot(
        self,
        symbol: str,
        *,
        limit: int = 100,
    ) -> pd.DataFrame:
        """取得現貨深度快照，計算交易成本與買賣盤失衡特徵。"""
        supported_limits = {5, 10, 20, 50, 100, 500, 1000, 5000}
        if limit not in supported_limits:
            raise ValueError(
                "Binance 深度 limit 必須是 5、10、20、50、100、500、1000 或 5000"
            )
        payload = self._get(
            "/api/v3/depth",
            {"symbol": to_binance_symbol(symbol), "limit": limit},
        )
        bids = [(float(price), float(quantity)) for price, quantity in payload.get("bids", [])]
        asks = [(float(price), float(quantity)) for price, quantity in payload.get("asks", [])]
        if not bids or not asks:
            raise BinanceApiError("Binance 深度快照沒有有效買賣盤")

        best_bid, best_bid_quantity = bids[0]
        best_ask, best_ask_quantity = asks[0]
        midpoint = (best_bid + best_ask) / 2
        top_quantity = best_bid_quantity + best_ask_quantity
        microprice = (
            (best_ask * best_bid_quantity + best_bid * best_ask_quantity) / top_quantity
            if top_quantity > 0
            else midpoint
        )
        record: dict[str, Any] = {
            "timestamp": utc_now_iso(),
            "symbol": symbol,
            "exchange": "binance",
            "last_update_id": _safe_int(payload.get("lastUpdateId")),
            "best_bid": best_bid,
            "best_ask": best_ask,
            "midpoint": midpoint,
            "spread_bps": (
                (best_ask - best_bid) / midpoint * 10_000 if midpoint > 0 else 0.0
            ),
            "microprice": microprice,
            "microprice_deviation_bps": (
                (microprice / midpoint - 1) * 10_000 if midpoint > 0 else 0.0
            ),
            "collected_at": utc_now_iso(),
        }
        for depth in (5, 10, 20):
            available_depth = min(depth, len(bids), len(asks))
            bid_notional = sum(price * quantity for price, quantity in bids[:available_depth])
            ask_notional = sum(price * quantity for price, quantity in asks[:available_depth])
            total_notional = bid_notional + ask_notional
            record[f"bid_depth_{depth}"] = bid_notional
            record[f"ask_depth_{depth}"] = ask_notional
            record[f"imbalance_{depth}"] = (
                (bid_notional - ask_notional) / total_notional
                if total_notional > 0
                else 0.0
            )
        return pd.DataFrame.from_records([record], columns=ORDER_BOOK_CONTEXT_COLUMNS)

    @staticmethod
    def _klines_to_dataframe(rows: list[list[Any]], symbol: str, interval: str) -> pd.DataFrame:
        """把 Binance tuple 格式 K 線轉成欄位清楚的表格。"""
        collected_at = utc_now_iso()
        records = []
        for row in rows:
            open_time_ms = int(row[0])
            close_time_ms = int(row[6])
            records.append(
                {
                    "timestamp": utc_ms_to_iso(open_time_ms),
                    "symbol": symbol,
                    "exchange": "binance",
                    "interval": interval,
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                    "quote_asset_volume": float(row[7]),
                    "number_of_trades": int(row[8]),
                    "taker_buy_base_volume": float(row[9]),
                    "taker_buy_quote_volume": float(row[10]),
                    "open_time_ms": open_time_ms,
                    "close_time_ms": close_time_ms,
                    "collected_at": collected_at,
                }
            )

        if not records:
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        frame = pd.DataFrame.from_records(records, columns=OHLCV_COLUMNS)
        return (
            frame.sort_values("open_time_ms").drop_duplicates("open_time_ms").reset_index(drop=True)
        )

    @staticmethod
    def _ticker_24h_to_dataframe(payload: dict[str, Any], symbol: str) -> pd.DataFrame:
        """把 Binance 24h ticker 回應轉成標準 market data 表格。"""
        close_time_ms = _safe_int(payload.get("closeTime"))
        timestamp = utc_ms_to_iso(close_time_ms) if close_time_ms is not None else utc_now_iso()
        record = {
            "timestamp": timestamp,
            "symbol": symbol,
            "exchange": "binance",
            "price_change": _safe_float(payload.get("priceChange")),
            "price_change_percent": _safe_float(payload.get("priceChangePercent")),
            "weighted_avg_price": _safe_float(payload.get("weightedAvgPrice")),
            "prev_close_price": _safe_float(payload.get("prevClosePrice")),
            "last_price": _safe_float(payload.get("lastPrice")),
            "last_qty": _safe_float(payload.get("lastQty")),
            "bid_price": _safe_float(payload.get("bidPrice")),
            "bid_qty": _safe_float(payload.get("bidQty")),
            "ask_price": _safe_float(payload.get("askPrice")),
            "ask_qty": _safe_float(payload.get("askQty")),
            "open_price": _safe_float(payload.get("openPrice")),
            "high_price": _safe_float(payload.get("highPrice")),
            "low_price": _safe_float(payload.get("lowPrice")),
            "volume": _safe_float(payload.get("volume")),
            "quote_volume": _safe_float(payload.get("quoteVolume")),
            "open_time_ms": _safe_int(payload.get("openTime")),
            "close_time_ms": close_time_ms,
            "first_trade_id": _safe_int(payload.get("firstId")),
            "last_trade_id": _safe_int(payload.get("lastId")),
            "trade_count": _safe_int(payload.get("count")),
            "collected_at": utc_now_iso(),
        }
        return pd.DataFrame.from_records([record], columns=MARKET_DATA_COLUMNS)
