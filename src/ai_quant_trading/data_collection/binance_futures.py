"""Binance USDⓈ-M 公開衍生品市場脈絡資料。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import requests

from ai_quant_trading.data_collection.binance import (
    INTERVAL_TO_MS,
    BinanceApiError,
    to_binance_symbol,
)
from ai_quant_trading.data_collection.http_client import create_secure_session, require_session
from ai_quant_trading.data_collection.schemas import (
    DERIVATIVES_CONTEXT_COLUMNS,
    MARKET_DATA_COLUMNS,
    OHLCV_COLUMNS,
    ORDER_BOOK_CONTEXT_COLUMNS,
)
from ai_quant_trading.data_collection.time_utils import (
    parse_datetime_to_utc_ms,
    utc_ms_to_iso,
    utc_now_iso,
)


OPEN_INTEREST_PERIODS = {
    "1s": "5m",
    "1m": "5m",
    "3m": "5m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "8h": "6h",
    "12h": "12h",
    "1d": "1d",
    "3d": "1d",
    "1w": "1d",
}


@dataclass(slots=True)
class BinanceFuturesPublicClient:
    """只讀取公開市場資料，不持有 API key，也不提供下單方法。"""

    base_url: str = "https://fapi.binance.com"
    timeout: int = 30
    session: requests.Session | None = None

    def __post_init__(self) -> None:
        if self.session is None:
            self.session = create_secure_session()

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        clean = {key: value for key, value in params.items() if value is not None}
        session = require_session(self.session)
        try:
            response = session.get(
                f"{self.base_url}{path}", params=clean, timeout=self.timeout
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise BinanceApiError(f"Binance Futures 公開 API 請求失敗：{exc}") from exc
        payload = response.json()
        if isinstance(payload, dict) and payload.get("code") not in (None, 0):
            raise BinanceApiError(f"Binance Futures API 回傳錯誤：{payload}")
        return payload

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "15m",
        start: str | None = None,
        end: str | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """下載 USDⓈ-M 永續合約 K 線；指定開始時間時自動向後分頁。"""
        if interval not in INTERVAL_TO_MS:
            raise ValueError(f"Binance Futures 不支援 K 線週期：{interval}")
        if not 1 <= limit <= 1500:
            raise ValueError("Binance Futures K 線 limit 必須介於 1 與 1500")
        start_ms = parse_datetime_to_utc_ms(start)
        end_ms = parse_datetime_to_utc_ms(end)
        next_start = start_ms
        rows: list[list[Any]] = []
        while True:
            payload = self._get(
                "/fapi/v1/klines",
                {
                    "symbol": to_binance_symbol(symbol),
                    "interval": interval,
                    "startTime": next_start,
                    "endTime": end_ms,
                    "limit": limit,
                },
            )
            if not payload:
                break
            rows.extend(payload)
            if start_ms is None or len(payload) < limit:
                break
            next_start = int(payload[-1][0]) + INTERVAL_TO_MS[interval]
            if end_ms is not None and next_start > end_ms:
                break

        collected_at = utc_now_iso()
        now_ms = int(datetime.now(UTC).timestamp() * 1000)
        completed_cutoff_ms = min(end_ms, now_ms) if end_ms is not None else now_ms
        records = [
            {
                "timestamp": utc_ms_to_iso(int(row[0])),
                "symbol": symbol,
                "exchange": "binance_futures",
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
                "open_time_ms": int(row[0]),
                "close_time_ms": int(row[6]),
                "collected_at": collected_at,
            }
            for row in rows
            if int(row[6]) <= completed_cutoff_ms
        ]
        if not records:
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        return (
            pd.DataFrame.from_records(records, columns=OHLCV_COLUMNS)
            .sort_values("open_time_ms")
            .drop_duplicates("open_time_ms", keep="last")
            .reset_index(drop=True)
        )

    def fetch_funding_history(
        self,
        symbol: str,
        *,
        start: str | None = None,
        end: str | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """取得歷史資金費率；指定開始日期時自動分頁。"""
        if not 1 <= limit <= 1000:
            raise ValueError("Funding rate limit 必須介於 1 與 1000")
        start_ms = parse_datetime_to_utc_ms(start)
        end_ms = parse_datetime_to_utc_ms(end)
        next_start = start_ms
        rows: list[dict[str, Any]] = []
        while True:
            payload = self._get(
                "/fapi/v1/fundingRate",
                {
                    "symbol": to_binance_symbol(symbol),
                    "startTime": next_start,
                    "endTime": end_ms,
                    "limit": limit,
                },
            )
            if not payload:
                break
            rows.extend(payload)
            if start_ms is None or len(payload) < limit:
                break
            last_time = int(payload[-1]["fundingTime"])
            next_start = last_time + 1
            if end_ms is not None and next_start > end_ms:
                break
        return pd.DataFrame(
            [
                {
                    "timestamp": utc_ms_to_iso(int(row["fundingTime"])),
                    "funding_rate": float(row["fundingRate"]),
                    "mark_price": float(row["markPrice"]) if row.get("markPrice") else None,
                }
                for row in rows
            ]
        )

    def fetch_24h_market_data(self, symbol: str) -> pd.DataFrame:
        """取得 USD-M 永續合約 24 小時市場統計快照。"""
        payload = self._get(
            "/fapi/v1/ticker/24hr",
            {"symbol": to_binance_symbol(symbol)},
        )

        def safe_float(key: str) -> float | None:
            value = payload.get(key)
            return float(value) if value not in (None, "") else None

        def safe_int(key: str) -> int | None:
            value = payload.get(key)
            return int(value) if value not in (None, "") else None

        close_time_ms = safe_int("closeTime")
        timestamp = (
            utc_ms_to_iso(close_time_ms)
            if close_time_ms is not None
            else utc_now_iso()
        )
        record = {
            "timestamp": timestamp,
            "symbol": symbol,
            "exchange": "binance_futures",
            "price_change": safe_float("priceChange"),
            "price_change_percent": safe_float("priceChangePercent"),
            "weighted_avg_price": safe_float("weightedAvgPrice"),
            "prev_close_price": safe_float("prevClosePrice"),
            "last_price": safe_float("lastPrice"),
            "last_qty": safe_float("lastQty"),
            "bid_price": safe_float("bidPrice"),
            "bid_qty": safe_float("bidQty"),
            "ask_price": safe_float("askPrice"),
            "ask_qty": safe_float("askQty"),
            "open_price": safe_float("openPrice"),
            "high_price": safe_float("highPrice"),
            "low_price": safe_float("lowPrice"),
            "volume": safe_float("volume"),
            "quote_volume": safe_float("quoteVolume"),
            "open_time_ms": safe_int("openTime"),
            "close_time_ms": close_time_ms,
            "first_trade_id": safe_int("firstId"),
            "last_trade_id": safe_int("lastId"),
            "trade_count": safe_int("count"),
            "collected_at": utc_now_iso(),
        }
        return pd.DataFrame.from_records([record], columns=MARKET_DATA_COLUMNS)

    def fetch_open_interest_history(
        self,
        symbol: str,
        interval: str,
        *,
        start: str | None = None,
        end: str | None = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """取得交易所目前可提供的 OI 歷史；官方端點最多回傳 500 筆。"""
        if interval not in OPEN_INTEREST_PERIODS:
            raise ValueError(f"Open interest 不支援 K 線週期：{interval}")
        if not 1 <= limit <= 500:
            raise ValueError("Open interest limit 必須介於 1 與 500")
        earliest_ms = int((datetime.now(UTC) - timedelta(days=30)).timestamp() * 1000)
        requested_start = parse_datetime_to_utc_ms(start)
        requested_end = parse_datetime_to_utc_ms(end)
        if requested_end is not None and requested_end < earliest_ms:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "open_interest",
                    "open_interest_value",
                    "open_interest_change",
                ]
            )
        effective_start = max(requested_start or earliest_ms, earliest_ms)
        payload = self._get(
            "/futures/data/openInterestHist",
            {
                "symbol": to_binance_symbol(symbol),
                "period": OPEN_INTEREST_PERIODS[interval],
                "startTime": effective_start,
                "endTime": requested_end,
                "limit": limit,
            },
        )
        result = pd.DataFrame(
            [
                {
                    "timestamp": utc_ms_to_iso(int(row["timestamp"])),
                    "open_interest": float(row["sumOpenInterest"]),
                    "open_interest_value": float(row["sumOpenInterestValue"]),
                }
                for row in payload
            ]
        )
        if not result.empty:
            result["open_interest_change"] = result["open_interest"].pct_change(
                fill_method=None
            )
        return result

    def fetch_book_ticker(self, symbol: str) -> dict[str, float]:
        """取得目前最佳買賣價並換算 spread basis points。"""
        payload = self._get(
            "/fapi/v1/ticker/bookTicker",
            {"symbol": to_binance_symbol(symbol)},
        )
        bid = float(payload["bidPrice"])
        ask = float(payload["askPrice"])
        midpoint = (bid + ask) / 2
        return {
            "bid_price": bid,
            "ask_price": ask,
            "spread_bps": (ask - bid) / midpoint * 10_000 if midpoint > 0 else 0.0,
        }

    def fetch_order_book_snapshot(
        self,
        symbol: str,
        *,
        limit: int = 100,
    ) -> pd.DataFrame:
        """取得 USDⓈ-M 永續合約深度，計算 spread、microprice 與失衡。"""
        supported_limits = {5, 10, 20, 50, 100, 500, 1000}
        if limit not in supported_limits:
            raise ValueError(
                "Binance Futures 深度 limit 必須是 5、10、20、50、100、500 或 1000"
            )
        payload = self._get(
            "/fapi/v1/depth",
            {"symbol": to_binance_symbol(symbol), "limit": limit},
        )
        bids = [
            (float(price), float(quantity))
            for price, quantity in payload.get("bids", [])
        ]
        asks = [
            (float(price), float(quantity))
            for price, quantity in payload.get("asks", [])
        ]
        if not bids or not asks:
            raise BinanceApiError("Binance Futures 深度快照沒有有效買賣盤")
        best_bid, best_bid_quantity = bids[0]
        best_ask, best_ask_quantity = asks[0]
        midpoint = (best_bid + best_ask) / 2
        top_quantity = best_bid_quantity + best_ask_quantity
        microprice = (
            (best_ask * best_bid_quantity + best_bid * best_ask_quantity)
            / top_quantity
            if top_quantity > 0
            else midpoint
        )
        record: dict[str, Any] = {
            "timestamp": utc_now_iso(),
            "symbol": symbol,
            "exchange": "binance_futures",
            "last_update_id": int(payload.get("lastUpdateId", 0) or 0),
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
            available = min(depth, len(bids), len(asks))
            bid_notional = sum(price * quantity for price, quantity in bids[:available])
            ask_notional = sum(price * quantity for price, quantity in asks[:available])
            total_notional = bid_notional + ask_notional
            record[f"bid_depth_{depth}"] = bid_notional
            record[f"ask_depth_{depth}"] = ask_notional
            record[f"imbalance_{depth}"] = (
                (bid_notional - ask_notional) / total_notional
                if total_notional > 0
                else 0.0
            )
        return pd.DataFrame.from_records([record], columns=ORDER_BOOK_CONTEXT_COLUMNS)

    def _fetch_30_day_series(
        self,
        path: str,
        symbol: str,
        interval: str,
        *,
        start: str | None,
        end: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        """讀取 Binance 僅保留最近 30 天的公開統計序列。"""
        if interval not in OPEN_INTEREST_PERIODS:
            raise ValueError(f"衍生品統計不支援 K 線週期：{interval}")
        if not 1 <= limit <= 500:
            raise ValueError("衍生品統計 limit 必須介於 1 與 500")
        earliest_ms = int((datetime.now(UTC) - timedelta(days=30)).timestamp() * 1000)
        requested_start = parse_datetime_to_utc_ms(start)
        requested_end = parse_datetime_to_utc_ms(end)
        if requested_end is not None and requested_end < earliest_ms:
            return []
        market_params: dict[str, Any]
        if path.endswith("/basis"):
            market_params = {
                "pair": to_binance_symbol(symbol),
                "contractType": "PERPETUAL",
            }
        else:
            market_params = {"symbol": to_binance_symbol(symbol)}
        time_params = (
            {}
            if path.endswith("/basis")
            else {
                "startTime": max(requested_start or earliest_ms, earliest_ms),
                "endTime": requested_end,
            }
        )
        payload = self._get(
            path,
            {
                **market_params,
                "period": OPEN_INTEREST_PERIODS[interval],
                **time_params,
                "limit": limit,
            },
        )
        return list(payload)

    def fetch_global_long_short_ratio(
        self,
        symbol: str,
        interval: str,
        *,
        start: str | None = None,
        end: str | None = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """取得全市場帳戶多空比，官方端點僅保留最近 30 天。"""
        rows = self._fetch_30_day_series(
            "/futures/data/globalLongShortAccountRatio",
            symbol,
            interval,
            start=start,
            end=end,
            limit=limit,
        )
        return pd.DataFrame(
            [
                {
                    "timestamp": utc_ms_to_iso(int(row["timestamp"])),
                    "global_long_short_ratio": float(row["longShortRatio"]),
                    "global_long_account": float(row["longAccount"]),
                    "global_short_account": float(row["shortAccount"]),
                }
                for row in rows
            ]
        )

    def fetch_taker_buy_sell_ratio(
        self,
        symbol: str,
        interval: str,
        *,
        start: str | None = None,
        end: str | None = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """取得主動買入／賣出量比，衡量短期交易方向壓力。"""
        rows = self._fetch_30_day_series(
            "/futures/data/takerlongshortRatio",
            symbol,
            interval,
            start=start,
            end=end,
            limit=limit,
        )
        return pd.DataFrame(
            [
                {
                    "timestamp": utc_ms_to_iso(int(row["timestamp"])),
                    "taker_buy_sell_ratio": float(row["buySellRatio"]),
                    "taker_buy_volume": float(row["buyVol"]),
                    "taker_sell_volume": float(row["sellVol"]),
                }
                for row in rows
            ]
        )

    def fetch_basis_history(
        self,
        symbol: str,
        interval: str,
        *,
        start: str | None = None,
        end: str | None = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """取得永續合約相對指數價格的基差。"""
        rows = self._fetch_30_day_series(
            "/futures/data/basis",
            symbol,
            interval,
            start=start,
            end=end,
            limit=limit,
        )
        return pd.DataFrame(
            [
                {
                    "timestamp": utc_ms_to_iso(int(row["timestamp"])),
                    "basis_rate": float(row["basisRate"]),
                    "basis_price": float(row["basis"]),
                    "index_price": float(row["indexPrice"]),
                }
                for row in rows
            ]
        )

    def fetch_market_context(
        self,
        symbol: str,
        interval: str,
        *,
        start: str | None = None,
        end: str | None = None,
        include_advanced: bool = False,
    ) -> pd.DataFrame:
        """合併 funding、OI、spread 與選用的進階期貨統計。"""
        funding = self.fetch_funding_history(symbol, start=start, end=end)
        open_interest = self.fetch_open_interest_history(
            symbol, interval, start=start, end=end
        )
        if funding.empty and open_interest.empty:
            context = pd.DataFrame(columns=["timestamp"])
        elif funding.empty:
            context = open_interest.copy()
        elif open_interest.empty:
            context = funding.copy()
        else:
            context = pd.merge(funding, open_interest, on="timestamp", how="outer")
        if include_advanced:
            advanced_fetchers = [
                lambda: self.fetch_global_long_short_ratio(
                    symbol, interval, start=start, end=end
                ),
                lambda: self.fetch_taker_buy_sell_ratio(
                    symbol, interval, start=start, end=end
                ),
                lambda: self.fetch_basis_history(
                    symbol, interval, start=start, end=end
                ),
            ]
            advanced_frames: list[pd.DataFrame] = []
            for fetch in advanced_fetchers:
                try:
                    advanced_frames.append(fetch())
                except BinanceApiError:
                    # 個別公開統計偶爾會維護或限流，保留其他已成功資料。
                    advanced_frames.append(pd.DataFrame())
            for advanced in advanced_frames:
                if advanced.empty:
                    continue
                if context.empty:
                    context = advanced.copy()
                else:
                    context = pd.merge(
                        context,
                        advanced,
                        on="timestamp",
                        how="outer",
                    )
        ticker = self.fetch_book_ticker(symbol)
        snapshot = {
            "timestamp": utc_now_iso(),
            **ticker,
        }
        context = pd.concat([context, pd.DataFrame([snapshot])], ignore_index=True, sort=False)
        context["timestamp"] = pd.to_datetime(
            context["timestamp"], utc=True, errors="coerce", format="mixed"
        )
        context = context.dropna(subset=["timestamp"]).sort_values("timestamp")
        for column in [
            "funding_rate",
            "mark_price",
            "open_interest",
            "open_interest_value",
            "global_long_short_ratio",
            "global_long_account",
            "global_short_account",
            "taker_buy_sell_ratio",
            "taker_buy_volume",
            "taker_sell_volume",
            "basis_rate",
            "basis_price",
            "index_price",
        ]:
            if column not in context:
                context[column] = pd.NA
            context[column] = pd.to_numeric(context[column], errors="coerce").ffill()
        if "open_interest_change" not in context:
            context["open_interest_change"] = pd.NA
        for column in ["bid_price", "ask_price", "spread_bps"]:
            if column not in context:
                context[column] = pd.NA
        context.insert(1, "symbol", symbol)
        context.insert(2, "exchange", "binance_futures")
        context.insert(3, "interval", interval)
        context["collected_at"] = utc_now_iso()
        context["timestamp"] = context["timestamp"].map(lambda value: value.isoformat())
        return context.reindex(columns=DERIVATIVES_CONTEXT_COLUMNS).reset_index(drop=True)


def attach_derivatives_context(
    ohlcv: pd.DataFrame,
    context: pd.DataFrame,
) -> pd.DataFrame:
    """只把當時已公開的最近一筆衍生品資料向後合併到 K 線。"""
    if ohlcv.empty:
        return ohlcv.copy()
    result = ohlcv.copy()
    result["timestamp"] = pd.to_datetime(
        result["timestamp"], utc=True, errors="coerce", format="mixed"
    )
    if context.empty:
        for column in [
            "funding_rate",
            "mark_price",
            "open_interest",
            "open_interest_value",
            "open_interest_change",
            "spread_bps",
            "global_long_short_ratio",
            "global_long_account",
            "global_short_account",
            "taker_buy_sell_ratio",
            "taker_buy_volume",
            "taker_sell_volume",
            "basis_rate",
            "basis_price",
            "index_price",
        ]:
            result[column] = pd.NA
        result["derivatives_context_available"] = 0
        return result
    right = context.copy()
    right["timestamp"] = pd.to_datetime(
        right["timestamp"], utc=True, errors="coerce", format="mixed"
    )
    columns = [
        "timestamp",
        "funding_rate",
        "mark_price",
        "open_interest",
        "open_interest_value",
        "open_interest_change",
        "spread_bps",
        "global_long_short_ratio",
        "global_long_account",
        "global_short_account",
        "taker_buy_sell_ratio",
        "taker_buy_volume",
        "taker_sell_volume",
        "basis_rate",
        "basis_price",
        "index_price",
    ]
    for column in columns[1:]:
        if column not in right:
            right[column] = pd.NA
    right = right[columns].dropna(subset=["timestamp"]).sort_values("timestamp")
    result = pd.merge_asof(
        result.sort_values("timestamp"),
        right,
        on="timestamp",
        direction="backward",
    )
    context_columns = columns[1:]
    result["derivatives_context_available"] = result[context_columns].notna().any(axis=1).astype("int8")
    return result.reset_index(drop=True)
