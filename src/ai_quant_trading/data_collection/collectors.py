"""高階資料收集流程。"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

from ai_quant_trading.data_collection.binance import BinanceSpotClient
from ai_quant_trading.data_collection.binance_futures import (
    BinanceFuturesPublicClient,
    attach_derivatives_context,
)
from ai_quant_trading.data_collection.csv_storage import (
    save_derivatives_context_csv,
    save_market_data_csv,
    save_ohlcv_csv,
)
from ai_quant_trading.data_collection.validators import (
    validate_market_data_dataframe,
    validate_ohlcv_dataframe,
)
from ai_quant_trading.data_collection.yahoo_finance import YahooFinanceClient
from ai_quant_trading.market_clock import BTC_MULTITIMEFRAME_INTERVALS, interval_duration


TimeframeProgressCallback = Callable[[str, int, int], None]


@dataclass(slots=True)
class DataCollectionResult:
    """一次資料收集任務的輸出檔案。"""

    ohlcv_files: list[Path] = field(default_factory=list)
    market_data_files: list[Path] = field(default_factory=list)
    derivatives_context_files: list[Path] = field(default_factory=list)


def collect_binance_data(
    symbols: list[str],
    interval: str,
    raw_dir: str | Path,
    start: str | None = None,
    end: str | None = None,
    limit: int = 1000,
    include_market_data: bool = True,
    include_derivatives_context: bool = False,
    client: BinanceSpotClient | None = None,
    futures_client: BinanceFuturesPublicClient | None = None,
) -> DataCollectionResult:
    """下載 Binance OHLCV 與 24 小時 market data，驗證後寫入 CSV。"""
    client = client or BinanceSpotClient()
    result = DataCollectionResult()
    futures_client = futures_client or (
        BinanceFuturesPublicClient() if include_derivatives_context else None
    )

    for symbol in symbols:
        ohlcv = client.fetch_ohlcv(symbol=symbol, interval=interval, start=start, end=end, limit=limit)
        if futures_client is not None:
            context = futures_client.fetch_market_context(
                symbol,
                interval,
                start=start,
                end=end,
            )
            result.derivatives_context_files.append(
                save_derivatives_context_csv(context, raw_dir, symbol, interval)
            )
            ohlcv = attach_derivatives_context(ohlcv, context)
        validate_ohlcv_dataframe(ohlcv)
        result.ohlcv_files.append(
            save_ohlcv_csv(
                frame=ohlcv,
                raw_dir=raw_dir,
                exchange="binance",
                symbol=symbol,
                interval=interval,
            )
        )

        if include_market_data:
            market_data = client.fetch_24h_market_data(symbol=symbol)
            validate_market_data_dataframe(market_data)
            result.market_data_files.append(
                save_market_data_csv(
                    frame=market_data,
                    raw_dir=raw_dir,
                    exchange="binance",
                    symbol=symbol,
                )
            )

    return result


def collect_binance_futures_data(
    symbols: list[str],
    interval: str,
    raw_dir: str | Path,
    start: str | None = None,
    end: str | None = None,
    limit: int = 1500,
    include_market_data: bool = True,
    include_derivatives_context: bool = True,
    client: BinanceFuturesPublicClient | None = None,
) -> DataCollectionResult:
    """下載 Binance USDⓈ-M 永續合約 K 線與當時可取得的合約市場脈絡。"""
    futures = client or BinanceFuturesPublicClient()
    result = DataCollectionResult()
    for symbol in symbols:
        ohlcv = futures.fetch_ohlcv(
            symbol=symbol,
            interval=interval,
            start=start,
            end=end,
            limit=limit,
        )
        if include_derivatives_context:
            context = futures.fetch_market_context(
                symbol,
                interval,
                start=start,
                end=end,
                include_advanced=True,
            )
            result.derivatives_context_files.append(
                save_derivatives_context_csv(context, raw_dir, symbol, interval)
            )
            ohlcv = attach_derivatives_context(ohlcv, context)
        validate_ohlcv_dataframe(ohlcv)
        result.ohlcv_files.append(
            save_ohlcv_csv(
                frame=ohlcv,
                raw_dir=raw_dir,
                exchange="binance_futures",
                symbol=symbol,
                interval=interval,
            )
        )
        if include_market_data:
            market_data = futures.fetch_24h_market_data(symbol)
            validate_market_data_dataframe(market_data)
            result.market_data_files.append(
                save_market_data_csv(
                    frame=market_data,
                    raw_dir=raw_dir,
                    exchange="binance_futures",
                    symbol=symbol,
                )
            )
    return result


def collect_binance_futures_timeframe_bundle(
    symbols: list[str],
    raw_dir: str | Path,
    *,
    intervals: Sequence[str] = ("5m", "15m", "1h", "4h", "1d"),
    start: str | None = None,
    end: str | None = None,
    limit: int = 1500,
    include_market_data: bool = True,
    include_derivatives_context: bool = False,
    client: BinanceFuturesPublicClient | None = None,
    progress_callback: TimeframeProgressCallback | None = None,
    decision_bars: int = 5_000,
    indicator_warmup_bars: int = 250,
) -> DataCollectionResult:
    """下載 BTC 15m 模型所需的 5m、15m、1h、4h、1d 永續合約資料。"""
    normalized = tuple(dict.fromkeys(str(value).strip().lower() for value in intervals))
    if "15m" not in normalized or len(normalized) < 2:
        raise ValueError("BTC 15m 永續資料包必須包含 15m 與至少一個輔助週期")
    unsupported = [value for value in normalized if value not in BTC_MULTITIMEFRAME_INTERVALS]
    if unsupported:
        raise ValueError(f"BTC 15m 永續資料包不支援：{unsupported}")
    if decision_bars < 500:
        raise ValueError("15m 決策歷史至少需要 500 根 K 線")
    if indicator_warmup_bars < 200:
        raise ValueError("多週期指標暖機至少需要 200 根 K 線")
    futures = client or BinanceFuturesPublicClient()
    combined = DataCollectionResult()
    reference = pd.Timestamp(end) if end else pd.Timestamp.now(tz="UTC")
    reference = (
        reference.tz_localize("UTC")
        if reference.tzinfo is None
        else reference.tz_convert("UTC")
    )
    requested_start = pd.Timestamp(start) if start else None
    if requested_start is not None:
        requested_start = (
            requested_start.tz_localize("UTC")
            if requested_start.tzinfo is None
            else requested_start.tz_convert("UTC")
        )
    decision_duration = interval_duration("15m")
    if decision_duration is None:
        raise RuntimeError("系統缺少 15m K 線週期定義")
    history_span = decision_duration * decision_bars
    for index, interval in enumerate(normalized, start=1):
        duration = interval_duration(interval)
        if duration is None:
            raise ValueError(f"無法解析 K 線週期：{interval}")
        aligned_bars = math.ceil(history_span / duration)
        required_bars = aligned_bars + indicator_warmup_bars
        warmup_start = reference - duration * required_bars
        interval_start = min(requested_start, warmup_start) if requested_start is not None else warmup_start
        current = collect_binance_futures_data(
            symbols,
            interval,
            raw_dir,
            start=interval_start.isoformat(),
            end=end,
            limit=limit,
            include_market_data=(include_market_data and index == 1),
            include_derivatives_context=(
                include_derivatives_context and interval == "15m"
            ),
            client=futures,
        )
        combined.ohlcv_files.extend(current.ohlcv_files)
        combined.market_data_files.extend(current.market_data_files)
        combined.derivatives_context_files.extend(current.derivatives_context_files)
        if progress_callback:
            progress_callback(interval, index, len(normalized))
    return combined


def collect_binance_timeframe_bundle(
    symbols: list[str],
    raw_dir: str | Path,
    *,
    intervals: Sequence[str] = BTC_MULTITIMEFRAME_INTERVALS,
    start: str | None = None,
    end: str | None = None,
    limit: int = 1000,
    include_market_data: bool = True,
    include_derivatives_context: bool = False,
    client: BinanceSpotClient | None = None,
    futures_client: BinanceFuturesPublicClient | None = None,
    progress_callback: TimeframeProgressCallback | None = None,
    minimum_bars_per_interval: int = 250,
) -> DataCollectionResult:
    """依序下載同一批標的的多週期 K 線，並替慢週期保留指標暖機資料。"""
    normalized = tuple(dict.fromkeys(str(value).strip().lower() for value in intervals))
    if len(normalized) < 2:
        raise ValueError("多週期資料包至少需要兩種 K 線週期")
    unsupported = [value for value in normalized if value not in BTC_MULTITIMEFRAME_INTERVALS]
    if unsupported:
        raise ValueError(f"BTC 多週期資料包不支援：{unsupported}")
    if minimum_bars_per_interval < 200:
        raise ValueError("多週期指標暖機至少需要 200 根 K 線")

    requested_start = pd.Timestamp(start) if start else None
    if requested_start is not None:
        requested_start = (
            requested_start.tz_localize("UTC")
            if requested_start.tzinfo is None
            else requested_start.tz_convert("UTC")
        )
    reference = pd.Timestamp(end) if end else pd.Timestamp.now(tz="UTC")
    reference = (
        reference.tz_localize("UTC")
        if reference.tzinfo is None
        else reference.tz_convert("UTC")
    )

    spot = client or BinanceSpotClient()
    futures = futures_client or (
        BinanceFuturesPublicClient() if include_derivatives_context else None
    )
    combined = DataCollectionResult()
    total = len(normalized)
    for index, interval in enumerate(normalized, start=1):
        duration = interval_duration(interval)
        if duration is None:
            raise ValueError(f"無法解析 K 線週期：{interval}")
        warmup_start = reference - duration * minimum_bars_per_interval
        interval_start = (
            min(requested_start, warmup_start).isoformat()
            if requested_start is not None
            else None
        )
        current = collect_binance_data(
            symbols=symbols,
            interval=interval,
            raw_dir=raw_dir,
            start=interval_start,
            end=end,
            limit=limit,
            include_market_data=include_market_data and index == 1,
            include_derivatives_context=include_derivatives_context,
            client=spot,
            futures_client=futures,
        )
        combined.ohlcv_files.extend(current.ohlcv_files)
        combined.market_data_files.extend(current.market_data_files)
        combined.derivatives_context_files.extend(current.derivatives_context_files)
        if progress_callback:
            progress_callback(interval, index, total)
    return combined


def collect_yahoo_finance_data(
    symbols: list[str],
    interval: str,
    raw_dir: str | Path,
    start: str | None = None,
    end: str | None = None,
    limit: int | None = 1000,
    client: YahooFinanceClient | None = None,
) -> DataCollectionResult:
    """下載 Yahoo Finance 美股 OHLCV，驗證後寫入 CSV。"""
    client = client or YahooFinanceClient()
    result = DataCollectionResult()

    for symbol in symbols:
        ohlcv = client.fetch_ohlcv(symbol=symbol, interval=interval, start=start, end=end, limit=limit)
        validate_ohlcv_dataframe(ohlcv)
        result.ohlcv_files.append(
            save_ohlcv_csv(
                frame=ohlcv,
                raw_dir=raw_dir,
                exchange="yahoo_finance",
                symbol=symbol,
                interval=interval,
            )
        )

    return result
