"""Binance BTC 即時行情串流與已收盤 K 線解析。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from threading import Event, Lock, Thread
from time import monotonic
from typing import Callable, Literal

import pandas as pd

from ai_quant_trading.data_collection.binance import INTERVAL_TO_MS, to_binance_symbol
from ai_quant_trading.data_collection.http_client import create_secure_ssl_context
from ai_quant_trading.data_collection.schemas import OHLCV_COLUMNS
from ai_quant_trading.data_collection.time_utils import utc_ms_to_iso, utc_now_iso
from ai_quant_trading.market_clock import BTC_MULTITIMEFRAME_INTERVALS
from ai_quant_trading.trading.market_events import MarketEvent


StreamEventKind = Literal["kline", "book_ticker"]
StreamChannel = Literal["kline", "book_ticker"]


@dataclass(frozen=True, slots=True)
class BinanceKlineUpdate:
    """單一 WebSocket K 線事件；closed=False 時不得送進模型。"""

    symbol: str
    interval: str
    open_time_ms: int
    close_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_asset_volume: float
    number_of_trades: int
    taker_buy_base_volume: float
    taker_buy_quote_volume: float
    closed: bool
    event_time_ms: int
    exchange: str = "binance_futures"

    def to_frame(self) -> pd.DataFrame:
        """轉成與 REST 收集器相同的 OHLCV 欄位。"""
        row = {
            "timestamp": utc_ms_to_iso(self.open_time_ms),
            "symbol": self.symbol,
            "exchange": self.exchange,
            "interval": self.interval,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "quote_asset_volume": self.quote_asset_volume,
            "number_of_trades": self.number_of_trades,
            "taker_buy_base_volume": self.taker_buy_base_volume,
            "taker_buy_quote_volume": self.taker_buy_quote_volume,
            "open_time_ms": self.open_time_ms,
            "close_time_ms": self.close_time_ms,
            "collected_at": utc_now_iso(),
        }
        return pd.DataFrame.from_records([row], columns=OHLCV_COLUMNS)

    def to_market_event(self) -> MarketEvent:
        """轉成歷史重播與即時推論共用的事件格式。"""
        return MarketEvent.closed_kline(
            source="websocket",
            exchange=self.exchange,
            symbol=self.symbol,
            interval=self.interval,
            open_time_ms=self.open_time_ms,
            close_time_ms=self.close_time_ms,
            observed_at=datetime.fromtimestamp(
                self.event_time_ms / 1000,
                tz=timezone.utc,
            ),
            payload={
                "open": self.open,
                "high": self.high,
                "low": self.low,
                "close": self.close,
                "volume": self.volume,
                "quote_asset_volume": self.quote_asset_volume,
                "number_of_trades": self.number_of_trades,
                "taker_buy_base_volume": self.taker_buy_base_volume,
                "taker_buy_quote_volume": self.taker_buy_quote_volume,
            },
        )


@dataclass(frozen=True, slots=True)
class BinanceBookTickerUpdate:
    """最佳買賣價快照，供延遲與交易成本防護使用。"""

    symbol: str
    update_id: int
    bid_price: float
    bid_quantity: float
    ask_price: float
    ask_quantity: float
    received_at: str

    @property
    def spread_bps(self) -> float:
        midpoint = (self.bid_price + self.ask_price) / 2
        if midpoint <= 0:
            return float("inf")
        return (self.ask_price - self.bid_price) / midpoint * 10_000

    def to_market_event(self) -> MarketEvent:
        observed_at = datetime.fromisoformat(self.received_at.replace("Z", "+00:00"))
        return MarketEvent.book_ticker(
            source="websocket",
            exchange="binance_futures",
            symbol=self.symbol,
            update_id=self.update_id,
            observed_at=observed_at,
            payload={
                "bid_price": self.bid_price,
                "bid_quantity": self.bid_quantity,
                "ask_price": self.ask_price,
                "ask_quantity": self.ask_quantity,
            },
        )


@dataclass(frozen=True, slots=True)
class ParsedBinanceStreamEvent:
    """解析後的聯集事件。"""

    kind: StreamEventKind
    kline: BinanceKlineUpdate | None = None
    book_ticker: BinanceBookTickerUpdate | None = None


@dataclass(frozen=True, slots=True)
class BinanceStreamConfig:
    """BTC 多週期行情串流設定。"""

    symbol: str = "BTC/USDT"
    intervals: tuple[str, ...] = BTC_MULTITIMEFRAME_INTERVALS
    include_book_ticker: bool = True
    base_url: str = "wss://fstream.binance.com"
    reconnect_initial_seconds: float = 1.0
    reconnect_max_seconds: float = 30.0
    status_publish_seconds: float = 1.0
    book_ticker_sample_seconds: float = 0.1

    def __post_init__(self) -> None:
        normalized = tuple(dict.fromkeys(value.strip().lower() for value in self.intervals))
        unsupported = [value for value in normalized if value not in INTERVAL_TO_MS]
        if unsupported:
            raise ValueError(f"WebSocket 不支援 K 線週期：{unsupported}")
        if not normalized:
            raise ValueError("WebSocket 至少需要一個 K 線週期")
        if self.reconnect_initial_seconds <= 0 or self.reconnect_max_seconds <= 0:
            raise ValueError("WebSocket 重連秒數必須大於 0")
        if self.book_ticker_sample_seconds < 0:
            raise ValueError("BookTicker 取樣秒數不可小於 0")
        object.__setattr__(self, "intervals", normalized)

    @property
    def kline_stream_url(self) -> str:
        """Binance 期貨市場 K 線在 2026 API 分流後使用 market 路徑。"""
        token = to_binance_symbol(self.symbol).lower()
        streams = [f"{token}@kline_{interval}" for interval in self.intervals]
        return f"{self.base_url.rstrip('/')}/market/stream?streams={'/'.join(streams)}"

    @property
    def book_ticker_stream_url(self) -> str:
        """最佳買賣價使用 public 路徑，不能再與 K 線共用 market 連線。"""
        token = to_binance_symbol(self.symbol).lower()
        return f"{self.base_url.rstrip('/')}/public/ws/{token}@bookTicker"

    @property
    def stream_url(self) -> str:
        """保留舊呼叫介面，代表主要的 K 線串流網址。"""
        return self.kline_stream_url

    @property
    def stream_endpoints(self) -> tuple[tuple[StreamChannel, str], ...]:
        endpoints: list[tuple[StreamChannel, str]] = [
            ("kline", self.kline_stream_url)
        ]
        if self.include_book_ticker:
            endpoints.append(("book_ticker", self.book_ticker_stream_url))
        return tuple(endpoints)


@dataclass(slots=True)
class BinanceStreamStatus:
    """可持久化到 UI 的串流健康狀態。"""

    state: str = "stopped"
    connected: bool = False
    kline_connected: bool = False
    book_ticker_connected: bool = False
    connected_at: str | None = None
    last_message_at: str | None = None
    last_event_latency_ms: float | None = None
    reconnect_count: int = 0
    messages_received: int = 0
    book_ticker_messages_processed: int = 0
    closed_klines_received: int = 0
    last_closed_by_interval: dict[str, str] = field(default_factory=dict)
    latest_klines_by_interval: dict[str, dict[str, object]] = field(
        default_factory=dict
    )
    bid_price: float | None = None
    ask_price: float | None = None
    spread_bps: float | None = None
    last_error: str | None = None


def _display_symbol(exchange_symbol: str) -> str:
    """目前專案聚焦 BTC/USDT，其他代碼仍保留原始 Binance 格式。"""
    return "BTC/USDT" if exchange_symbol.upper() == "BTCUSDT" else exchange_symbol.upper()


def parse_binance_stream_message(message: str | bytes) -> ParsedBinanceStreamEvent | None:
    """解析 combined/raw stream；控制回應或未知事件回傳 None。"""
    if isinstance(message, bytes):
        message = message.decode("utf-8")
    payload = json.loads(message)
    data = payload.get("data", payload)
    if not isinstance(data, dict):
        return None

    if data.get("e") == "kline" and isinstance(data.get("k"), dict):
        kline = data["k"]
        update = BinanceKlineUpdate(
            symbol=_display_symbol(str(kline["s"])),
            interval=str(kline["i"]).lower(),
            open_time_ms=int(kline["t"]),
            close_time_ms=int(kline["T"]),
            open=float(kline["o"]),
            high=float(kline["h"]),
            low=float(kline["l"]),
            close=float(kline["c"]),
            volume=float(kline["v"]),
            quote_asset_volume=float(kline["q"]),
            number_of_trades=int(kline["n"]),
            taker_buy_base_volume=float(kline["V"]),
            taker_buy_quote_volume=float(kline["Q"]),
            closed=bool(kline["x"]),
            event_time_ms=int(data.get("E", kline["T"])),
        )
        return ParsedBinanceStreamEvent("kline", kline=update)

    stream_name = str(payload.get("stream", ""))
    is_book_ticker = data.get("e") == "bookTicker" or stream_name.endswith(
        "@bookTicker"
    ) or (
        {"u", "s", "b", "B", "a", "A"}.issubset(data)
        and data.get("e") is None
    )
    if is_book_ticker:
        update = BinanceBookTickerUpdate(
            symbol=_display_symbol(str(data["s"])),
            update_id=int(data["u"]),
            bid_price=float(data["b"]),
            bid_quantity=float(data["B"]),
            ask_price=float(data["a"]),
            ask_quantity=float(data["A"]),
            received_at=datetime.now(timezone.utc).isoformat(),
        )
        return ParsedBinanceStreamEvent("book_ticker", book_ticker=update)
    return None


class BinanceRealtimeStream:
    """背景維持 Binance WebSocket，並以 callback 傳出已收盤 K 線。"""

    def __init__(
        self,
        config: BinanceStreamConfig,
        *,
        on_closed_kline: Callable[[BinanceKlineUpdate], None],
        on_book_ticker: Callable[[BinanceBookTickerUpdate], None] | None = None,
        on_status: Callable[[dict[str, object]], None] | None = None,
        on_market_event: Callable[[MarketEvent], None] | None = None,
    ) -> None:
        self.config = config
        self.on_closed_kline = on_closed_kline
        self.on_book_ticker = on_book_ticker
        self.on_status = on_status
        self.on_market_event = on_market_event
        self._status = BinanceStreamStatus()
        self._status_lock = Lock()
        self._stop = Event()
        self._threads: dict[StreamChannel, Thread] = {}
        self._apps: dict[StreamChannel, object] = {}
        self._last_status_publish = 0.0
        self._last_book_ticker_process = 0.0
        self._pending_book_ticker_messages = 0

    def snapshot(self) -> dict[str, object]:
        """取得不會與 WebSocket thread 互相修改的狀態副本。"""
        with self._status_lock:
            return asdict(self._status)

    def start(self) -> None:
        """分別啟動 K 線與報價執行緒；重複呼叫不會建立第二組連線。"""
        if any(thread.is_alive() for thread in self._threads.values()):
            return
        self._stop.clear()
        self._threads = {}
        for channel, url in self.config.stream_endpoints:
            thread = Thread(
                target=self._run,
                args=(channel, url),
                name=f"binance-btc-{channel}-stream",
                daemon=True,
            )
            self._threads[channel] = thread
            thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        """要求安全停止並關閉全部 socket。"""
        self._stop.set()
        for app in tuple(self._apps.values()):
            try:
                app.close()
            except Exception:
                pass
        for thread in tuple(self._threads.values()):
            thread.join(timeout=timeout)
        self._update_status(
            state="stopped",
            connected=False,
            kline_connected=False,
            book_ticker_connected=False,
        )
        self._publish_status(force=True)

    def _update_status(self, **values: object) -> None:
        with self._status_lock:
            for key, value in values.items():
                setattr(self._status, key, value)

    def _publish_status(self, *, force: bool = False) -> None:
        if self.on_status is None:
            return
        now = monotonic()
        if not force and now - self._last_status_publish < self.config.status_publish_seconds:
            return
        self._last_status_publish = now
        self.on_status(self.snapshot())

    def _set_channel_connection(
        self,
        channel: StreamChannel,
        connected: bool,
        *,
        fallback_state: str,
        error: str | None = None,
    ) -> None:
        """合併兩條連線的健康狀態，只有必要資料都到齊才算已連線。"""
        now = datetime.now(timezone.utc).isoformat()
        with self._status_lock:
            if channel == "kline":
                self._status.kline_connected = connected
            else:
                self._status.book_ticker_connected = connected
            ready = self._status.kline_connected and (
                self._status.book_ticker_connected
                if self.config.include_book_ticker
                else True
            )
            was_connected = self._status.connected
            self._status.connected = ready
            if self._stop.is_set():
                self._status.state = "stopped"
            elif ready:
                self._status.state = "connected"
                if not was_connected:
                    self._status.connected_at = now
                self._status.last_error = None
            else:
                self._status.state = fallback_state
                if error is not None:
                    self._status.last_error = error

    def _on_open(
        self,
        _app: object,
        channel: StreamChannel = "kline",
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._set_channel_connection(
            channel,
            True,
            fallback_state="connecting",
        )
        self._update_status(last_message_at=now)
        self._publish_status(force=True)

    def _on_message(
        self,
        _app: object,
        message: str | bytes,
        channel: StreamChannel = "kline",
    ) -> None:
        try:
            received_count = 1
            if channel == "book_ticker":
                self._pending_book_ticker_messages += 1
                now_monotonic = monotonic()
                if (
                    self.config.book_ticker_sample_seconds > 0
                    and now_monotonic - self._last_book_ticker_process
                    < self.config.book_ticker_sample_seconds
                ):
                    return
                self._last_book_ticker_process = now_monotonic
                received_count = self._pending_book_ticker_messages
                self._pending_book_ticker_messages = 0
            event = parse_binance_stream_message(message)
            if event is None:
                return
            now = datetime.now(timezone.utc)
            with self._status_lock:
                self._status.messages_received += received_count
                self._status.last_message_at = now.isoformat()
            if event.kline is not None:
                latency = max(0.0, now.timestamp() * 1000 - event.kline.event_time_ms)
                with self._status_lock:
                    self._status.last_event_latency_ms = latency
                    self._status.latest_klines_by_interval[event.kline.interval] = {
                        "timestamp": utc_ms_to_iso(event.kline.open_time_ms),
                        "symbol": event.kline.symbol,
                        "exchange": event.kline.exchange,
                        "interval": event.kline.interval,
                        "open": event.kline.open,
                        "high": event.kline.high,
                        "low": event.kline.low,
                        "close": event.kline.close,
                        "volume": event.kline.volume,
                        "quote_asset_volume": event.kline.quote_asset_volume,
                        "number_of_trades": event.kline.number_of_trades,
                        "closed": event.kline.closed,
                        "event_time_ms": event.kline.event_time_ms,
                    }
                if event.kline.closed:
                    with self._status_lock:
                        self._status.closed_klines_received += 1
                        self._status.last_closed_by_interval[event.kline.interval] = (
                            utc_ms_to_iso(event.kline.close_time_ms)
                        )
                    if self.on_market_event is not None:
                        self.on_market_event(event.kline.to_market_event())
                    self.on_closed_kline(event.kline)
                    self._publish_status(force=True)
                    return
            elif event.book_ticker is not None:
                ticker = event.book_ticker
                with self._status_lock:
                    self._status.book_ticker_messages_processed += 1
                    self._status.bid_price = ticker.bid_price
                    self._status.ask_price = ticker.ask_price
                    self._status.spread_bps = ticker.spread_bps
                if self.on_book_ticker is not None:
                    self.on_book_ticker(ticker)
                if self.on_market_event is not None:
                    self.on_market_event(ticker.to_market_event())
            self._publish_status()
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._update_status(last_error=f"行情訊息解析失敗：{exc}")
            self._publish_status(force=True)
        except Exception as exc:
            self._update_status(last_error=f"行情 callback 失敗：{exc}")
            self._publish_status(force=True)

    def _on_error(
        self,
        _app: object,
        error: object,
        channel: StreamChannel = "kline",
    ) -> None:
        self._set_channel_connection(
            channel,
            False,
            fallback_state="reconnecting",
            error=f"{channel}：{error}",
        )
        self._publish_status(force=True)

    def _on_close(
        self,
        _app: object,
        _status_code: int | None,
        _message: str | None,
        channel: StreamChannel = "kline",
    ) -> None:
        state = "stopped" if self._stop.is_set() else "reconnecting"
        self._set_channel_connection(
            channel,
            False,
            fallback_state=state,
        )
        self._publish_status(force=True)

    def _run(
        self,
        channel: StreamChannel = "kline",
        url: str | None = None,
    ) -> None:
        try:
            import websocket
        except ImportError:
            self._update_status(
                state="failed",
                connected=False,
                last_error="缺少 websocket-client，請重新安裝專案依賴",
            )
            self._publish_status(force=True)
            return

        delay = self.config.reconnect_initial_seconds
        ssl_options = {"context": create_secure_ssl_context()}
        endpoint = url or (
            self.config.kline_stream_url
            if channel == "kline"
            else self.config.book_ticker_stream_url
        )
        while not self._stop.is_set():
            self._set_channel_connection(
                channel,
                False,
                fallback_state="connecting",
            )
            self._publish_status(force=True)
            app = websocket.WebSocketApp(
                endpoint,
                on_open=lambda active_app: self._on_open(active_app, channel),
                on_message=lambda active_app, message: self._on_message(
                    active_app,
                    message,
                    channel,
                ),
                on_error=lambda active_app, error: self._on_error(
                    active_app,
                    error,
                    channel,
                ),
                on_close=lambda active_app, status_code, message: self._on_close(
                    active_app,
                    status_code,
                    message,
                    channel,
                ),
            )
            self._apps[channel] = app
            app.run_forever(
                ping_interval=20,
                ping_timeout=10,
                sslopt=ssl_options,
            )
            self._apps.pop(channel, None)
            if self._stop.is_set():
                break
            with self._status_lock:
                self._status.reconnect_count += 1
            self._set_channel_connection(
                channel,
                False,
                fallback_state="reconnecting",
            )
            self._publish_status(force=True)
            if self._stop.wait(delay):
                break
            delay = min(delay * 2, self.config.reconnect_max_seconds)
