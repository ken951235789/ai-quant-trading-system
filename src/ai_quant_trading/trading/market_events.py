"""歷史重播與即時行情共用的市場事件合約。"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from threading import Lock
from types import MappingProxyType
from typing import Callable, Literal, Mapping


MarketEventKind = Literal["kline_closed", "book_ticker"]
MarketEventSource = Literal["historical", "rest", "websocket"]
MarketEventConsumer = Callable[["MarketEvent"], None]


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _canonical_hash(values: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(values),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_number(payload: Mapping[str, object], name: str) -> float:
    try:
        value = float(payload[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"市場事件 {name} 必須是數字") from exc
    if not math.isfinite(value):
        raise ValueError(f"市場事件 {name} 必須是有限數值")
    return value


@dataclass(frozen=True, slots=True)
class MarketEvent:
    """來源無關的市場事件；同一根 K 線跨 REST/WebSocket 使用相同 ID。"""

    event_id: str
    kind: MarketEventKind
    source: MarketEventSource
    exchange: str
    symbol: str
    event_time: datetime
    observed_at: datetime
    payload: Mapping[str, object]
    interval: str | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("不支援的 MarketEvent schema_version")
        if len(self.event_id) != 64:
            raise ValueError("MarketEvent event_id 必須是 SHA-256")
        if not self.exchange.strip() or not self.symbol.strip():
            raise ValueError("MarketEvent exchange 與 symbol 不可為空")
        object.__setattr__(self, "event_time", _utc(self.event_time))
        object.__setattr__(self, "observed_at", _utc(self.observed_at))
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))
        if self.kind == "kline_closed":
            self._validate_kline()
        elif self.kind == "book_ticker":
            self._validate_book_ticker()

    def _validate_kline(self) -> None:
        if not self.interval:
            raise ValueError("K 線事件必須包含 interval")
        values = {
            name: _finite_number(self.payload, name)
            for name in ("open", "high", "low", "close", "volume")
        }
        if min(values[name] for name in ("open", "high", "low", "close")) <= 0:
            raise ValueError("K 線價格必須大於 0")
        if values["volume"] < 0:
            raise ValueError("K 線成交量不可小於 0")
        if values["high"] < max(values["open"], values["low"], values["close"]):
            raise ValueError("K 線 high 不可低於 open、low 或 close")
        if values["low"] > min(values["open"], values["high"], values["close"]):
            raise ValueError("K 線 low 不可高於 open、high 或 close")
        for name in ("open_time_ms", "close_time_ms"):
            try:
                int(self.payload[name])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"K 線事件缺少有效的 {name}") from exc

    def _validate_book_ticker(self) -> None:
        bid = _finite_number(self.payload, "bid_price")
        ask = _finite_number(self.payload, "ask_price")
        if bid <= 0 or ask <= 0 or ask < bid:
            raise ValueError("BookTicker 買賣價不合法")

    @classmethod
    def closed_kline(
        cls,
        *,
        source: MarketEventSource,
        exchange: str,
        symbol: str,
        interval: str,
        open_time_ms: int,
        close_time_ms: int,
        observed_at: datetime,
        payload: Mapping[str, object],
    ) -> "MarketEvent":
        """建立已收盤 K 線；事件 ID 不包含傳輸來源。"""
        identity = {
            "schema_version": 1,
            "kind": "kline_closed",
            "exchange": exchange.lower(),
            "symbol": symbol.upper(),
            "interval": interval.lower(),
            "open_time_ms": int(open_time_ms),
        }
        values = dict(payload)
        values.update(
            {
                "open_time_ms": int(open_time_ms),
                "close_time_ms": int(close_time_ms),
            }
        )
        return cls(
            event_id=_canonical_hash(identity),
            kind="kline_closed",
            source=source,
            exchange=exchange.lower(),
            symbol=symbol.upper(),
            interval=interval.lower(),
            event_time=datetime.fromtimestamp(close_time_ms / 1000, tz=timezone.utc),
            observed_at=observed_at,
            payload=values,
        )

    @classmethod
    def book_ticker(
        cls,
        *,
        source: MarketEventSource,
        exchange: str,
        symbol: str,
        update_id: int,
        observed_at: datetime,
        payload: Mapping[str, object],
    ) -> "MarketEvent":
        identity = {
            "schema_version": 1,
            "kind": "book_ticker",
            "exchange": exchange.lower(),
            "symbol": symbol.upper(),
            "update_id": int(update_id),
        }
        values = dict(payload)
        values["update_id"] = int(update_id)
        return cls(
            event_id=_canonical_hash(identity),
            kind="book_ticker",
            source=source,
            exchange=exchange.lower(),
            symbol=symbol.upper(),
            event_time=observed_at,
            observed_at=observed_at,
            payload=values,
        )

    @property
    def stream_key(self) -> tuple[str, str, str, str]:
        return (self.exchange, self.symbol, self.kind, self.interval or "")

    def to_ohlcv_row(self) -> dict[str, object]:
        """轉回既有特徵工程可使用的 OHLCV 列。"""
        if self.kind != "kline_closed":
            raise ValueError("只有 kline_closed 可以轉成 OHLCV")
        row = dict(self.payload)
        open_time_ms = int(row["open_time_ms"])
        row.update(
            {
                "timestamp": datetime.fromtimestamp(
                    open_time_ms / 1000,
                    tz=timezone.utc,
                ).isoformat(),
                "symbol": self.symbol,
                "exchange": self.exchange,
                "interval": self.interval,
                "collected_at": self.observed_at.isoformat(),
            }
        )
        return row

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "kind": self.kind,
            "source": self.source,
            "exchange": self.exchange,
            "symbol": self.symbol,
            "interval": self.interval,
            "event_time": self.event_time.isoformat(),
            "observed_at": self.observed_at.isoformat(),
            "payload": dict(self.payload),
        }


class DuplicateMarketEvent(RuntimeError):
    """保留給需要把重複事件視為錯誤的呼叫端。"""


class OutOfOrderMarketEvent(RuntimeError):
    """事件時間比同一資料流上一筆更早。"""


class MarketEventGap(RuntimeError):
    """同週期 K 線之間出現缺口。"""


class ConsumerDeliveryError(RuntimeError):
    """Consumer 尚未成功處理事件，checkpoint 不得前進。"""

    def __init__(self, event: MarketEvent, consumer: object, cause: Exception) -> None:
        super().__init__(f"Consumer {consumer!r} 處理事件 {event.event_id[:12]} 失敗：{cause}")
        self.event = event
        self.consumer = consumer
        self.cause = cause


@dataclass(slots=True)
class MarketEventMetrics:
    received: int = 0
    delivered: int = 0
    duplicates: int = 0
    out_of_order: int = 0
    gaps: int = 0
    consumer_failures: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "received": self.received,
            "delivered": self.delivered,
            "duplicates": self.duplicates,
            "out_of_order": self.out_of_order,
            "gaps": self.gaps,
            "consumer_failures": self.consumer_failures,
        }


@dataclass(frozen=True, slots=True)
class DeliveryReceipt:
    event_id: str
    status: Literal["delivered", "duplicate"]
    consumers: int


@dataclass(slots=True)
class MarketEventBus:
    """歷史與即時共用的排序、去重及投遞入口。"""

    consumers: list[MarketEventConsumer] = field(default_factory=list)
    strict_ordering: bool = True
    strict_continuity: bool = False
    dedup_capacity: int = 100_000
    metrics: MarketEventMetrics = field(default_factory=MarketEventMetrics, init=False)
    _seen: OrderedDict[str, None] = field(default_factory=OrderedDict, init=False)
    _last_event_time: dict[tuple[str, str, str, str], datetime] = field(
        default_factory=dict,
        init=False,
    )
    _last_open_time_ms: dict[tuple[str, str, str, str], int] = field(
        default_factory=dict,
        init=False,
    )
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.dedup_capacity < 1:
            raise ValueError("dedup_capacity 必須大於 0")

    def publish(self, event: MarketEvent) -> DeliveryReceipt:
        """成功投遞全部 consumer 後才把事件標記為已完成。"""
        with self._lock:
            return self._publish_locked(event)

    def _publish_locked(self, event: MarketEvent) -> DeliveryReceipt:
        self.metrics.received += 1
        if event.event_id in self._seen:
            self.metrics.duplicates += 1
            self._seen.move_to_end(event.event_id)
            return DeliveryReceipt(event.event_id, "duplicate", 0)

        key = event.stream_key
        previous_time = self._last_event_time.get(key)
        if previous_time is not None and event.event_time < previous_time:
            self.metrics.out_of_order += 1
            if self.strict_ordering:
                raise OutOfOrderMarketEvent(
                    f"{key} 事件時間倒退：{event.event_time.isoformat()} < "
                    f"{previous_time.isoformat()}"
                )

        if event.kind == "kline_closed":
            current_open = int(event.payload["open_time_ms"])
            previous_open = self._last_open_time_ms.get(key)
            if previous_open is not None:
                expected_step = int(event.payload["close_time_ms"]) - current_open + 1
                if current_open - previous_open > expected_step:
                    self.metrics.gaps += 1
                    if self.strict_continuity:
                        raise MarketEventGap(
                            f"{key} K 線缺口：{previous_open} -> {current_open}"
                        )

        for consumer in self.consumers:
            try:
                consumer(event)
            except Exception as exc:
                self.metrics.consumer_failures += 1
                raise ConsumerDeliveryError(event, consumer, exc) from exc

        self._seen[event.event_id] = None
        self._seen.move_to_end(event.event_id)
        while len(self._seen) > self.dedup_capacity:
            self._seen.popitem(last=False)
        if previous_time is None or event.event_time > previous_time:
            self._last_event_time[key] = event.event_time
        if event.kind == "kline_closed":
            self._last_open_time_ms[key] = int(event.payload["open_time_ms"])
        self.metrics.delivered += 1
        return DeliveryReceipt(event.event_id, "delivered", len(self.consumers))
