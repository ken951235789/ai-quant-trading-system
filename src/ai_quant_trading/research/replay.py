"""歷史 OHLCV 轉 MarketEvent、可控重播與成功後 checkpoint。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import monotonic, sleep
from typing import Iterable, Sequence

import pandas as pd

from ai_quant_trading.data_collection.validators import validate_ohlcv_dataframe
from ai_quant_trading.market_clock import interval_duration
from ai_quant_trading.operations.integrity import sha256_file
from ai_quant_trading.persistence import read_csv_snapshot, write_json_atomic
from ai_quant_trading.trading.market_events import MarketEvent, MarketEventBus


def _timestamp(value: object, *, fallback: datetime | None = None) -> datetime:
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        if fallback is None:
            raise ValueError(f"無法解析 UTC 時間：{value}")
        return fallback
    return pd.Timestamp(parsed).to_pydatetime()


def _json_value(value: object) -> object:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            value = value.item()  # type: ignore[union-attr]
        except (TypeError, ValueError):
            pass
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class HistoricalMarketEventSource:
    """把歷史 CSV 轉成與 WebSocket 相同的已收盤 K 線事件。"""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        source_path: str | Path | None = None,
        default_exchange: str = "binance_futures",
        default_symbol: str = "BTC/USDT",
        default_interval: str = "15m",
    ) -> None:
        if frame.empty:
            raise ValueError("歷史事件來源不可為空")
        self.source_path = Path(source_path).resolve() if source_path else None
        self.default_exchange = default_exchange
        self.default_symbol = default_symbol
        self.default_interval = default_interval
        self.frame = self._prepare(frame)
        self._events = tuple(self._build_events())
        digest = hashlib.sha256()
        for event in self._events:
            digest.update(event.event_id.encode("ascii"))
        self.event_sequence_hash = digest.hexdigest()
        self.content_hash = (
            sha256_file(self.source_path) if self.source_path else self.event_sequence_hash
        )
        self.source_id = f"sha256:{self.event_sequence_hash}"

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        limit: int | None = None,
        **kwargs: object,
    ) -> "HistoricalMarketEventSource":
        source = Path(path).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"找不到歷史重播資料：{source}")
        frame = read_csv_snapshot(source)
        if limit is not None:
            if limit < 2:
                raise ValueError("歷史重播 limit 至少為 2")
            frame = frame.tail(limit).reset_index(drop=True)
        return cls(frame, source_path=source, **kwargs)

    def _prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        defaults = {
            "exchange": self.default_exchange,
            "symbol": self.default_symbol,
            "interval": self.default_interval,
        }
        for column, value in defaults.items():
            if column not in result:
                result[column] = value
            else:
                result[column] = result[column].fillna(value).replace("", value)
        result["timestamp"] = pd.to_datetime(
            result["timestamp"],
            utc=True,
            errors="coerce",
            format="mixed",
        )
        if result["timestamp"].isna().any():
            raise ValueError("歷史資料含有無法解析的 timestamp")
        group_columns = ["exchange", "symbol", "interval"]
        for _key, group in result.groupby(group_columns, sort=False, dropna=False):
            validate_ohlcv_dataframe(group)
        return result

    def _build_events(self) -> Iterable[MarketEvent]:
        events: list[MarketEvent] = []
        for row in self.frame.to_dict(orient="records"):
            opened_at = _timestamp(row["timestamp"])
            exchange = str(row.get("exchange") or self.default_exchange).lower()
            symbol = str(row.get("symbol") or self.default_symbol).upper()
            interval = str(row.get("interval") or self.default_interval).lower()
            open_raw = row.get("open_time_ms")
            try:
                open_time_ms = int(float(open_raw))
            except (TypeError, ValueError):
                open_time_ms = int(opened_at.timestamp() * 1000)
            close_raw = row.get("close_time_ms")
            try:
                close_time_ms = int(float(close_raw))
            except (TypeError, ValueError):
                duration = interval_duration(interval)
                if duration is None:
                    raise ValueError(f"無法推算 {interval} 的收盤時間")
                close_time_ms = open_time_ms + int(duration.total_seconds() * 1000) - 1
            event_time = datetime.fromtimestamp(close_time_ms / 1000, tz=timezone.utc)
            observed_at = _timestamp(row.get("collected_at"), fallback=event_time)
            ignored = {"timestamp", "exchange", "symbol", "interval", "collected_at"}
            payload = {
                str(name): _json_value(value)
                for name, value in row.items()
                if name not in ignored
            }
            events.append(
                MarketEvent.closed_kline(
                    source="historical",
                    exchange=exchange,
                    symbol=symbol,
                    interval=interval,
                    open_time_ms=open_time_ms,
                    close_time_ms=close_time_ms,
                    observed_at=observed_at,
                    payload=payload,
                )
            )
        return sorted(events, key=lambda item: (item.event_time, item.event_id))

    @property
    def events(self) -> tuple[MarketEvent, ...]:
        return self._events

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "source_path": str(self.source_path) if self.source_path else None,
            "source_id": self.source_id,
            "sha256": self.content_hash,
            "event_sequence_sha256": self.event_sequence_hash,
            "rows": len(self._events),
            "first_event_time": self._events[0].event_time.isoformat(),
            "last_event_time": self._events[-1].event_time.isoformat(),
            "event_schema_version": 1,
        }


@dataclass(frozen=True, slots=True)
class ReplayCheckpoint:
    source_id: str
    last_index: int
    last_event_id: str
    updated_at: str


class ReplayCheckpointStore:
    """原子保存最後成功投遞位置；consumer 失敗時不前進。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self, source_id: str) -> ReplayCheckpoint | None:
        if not self.path.is_file():
            return None
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("Replay checkpoint 版本不支援")
        if payload.get("source_id") != source_id:
            raise ValueError("Replay checkpoint 與目前輸入資料不一致")
        return ReplayCheckpoint(
            source_id=str(payload["source_id"]),
            last_index=int(payload["last_index"]),
            last_event_id=str(payload["last_event_id"]),
            updated_at=str(payload["updated_at"]),
        )

    def save(self, source_id: str, index: int, event_id: str) -> None:
        write_json_atomic(
            self.path,
            {
                "schema_version": 1,
                "source_id": source_id,
                "last_index": index,
                "last_event_id": event_id,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )


class InjectedTransportDisconnect(ConnectionError):
    """離線實驗刻意注入的傳輸中斷。"""

    def __init__(self, index: int) -> None:
        super().__init__(f"在事件索引 {index} 注入傳輸中斷")
        self.index = index


@dataclass(frozen=True, slots=True)
class ReplayResult:
    source_id: str
    start_index: int
    end_index: int
    events_available: int
    attempted: int
    delivered: int
    duplicates: int
    elapsed_seconds: float
    resumed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class MarketEventReplayer:
    """以虛擬時間或指定倍速，把歷史事件送進正式 Event Bus。"""

    def __init__(
        self,
        events: Sequence[MarketEvent],
        *,
        source_id: str,
        speed_multiplier: float = 0.0,
        maximum_sleep_seconds: float = 1.0,
    ) -> None:
        if not events:
            raise ValueError("重播事件不可為空")
        if speed_multiplier < 0 or maximum_sleep_seconds < 0:
            raise ValueError("重播速度與最大等待不可小於 0")
        self.events = tuple(events)
        self.source_id = source_id
        self.speed_multiplier = speed_multiplier
        self.maximum_sleep_seconds = maximum_sleep_seconds

    def run(
        self,
        bus: MarketEventBus,
        *,
        checkpoint_store: ReplayCheckpointStore | None = None,
        resume: bool = True,
        disconnect_before_index: int | None = None,
        duplicate_at_index: int | None = None,
        checkpoint_every: int = 1,
    ) -> ReplayResult:
        if checkpoint_every < 1:
            raise ValueError("checkpoint_every 至少為 1")
        checkpoint = checkpoint_store.load(self.source_id) if checkpoint_store and resume else None
        start_index = checkpoint.last_index + 1 if checkpoint else 0
        if start_index > len(self.events):
            raise ValueError("Replay checkpoint 超出事件範圍")
        started = monotonic()
        before_delivered = bus.metrics.delivered
        before_duplicates = bus.metrics.duplicates
        attempted = 0
        previous_time: datetime | None = None
        end_index = start_index - 1
        for index in range(start_index, len(self.events)):
            if disconnect_before_index == index:
                raise InjectedTransportDisconnect(index)
            event = self.events[index]
            if self.speed_multiplier > 0 and previous_time is not None:
                market_delay = max((event.event_time - previous_time).total_seconds(), 0.0)
                sleep(min(market_delay / self.speed_multiplier, self.maximum_sleep_seconds))
            attempted += 1
            bus.publish(event)
            if duplicate_at_index == index:
                bus.publish(event)
            if checkpoint_store is not None and (
                (index + 1) % checkpoint_every == 0 or index == len(self.events) - 1
            ):
                checkpoint_store.save(self.source_id, index, event.event_id)
            previous_time = event.event_time
            end_index = index
        return ReplayResult(
            source_id=self.source_id,
            start_index=start_index,
            end_index=end_index,
            events_available=len(self.events),
            attempted=attempted,
            delivered=bus.metrics.delivered - before_delivered,
            duplicates=bus.metrics.duplicates - before_duplicates,
            elapsed_seconds=monotonic() - started,
            resumed=checkpoint is not None,
        )


class MarketFrameAccumulator:
    """將統一事件還原成既有模型可接收的 DataFrame。"""

    def __init__(self, *, max_rows: int | None = None) -> None:
        if max_rows is not None and max_rows < 1:
            raise ValueError("max_rows 必須大於 0")
        self.max_rows = max_rows
        self._rows: dict[tuple[str, str, str], dict[int, dict[str, object]]] = {}

    def __call__(self, event: MarketEvent) -> None:
        if event.kind != "kline_closed" or event.interval is None:
            return
        key = (event.exchange, event.symbol, event.interval)
        rows = self._rows.setdefault(key, {})
        rows[int(event.payload["open_time_ms"])] = event.to_ohlcv_row()
        if self.max_rows is not None:
            for open_time in sorted(rows)[: max(len(rows) - self.max_rows, 0)]:
                rows.pop(open_time, None)

    def frame(
        self,
        *,
        exchange: str,
        symbol: str,
        interval: str,
    ) -> pd.DataFrame:
        rows = self._rows.get((exchange.lower(), symbol.upper(), interval.lower()), {})
        return pd.DataFrame([rows[key] for key in sorted(rows)]).reset_index(drop=True)
