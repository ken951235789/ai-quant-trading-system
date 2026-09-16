"""交易事件與部位的 PostgreSQL Repository。"""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from functools import lru_cache
import hashlib
import json
import math
from typing import Any, Mapping

import pandas as pd

from ai_quant_trading.database.config import DatabaseSettings
from ai_quant_trading.database.engine import engine_from_url


def _json_safe(value: Any) -> Any:
    """把 Pandas、NumPy、Decimal 與 NaN 轉成 PostgreSQL JSON 可接受的值。"""
    if value is None:
        return None
    if isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _record_key(record_type: str, environment: str, payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        {
            "record_type": record_type,
            "environment": environment,
            "payload": _json_safe(payload),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _timestamp(value: Any) -> datetime | None:
    if value in {None, ""}:
        return None
    if isinstance(value, int | float) and abs(float(value)) >= 100_000_000_000:
        parsed = pd.to_datetime(value, unit="ms", utc=True, errors="coerce")
    else:
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def _decimal(value: Any) -> Decimal | None:
    if value in {None, ""}:
        return None
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None
    return number if number.is_finite() else None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def _boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


class PostgresTradingRepository:
    """集中管理 SQL transaction，禁止交易模組自行拼接 SQL。"""

    def __init__(self, engine: Any) -> None:
        try:
            from sqlalchemy.orm import sessionmaker
        except ImportError as exc:
            raise RuntimeError(
                '缺少 SQL 套件，請執行 python -m pip install -e ".[database]"'
            ) from exc
        self.engine = engine
        self._session_factory = sessionmaker(
            bind=engine,
            expire_on_commit=False,
            autoflush=False,
        )

    def append_record(
        self,
        record_type: str,
        environment: str,
        row: Mapping[str, Any],
    ) -> bool:
        """冪等新增交易事件；回傳 False 表示相同事件已存在。"""
        from sqlalchemy import select

        from ai_quant_trading.database.models import Order, TradingRecord

        payload = {str(key): _json_safe(value) for key, value in row.items()}
        key = _record_key(record_type, environment, payload)
        record_values = {
            "record_key": key,
            "record_type": record_type,
            "environment": environment,
            "market_type": _text(payload.get("market_type")),
            "symbol": _text(payload.get("symbol")),
            "occurred_at": _timestamp(payload.get("timestamp")),
            "payload": _json_safe(payload),
        }
        with self._session_factory.begin() as session:
            if session.bind.dialect.name == "postgresql":
                from sqlalchemy.dialects.postgresql import insert

                statement = (
                    insert(TradingRecord)
                    .values(**record_values)
                    .on_conflict_do_nothing(index_elements=[TradingRecord.record_key])
                    .returning(TradingRecord.id)
                )
                inserted = session.execute(statement).scalar_one_or_none() is not None
            else:
                existing = session.scalar(
                    select(TradingRecord.id).where(TradingRecord.record_key == key)
                )
                inserted = existing is None
                if inserted:
                    session.add(TradingRecord(**record_values))
            if record_type == "orders":
                self._upsert_order(session, Order, key, environment, payload)
        return inserted

    def append_exchange_event(
        self,
        environment: str,
        stream_name: str,
        event: Mapping[str, Any],
        *,
        received_at: datetime | None = None,
    ) -> tuple[bool, str]:
        """先冪等保存 Binance 原始事件；listenKey 與 API 憑證不可放入 event。"""
        from sqlalchemy import select

        from ai_quant_trading.database.models import ExchangeEvent

        payload = {str(key): _json_safe(value) for key, value in event.items()}
        payload_json = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        event_type = _text(payload.get("e")) or "UNKNOWN"
        order = payload.get("o") if isinstance(payload.get("o"), Mapping) else {}
        event_key = _record_key(
            f"exchange:{stream_name}",
            environment,
            {"payload_hash": payload_hash},
        )
        values = {
            "event_key": event_key,
            "venue": "binance_futures",
            "environment": environment,
            "stream_name": stream_name,
            "event_type": event_type,
            "symbol": _text(order.get("s") or payload.get("s")),
            "exchange_order_id": _text(order.get("i")),
            "client_order_id": _text(order.get("c")),
            "exchange_trade_id": _text(order.get("t")),
            "event_time": _timestamp(payload.get("E")),
            "transaction_time": _timestamp(payload.get("T")),
            "received_at": received_at or datetime.now(timezone.utc),
            "payload_hash": payload_hash,
            "schema_version": 1,
            "payload": _json_safe(payload),
        }
        with self._session_factory.begin() as session:
            if session.bind.dialect.name == "postgresql":
                from sqlalchemy.dialects.postgresql import insert

                statement = (
                    insert(ExchangeEvent)
                    .values(**values)
                    .on_conflict_do_nothing(index_elements=[ExchangeEvent.event_key])
                    .returning(ExchangeEvent.id)
                )
                inserted = session.execute(statement).scalar_one_or_none() is not None
            else:
                existing = session.scalar(
                    select(ExchangeEvent.id).where(ExchangeEvent.event_key == event_key)
                )
                inserted = existing is None
                if inserted:
                    session.add(ExchangeEvent(**values))
        return inserted, event_key

    def project_exchange_event(self, event_key: str) -> bool:
        """把已保存事件投影成訂單與成交；失敗會保留原始事件供重播。"""
        from sqlalchemy import select

        from ai_quant_trading.database.models import ExchangeEvent, Fill, Order

        try:
            with self._session_factory.begin() as session:
                event = session.scalar(
                    select(ExchangeEvent)
                    .where(ExchangeEvent.event_key == event_key)
                    .with_for_update()
                )
                if event is None:
                    raise ValueError("找不到待投影的交易所事件")
                if event.processed_at is not None:
                    return False
                payload = dict(event.payload)
                if event.event_type == "ORDER_TRADE_UPDATE":
                    order = payload.get("o")
                    if not isinstance(order, Mapping):
                        raise ValueError("ORDER_TRADE_UPDATE 缺少 o 欄位")
                    occurred_at = _timestamp(payload.get("E")) or event.received_at
                    normalized_order = {
                        "timestamp": occurred_at.isoformat(),
                        "market_type": "usd_m_futures",
                        "symbol": _text(order.get("s")) or "UNKNOWN",
                        "client_order_id": _text(order.get("c")),
                        "order_id": _text(order.get("i")),
                        "side": _text(order.get("S")),
                        "position_side": _text(order.get("ps")),
                        "reduce_only": _boolean(order.get("R")),
                        "quantity": _decimal(order.get("q")),
                        "reference_price": _decimal(order.get("ap") or order.get("p")),
                        "executed_quantity": _decimal(order.get("z")),
                        "status": _text(order.get("X")),
                        "source": "user_data_stream",
                        "raw_execution_type": _text(order.get("x")),
                    }
                    self._upsert_order(
                        session,
                        Order,
                        event.event_key,
                        event.environment,
                        normalized_order,
                    )
                    trade_id = _text(order.get("t"))
                    fill_quantity = _decimal(order.get("l")) or Decimal("0")
                    fill_price = _decimal(order.get("L")) or Decimal("0")
                    if trade_id not in {None, "0", "-1"} and fill_quantity > 0:
                        fill_values = {
                            "venue": "binance_futures",
                            "environment": event.environment,
                            "exchange_trade_id": trade_id,
                            "client_order_id": _text(order.get("c")),
                            "symbol": _text(order.get("s")) or "UNKNOWN",
                            "side": _text(order.get("S")) or "UNKNOWN",
                            "price": fill_price,
                            "quantity": fill_quantity,
                            "commission": _decimal(order.get("n")),
                            "commission_asset": _text(order.get("N")),
                            "occurred_at": occurred_at,
                            "payload": payload,
                        }
                        if session.bind.dialect.name == "postgresql":
                            from sqlalchemy.dialects.postgresql import insert

                            session.execute(
                                insert(Fill)
                                .values(**fill_values)
                                .on_conflict_do_nothing(
                                    constraint="uq_fills_environment_venue_trade"
                                )
                            )
                        else:
                            duplicate = session.scalar(
                                select(Fill.id).where(
                                    Fill.environment == event.environment,
                                    Fill.venue == "binance_futures",
                                    Fill.exchange_trade_id == trade_id,
                                )
                            )
                            if duplicate is None:
                                session.add(Fill(**fill_values))
                event.processed_at = datetime.now(timezone.utc)
                event.projection_error = None
            return True
        except Exception as exc:
            with self._session_factory.begin() as session:
                event = session.scalar(
                    select(ExchangeEvent)
                    .where(ExchangeEvent.event_key == event_key)
                    .with_for_update()
                )
                if event is not None:
                    event.projection_error = f"{type(exc).__name__}: {exc}"[:1000]
            raise

    def update_stream_checkpoint(
        self,
        environment: str,
        stream_name: str,
        *,
        state: str,
        heartbeat_at: datetime,
        last_event_key: str | None = None,
        last_event_time: datetime | None = None,
        connected_at: datetime | None = None,
        reconnect_count: int = 0,
        gap_count: int = 0,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """更新串流 checkpoint；details 僅保存非敏感健康資訊。"""
        from sqlalchemy import select

        from ai_quant_trading.database.models import StreamCheckpoint

        values = {
            "state": state,
            "last_event_key": last_event_key,
            "last_event_time": last_event_time,
            "connected_at": connected_at,
            "heartbeat_at": heartbeat_at,
            "reconnect_count": reconnect_count,
            "gap_count": gap_count,
            "details": _json_safe(dict(details or {})),
        }
        with self._session_factory.begin() as session:
            row = session.scalar(
                select(StreamCheckpoint).where(
                    StreamCheckpoint.environment == environment,
                    StreamCheckpoint.stream_name == stream_name,
                )
            )
            if row is None:
                session.add(
                    StreamCheckpoint(
                        environment=environment,
                        stream_name=stream_name,
                        **values,
                    )
                )
                return
            for field, value in values.items():
                setattr(row, field, value)

    def stream_checkpoint(self, environment: str, stream_name: str) -> dict[str, Any] | None:
        """讀取 Watchdog 所需的串流健康資訊。"""
        from sqlalchemy import select

        from ai_quant_trading.database.models import StreamCheckpoint

        with self._session_factory() as session:
            row = session.scalar(
                select(StreamCheckpoint).where(
                    StreamCheckpoint.environment == environment,
                    StreamCheckpoint.stream_name == stream_name,
                )
            )
            if row is None:
                return None
            return {
                "state": row.state,
                "last_event_key": row.last_event_key,
                "last_event_time": row.last_event_time,
                "heartbeat_at": row.heartbeat_at,
                "connected_at": row.connected_at,
                "reconnect_count": row.reconnect_count,
                "gap_count": row.gap_count,
                "details": dict(row.details),
            }

    def record_reconciliation(
        self,
        environment: str,
        symbol: str,
        *,
        consistent: bool,
        local_quantity: Decimal,
        exchange_quantity: Decimal,
        reason: str,
        payload: Mapping[str, Any],
    ) -> None:
        """保存啟動與重連對帳結果，供 Watchdog 判斷是否允許交易。"""
        from ai_quant_trading.database.models import ReconciliationRun

        with self._session_factory.begin() as session:
            session.add(
                ReconciliationRun(
                    environment=environment,
                    symbol=symbol,
                    consistent=consistent,
                    local_quantity=local_quantity,
                    exchange_quantity=exchange_quantity,
                    reason=reason,
                    occurred_at=datetime.now(timezone.utc),
                    payload=_json_safe(dict(payload)),
                )
            )

    def replay_unprocessed_exchange_events(self, *, limit: int = 1000) -> int:
        """重播落盤後尚未完成投影的事件，供程序崩潰後自動修復。"""
        from sqlalchemy import select

        from ai_quant_trading.database.models import ExchangeEvent

        if limit <= 0:
            raise ValueError("重播筆數必須大於 0")
        with self._session_factory() as session:
            keys = list(
                session.scalars(
                    select(ExchangeEvent.event_key)
                    .where(ExchangeEvent.processed_at.is_(None))
                    .order_by(ExchangeEvent.id)
                    .limit(limit)
                ).all()
            )
        replayed = 0
        for event_key in keys:
            if self.project_exchange_event(event_key):
                replayed += 1
        return replayed

    @staticmethod
    def _upsert_order(
        session: Any,
        order_model: Any,
        record_key: str,
        environment: str,
        payload: dict[str, Any],
    ) -> None:
        """以 client_order_id 更新目前訂單狀態，避免 timeout 重送形成重複單。"""
        from sqlalchemy import or_, select

        client_id = _text(payload.get("client_order_id"))
        exchange_id = _text(payload.get("order_id"))
        predicates = [order_model.record_key == record_key]
        if client_id:
            predicates.append(
                (order_model.environment == environment)
                & (order_model.client_order_id == client_id)
            )
        if exchange_id:
            predicates.append(
                (order_model.environment == environment)
                & (order_model.exchange_order_id == exchange_id)
            )
        values = {
            "record_key": record_key,
            "environment": environment,
            "market_type": _text(payload.get("market_type")),
            "symbol": _text(payload.get("symbol")) or "UNKNOWN",
            "client_order_id": client_id,
            "exchange_order_id": exchange_id,
            "side": _text(payload.get("side")),
            "position_side": _text(payload.get("position_side")),
            "reduce_only": _boolean(payload.get("reduce_only")),
            "quantity": _decimal(payload.get("quantity")),
            "reference_price": _decimal(payload.get("reference_price")),
            "executed_quantity": _decimal(payload.get("executed_quantity")),
            "status": _text(payload.get("status")),
            "occurred_at": _timestamp(payload.get("timestamp")),
            "payload": _json_safe(payload),
        }
        if session.bind.dialect.name == "postgresql" and client_id:
            from sqlalchemy.dialects.postgresql import insert

            statement = insert(order_model).values(**values)
            update_values = {
                field: getattr(statement.excluded, field)
                for field in values
                if field not in {"environment", "client_order_id"}
            }
            session.execute(
                statement.on_conflict_do_update(
                    constraint="uq_orders_environment_client_id",
                    set_=update_values,
                    where=(order_model.occurred_at.is_(None))
                    | (statement.excluded.occurred_at >= order_model.occurred_at),
                )
            )
            return
        if session.bind.dialect.name == "postgresql" and exchange_id:
            from sqlalchemy.dialects.postgresql import insert

            statement = insert(order_model).values(**values)
            update_values = {
                field: getattr(statement.excluded, field)
                for field in values
                if field not in {"environment", "exchange_order_id"}
            }
            session.execute(
                statement.on_conflict_do_update(
                    constraint="uq_orders_environment_exchange_id",
                    set_=update_values,
                    where=(order_model.occurred_at.is_(None))
                    | (statement.excluded.occurred_at >= order_model.occurred_at),
                )
            )
            return
        order = session.scalar(select(order_model).where(or_(*predicates)))
        if order is None:
            session.add(order_model(**values))
            return
        if order.occurred_at and values["occurred_at"]:
            current_time = order.occurred_at
            incoming_time = values["occurred_at"]
            if current_time.tzinfo is None:
                current_time = current_time.replace(tzinfo=timezone.utc)
            if incoming_time.tzinfo is None:
                incoming_time = incoming_time.replace(tzinfo=timezone.utc)
            if incoming_time < current_time:
                return
        for field, value in values.items():
            setattr(order, field, value)

    def read_records(
        self,
        record_type: str,
        environment: str,
        *,
        limit: int | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """依原始發生順序讀取事件，供既有 Dashboard 與驗收報表相容使用。"""
        from sqlalchemy import select

        from ai_quant_trading.database.models import TradingRecord

        if limit is not None and limit <= 0:
            raise ValueError("limit 必須大於 0")
        predicates = [
            TradingRecord.record_type == record_type,
            TradingRecord.environment == environment,
        ]
        if since is not None:
            predicates.append(TradingRecord.occurred_at >= since)
        if until is not None:
            predicates.append(TradingRecord.occurred_at < until)
        statement = select(TradingRecord.payload).where(*predicates)
        if limit is not None:
            statement = statement.order_by(TradingRecord.id.desc()).limit(limit)
        else:
            statement = statement.order_by(TradingRecord.id)
        with self._session_factory() as session:
            records = [dict(payload) for payload in session.scalars(statement).all()]
        if limit is not None:
            records.reverse()
        return records

    def load_positions(self, environment: str) -> dict[str, dict[str, Any]]:
        """讀取策略管理部位；不把帳戶其他部位納入策略。"""
        from sqlalchemy import select

        from ai_quant_trading.database.models import ManagedPosition

        statement = select(ManagedPosition).where(ManagedPosition.environment == environment)
        with self._session_factory() as session:
            rows = session.scalars(statement).all()
            return {row.position_key: dict(row.payload) for row in rows}

    def save_positions(
        self,
        environment: str,
        positions: Mapping[str, Mapping[str, Any]],
    ) -> None:
        """在單一 transaction 內鎖定並更新整組策略部位。"""
        from sqlalchemy import select

        from ai_quant_trading.database.models import ManagedPosition

        payloads = {
            str(key): {
                str(field): _json_safe(field_value)
                for field, field_value in value.items()
            }
            for key, value in positions.items()
        }
        with self._session_factory.begin() as session:
            current = {
                row.position_key: row
                for row in session.scalars(
                    select(ManagedPosition)
                    .where(ManagedPosition.environment == environment)
                    .with_for_update()
                ).all()
            }
            for stale_key in current.keys() - payloads.keys():
                session.delete(current[stale_key])
            for position_key, payload in payloads.items():
                row = current.get(position_key)
                values = {
                    "market_type": _text(payload.get("market_type")) or "unknown",
                    "symbol": _text(payload.get("symbol")) or position_key.split(":")[-1],
                    "side": _text(payload.get("side")) or "LONG",
                    "quantity": _decimal(payload.get("quantity")) or Decimal("0"),
                    "entry_price": _decimal(payload.get("entry_price")) or Decimal("0"),
                    "leverage": int(payload.get("leverage") or 1),
                    "payload": payload,
                }
                if row is None:
                    session.add(
                        ManagedPosition(
                            environment=environment,
                            position_key=position_key,
                            version=1,
                            **values,
                        )
                    )
                    continue
                for field, value in values.items():
                    setattr(row, field, value)
                row.version += 1


@lru_cache(maxsize=4)
def repository_from_url(
    database_url: str,
    pool_size: int = 5,
    max_overflow: int = 5,
    connect_timeout_seconds: int = 5,
) -> PostgresTradingRepository:
    """重用 Repository 與連線池。"""
    engine = engine_from_url(
        database_url,
        pool_size,
        max_overflow,
        connect_timeout_seconds,
    )
    return PostgresTradingRepository(engine)


def repository_from_settings(
    settings: DatabaseSettings | None = None,
) -> PostgresTradingRepository:
    """依環境建立 Repository；file 模式不得誤呼叫。"""
    resolved = settings or DatabaseSettings.from_environment()
    if not resolved.enabled:
        raise ValueError("目前不是 postgres 儲存模式")
    return repository_from_url(
        resolved.database_url,
        resolved.pool_size,
        resolved.max_overflow,
        resolved.connect_timeout_seconds,
    )
