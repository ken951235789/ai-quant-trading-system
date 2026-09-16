"""交易狀態中心的 SQLAlchemy 資料表。"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Index,
    Integer,
    JSON,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


JSON_DOCUMENT = JSON().with_variant(JSONB, "postgresql")
MONEY = Numeric(28, 12)
IDENTITY = BigInteger().with_variant(Integer, "sqlite")


class Base(DeclarativeBase):
    """所有交易資料表的 Declarative Base。"""


class TradingRecord(Base):
    """現有 CSV 事件的不可變相容紀錄，方便安全遷移與稽核。"""

    __tablename__ = "trading_records"
    __table_args__ = (
        UniqueConstraint("record_key", name="uq_trading_records_record_key"),
        Index(
            "ix_trading_records_lookup",
            "environment",
            "record_type",
            "occurred_at",
        ),
    )

    id: Mapped[int] = mapped_column(IDENTITY, primary_key=True, autoincrement=True)
    record_key: Mapped[str] = mapped_column(String(64), nullable=False)
    record_type: Mapped[str] = mapped_column(String(40), nullable=False)
    environment: Mapped[str] = mapped_column(String(16), nullable=False)
    market_type: Mapped[str | None] = mapped_column(String(32))
    symbol: Mapped[str | None] = mapped_column(String(32))
    occurred_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON_DOCUMENT, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ManagedPosition(Base):
    """由本系統管理的當前部位；交易所部位仍需定期對帳。"""

    __tablename__ = "managed_positions"

    environment: Mapped[str] = mapped_column(String(16), primary_key=True)
    position_key: Mapped[str] = mapped_column(String(96), primary_key=True)
    market_type: Mapped[str] = mapped_column(String(32), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    entry_price: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    leverage: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON_DOCUMENT, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class Order(Base):
    """交易所訂單的目前狀態，client_order_id 是冪等鍵。"""

    __tablename__ = "orders"
    __table_args__ = (
        UniqueConstraint("record_key", name="uq_orders_record_key"),
        UniqueConstraint(
            "environment", "client_order_id", name="uq_orders_environment_client_id"
        ),
        UniqueConstraint(
            "environment", "exchange_order_id", name="uq_orders_environment_exchange_id"
        ),
        Index("ix_orders_symbol_time", "environment", "symbol", "occurred_at"),
    )

    id: Mapped[int] = mapped_column(IDENTITY, primary_key=True, autoincrement=True)
    record_key: Mapped[str] = mapped_column(String(64), nullable=False)
    environment: Mapped[str] = mapped_column(String(16), nullable=False)
    market_type: Mapped[str | None] = mapped_column(String(32))
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    client_order_id: Mapped[str | None] = mapped_column(String(64))
    exchange_order_id: Mapped[str | None] = mapped_column(String(64))
    side: Mapped[str | None] = mapped_column(String(8))
    position_side: Mapped[str | None] = mapped_column(String(8))
    reduce_only: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    quantity: Mapped[Decimal | None] = mapped_column(MONEY)
    reference_price: Mapped[Decimal | None] = mapped_column(MONEY)
    executed_quantity: Mapped[Decimal | None] = mapped_column(MONEY)
    status: Mapped[str | None] = mapped_column(String(32))
    occurred_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON_DOCUMENT, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class Fill(Base):
    """User Data Stream 或交易所查詢取得的實際成交。"""

    __tablename__ = "fills"
    __table_args__ = (
        UniqueConstraint(
            "environment",
            "venue",
            "exchange_trade_id",
            name="uq_fills_environment_venue_trade",
        ),
        Index("ix_fills_symbol_time", "symbol", "occurred_at"),
    )

    id: Mapped[int] = mapped_column(IDENTITY, primary_key=True, autoincrement=True)
    venue: Mapped[str] = mapped_column(String(32), nullable=False)
    environment: Mapped[str] = mapped_column(String(16), nullable=False)
    exchange_trade_id: Mapped[str] = mapped_column(String(64), nullable=False)
    client_order_id: Mapped[str | None] = mapped_column(String(64))
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    price: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    quantity: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    commission: Mapped[Decimal | None] = mapped_column(MONEY)
    commission_asset: Mapped[str | None] = mapped_column(String(16))
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON_DOCUMENT, nullable=False)


class RiskEvent(Base):
    """停機、限制觸發與人工解除等風控稽核事件。"""

    __tablename__ = "risk_events"

    id: Mapped[int] = mapped_column(IDENTITY, primary_key=True, autoincrement=True)
    environment: Mapped[str] = mapped_column(String(16), nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    symbol: Mapped[str | None] = mapped_column(String(32))
    message: Mapped[str] = mapped_column(Text, nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON_DOCUMENT, nullable=False)


class ReconciliationRun(Base):
    """本機、資料庫與 Binance 部位核對結果。"""

    __tablename__ = "reconciliation_runs"

    id: Mapped[int] = mapped_column(IDENTITY, primary_key=True, autoincrement=True)
    environment: Mapped[str] = mapped_column(String(16), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    consistent: Mapped[bool] = mapped_column(Boolean, nullable=False)
    local_quantity: Mapped[Decimal | None] = mapped_column(MONEY)
    exchange_quantity: Mapped[Decimal | None] = mapped_column(MONEY)
    reason: Mapped[str | None] = mapped_column(Text)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON_DOCUMENT, nullable=False)


class ExchangeEvent(Base):
    """Binance 私有串流原始事件；只新增、不覆寫 payload。"""

    __tablename__ = "exchange_events"
    __table_args__ = (
        UniqueConstraint("event_key", name="uq_exchange_events_event_key"),
        Index(
            "ix_exchange_events_stream_time",
            "environment",
            "stream_name",
            "event_time",
        ),
    )

    id: Mapped[int] = mapped_column(IDENTITY, primary_key=True, autoincrement=True)
    event_key: Mapped[str] = mapped_column(String(64), nullable=False)
    venue: Mapped[str] = mapped_column(String(32), nullable=False)
    environment: Mapped[str] = mapped_column(String(16), nullable=False)
    stream_name: Mapped[str] = mapped_column(String(64), nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    symbol: Mapped[str | None] = mapped_column(String(32))
    exchange_order_id: Mapped[str | None] = mapped_column(String(64))
    client_order_id: Mapped[str | None] = mapped_column(String(64))
    exchange_trade_id: Mapped[str | None] = mapped_column(String(64))
    event_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    transaction_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON_DOCUMENT, nullable=False)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    projection_error: Mapped[str | None] = mapped_column(Text)


class StreamCheckpoint(Base):
    """私有串流健康狀態；listenKey 不得保存於此。"""

    __tablename__ = "stream_checkpoints"

    environment: Mapped[str] = mapped_column(String(16), primary_key=True)
    stream_name: Mapped[str] = mapped_column(String(64), primary_key=True)
    state: Mapped[str] = mapped_column(String(24), nullable=False)
    last_event_key: Mapped[str | None] = mapped_column(String(64))
    last_event_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    connected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    reconnect_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    gap_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    details: Mapped[dict[str, Any]] = mapped_column(JSON_DOCUMENT, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
