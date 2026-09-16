"""新增 Binance 私有事件流與 checkpoint。

Revision ID: 20260829_0002
Revises: 20260828_0001
Create Date: 2026-08-29
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260829_0002"
down_revision = "20260828_0001"
branch_labels = None
depends_on = None

JSONB = postgresql.JSONB(astext_type=sa.Text())


def upgrade() -> None:
    """建立只追加事件表與單列更新的串流 checkpoint。"""
    op.drop_constraint("uq_fills_venue_exchange_trade", "fills", type_="unique")
    op.create_unique_constraint(
        "uq_fills_environment_venue_trade",
        "fills",
        ["environment", "venue", "exchange_trade_id"],
    )
    op.create_table(
        "exchange_events",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("event_key", sa.String(64), nullable=False),
        sa.Column("venue", sa.String(32), nullable=False),
        sa.Column("environment", sa.String(16), nullable=False),
        sa.Column("stream_name", sa.String(64), nullable=False),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("symbol", sa.String(32), nullable=True),
        sa.Column("exchange_order_id", sa.String(64), nullable=True),
        sa.Column("client_order_id", sa.String(64), nullable=True),
        sa.Column("exchange_trade_id", sa.String(64), nullable=True),
        sa.Column("event_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("transaction_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payload_hash", sa.String(64), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("projection_error", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("event_key", name="uq_exchange_events_event_key"),
    )
    op.create_index(
        "ix_exchange_events_stream_time",
        "exchange_events",
        ["environment", "stream_name", "event_time"],
    )
    op.create_table(
        "stream_checkpoints",
        sa.Column("environment", sa.String(16), nullable=False),
        sa.Column("stream_name", sa.String(64), nullable=False),
        sa.Column("state", sa.String(24), nullable=False),
        sa.Column("last_event_key", sa.String(64), nullable=True),
        sa.Column("last_event_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("connected_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("reconnect_count", sa.Integer(), nullable=False),
        sa.Column("gap_count", sa.Integer(), nullable=False),
        sa.Column("details", JSONB, nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("environment", "stream_name"),
    )


def downgrade() -> None:
    """移除私有事件流資料表。"""
    op.drop_table("stream_checkpoints")
    op.drop_index("ix_exchange_events_stream_time", table_name="exchange_events")
    op.drop_table("exchange_events")
    op.drop_constraint("uq_fills_environment_venue_trade", "fills", type_="unique")
    op.create_unique_constraint(
        "uq_fills_venue_exchange_trade",
        "fills",
        ["venue", "exchange_trade_id"],
    )
