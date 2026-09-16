"""建立 BTC Futures 交易狀態中心。

Revision ID: 20260828_0001
Revises:
Create Date: 2026-08-28
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260828_0001"
down_revision = None
branch_labels = None
depends_on = None

JSONB = postgresql.JSONB(astext_type=sa.Text())
MONEY = sa.Numeric(28, 12)


def upgrade() -> None:
    """建立事件、部位、訂單、成交與風控資料表。"""
    op.create_table(
        "trading_records",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("record_key", sa.String(64), nullable=False),
        sa.Column("record_type", sa.String(40), nullable=False),
        sa.Column("environment", sa.String(16), nullable=False),
        sa.Column("market_type", sa.String(32), nullable=True),
        sa.Column("symbol", sa.String(32), nullable=True),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("record_key", name="uq_trading_records_record_key"),
    )
    op.create_index(
        "ix_trading_records_lookup",
        "trading_records",
        ["environment", "record_type", "occurred_at"],
    )

    op.create_table(
        "managed_positions",
        sa.Column("environment", sa.String(16), nullable=False),
        sa.Column("position_key", sa.String(96), nullable=False),
        sa.Column("market_type", sa.String(32), nullable=False),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("side", sa.String(8), nullable=False),
        sa.Column("quantity", MONEY, nullable=False),
        sa.Column("entry_price", MONEY, nullable=False),
        sa.Column("leverage", sa.Integer(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("environment", "position_key"),
    )

    op.create_table(
        "orders",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("record_key", sa.String(64), nullable=False),
        sa.Column("environment", sa.String(16), nullable=False),
        sa.Column("market_type", sa.String(32), nullable=True),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("client_order_id", sa.String(64), nullable=True),
        sa.Column("exchange_order_id", sa.String(64), nullable=True),
        sa.Column("side", sa.String(8), nullable=True),
        sa.Column("position_side", sa.String(8), nullable=True),
        sa.Column("reduce_only", sa.Boolean(), nullable=False),
        sa.Column("quantity", MONEY, nullable=True),
        sa.Column("reference_price", MONEY, nullable=True),
        sa.Column("executed_quantity", MONEY, nullable=True),
        sa.Column("status", sa.String(32), nullable=True),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("record_key", name="uq_orders_record_key"),
        sa.UniqueConstraint(
            "environment", "client_order_id", name="uq_orders_environment_client_id"
        ),
        sa.UniqueConstraint(
            "environment", "exchange_order_id", name="uq_orders_environment_exchange_id"
        ),
    )
    op.create_index(
        "ix_orders_symbol_time", "orders", ["environment", "symbol", "occurred_at"]
    )

    op.create_table(
        "fills",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("venue", sa.String(32), nullable=False),
        sa.Column("environment", sa.String(16), nullable=False),
        sa.Column("exchange_trade_id", sa.String(64), nullable=False),
        sa.Column("client_order_id", sa.String(64), nullable=True),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("side", sa.String(8), nullable=False),
        sa.Column("price", MONEY, nullable=False),
        sa.Column("quantity", MONEY, nullable=False),
        sa.Column("commission", MONEY, nullable=True),
        sa.Column("commission_asset", sa.String(16), nullable=True),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payload", JSONB, nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "venue", "exchange_trade_id", name="uq_fills_venue_exchange_trade"
        ),
    )
    op.create_index("ix_fills_symbol_time", "fills", ["symbol", "occurred_at"])

    op.create_table(
        "risk_events",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("environment", sa.String(16), nullable=False),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("severity", sa.String(16), nullable=False),
        sa.Column("symbol", sa.String(32), nullable=True),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payload", JSONB, nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "reconciliation_runs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("environment", sa.String(16), nullable=False),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("consistent", sa.Boolean(), nullable=False),
        sa.Column("local_quantity", MONEY, nullable=True),
        sa.Column("exchange_quantity", MONEY, nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payload", JSONB, nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    """依相依順序移除本版本資料表。"""
    op.drop_table("reconciliation_runs")
    op.drop_table("risk_events")
    op.drop_index("ix_fills_symbol_time", table_name="fills")
    op.drop_table("fills")
    op.drop_index("ix_orders_symbol_time", table_name="orders")
    op.drop_table("orders")
    op.drop_table("managed_positions")
    op.drop_index("ix_trading_records_lookup", table_name="trading_records")
    op.drop_table("trading_records")
