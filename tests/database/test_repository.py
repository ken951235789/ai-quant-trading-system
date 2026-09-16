"""PostgreSQL Repository 的交易語意測試；SQLite 只作快速單元測試。"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine, func, select

from ai_quant_trading.database.models import Base, Order
from ai_quant_trading.database.repository import PostgresTradingRepository


def _repository() -> PostgresTradingRepository:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    return PostgresTradingRepository(engine)


def test_append_record_is_idempotent_and_updates_order() -> None:
    repository = _repository()
    submitted = {
        "timestamp": "2026-08-28T00:00:00Z",
        "market_type": "usd_m_futures",
        "symbol": "BTC/USDT",
        "client_order_id": "aqt-001",
        "order_id": "1001",
        "side": "BUY",
        "position_side": "BOTH",
        "quantity": "0.001",
        "status": "NEW",
    }

    assert repository.append_record("orders", "testnet", submitted)
    assert not repository.append_record("orders", "testnet", submitted)

    filled = {**submitted, "status": "FILLED", "executed_quantity": "0.001"}
    assert repository.append_record("orders", "testnet", filled)

    records = repository.read_records("orders", "testnet")
    assert len(records) == 2
    with repository._session_factory() as session:
        assert session.scalar(select(func.count()).select_from(Order)) == 1
        order = session.scalar(select(Order))
        assert order is not None
        assert order.status == "FILLED"


def test_positions_are_replaced_in_one_repository_call() -> None:
    repository = _repository()
    position = {
        "symbol": "BTC/USDT",
        "market_type": "usd_m_futures",
        "side": "SHORT",
        "quantity": 0.002,
        "entry_price": 60_000,
        "leverage": 2,
        "stop_loss": 61_000,
        "take_profit": 59_000,
        "entry_order_id": "1001",
        "opened_at": "2026-08-28T00:00:00Z",
        "updated_at": "2026-08-28T00:00:00Z",
    }

    repository.save_positions("testnet", {"usd_m_futures:BTC/USDT": position})
    loaded = repository.load_positions("testnet")
    assert loaded["usd_m_futures:BTC/USDT"]["side"] == "SHORT"

    repository.save_positions("testnet", {})
    assert repository.load_positions("testnet") == {}


def test_read_records_applies_time_range_and_limit_in_order() -> None:
    repository = _repository()
    for day in range(1, 5):
        repository.append_record(
            "cycles",
            "testnet",
            {
                "timestamp": f"2026-08-0{day}T00:00:00Z",
                "symbol": "BTC/USDT",
                "sequence": day,
            },
        )

    records = repository.read_records(
        "cycles",
        "testnet",
        since=pd.Timestamp("2026-08-02T00:00:00Z").to_pydatetime(),
        until=pd.Timestamp("2026-08-05T00:00:00Z").to_pydatetime(),
        limit=2,
    )

    assert [record["sequence"] for record in records] == [3, 4]
