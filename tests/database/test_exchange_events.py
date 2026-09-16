from datetime import datetime, timezone

from sqlalchemy import create_engine, select

from ai_quant_trading.database.models import Base, ExchangeEvent, Fill, Order
from ai_quant_trading.database.repository import PostgresTradingRepository


def _repository() -> PostgresTradingRepository:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    return PostgresTradingRepository(engine)


def _order_event(status: str, event_time: int, cumulative: str, last: str, trade_id: int):
    return {
        "e": "ORDER_TRADE_UPDATE",
        "E": event_time,
        "T": event_time,
        "o": {
            "s": "BTCUSDT",
            "c": "aqt-test-order",
            "S": "BUY",
            "ps": "BOTH",
            "R": False,
            "q": "0.010",
            "p": "0",
            "ap": "50000",
            "x": "TRADE" if trade_id else "NEW",
            "X": status,
            "i": 123,
            "l": last,
            "z": cumulative,
            "L": "50000",
            "t": trade_id,
            "n": "0.01",
            "N": "USDT",
        },
    }


def test_event_is_append_only_idempotent_and_projects_partial_fills() -> None:
    repository = _repository()
    first = _order_event("PARTIALLY_FILLED", 1_780_000_000_000, "0.004", "0.004", 91)
    inserted, key = repository.append_exchange_event("testnet", "usd_m_user_data", first)
    assert inserted
    assert repository.project_exchange_event(key)
    assert repository.append_exchange_event("testnet", "usd_m_user_data", first) == (False, key)

    second = _order_event("FILLED", 1_780_000_001_000, "0.010", "0.006", 92)
    _, second_key = repository.append_exchange_event("testnet", "usd_m_user_data", second)
    repository.project_exchange_event(second_key)

    with repository._session_factory() as session:
        assert session.scalar(select(Order)).status == "FILLED"
        assert len(session.scalars(select(Fill)).all()) == 2
        assert len(session.scalars(select(ExchangeEvent)).all()) == 2


def test_older_order_event_cannot_regress_terminal_state() -> None:
    repository = _repository()
    filled = _order_event("FILLED", 1_780_000_001_000, "0.010", "0.010", 92)
    _, filled_key = repository.append_exchange_event("testnet", "usd_m_user_data", filled)
    repository.project_exchange_event(filled_key)
    stale = _order_event("NEW", 1_780_000_000_000, "0", "0", 0)
    _, stale_key = repository.append_exchange_event("testnet", "usd_m_user_data", stale)
    repository.project_exchange_event(stale_key)
    with repository._session_factory() as session:
        assert session.scalar(select(Order)).status == "FILLED"


def test_stream_checkpoint_never_requires_listen_key() -> None:
    repository = _repository()
    now = datetime.now(timezone.utc)
    repository.update_stream_checkpoint(
        "testnet",
        "usd_m_user_data",
        state="connected",
        heartbeat_at=now,
        details={"schema_version": 1},
    )
    checkpoint = repository.stream_checkpoint("testnet", "usd_m_user_data")
    assert checkpoint is not None
    assert checkpoint["state"] == "connected"
    assert "listen" not in str(checkpoint).lower()


def test_startup_replays_event_committed_before_projection() -> None:
    repository = _repository()
    payload = _order_event("FILLED", 1_780_000_001_000, "0.010", "0.010", 92)
    inserted, event_key = repository.append_exchange_event(
        "testnet",
        "usd_m_user_data",
        payload,
    )

    assert inserted
    assert repository.replay_unprocessed_exchange_events() == 1
    assert repository.replay_unprocessed_exchange_events() == 0
    with repository._session_factory() as session:
        event = session.scalar(select(ExchangeEvent).where(ExchangeEvent.event_key == event_key))
        assert event is not None
        assert event.processed_at is not None
        assert session.scalar(select(Order)).status == "FILLED"
        assert len(session.scalars(select(Fill)).all()) == 1
