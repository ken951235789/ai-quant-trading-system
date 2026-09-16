"""私有事件流連線健康與持久化投影重試測試。"""

from __future__ import annotations

import json

import pytest

from ai_quant_trading.live_trading.user_stream import (
    BinanceFuturesUserStream,
    UserStreamConfig,
)


class FakeSocket:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeRepository:
    def __init__(self, *, fail_first_projection: bool = False) -> None:
        self.fail_first_projection = fail_first_projection
        self.project_calls = 0
        self.inserted = False
        self.checkpoints: list[dict[str, object]] = []
        self.replay_results: list[int] = [0]

    def update_stream_checkpoint(self, environment, stream_name, **payload) -> None:
        self.checkpoints.append({"environment": environment, "stream": stream_name, **payload})

    def replay_unprocessed_exchange_events(self, *, limit: int = 1_000) -> int:
        return self.replay_results.pop(0) if self.replay_results else 0

    def append_exchange_event(self, environment, stream_name, payload):
        inserted = not self.inserted
        self.inserted = True
        return inserted, "event-1"

    def project_exchange_event(self, event_key: str) -> bool:
        self.project_calls += 1
        if self.fail_first_projection and self.project_calls == 1:
            raise RuntimeError("temporary database failure")
        return True


def test_duplicate_delivery_retries_an_unprocessed_event() -> None:
    repository = FakeRepository(fail_first_projection=True)
    stream = BinanceFuturesUserStream(
        object(),
        repository,
        UserStreamConfig(environment="testnet"),
    )
    stream.connection_ready = True
    stream.projection_ready = True
    socket = FakeSocket()
    message = json.dumps({"e": "ORDER_TRADE_UPDATE", "E": 1, "o": {"i": 1}})

    with pytest.raises(RuntimeError, match="temporary database failure"):
        stream._on_message(socket, message)
    stream._on_pong(socket, None)
    assert repository.checkpoints[-1]["state"] == "degraded"

    stream.connection_ready = True
    stream.projection_ready = True
    stream._on_message(socket, message)

    assert repository.project_calls == 2
    assert repository.checkpoints[-1]["state"] == "connected"


def test_open_drains_all_replay_batches_before_connected() -> None:
    repository = FakeRepository()
    repository.replay_results = [2, 2, 0]
    stream = BinanceFuturesUserStream(
        object(),
        repository,
        UserStreamConfig(
            environment="testnet",
            replay_batch_size=2,
            replay_max_batches=4,
        ),
    )
    socket = FakeSocket()

    stream._on_open(socket)

    assert stream.connection_ready
    assert stream.projection_ready
    assert repository.checkpoints[-1]["state"] == "connected"


def test_projection_and_checkpoint_failure_still_closes_socket() -> None:
    class FailingRepository(FakeRepository):
        def update_stream_checkpoint(self, environment, stream_name, **payload) -> None:
            raise RuntimeError("database unavailable")

    repository = FailingRepository(fail_first_projection=True)
    stream = BinanceFuturesUserStream(
        object(),
        repository,
        UserStreamConfig(environment="testnet"),
    )
    stream.connection_ready = True
    stream.projection_ready = True
    socket = FakeSocket()

    with pytest.raises(RuntimeError, match="temporary database failure"):
        stream._on_message(
            socket,
            json.dumps({"e": "ORDER_TRADE_UPDATE", "E": 1, "o": {"i": 1}}),
        )

    assert socket.closed
    assert not stream.connection_ready
    assert not stream.projection_ready
