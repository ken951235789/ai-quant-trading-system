"""既有實盤儲存 API 切換 PostgreSQL 的整合測試。"""

from __future__ import annotations

from dataclasses import asdict

import pytest

from ai_quant_trading.database import repository as repository_module
from ai_quant_trading.live_trading.storage import (
    ORDER_COLUMNS,
    LivePosition,
    append_csv_row,
    live_trading_paths,
    load_positions,
    read_live_csv,
    save_positions,
)


class _RepositoryStub:
    def __init__(self) -> None:
        self.records: dict[tuple[str, str], list[dict[str, object]]] = {}
        self.positions: dict[str, dict[str, dict[str, object]]] = {}

    def append_record(self, kind: str, environment: str, row: dict[str, object]) -> bool:
        self.records.setdefault((kind, environment), []).append(dict(row))
        return True

    def read_records(
        self,
        kind: str,
        environment: str,
        *,
        limit: int | None = None,
        since=None,
        until=None,
    ) -> list[dict[str, object]]:
        records = self.records.get((kind, environment), [])
        return records[-limit:] if limit is not None else records

    def save_positions(self, environment: str, positions: dict[str, dict[str, object]]) -> None:
        self.positions[environment] = positions

    def load_positions(self, environment: str) -> dict[str, dict[str, object]]:
        return self.positions.get(environment, {})


def test_live_storage_routes_to_postgres_without_writing_csv(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = _RepositoryStub()
    monkeypatch.setenv("AI_QUANT_STORAGE_BACKEND", "postgres")
    monkeypatch.setenv(
        "AI_QUANT_DATABASE_URL",
        "postgresql+psycopg://robot:secret@127.0.0.1:5432/ai_quant",
    )
    monkeypatch.setattr(
        repository_module,
        "repository_from_settings",
        lambda settings: stub,
    )
    paths = live_trading_paths(tmp_path, "testnet")
    order = {
        "timestamp": "2026-08-28T00:00:00Z",
        "environment": "testnet",
        "market_type": "usd_m_futures",
        "symbol": "BTC/USDT",
        "status": "NEW",
    }

    append_csv_row(paths.orders_csv, ORDER_COLUMNS, order)

    assert not paths.orders_csv.exists()
    frame = read_live_csv(paths.orders_csv, ORDER_COLUMNS)
    assert len(frame) == 1
    assert frame.iloc[0]["symbol"] == "BTC/USDT"

    position = LivePosition(
        symbol="BTC/USDT",
        quantity=0.001,
        entry_price=60_000,
        stop_loss=59_000,
        take_profit=61_000,
        entry_order_id="1001",
        opened_at="2026-08-28T00:00:00Z",
        updated_at="2026-08-28T00:00:00Z",
        market_type="usd_m_futures",
        side="LONG",
        leverage=2,
        margin_type="ISOLATED",
    )
    save_positions(paths, {"usd_m_futures:BTC/USDT": position})

    loaded = load_positions(paths)
    assert asdict(loaded["usd_m_futures:BTC/USDT"]) == asdict(position)
    assert not paths.positions_json.exists()
