"""期貨實盤對帳的 fail-closed 行為。"""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

from ai_quant_trading.live_trading.reconciliation import reconcile_futures_state


class _Repository:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def record_reconciliation(self, *args, **kwargs) -> None:
        self.records.append({"args": args, "kwargs": kwargs})


class _Client:
    def __init__(self, *, orders=None, protections=None) -> None:
        self.orders = list(orders or [])
        self.protections = list(protections or [])

    def fetch_open_orders(self, _symbol):
        return self.orders

    def fetch_user_trades(self, _symbol, limit=100):
        return []

    def fetch_open_algo_orders(self, _symbol):
        return self.protections


class _Gateway:
    def __init__(self, client: _Client, quantity: str = "0") -> None:
        self.client = client
        self.config = SimpleNamespace(environment="testnet", exchange_protection_enabled=True)
        self.quantity = Decimal(quantity)

    def portfolio(self, _symbol):
        return SimpleNamespace(symbol="BTC/USDT", position_quantity=self.quantity)


def test_reconciliation_rejects_orphan_regular_order(monkeypatch) -> None:
    gateway = _Gateway(_Client(orders=[{"clientOrderId": "unknown"}]))
    repository = _Repository()
    halted: list[str] = []
    monkeypatch.setattr(
        "ai_quant_trading.live_trading.reconciliation.load_positions",
        lambda _paths: {},
    )
    monkeypatch.setattr(
        "ai_quant_trading.live_trading.reconciliation.activate_emergency_halt",
        lambda _paths, reason: halted.append(reason),
    )

    result = reconcile_futures_state(gateway, repository, SimpleNamespace())

    assert not result.consistent
    assert "未完成的一般委託" in result.reason
    assert halted


def test_reconciliation_rejects_orphan_protection_when_flat(monkeypatch) -> None:
    gateway = _Gateway(_Client(protections=[{"clientAlgoId": "aqs-orphan"}]))
    repository = _Repository()
    monkeypatch.setattr(
        "ai_quant_trading.live_trading.reconciliation.load_positions",
        lambda _paths: {},
    )
    monkeypatch.setattr(
        "ai_quant_trading.live_trading.reconciliation.activate_emergency_halt",
        lambda _paths, _reason: None,
    )

    result = reconcile_futures_state(gateway, repository, SimpleNamespace())

    assert not result.consistent
    assert "未知保護單" in result.reason
