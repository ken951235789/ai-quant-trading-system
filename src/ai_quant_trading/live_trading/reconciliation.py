"""Binance USD-M 啟動、重連與週期性狀態對帳。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal
from ai_quant_trading.database.repository import PostgresTradingRepository
from ai_quant_trading.live_trading.futures_gateway import FuturesTradingGateway
from ai_quant_trading.live_trading.storage import (
    activate_emergency_halt,
    load_positions,
    LiveTradingPaths,
)


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    """一次本機、資料庫與交易所對帳摘要。"""

    consistent: bool
    reason: str
    local_quantity: Decimal
    exchange_quantity: Decimal
    open_order_count: int
    recent_fill_count: int
    open_protection_count: int


def reconcile_futures_state(
    gateway: FuturesTradingGateway,
    repository: PostgresTradingRepository,
    paths: LiveTradingPaths,
    *,
    symbol: str = "BTC/USDT",
) -> ReconciliationResult:
    """以 REST 強制核對部位、一般單、近期成交與交易所保護單。"""
    snapshot = gateway.portfolio(symbol)
    positions = load_positions(paths)
    local = positions.get(f"usd_m_futures:{snapshot.symbol}")
    local_quantity = Decimal(str(local.signed_quantity if local else 0))
    exchange_quantity = snapshot.position_quantity
    open_orders = gateway.client.fetch_open_orders(symbol)
    recent_fills = gateway.client.fetch_user_trades(symbol, limit=100)
    open_protections = gateway.client.fetch_open_algo_orders(symbol)

    tolerance = max(abs(local_quantity) * Decimal("0.001"), Decimal("0.00000001"))
    quantity_matches = abs(local_quantity - exchange_quantity) <= tolerance
    regular_orders_clear = len(open_orders) == 0
    protection_matches = True
    expected_ids: set[str] = set()
    if local is not None:
        expected_ids = {
            value
            for value in (local.stop_client_algo_id, local.take_profit_client_algo_id)
            if value
        }
    open_ids = {
        str(row.get("clientAlgoId") or row.get("clientOrderId") or "")
        for row in open_protections
    }
    if exchange_quantity != 0 and gateway.config.exchange_protection_enabled:
        protection_matches = len(expected_ids) == 2 and open_ids == expected_ids
    elif exchange_quantity != 0:
        protection_matches = not open_ids
    if exchange_quantity == 0:
        protection_matches = local is None and not open_ids

    problems: list[str] = []
    if not quantity_matches:
        problems.append("本機管理部位與 Binance 部位不一致")
    if not regular_orders_clear:
        problems.append("Binance 存在未完成的一般委託，禁止在狀態不明時繼續交易")
    if not protection_matches:
        problems.append("持倉保護單不完整、存在未知保護單或平倉後本機狀態未清除")
    consistent = not problems
    reason = "對帳一致" if consistent else "；".join(problems)
    result = ReconciliationResult(
        consistent,
        reason,
        local_quantity,
        exchange_quantity,
        len(open_orders),
        len(recent_fills),
        len(open_protections),
    )
    repository.record_reconciliation(
        gateway.config.environment,
        snapshot.symbol,
        consistent=consistent,
        local_quantity=local_quantity,
        exchange_quantity=exchange_quantity,
        reason=reason,
        payload={
            **asdict(result),
            "open_order_client_ids": [
                str(row.get("clientOrderId") or "") for row in open_orders
            ],
            "recent_trade_ids": [str(row.get("id") or "") for row in recent_fills],
            "open_protection_ids": sorted(open_ids),
        },
    )
    if not consistent:
        activate_emergency_halt(paths, f"交易所狀態對帳失敗：{reason}")
    return result
