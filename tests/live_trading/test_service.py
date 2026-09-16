"""RL 訊號實盤輪次、風控與持倉狀態測試。"""

from __future__ import annotations

from decimal import Decimal
import tempfile
import unittest

from ai_quant_trading.live_trading.binance_client import BinancePrivateApiError
from ai_quant_trading.live_trading.config import LiveTradingConfig
from ai_quant_trading.live_trading.gateway import LiveTradingGateway, PortfolioSnapshot
from ai_quant_trading.live_trading.operations import OperationalRiskLimits
from ai_quant_trading.live_trading.service import (
    LiveSignal,
    reconcile_managed_spot_position,
    run_live_signal_cycle,
)
from ai_quant_trading.live_trading.storage import (
    SNAPSHOT_COLUMNS,
    LivePosition,
    append_csv_row,
    load_positions,
    live_trading_paths,
)
from ai_quant_trading.risk import RiskConfig


def make_signal(timestamp: str, target: float) -> LiveSignal:
    signal = 1 if target >= 0.7 else -1 if target <= 0.3 else 0
    return LiveSignal(
        timestamp=timestamp,
        symbol="BTC/USDT",
        target_fraction=target,
        action_signal=signal,
        close=50_500,
        atr=1_000,
    )


class FakeClient:
    def __init__(self):
        self.price = Decimal("50000")
        self.placed = []
        self.oco = None
        self.oco_cancelled = False
        self.partial_sell = False

    def sync_server_time(self):
        return 0

    def fetch_account(self):
        return {
            "canTrade": True,
            "balances": [
                {"asset": "BTC", "free": "1", "locked": "0"},
                {"asset": "USDT", "free": "1000", "locked": "0"},
            ],
        }

    def fetch_exchange_info(self, symbol):
        return {
            "symbol": "BTCUSDT",
            "status": "TRADING",
            "baseAsset": "BTC",
            "quoteAsset": "USDT",
            "filters": [
                {
                    "filterType": "LOT_SIZE",
                    "minQty": "0.00001",
                    "maxQty": "10",
                    "stepSize": "0.00001",
                },
                {"filterType": "MIN_NOTIONAL", "minNotional": "10"},
                {
                    "filterType": "PRICE_FILTER",
                    "minPrice": "0.01",
                    "maxPrice": "1000000",
                    "tickSize": "0.01",
                },
            ],
        }

    def fetch_price(self, symbol):
        return self.price

    def test_market_order(self, symbol, side, quantity, client_order_id):
        return {}

    def query_order(self, symbol, client_order_id):
        if self.oco and client_order_id in {
            self.oco["take_client_order_id"],
            self.oco["stop_client_order_id"],
        }:
            stop_filled = self.price <= self.oco["stop_price"]
            is_stop = client_order_id == self.oco["stop_client_order_id"]
            return {
                "status": "FILLED" if stop_filled and is_stop else "CANCELED",
                "executedQty": str(self.oco["quantity"] if stop_filled and is_stop else 0),
                "clientOrderId": client_order_id,
            }
        raise BinancePrivateApiError("not found", code=-2013)

    def place_market_order(self, symbol, side, quantity, client_order_id):
        self.placed.append(client_order_id)
        executed_quantity = (
            quantity / Decimal("2") if side == "SELL" and self.partial_sell else quantity
        )
        return {
            "status": (
                "PARTIALLY_FILLED" if side == "SELL" and self.partial_sell else "FILLED"
            ),
            "clientOrderId": client_order_id,
            "executedQty": str(executed_quantity),
            "cummulativeQuoteQty": str(executed_quantity * self.price),
        }

    def query_order_list(self, list_client_order_id):
        if self.oco is None or self.oco["list_client_order_id"] != list_client_order_id:
            raise BinancePrivateApiError("not found", code=-2013)
        all_done = self.oco_cancelled or self.price <= self.oco["stop_price"]
        return {
            "orderListId": 10,
            "listClientOrderId": list_client_order_id,
            "listOrderStatus": "ALL_DONE" if all_done else "EXECUTING",
            "symbol": "BTCUSDT",
            "orders": [
                {"symbol": "BTCUSDT", "clientOrderId": self.oco["take_client_order_id"]},
                {"symbol": "BTCUSDT", "clientOrderId": self.oco["stop_client_order_id"]},
            ],
        }

    def place_oco_sell_order(
        self,
        symbol,
        quantity,
        take_profit_price,
        stop_price,
        list_client_order_id,
        take_profit_client_order_id,
        stop_client_order_id,
    ):
        self.oco = {
            "quantity": quantity,
            "take_profit_price": take_profit_price,
            "stop_price": stop_price,
            "list_client_order_id": list_client_order_id,
            "take_client_order_id": take_profit_client_order_id,
            "stop_client_order_id": stop_client_order_id,
        }
        return {
            "orderListId": 10,
            "listClientOrderId": list_client_order_id,
            "listStatusType": "EXEC_STARTED",
            "listOrderStatus": "EXECUTING",
        }

    def cancel_order_list(self, symbol, list_client_order_id):
        self.oco_cancelled = True
        return {"listClientOrderId": list_client_order_id, "listOrderStatus": "ALL_DONE"}


class LiveSignalCycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = self.enterContext(tempfile.TemporaryDirectory())
        self.risk = RiskConfig(max_risk_per_trade=0.01, fixed_stop_loss_pct=0.03)

    def make_gateway(self, client, enabled=False, environment="testnet"):
        return LiveTradingGateway(
            client,
            LiveTradingConfig(
                environment=environment,
                trading_enabled=enabled,
                max_order_quote=100,
                max_balance_fraction=0.25,
            ),
        )

    def test_default_cycle_only_validates_order(self) -> None:
        client = FakeClient()
        result = run_live_signal_cycle(
            make_signal("2026-01-01", 0.8),
            model_dir="rl/ppo",
            gateway=self.make_gateway(client),
            storage_root=self.temp_dir,
            risk_config=self.risk,
        )

        self.assertTrue(result.submission.validated_only)
        self.assertEqual(len(client.placed), 0)
        self.assertFalse(result.paths.positions_json.exists())

    def test_execute_creates_position_and_duplicate_is_ignored(self) -> None:
        client = FakeClient()
        gateway = self.make_gateway(client, enabled=True)
        signal = make_signal("2026-01-01", 0.8)
        first = run_live_signal_cycle(
            signal,
            model_dir="rl/ppo",
            gateway=gateway,
            storage_root=self.temp_dir,
            risk_config=self.risk,
            execute=True,
            confirmation="EXECUTE TESTNET",
        )
        duplicate = run_live_signal_cycle(
            signal,
            model_dir="rl/ppo",
            gateway=gateway,
            storage_root=self.temp_dir,
            risk_config=self.risk,
            execute=True,
            confirmation="EXECUTE TESTNET",
        )

        positions = load_positions(first.paths)
        self.assertIn("BTC/USDT", positions)
        self.assertEqual(positions["BTC/USDT"].protection_status, "active")
        self.assertIsNotNone(client.oco)
        self.assertEqual(len(client.placed), 1)
        self.assertFalse(duplicate.processed)
        self.assertEqual(duplicate.reason, "duplicate")

    def test_sell_signal_without_managed_position_does_not_sell_account_assets(self) -> None:
        client = FakeClient()
        result = run_live_signal_cycle(
            make_signal("2026-01-01", 0.2),
            model_dir="rl/ppo",
            gateway=self.make_gateway(client, enabled=True),
            storage_root=self.temp_dir,
            risk_config=self.risk,
            execute=True,
            confirmation="EXECUTE TESTNET",
        )

        self.assertEqual(result.reason, "no_position")
        self.assertEqual(len(client.placed), 0)

    def test_stop_loss_closes_managed_position(self) -> None:
        client = FakeClient()
        gateway = self.make_gateway(client, enabled=True)
        first = run_live_signal_cycle(
            make_signal("2026-01-01", 0.8),
            model_dir="rl/ppo",
            gateway=gateway,
            storage_root=self.temp_dir,
            risk_config=self.risk,
            execute=True,
            confirmation="EXECUTE TESTNET",
        )
        client.price = Decimal("48000")
        stopped = run_live_signal_cycle(
            make_signal("2026-01-02", 0.8),
            model_dir="rl/ppo",
            gateway=gateway,
            storage_root=self.temp_dir,
            risk_config=self.risk,
            execute=True,
            confirmation="EXECUTE TESTNET",
        )

        self.assertEqual(stopped.reason, "exchange_protection_filled")
        self.assertIsNone(stopped.preview)
        self.assertEqual(len(client.placed), 1)
        self.assertEqual(load_positions(first.paths), {})

    def test_partial_sell_updates_remaining_local_position(self) -> None:
        client = FakeClient()
        gateway = self.make_gateway(client, enabled=True)
        opened = run_live_signal_cycle(
            make_signal("2026-01-01", 0.8),
            model_dir="rl/sac",
            gateway=gateway,
            storage_root=self.temp_dir,
            risk_config=self.risk,
            execute=True,
            confirmation="EXECUTE TESTNET",
        )
        original = load_positions(opened.paths)["BTC/USDT"]
        client.partial_sell = True

        run_live_signal_cycle(
            make_signal("2026-01-02", 0.2),
            model_dir="rl/sac",
            gateway=gateway,
            storage_root=self.temp_dir,
            risk_config=self.risk,
            execute=True,
            confirmation="EXECUTE TESTNET",
        )

        remaining = load_positions(opened.paths)["BTC/USDT"]
        self.assertEqual(remaining.quantity, original.quantity / 2)
        self.assertEqual(remaining.protection_status, "pending")

    def test_position_reconciliation_detects_exchange_shortage(self) -> None:
        snapshot = PortfolioSnapshot(
            "BTC/USDT",
            "BTC",
            "USDT",
            Decimal("0.004"),
            Decimal("0.001"),
            Decimal("1000"),
            Decimal("0"),
            Decimal("50000"),
            True,
        )
        position = LivePosition(
            symbol="BTC/USDT",
            quantity=0.01,
            entry_price=50_000,
            stop_loss=48_000,
            take_profit=52_000,
            entry_order_id="aq-test",
            opened_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
        )

        result = reconcile_managed_spot_position(snapshot, position)

        self.assertFalse(result.consistent)
        self.assertAlmostEqual(result.shortage, 0.005)

    def test_live_execute_requires_rl_quality_callback(self) -> None:
        with self.assertRaisesRegex(ValueError, "缺少 RL 品質檢查"):
            run_live_signal_cycle(
                make_signal("2026-01-01", 0.8),
                model_dir="rl/ppo",
                gateway=self.make_gateway(FakeClient(), enabled=True, environment="live"),
                storage_root=self.temp_dir,
                risk_config=self.risk,
                execute=True,
                confirmation="EXECUTE TESTNET",
            )

    def test_daily_loss_limit_blocks_new_position_and_activates_halt(self) -> None:
        paths = live_trading_paths(self.temp_dir, "testnet")
        append_csv_row(
            paths.account_snapshots_csv,
            SNAPSHOT_COLUMNS,
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "estimated_equity": 60_000,
            },
        )
        client = FakeClient()

        result = run_live_signal_cycle(
            make_signal("2026-01-01T01:00:00Z", 0.8),
            model_dir="rl/sac",
            gateway=self.make_gateway(client, enabled=True),
            storage_root=self.temp_dir,
            risk_config=self.risk,
            execute=True,
            confirmation="EXECUTE TESTNET",
            operational_limits=OperationalRiskLimits(
                maximum_daily_loss=0.01,
                maximum_drawdown=0.08,
            ),
        )

        self.assertEqual(result.reason, "operational_risk_halt")
        self.assertEqual(client.placed, [])
        self.assertTrue(paths.emergency_halt_json.is_file())


if __name__ == "__main__":
    unittest.main()
