"""USD-M 多空、逐倉槓桿與保護單閘門測試。"""

from decimal import Decimal

import pytest

from ai_quant_trading.live_trading.binance_client import BinancePrivateApiError
from ai_quant_trading.live_trading.config import LiveTradingConfig
from ai_quant_trading.live_trading.futures_gateway import FuturesTradingGateway


class FakeFuturesClient:
    def __init__(self, position: str = "0"):
        self.position = Decimal(position)
        self.orders: dict[str, dict] = {}
        self.algos: dict[str, dict] = {}
        self.tested: list[tuple] = []
        self.placed: list[tuple] = []
        self.cancelled_algos: list[str] = []
        self.hedge_mode = False

    def sync_server_time(self):
        return 0

    def fetch_account(self):
        return {
            "canTrade": True,
            "totalMarginBalance": "1000",
            "availableBalance": "900",
            "assets": [
                {
                    "asset": "USDT",
                    "walletBalance": "1000",
                    "marginBalance": "1000",
                    "availableBalance": "900",
                }
            ],
        }

    def fetch_position_risk(self, symbol):
        return [
            {
                "symbol": "BTCUSDT",
                "positionSide": "BOTH",
                "positionAmt": str(self.position),
                "entryPrice": "50000" if self.position else "0",
                "markPrice": "50000",
                "unRealizedProfit": "0",
                "liquidationPrice": "30000" if self.position else "0",
                "leverage": "2",
                "marginType": "isolated",
            }
        ]

    def fetch_exchange_info(self, symbol):
        return {
            "symbol": "BTCUSDT",
            "status": "TRADING",
            "baseAsset": "BTC",
            "quoteAsset": "USDT",
            "filters": [
                {
                    "filterType": "MARKET_LOT_SIZE",
                    "minQty": "0.0001",
                    "maxQty": "100",
                    "stepSize": "0.0001",
                },
                {"filterType": "MIN_NOTIONAL", "notional": "5"},
                {
                    "filterType": "PRICE_FILTER",
                    "minPrice": "1",
                    "maxPrice": "1000000",
                    "tickSize": "0.1",
                },
            ],
        }

    def fetch_price(self, symbol):
        return Decimal("50000")

    def fetch_position_mode(self):
        return {"dualSidePosition": self.hedge_mode}

    def fetch_multi_asset_mode(self):
        return {"multiAssetsMargin": False}

    def change_margin_type(self, symbol, margin_type):
        raise BinancePrivateApiError("already isolated", -4046)

    def change_leverage(self, symbol, leverage):
        return {"symbol": "BTCUSDT", "leverage": leverage}

    def test_market_order(self, symbol, side, quantity, client_order_id, *, reduce_only=False):
        self.tested.append((symbol, side, quantity, reduce_only))
        return {}

    def query_order(self, symbol, client_order_id):
        if client_order_id in self.orders:
            return self.orders[client_order_id]
        raise BinancePrivateApiError("not found", -2013)

    def place_market_order(self, symbol, side, quantity, client_order_id, *, reduce_only=False):
        signed = quantity if side == "BUY" else -quantity
        if reduce_only:
            if self.position > 0:
                self.position = max(self.position + signed, Decimal("0"))
            elif self.position < 0:
                self.position = min(self.position + signed, Decimal("0"))
        else:
            self.position += signed
        response = {
            "status": "FILLED",
            "executedQty": str(quantity),
            "avgPrice": "50000",
            "cumQuote": str(quantity * Decimal("50000")),
            "clientOrderId": client_order_id,
        }
        self.orders[client_order_id] = response
        self.placed.append((side, quantity, reduce_only))
        return response

    def query_algo_order(self, client_id):
        if client_id in self.algos:
            return self.algos[client_id]
        raise BinancePrivateApiError("not found", -2013)

    def place_close_position_algo_order(
        self, symbol, side, order_type, trigger_price, client_id
    ):
        response = {
            "algoStatus": "NEW",
            "clientAlgoId": client_id,
            "side": side,
            "type": order_type,
            "triggerPrice": str(trigger_price),
        }
        self.algos[client_id] = response
        return response

    def cancel_algo_order(self, client_id):
        self.cancelled_algos.append(client_id)
        return {"algoStatus": "CANCELED", "clientAlgoId": client_id}


def make_gateway(client, *, enabled=False, protected=True):
    return FuturesTradingGateway(
        client,
        LiveTradingConfig(
            market_type="usd_m_futures",
            trading_enabled=enabled,
            max_order_quote=1000,
            max_balance_fraction=1.0,
            exchange_protection_enabled=protected,
            leverage=2,
        ),
    )


def test_flat_account_can_preview_short_and_short_close_is_reduce_only() -> None:
    client = FakeFuturesClient()
    gateway = make_gateway(client)

    opening = gateway.preview_market_order("BTC/USDT", "SELL", quote_amount=100)
    assert opening.market_type == "usd_m_futures"
    assert not opening.reduce_only
    assert opening.quantity == Decimal("0.002")

    client.position = Decimal("-0.002")
    closing = gateway.preview_market_order(
        "BTC/USDT", "BUY", full_exit=True, reduce_only=True
    )
    assert closing.reduce_only
    assert closing.quantity == Decimal("0.002")


def test_direct_reversal_is_rejected() -> None:
    gateway = make_gateway(FakeFuturesClient("0.002"))
    with pytest.raises(ValueError, match="先用 reduce-only 平倉"):
        gateway.preview_market_order("BTC/USDT", "SELL", quote_amount=100)


def test_hedge_mode_is_rejected_before_order_validation() -> None:
    client = FakeFuturesClient()
    client.hedge_mode = True
    gateway = make_gateway(client)
    preview = gateway.preview_market_order("BTC/USDT", "BUY", quote_amount=100)
    with pytest.raises(ValueError, match="One-way Mode"):
        gateway.validate_order(preview)


def test_short_protection_uses_buy_exit_with_prices_on_correct_sides() -> None:
    client = FakeFuturesClient("-0.002")
    gateway = make_gateway(client, enabled=True)

    result = gateway.submit_position_protection(
        symbol="BTC/USDT",
        position_side="SHORT",
        stop_price=51000,
        take_profit_price=49000,
        confirmation="EXECUTE TESTNET",
        seed="short-entry",
    )

    assert result.stop_response["side"] == "BUY"
    assert result.take_profit_response["side"] == "BUY"
    assert result.stop_response["type"] == "STOP_MARKET"
    assert result.take_profit_response["type"] == "TAKE_PROFIT_MARKET"
