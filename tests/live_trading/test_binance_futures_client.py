"""Binance USD-M 私有 API 路徑與下單參數測試。"""

from decimal import Decimal
from urllib.parse import parse_qs

from ai_quant_trading.live_trading.binance_futures_client import (
    BinanceFuturesPrivateClient,
)
from ai_quant_trading.live_trading.credentials import BinanceCredentials
from tests.live_trading.test_binance_client import FakeResponse, FakeSession


def make_client(responses):
    session = FakeSession(responses)
    client = BinanceFuturesPrivateClient(
        BinanceCredentials("key", "secret"),
        "https://demo-fapi.binance.com",
        session=session,
        time_provider=lambda: 1_700_000_000_000,
    )
    return client, session


def test_market_order_uses_fapi_and_reduce_only() -> None:
    client, session = make_client([FakeResponse({"status": "FILLED"})])

    client.place_market_order(
        "BTC/USDT",
        "BUY",
        Decimal("0.002"),
        "aq-close-short",
        reduce_only=True,
    )

    call = session.calls[0]
    values = parse_qs(call["params"])
    assert call["url"].endswith("/fapi/v1/order")
    assert values["symbol"] == ["BTCUSDT"]
    assert values["positionSide"] == ["BOTH"]
    assert values["reduceOnly"] == ["true"]
    assert values["newOrderRespType"] == ["RESULT"]


def test_protection_uses_mark_price_close_position_algo_order() -> None:
    client, session = make_client([FakeResponse({"algoStatus": "NEW"})])

    client.place_close_position_algo_order(
        "BTC/USDT",
        "SELL",
        "STOP_MARKET",
        Decimal("49000"),
        "aq-stop-long",
    )

    call = session.calls[0]
    values = parse_qs(call["params"])
    assert call["url"].endswith("/fapi/v1/algoOrder")
    assert values["algoType"] == ["CONDITIONAL"]
    assert values["workingType"] == ["MARK_PRICE"]
    assert values["closePosition"] == ["true"]
    assert values["priceProtect"] == ["true"]


def test_user_stream_listen_key_uses_api_key_without_signature() -> None:
    client, session = make_client([FakeResponse({"listenKey": "stream-key"})])

    result = client.create_listen_key()

    call = session.calls[0]
    values = parse_qs(call["params"])
    assert result == "stream-key"
    assert call["url"].endswith("/fapi/v1/listenKey")
    assert call["headers"]["X-MBX-APIKEY"] == "key"
    assert "signature" not in values
