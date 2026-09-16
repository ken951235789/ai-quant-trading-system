"""Binance USD-M 永續合約私有 REST API 客戶端。"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from ai_quant_trading.data_collection.binance import to_binance_symbol
from ai_quant_trading.live_trading.binance_client import BinancePrivateClient


class BinanceFuturesPrivateClient(BinancePrivateClient):
    """封裝帳戶、持倉、槓桿、訂單與交易所條件保護單。"""

    def ping(self) -> bool:
        self._request("GET", "/fapi/v1/ping")
        return True

    def sync_server_time(self) -> int:
        payload = self._request("GET", "/fapi/v1/time")
        server_time = int(payload["serverTime"])
        self._server_offset_ms = server_time - self.time_provider()
        return self._server_offset_ms

    def fetch_account(self) -> dict[str, Any]:
        return dict(self._request("GET", "/fapi/v3/account", signed=True))

    def fetch_balances(self) -> list[dict[str, Any]]:
        return list(self._request("GET", "/fapi/v3/balance", signed=True))

    def fetch_position_risk(self, symbol: str) -> list[dict[str, Any]]:
        payload = self._request(
            "GET",
            "/fapi/v3/positionRisk",
            {"symbol": to_binance_symbol(symbol)},
            signed=True,
        )
        return list(payload)

    def fetch_position_mode(self) -> dict[str, Any]:
        return dict(self._request("GET", "/fapi/v1/positionSide/dual", signed=True))

    def fetch_multi_asset_mode(self) -> dict[str, Any]:
        return dict(self._request("GET", "/fapi/v1/multiAssetsMargin", signed=True))

    def fetch_exchange_info(self, symbol: str) -> dict[str, Any]:
        payload = self._request(
            "GET",
            "/fapi/v1/exchangeInfo",
            {"symbol": to_binance_symbol(symbol)},
        )
        symbols = list(payload.get("symbols", []))
        if not symbols:
            from ai_quant_trading.live_trading.binance_client import BinancePrivateApiError

            raise BinancePrivateApiError(f"交易所找不到 USD-M 永續標的：{symbol}")
        return dict(symbols[0])

    def fetch_price(self, symbol: str) -> Decimal:
        payload = self._request(
            "GET",
            "/fapi/v1/ticker/price",
            {"symbol": to_binance_symbol(symbol)},
        )
        return Decimal(str(payload["price"]))

    def change_leverage(self, symbol: str, leverage: int) -> dict[str, Any]:
        return dict(
            self._request(
                "POST",
                "/fapi/v1/leverage",
                {"symbol": to_binance_symbol(symbol), "leverage": leverage},
                signed=True,
            )
        )

    def change_margin_type(self, symbol: str, margin_type: str) -> dict[str, Any]:
        return dict(
            self._request(
                "POST",
                "/fapi/v1/marginType",
                {
                    "symbol": to_binance_symbol(symbol),
                    "marginType": margin_type.upper(),
                },
                signed=True,
            )
        )

    def test_market_order(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        client_order_id: str,
        *,
        reduce_only: bool = False,
    ) -> dict[str, Any]:
        return dict(
            self._request(
                "POST",
                "/fapi/v1/order/test",
                {
                    "symbol": to_binance_symbol(symbol),
                    "side": side.upper(),
                    "positionSide": "BOTH",
                    "type": "MARKET",
                    "quantity": quantity,
                    "reduceOnly": reduce_only,
                    "newClientOrderId": client_order_id,
                },
                signed=True,
            )
        )

    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        client_order_id: str,
        *,
        reduce_only: bool = False,
    ) -> dict[str, Any]:
        return dict(
            self._request(
                "POST",
                "/fapi/v1/order",
                {
                    "symbol": to_binance_symbol(symbol),
                    "side": side.upper(),
                    "positionSide": "BOTH",
                    "type": "MARKET",
                    "quantity": quantity,
                    "reduceOnly": reduce_only,
                    "newClientOrderId": client_order_id,
                    "newOrderRespType": "RESULT",
                },
                signed=True,
                order_request=True,
            )
        )

    def query_order(self, symbol: str, client_order_id: str) -> dict[str, Any]:
        return dict(
            self._request(
                "GET",
                "/fapi/v1/order",
                {
                    "symbol": to_binance_symbol(symbol),
                    "origClientOrderId": client_order_id,
                },
                signed=True,
            )
        )

    def cancel_order(self, symbol: str, client_order_id: str) -> dict[str, Any]:
        return dict(
            self._request(
                "DELETE",
                "/fapi/v1/order",
                {
                    "symbol": to_binance_symbol(symbol),
                    "origClientOrderId": client_order_id,
                },
                signed=True,
                order_request=True,
            )
        )

    def fetch_open_orders(self, symbol: str) -> list[dict[str, Any]]:
        return list(
            self._request(
                "GET",
                "/fapi/v1/openOrders",
                {"symbol": to_binance_symbol(symbol)},
                signed=True,
            )
        )

    def fetch_user_trades(self, symbol: str, *, limit: int = 100) -> list[dict[str, Any]]:
        return list(
            self._request(
                "GET",
                "/fapi/v1/userTrades",
                {"symbol": to_binance_symbol(symbol), "limit": limit},
                signed=True,
            )
        )

    def place_close_position_algo_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        trigger_price: Decimal,
        client_algo_id: str,
    ) -> dict[str, Any]:
        """以 Mark Price 觸發平掉整個單向持倉的停損或停利單。"""
        return dict(
            self._request(
                "POST",
                "/fapi/v1/algoOrder",
                {
                    "algoType": "CONDITIONAL",
                    "symbol": to_binance_symbol(symbol),
                    "side": side.upper(),
                    "positionSide": "BOTH",
                    "type": order_type.upper(),
                    "triggerPrice": trigger_price,
                    "workingType": "MARK_PRICE",
                    "closePosition": True,
                    "priceProtect": True,
                    "clientAlgoId": client_algo_id,
                },
                signed=True,
                order_request=True,
            )
        )

    def query_algo_order(self, client_algo_id: str) -> dict[str, Any]:
        return dict(
            self._request(
                "GET",
                "/fapi/v1/algoOrder",
                {"clientAlgoId": client_algo_id},
                signed=True,
            )
        )

    def cancel_algo_order(self, client_algo_id: str) -> dict[str, Any]:
        return dict(
            self._request(
                "DELETE",
                "/fapi/v1/algoOrder",
                {"clientAlgoId": client_algo_id},
                signed=True,
                order_request=True,
            )
        )

    def fetch_open_algo_orders(self, symbol: str) -> list[dict[str, Any]]:
        return list(
            self._request(
                "GET",
                "/fapi/v1/openAlgoOrders",
                {"symbol": to_binance_symbol(symbol)},
                signed=True,
            )
        )

    def create_listen_key(self) -> str:
        payload = self._request("POST", "/fapi/v1/listenKey", api_key=True)
        return str(payload["listenKey"])

    def keepalive_listen_key(self) -> None:
        self._request("PUT", "/fapi/v1/listenKey", api_key=True)

    def close_listen_key(self) -> None:
        self._request("DELETE", "/fapi/v1/listenKey", api_key=True)
