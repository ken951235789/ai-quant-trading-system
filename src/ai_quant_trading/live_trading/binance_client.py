"""Binance Spot 私有 REST API 與 HMAC 簽章客戶端。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
import time
from typing import Any
from urllib.parse import urlencode, urlsplit

import requests

from ai_quant_trading.data_collection.binance import to_binance_symbol
from ai_quant_trading.data_collection.http_client import create_secure_session, require_session
from ai_quant_trading.live_trading.credentials import BinanceCredentials


class BinancePrivateApiError(RuntimeError):
    """Binance 私有 API 明確拒絕請求。"""

    def __init__(self, message: str, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code


class BinanceExecutionUnknown(RuntimeError):
    """送單逾時或伺服器錯誤，成交狀態目前未知。"""

    def __init__(self, message: str, client_order_id: str | None = None) -> None:
        super().__init__(message)
        self.client_order_id = client_order_id


def _milliseconds_now() -> int:
    return int(time.time() * 1000)


def _api_value(value: object) -> object:
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


@dataclass(slots=True)
class BinancePrivateClient:
    """支援帳戶、查價、驗證單、送單、查單與取消單。"""

    credentials: BinanceCredentials
    base_url: str
    recv_window_ms: int = 5000
    timeout: int = 15
    session: requests.Session | None = None
    time_provider: Callable[[], int] = _milliseconds_now
    _server_offset_ms: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        if self.session is None:
            host = (urlsplit(self.base_url).hostname or "").lower()
            allowed_hosts = {
                "api.binance.com",
                "fapi.binance.com",
                "testnet.binance.vision",
                "demo-api.binance.com",
                "demo-fapi.binance.com",
            }
            if urlsplit(self.base_url).scheme != "https" or host not in allowed_hosts:
                raise ValueError("私有 Binance API 只允許官方 HTTPS 主機")
            # 私有簽章請求不採用 HTTP_PROXY/HTTPS_PROXY，避免 API Key 經未授權 Proxy。
            self.session = create_secure_session(trust_environment=False)

    def _signature(self, payload: str) -> str:
        # 封裝版延後載入 OpenSSL 相關模組，避免與 PyArrow 啟動順序衝突。
        import hashlib
        import hmac

        return hmac.new(
            self.credentials.api_secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, object] | None = None,
        *,
        signed: bool = False,
        api_key: bool = False,
        order_request: bool = False,
    ) -> Any:
        """送出請求；訂單逾時一律標成未知狀態，不自動重送。"""
        clean = {
            key: _api_value(value)
            for key, value in (params or {}).items()
            if value is not None
        }
        headers: dict[str, str] = {}
        if signed:
            clean["recvWindow"] = self.recv_window_ms
            clean["timestamp"] = self.time_provider() + self._server_offset_ms
            payload = urlencode(clean)
            clean["signature"] = self._signature(payload)
        if signed or api_key:
            headers["X-MBX-APIKEY"] = self.credentials.api_key
        query = urlencode(clean)
        url = f"{self.base_url}{path}"
        session = require_session(self.session)
        try:
            response = session.request(
                method.upper(),
                url,
                params=query,
                headers=headers,
                timeout=self.timeout,
            )
        except requests.Timeout as exc:
            if order_request:
                client_id = str(
                    clean.get("newClientOrderId")
                    or clean.get("listClientOrderId")
                    or clean.get("clientAlgoId")
                    or ""
                ) or None
                raise BinanceExecutionUnknown("訂單請求逾時，成交狀態未知", client_id) from exc
            raise BinancePrivateApiError(f"Binance API 請求逾時：{path}") from exc
        except requests.RequestException as exc:
            raise BinancePrivateApiError(f"Binance API 網路錯誤：{exc}") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise BinancePrivateApiError("Binance API 回傳非 JSON 資料") from exc
        status_code = int(getattr(response, "status_code", 200))
        if order_request and status_code >= 500:
            client_id = str(
                clean.get("newClientOrderId")
                or clean.get("listClientOrderId")
                or clean.get("clientAlgoId")
                or ""
            ) or None
            raise BinanceExecutionUnknown(
                f"Binance 回傳 {status_code}，訂單成交狀態未知",
                client_id,
            )
        if isinstance(payload, dict) and int(payload.get("code", 0) or 0) < 0:
            code = int(payload["code"])
            raise BinancePrivateApiError(
                f"Binance API 錯誤 {code}：{payload.get('msg', '未知錯誤')}",
                code,
            )
        try:
            response.raise_for_status()
        except requests.RequestException as exc:
            raise BinancePrivateApiError(f"Binance API HTTP 錯誤：{status_code}") from exc
        return payload

    def ping(self) -> bool:
        self._request("GET", "/api/v3/ping")
        return True

    def sync_server_time(self) -> int:
        """校正本機與 Binance 的毫秒時間差。"""
        payload = self._request("GET", "/api/v3/time")
        server_time = int(payload["serverTime"])
        self._server_offset_ms = server_time - self.time_provider()
        return self._server_offset_ms

    def fetch_account(self) -> dict[str, Any]:
        return dict(self._request("GET", "/api/v3/account", signed=True))

    def fetch_exchange_info(self, symbol: str) -> dict[str, Any]:
        payload = self._request(
            "GET",
            "/api/v3/exchangeInfo",
            {"symbol": to_binance_symbol(symbol)},
        )
        symbols = list(payload.get("symbols", []))
        if not symbols:
            raise BinancePrivateApiError(f"交易所找不到標的：{symbol}")
        return dict(symbols[0])

    def fetch_price(self, symbol: str) -> Decimal:
        payload = self._request(
            "GET",
            "/api/v3/ticker/price",
            {"symbol": to_binance_symbol(symbol)},
        )
        return Decimal(str(payload["price"]))

    def test_market_order(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        client_order_id: str,
    ) -> dict[str, Any]:
        return dict(
            self._request(
                "POST",
                "/api/v3/order/test",
                {
                    "symbol": to_binance_symbol(symbol),
                    "side": side.upper(),
                    "type": "MARKET",
                    "quantity": quantity,
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
    ) -> dict[str, Any]:
        return dict(
            self._request(
                "POST",
                "/api/v3/order",
                {
                    "symbol": to_binance_symbol(symbol),
                    "side": side.upper(),
                    "type": "MARKET",
                    "quantity": quantity,
                    "newClientOrderId": client_order_id,
                    "newOrderRespType": "FULL",
                },
                signed=True,
                order_request=True,
            )
        )

    def query_order(self, symbol: str, client_order_id: str) -> dict[str, Any]:
        return dict(
            self._request(
                "GET",
                "/api/v3/order",
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
                "/api/v3/order",
                {
                    "symbol": to_binance_symbol(symbol),
                    "origClientOrderId": client_order_id,
                },
                signed=True,
            )
        )

    def place_oco_sell_order(
        self,
        symbol: str,
        quantity: Decimal,
        take_profit_price: Decimal,
        stop_price: Decimal,
        list_client_order_id: str,
        take_profit_client_order_id: str,
        stop_client_order_id: str,
    ) -> dict[str, Any]:
        """建立 LIMIT_MAKER 停利與 STOP_LOSS 停損組成的現貨 OCO 賣單。"""
        return dict(
            self._request(
                "POST",
                "/api/v3/orderList/oco",
                {
                    "symbol": to_binance_symbol(symbol),
                    "listClientOrderId": list_client_order_id,
                    "side": "SELL",
                    "quantity": quantity,
                    "aboveType": "LIMIT_MAKER",
                    "aboveClientOrderId": take_profit_client_order_id,
                    "abovePrice": take_profit_price,
                    "belowType": "STOP_LOSS",
                    "belowClientOrderId": stop_client_order_id,
                    "belowStopPrice": stop_price,
                    "newOrderRespType": "RESULT",
                },
                signed=True,
                order_request=True,
            )
        )

    def query_order_list(self, list_client_order_id: str) -> dict[str, Any]:
        """依客戶端識別碼查詢 OCO，供逾時恢復與持倉對帳。"""
        return dict(
            self._request(
                "GET",
                "/api/v3/orderList",
                {"origClientOrderId": list_client_order_id},
                signed=True,
            )
        )

    def cancel_order_list(self, symbol: str, list_client_order_id: str) -> dict[str, Any]:
        """取消尚未觸發的 OCO，讓模型可以正常平倉。"""
        return dict(
            self._request(
                "DELETE",
                "/api/v3/orderList",
                {
                    "symbol": to_binance_symbol(symbol),
                    "listClientOrderId": list_client_order_id,
                },
                signed=True,
                order_request=True,
            )
        )
