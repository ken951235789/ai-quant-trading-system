"""實盤交易安全閘門、帳戶快照與市場單流程。"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import os
import time
from typing import Any

from ai_quant_trading.data_collection.assets import normalize_crypto_symbol
from ai_quant_trading.live_trading.binance_client import (
    BinanceExecutionUnknown,
    BinancePrivateApiError,
    BinancePrivateClient,
)
from ai_quant_trading.live_trading.config import LiveTradingConfig
from ai_quant_trading.live_trading.rules import SymbolRules


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    """單一交易對相關的現貨帳戶餘額。"""

    symbol: str
    base_asset: str
    quote_asset: str
    base_free: Decimal
    base_locked: Decimal
    quote_free: Decimal
    quote_locked: Decimal
    price: Decimal
    can_trade: bool

    @property
    def estimated_equity(self) -> Decimal:
        return self.quote_free + self.quote_locked + (self.base_free + self.base_locked) * self.price


@dataclass(frozen=True, slots=True)
class OrderPreview:
    """通過本地白名單、餘額與交易所規則的市場單。"""

    symbol: str
    side: str
    quantity: Decimal
    reference_price: Decimal
    estimated_notional: Decimal
    base_asset: str
    quote_asset: str
    market_type: str = "spot"
    reduce_only: bool = False
    position_side: str = "BOTH"


@dataclass(frozen=True, slots=True)
class OrderSubmission:
    """驗證單或實際訂單的回傳結果。"""

    preview: OrderPreview
    client_order_id: str
    validated_only: bool
    response: dict[str, Any]
    recovered_after_unknown: bool = False


@dataclass(frozen=True, slots=True)
class ProtectionSubmission:
    """買入成交後建立的交易所原生 OCO 停損停利。"""

    symbol: str
    quantity: Decimal
    take_profit_price: Decimal
    stop_price: Decimal
    list_client_order_id: str
    response: dict[str, Any]
    recovered_after_unknown: bool = False


def _balance(account: dict[str, Any], asset: str) -> tuple[Decimal, Decimal]:
    for item in account.get("balances", []):
        if str(item.get("asset")) == asset:
            return Decimal(str(item.get("free", "0"))), Decimal(str(item.get("locked", "0")))
    return Decimal("0"), Decimal("0")


def make_client_order_id(symbol: str, side: str, seed: str | None = None) -> str:
    """產生最長 36 字元且可用於查單的唯一識別碼。"""
    # 封裝版只有在建立訂單識別碼時才需要載入雜湊實作。
    import hashlib

    source = seed or str(time.time_ns())
    digest = hashlib.sha256(f"{symbol}|{side}|{source}".encode()).hexdigest()
    uniqueness = digest[:26] if seed is not None else f"{str(int(time.time() * 1000))[-12:]}{digest[:14]}"
    return f"aq{side.lower()[0]}{uniqueness}"[:36]


@dataclass(slots=True)
class LiveTradingGateway:
    """所有實際送單都必須通過此安全閘門。"""

    client: BinancePrivateClient
    config: LiveTradingConfig

    def account_status(self) -> dict[str, Any]:
        self.client.sync_server_time()
        return self.client.fetch_account()

    def portfolio(self, symbol: str) -> PortfolioSnapshot:
        normalized = normalize_crypto_symbol(symbol)
        self._validate_allowed_symbol(normalized)
        info = self.client.fetch_exchange_info(normalized)
        rules = SymbolRules.from_exchange_info(info)
        price = self.client.fetch_price(normalized)
        account = self.account_status()
        base_free, base_locked = _balance(account, rules.base_asset)
        quote_free, quote_locked = _balance(account, rules.quote_asset)
        return PortfolioSnapshot(
            normalized,
            rules.base_asset,
            rules.quote_asset,
            base_free,
            base_locked,
            quote_free,
            quote_locked,
            price,
            bool(account.get("canTrade", False)),
        )

    def _validate_allowed_symbol(self, symbol: str) -> None:
        if symbol not in self.config.allowed_symbols:
            allowed = ", ".join(self.config.allowed_symbols)
            raise ValueError(f"{symbol} 不在實盤白名單；允許：{allowed}")

    def preview_market_order(
        self,
        symbol: str,
        side: str,
        *,
        quote_amount: float | Decimal | None = None,
        quantity: float | Decimal | None = None,
        full_exit: bool = False,
    ) -> OrderPreview:
        """以最新餘額與 exchangeInfo 建立市場單預覽。"""
        normalized = normalize_crypto_symbol(symbol)
        self._validate_allowed_symbol(normalized)
        normalized_side = side.upper()
        if normalized_side not in {"BUY", "SELL"}:
            raise ValueError("side 必須是 BUY 或 SELL")

        info = self.client.fetch_exchange_info(normalized)
        rules = SymbolRules.from_exchange_info(info)
        price = self.client.fetch_price(normalized)
        account = self.account_status()
        if not bool(account.get("canTrade", False)):
            raise ValueError("此 Binance 帳戶目前不可交易")
        base_free, _ = _balance(account, rules.base_asset)
        quote_free, _ = _balance(account, rules.quote_asset)

        if normalized_side == "BUY":
            requested = Decimal(str(quote_amount or self.config.max_order_quote))
            spending_limit = min(
                Decimal(str(self.config.max_order_quote)),
                quote_free * Decimal(str(self.config.max_balance_fraction)),
            )
            spend = min(requested, spending_limit)
            raw_quantity = spend / price
        else:
            requested_quantity = base_free if quantity is None else Decimal(str(quantity))
            if full_exit:
                raw_quantity = min(requested_quantity, base_free)
            else:
                quote_cap = Decimal(str(self.config.max_order_quote)) / price
                raw_quantity = min(requested_quantity, base_free, quote_cap)

        normalized_quantity = rules.normalize_quantity(raw_quantity)
        notional = rules.validate(normalized_quantity, price)
        return OrderPreview(
            normalized,
            normalized_side,
            normalized_quantity,
            price,
            notional,
            rules.base_asset,
            rules.quote_asset,
        )

    def validate_order(
        self,
        preview: OrderPreview,
        client_order_id: str | None = None,
    ) -> OrderSubmission:
        """呼叫 `/order/test` 驗證簽章與參數，不送進撮合引擎。"""
        order_id = client_order_id or make_client_order_id(preview.symbol, preview.side)
        response = self.client.test_market_order(
            preview.symbol,
            preview.side,
            preview.quantity,
            order_id,
        )
        return OrderSubmission(preview, order_id, True, response)

    def submit_order(
        self,
        preview: OrderPreview,
        confirmation: str,
        client_order_id: str | None = None,
    ) -> OrderSubmission:
        """通過雙重開關後送單；未知狀態只查單，不重新送單。"""
        self._authorize_execution(confirmation)

        order_id = client_order_id or make_client_order_id(preview.symbol, preview.side)
        try:
            existing = self.client.query_order(preview.symbol, order_id)
        except BinancePrivateApiError as exc:
            if exc.code != -2013:
                raise
        else:
            return OrderSubmission(preview, order_id, False, existing, True)
        try:
            response = self.client.place_market_order(
                preview.symbol,
                preview.side,
                preview.quantity,
                order_id,
            )
            return OrderSubmission(preview, order_id, False, response)
        except BinanceExecutionUnknown as unknown:
            try:
                recovered = self.client.query_order(preview.symbol, order_id)
            except BinancePrivateApiError:
                raise unknown
            return OrderSubmission(preview, order_id, False, recovered, True)

    def _authorize_execution(self, confirmation: str) -> None:
        """所有市場單與保護單共用同一組雙重安全開關。"""
        if not self.config.trading_enabled:
            raise PermissionError("實際下單功能尚未啟用")
        if confirmation.strip() != self.config.confirmation_phrase:
            raise PermissionError(f"確認文字必須完全等於 {self.config.confirmation_phrase}")
        if self.config.environment == "live" and os.getenv(
            "AI_QUANT_LIVE_TRADING_ENABLED", ""
        ).upper() != "YES":
            raise PermissionError("Live 環境還需要 AI_QUANT_LIVE_TRADING_ENABLED=YES")

    def submit_oco_protection(
        self,
        *,
        symbol: str,
        quantity: float | Decimal,
        take_profit_price: float | Decimal,
        stop_price: float | Decimal,
        confirmation: str,
        seed: str,
    ) -> ProtectionSubmission:
        """將已成交的多單數量鎖入 Binance OCO，任一邊成交會取消另一邊。"""
        self._authorize_execution(confirmation)
        normalized = normalize_crypto_symbol(symbol)
        self._validate_allowed_symbol(normalized)
        info = self.client.fetch_exchange_info(normalized)
        rules = SymbolRules.from_exchange_info(info)
        current_price = self.client.fetch_price(normalized)
        protected_quantity = rules.normalize_quantity(Decimal(str(quantity)))
        target = rules.normalize_price(Decimal(str(take_profit_price)))
        stop = rules.normalize_price(Decimal(str(stop_price)))
        rules.validate(protected_quantity, current_price)
        if not target > current_price > stop:
            raise ValueError(
                f"SELL OCO 必須符合停利 {target} > 市價 {current_price} > 停損 {stop}"
            )

        list_id = make_client_order_id(normalized, "OCO", f"{seed}|list")
        target_id = make_client_order_id(normalized, "SELL", f"{seed}|take")
        stop_id = make_client_order_id(normalized, "SELL", f"{seed}|stop")
        try:
            existing = self.client.query_order_list(list_id)
        except BinancePrivateApiError as exc:
            if exc.code != -2013:
                raise
        else:
            return ProtectionSubmission(
                normalized,
                protected_quantity,
                target,
                stop,
                list_id,
                existing,
                True,
            )
        try:
            response = self.client.place_oco_sell_order(
                normalized,
                protected_quantity,
                target,
                stop,
                list_id,
                target_id,
                stop_id,
            )
            return ProtectionSubmission(
                normalized,
                protected_quantity,
                target,
                stop,
                list_id,
                response,
            )
        except BinanceExecutionUnknown as unknown:
            try:
                recovered = self.client.query_order_list(list_id)
            except BinancePrivateApiError:
                raise unknown
            return ProtectionSubmission(
                normalized,
                protected_quantity,
                target,
                stop,
                list_id,
                recovered,
                True,
            )

    def query_oco_protection(self, list_client_order_id: str) -> dict[str, Any]:
        """查詢交易所保護單狀態。"""
        payload = self.client.query_order_list(list_client_order_id)
        list_status = str(
            payload.get("listOrderStatus") or payload.get("listStatusType") or ""
        ).upper()
        if list_status == "ALL_DONE":
            reports = []
            for order in payload.get("orders", []):
                client_order_id = str(order.get("clientOrderId") or "")
                symbol = str(order.get("symbol") or payload.get("symbol") or "")
                if client_order_id and symbol:
                    reports.append(self.client.query_order(symbol, client_order_id))
            payload["orderReports"] = reports
        return payload

    def cancel_oco_protection(
        self,
        symbol: str,
        list_client_order_id: str,
        confirmation: str,
    ) -> dict[str, Any]:
        """模型需要主動平倉前，先解除鎖住現貨數量的 OCO。"""
        self._authorize_execution(confirmation)
        normalized = normalize_crypto_symbol(symbol)
        self._validate_allowed_symbol(normalized)
        return self.client.cancel_order_list(normalized, list_client_order_id)
