"""解析並套用 Binance 交易對的數量與名目金額規則。"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from typing import Any


ZERO = Decimal("0")


@dataclass(frozen=True, slots=True)
class SymbolRules:
    """市場單送出前需要驗證的交易對規則。"""

    symbol: str
    status: str
    base_asset: str
    quote_asset: str
    min_quantity: Decimal
    max_quantity: Decimal
    step_size: Decimal
    min_notional: Decimal
    max_notional: Decimal | None = None
    min_price: Decimal = ZERO
    max_price: Decimal = ZERO
    tick_size: Decimal = ZERO

    @classmethod
    def from_exchange_info(cls, payload: dict[str, Any]) -> "SymbolRules":
        filters = {item["filterType"]: item for item in payload.get("filters", [])}
        lot = filters.get("MARKET_LOT_SIZE") or filters.get("LOT_SIZE")
        if lot is None:
            raise ValueError("exchangeInfo 缺少 LOT_SIZE／MARKET_LOT_SIZE")
        if Decimal(str(lot.get("stepSize", "0"))) == ZERO and "LOT_SIZE" in filters:
            lot = filters["LOT_SIZE"]
        notional = filters.get("NOTIONAL") or filters.get("MIN_NOTIONAL") or {}
        price = filters.get("PRICE_FILTER") or {}
        max_notional_value = Decimal(str(notional.get("maxNotional", "0")))
        return cls(
            symbol=str(payload["symbol"]),
            status=str(payload.get("status", "")),
            base_asset=str(payload["baseAsset"]),
            quote_asset=str(payload["quoteAsset"]),
            min_quantity=Decimal(str(lot.get("minQty", "0"))),
            max_quantity=Decimal(str(lot.get("maxQty", "0"))),
            step_size=Decimal(str(lot.get("stepSize", "0"))),
            min_notional=Decimal(
                str(notional.get("minNotional") or notional.get("notional") or "0")
            ),
            max_notional=max_notional_value if max_notional_value > ZERO else None,
            min_price=Decimal(str(price.get("minPrice", "0"))),
            max_price=Decimal(str(price.get("maxPrice", "0"))),
            tick_size=Decimal(str(price.get("tickSize", "0"))),
        )

    def normalize_quantity(self, quantity: Decimal) -> Decimal:
        """向下截斷到 stepSize，避免超出使用者的資金上限。"""
        if quantity <= ZERO:
            return ZERO
        if self.step_size <= ZERO:
            return quantity
        steps = (quantity / self.step_size).to_integral_value(rounding=ROUND_DOWN)
        return steps * self.step_size

    def validate(self, quantity: Decimal, price: Decimal) -> Decimal:
        """回傳名目金額，規則不符時提供明確中文錯誤。"""
        if self.status != "TRADING":
            raise ValueError(f"{self.symbol} 目前不可交易，狀態：{self.status}")
        if quantity < self.min_quantity:
            raise ValueError(f"數量 {quantity} 小於最小下單量 {self.min_quantity}")
        if self.max_quantity > ZERO and quantity > self.max_quantity:
            raise ValueError(f"數量 {quantity} 大於最大下單量 {self.max_quantity}")
        notional = quantity * price
        if self.min_notional > ZERO and notional < self.min_notional:
            raise ValueError(f"訂單金額 {notional} 小於最小金額 {self.min_notional}")
        if self.max_notional is not None and notional > self.max_notional:
            raise ValueError(f"訂單金額 {notional} 大於最大金額 {self.max_notional}")
        return notional

    def normalize_price(self, price: Decimal) -> Decimal:
        """向下截斷到 tickSize，供停損與停利保護單使用。"""
        if price <= ZERO:
            raise ValueError("保護單價格必須大於 0")
        normalized = price
        if self.tick_size > ZERO:
            ticks = (price / self.tick_size).to_integral_value(rounding=ROUND_DOWN)
            normalized = ticks * self.tick_size
        if self.min_price > ZERO and normalized < self.min_price:
            raise ValueError(f"價格 {normalized} 小於最小價格 {self.min_price}")
        if self.max_price > ZERO and normalized > self.max_price:
            raise ValueError(f"價格 {normalized} 大於最大價格 {self.max_price}")
        return normalized
