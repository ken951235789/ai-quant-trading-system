"""Binance USD-M 永續合約下單、安全限制與持倉對帳閘門。"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import os
from typing import Any

from ai_quant_trading.data_collection.assets import normalize_crypto_symbol
from ai_quant_trading.live_trading.binance_client import (
    BinanceExecutionUnknown,
    BinancePrivateApiError,
)
from ai_quant_trading.live_trading.binance_futures_client import (
    BinanceFuturesPrivateClient,
)
from ai_quant_trading.live_trading.config import LiveTradingConfig
from ai_quant_trading.live_trading.gateway import (
    OrderPreview,
    OrderSubmission,
    make_client_order_id,
)
from ai_quant_trading.live_trading.rules import SymbolRules


ZERO = Decimal("0")


def _truthy(value: object) -> bool:
    return value is True or str(value).strip().lower() == "true"


@dataclass(frozen=True, slots=True)
class FuturesPortfolioSnapshot:
    """單一 USD-M 永續合約的帳戶與帶方向持倉快照。"""

    symbol: str
    base_asset: str
    quote_asset: str
    wallet_balance: Decimal
    available_balance: Decimal
    margin_balance: Decimal
    position_quantity: Decimal
    entry_price: Decimal
    mark_price: Decimal
    unrealized_pnl: Decimal
    liquidation_price: Decimal
    leverage: int
    margin_type: str
    can_trade: bool
    initial_margin: Decimal = ZERO
    maintenance_margin: Decimal = ZERO

    @property
    def estimated_equity(self) -> Decimal:
        return self.margin_balance

    @property
    def price(self) -> Decimal:
        return self.mark_price

    @property
    def position_side(self) -> str:
        if self.position_quantity > ZERO:
            return "LONG"
        if self.position_quantity < ZERO:
            return "SHORT"
        return "FLAT"

    @property
    def position_notional(self) -> Decimal:
        return abs(self.position_quantity) * self.mark_price

    @property
    def position_fraction(self) -> float:
        if self.margin_balance <= ZERO:
            return 0.0
        return float(self.position_quantity * self.mark_price / self.margin_balance)

    # 這四個相容屬性只供既有報表讀取；Futures 流程不以現貨餘額決策。
    @property
    def base_free(self) -> Decimal:
        return ZERO

    @property
    def base_locked(self) -> Decimal:
        return ZERO

    @property
    def quote_free(self) -> Decimal:
        return self.available_balance

    @property
    def quote_locked(self) -> Decimal:
        return max(self.wallet_balance - self.available_balance, ZERO)


@dataclass(frozen=True, slots=True)
class FuturesProtectionSubmission:
    """永續持倉的交易所端停損與停利條件單。"""

    symbol: str
    side: str
    stop_price: Decimal
    take_profit_price: Decimal
    stop_client_algo_id: str
    take_profit_client_algo_id: str
    stop_response: dict[str, Any]
    take_profit_response: dict[str, Any]
    recovered_after_unknown: bool = False


@dataclass(slots=True)
class FuturesTradingGateway:
    """把研究中的帶方向曝險映射到 Binance USD-M 單向逐倉帳戶。"""

    client: BinanceFuturesPrivateClient
    config: LiveTradingConfig

    def __post_init__(self) -> None:
        if self.config.market_type != "usd_m_futures":
            raise ValueError("FuturesTradingGateway 只接受 usd_m_futures 設定")

    def _validate_allowed_symbol(self, symbol: str) -> None:
        if symbol not in self.config.allowed_symbols:
            allowed = "、".join(self.config.allowed_symbols)
            raise ValueError(f"{symbol} 不在 USD-M 實盤白名單；允許：{allowed}")

    def _authorize_execution(self, confirmation: str) -> None:
        if not self.config.trading_enabled:
            raise PermissionError("實際下單功能尚未啟用")
        if confirmation.strip() != self.config.confirmation_phrase:
            raise PermissionError(f"確認文字必須完全等於 {self.config.confirmation_phrase}")
        if self.config.environment == "live" and os.getenv(
            "AI_QUANT_LIVE_TRADING_ENABLED", ""
        ).upper() != "YES":
            raise PermissionError("Live 環境還需要 AI_QUANT_LIVE_TRADING_ENABLED=YES")

    def _validate_account_modes(self) -> None:
        position_mode = self.client.fetch_position_mode()
        if _truthy(position_mode.get("dualSidePosition")):
            raise ValueError(
                "帳戶目前是 Hedge Mode；本模型使用 signed target，請先在 Binance 改為 One-way Mode"
            )
        multi_asset = self.client.fetch_multi_asset_mode()
        if _truthy(multi_asset.get("multiAssetsMargin")):
            raise ValueError("目前不支援 Multi-Assets Mode，請改為 Single-Asset Mode")

    def account_status(self) -> dict[str, Any]:
        self.client.sync_server_time()
        return self.client.fetch_account()

    def portfolio(self, symbol: str) -> FuturesPortfolioSnapshot:
        normalized = normalize_crypto_symbol(symbol)
        self._validate_allowed_symbol(normalized)
        info = self.client.fetch_exchange_info(normalized)
        rules = SymbolRules.from_exchange_info(info)
        account = self.account_status()
        position_rows = self.client.fetch_position_risk(normalized)
        position = next(
            (
                item
                for item in position_rows
                if str(item.get("symbol")) == rules.symbol
                and str(item.get("positionSide", "BOTH")) == "BOTH"
            ),
            {},
        )
        asset = next(
            (
                item
                for item in account.get("assets", [])
                if str(item.get("asset")) == rules.quote_asset
            ),
            {},
        )
        price = Decimal(str(position.get("markPrice") or self.client.fetch_price(normalized)))
        wallet = Decimal(str(asset.get("walletBalance", "0")))
        unrealized = Decimal(str(position.get("unRealizedProfit", "0")))
        margin_balance = Decimal(
            str(asset.get("marginBalance") or account.get("totalMarginBalance") or wallet + unrealized)
        )
        available = Decimal(
            str(asset.get("availableBalance") or account.get("availableBalance") or "0")
        )
        return FuturesPortfolioSnapshot(
            symbol=normalized,
            base_asset=rules.base_asset,
            quote_asset=rules.quote_asset,
            wallet_balance=wallet,
            available_balance=available,
            margin_balance=margin_balance,
            position_quantity=Decimal(str(position.get("positionAmt", "0"))),
            entry_price=Decimal(str(position.get("entryPrice", "0"))),
            mark_price=price,
            unrealized_pnl=unrealized,
            liquidation_price=Decimal(str(position.get("liquidationPrice", "0"))),
            leverage=int(position.get("leverage") or self.config.leverage),
            margin_type=str(position.get("marginType") or self.config.margin_type).upper(),
            can_trade=bool(account.get("canTrade", False)),
            initial_margin=Decimal(str(position.get("positionInitialMargin", "0"))),
            maintenance_margin=Decimal(str(position.get("maintMargin", "0"))),
        )

    def prepare_symbol_for_execution(self, symbol: str, confirmation: str) -> None:
        """實際送單前鎖定 One-way、Single-Asset、逐倉與指定槓桿。"""
        self._authorize_execution(confirmation)
        normalized = normalize_crypto_symbol(symbol)
        self._validate_allowed_symbol(normalized)
        self._validate_account_modes()
        try:
            self.client.change_margin_type(normalized, self.config.margin_type)
        except BinancePrivateApiError as exc:
            # -4046 代表不需要變更，並不是設定失敗。
            if exc.code != -4046:
                raise
        response = self.client.change_leverage(normalized, self.config.leverage)
        actual = int(response.get("leverage") or 0)
        if actual != self.config.leverage:
            raise RuntimeError(
                f"Binance 回報槓桿 {actual} 倍，與要求的 {self.config.leverage} 倍不一致"
            )

    def preview_market_order(
        self,
        symbol: str,
        side: str,
        *,
        quote_amount: float | Decimal | None = None,
        quantity: float | Decimal | None = None,
        full_exit: bool = False,
        reduce_only: bool = False,
    ) -> OrderPreview:
        """建立開倉、加倉或 reduce-only 減倉的市場單預覽。"""
        normalized = normalize_crypto_symbol(symbol)
        self._validate_allowed_symbol(normalized)
        normalized_side = side.upper()
        if normalized_side not in {"BUY", "SELL"}:
            raise ValueError("side 必須是 BUY 或 SELL")
        info = self.client.fetch_exchange_info(normalized)
        rules = SymbolRules.from_exchange_info(info)
        snapshot = self.portfolio(normalized)
        if not snapshot.can_trade:
            raise ValueError("此 Binance USD-M 帳戶目前不可交易")

        current = snapshot.position_quantity
        reducing_side = "SELL" if current > ZERO else "BUY" if current < ZERO else ""
        use_reduce_only = bool(reduce_only or full_exit)
        if use_reduce_only:
            if current == ZERO:
                raise ValueError("目前沒有合約持倉可供 reduce-only 平倉")
            if normalized_side != reducing_side:
                raise ValueError(f"目前為 {snapshot.position_side}，減倉方向必須是 {reducing_side}")
            requested_quantity = abs(current) if full_exit or quantity is None else Decimal(str(quantity))
            raw_quantity = min(requested_quantity, abs(current))
        else:
            if current > ZERO and normalized_side == "SELL":
                raise ValueError("由多翻空必須先用 reduce-only 平倉，禁止單筆直接反手")
            if current < ZERO and normalized_side == "BUY":
                raise ValueError("由空翻多必須先用 reduce-only 平倉，禁止單筆直接反手")
            requested_notional = Decimal(str(quote_amount or self.config.max_order_quote))
            margin_cap = (
                snapshot.available_balance
                * Decimal(str(self.config.max_balance_fraction))
                * Decimal(self.config.leverage)
            )
            notional = min(
                requested_notional,
                Decimal(str(self.config.max_order_quote)),
                margin_cap,
            )
            raw_quantity = Decimal(str(quantity)) if quantity is not None else notional / snapshot.mark_price

        normalized_quantity = rules.normalize_quantity(raw_quantity)
        notional = rules.validate(normalized_quantity, snapshot.mark_price)
        return OrderPreview(
            normalized,
            normalized_side,
            normalized_quantity,
            snapshot.mark_price,
            notional,
            rules.base_asset,
            rules.quote_asset,
            "usd_m_futures",
            use_reduce_only,
            "BOTH",
        )

    def validate_order(
        self,
        preview: OrderPreview,
        client_order_id: str | None = None,
    ) -> OrderSubmission:
        self._validate_account_modes()
        order_id = client_order_id or make_client_order_id(preview.symbol, preview.side)
        response = self.client.test_market_order(
            preview.symbol,
            preview.side,
            preview.quantity,
            order_id,
            reduce_only=preview.reduce_only,
        )
        return OrderSubmission(preview, order_id, True, response)

    def submit_order(
        self,
        preview: OrderPreview,
        confirmation: str,
        client_order_id: str | None = None,
    ) -> OrderSubmission:
        self.prepare_symbol_for_execution(preview.symbol, confirmation)
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
                reduce_only=preview.reduce_only,
            )
            return OrderSubmission(preview, order_id, False, response)
        except BinanceExecutionUnknown as unknown:
            try:
                recovered = self.client.query_order(preview.symbol, order_id)
            except BinancePrivateApiError:
                raise unknown
            return OrderSubmission(preview, order_id, False, recovered, True)

    def submit_position_protection(
        self,
        *,
        symbol: str,
        position_side: str,
        stop_price: float | Decimal,
        take_profit_price: float | Decimal,
        confirmation: str,
        seed: str,
    ) -> FuturesProtectionSubmission:
        """同時建立停損與停利；兩者都只會平掉當下單向持倉。"""
        self._authorize_execution(confirmation)
        normalized = normalize_crypto_symbol(symbol)
        self._validate_allowed_symbol(normalized)
        snapshot = self.portfolio(normalized)
        side = position_side.upper()
        if side not in {"LONG", "SHORT"} or snapshot.position_side != side:
            raise ValueError("保護單方向與交易所實際持倉不一致")
        rules = SymbolRules.from_exchange_info(self.client.fetch_exchange_info(normalized))
        stop = rules.normalize_price(Decimal(str(stop_price)))
        take = rules.normalize_price(Decimal(str(take_profit_price)))
        if side == "LONG" and not stop < snapshot.mark_price < take:
            raise ValueError("多單保護價必須符合停損 < Mark Price < 停利")
        if side == "SHORT" and not take < snapshot.mark_price < stop:
            raise ValueError("空單保護價必須符合停利 < Mark Price < 停損")
        exit_side = "SELL" if side == "LONG" else "BUY"
        stop_id = make_client_order_id(normalized, "STOP", f"{seed}|stop")
        take_id = make_client_order_id(normalized, "TAKE", f"{seed}|take")
        recovered = False

        def get_or_place(client_id: str, order_type: str, price: Decimal) -> dict[str, Any]:
            nonlocal recovered
            try:
                existing = self.client.query_algo_order(client_id)
            except BinancePrivateApiError as exc:
                if exc.code not in {-2011, -2013}:
                    raise
            else:
                recovered = True
                return existing
            try:
                return self.client.place_close_position_algo_order(
                    normalized,
                    exit_side,
                    order_type,
                    price,
                    client_id,
                )
            except BinanceExecutionUnknown as unknown:
                try:
                    result = self.client.query_algo_order(client_id)
                except BinancePrivateApiError:
                    raise unknown
                recovered = True
                return result

        stop_response = get_or_place(stop_id, "STOP_MARKET", stop)
        take_response = get_or_place(take_id, "TAKE_PROFIT_MARKET", take)
        return FuturesProtectionSubmission(
            normalized,
            side,
            stop,
            take,
            stop_id,
            take_id,
            stop_response,
            take_response,
            recovered,
        )

    def query_position_protection(
        self,
        stop_client_algo_id: str,
        take_profit_client_algo_id: str,
    ) -> dict[str, dict[str, Any]]:
        return {
            "stop": self.client.query_algo_order(stop_client_algo_id),
            "take_profit": self.client.query_algo_order(take_profit_client_algo_id),
        }

    def cancel_position_protection(
        self,
        stop_client_algo_id: str | None,
        take_profit_client_algo_id: str | None,
        confirmation: str,
    ) -> dict[str, dict[str, Any]]:
        self._authorize_execution(confirmation)
        results: dict[str, dict[str, Any]] = {}
        for name, client_id in (
            ("stop", stop_client_algo_id),
            ("take_profit", take_profit_client_algo_id),
        ):
            if not client_id:
                continue
            try:
                results[name] = self.client.cancel_algo_order(client_id)
            except BinancePrivateApiError as exc:
                if exc.code not in {-2011, -2013}:
                    raise
                results[name] = {"status": "NOT_FOUND", "clientAlgoId": client_id}
        return results
