"""實盤交易環境與安全限制設定。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ai_quant_trading.data_collection.assets import normalize_crypto_symbol


TradingEnvironment = Literal["testnet", "demo", "live"]
TradingMarketType = Literal["spot", "usd_m_futures"]

SPOT_ENVIRONMENT_BASE_URLS: dict[TradingEnvironment, str] = {
    "testnet": "https://testnet.binance.vision",
    "demo": "https://demo-api.binance.com",
    "live": "https://api.binance.com",
}

FUTURES_ENVIRONMENT_BASE_URLS: dict[TradingEnvironment, str] = {
    "testnet": "https://demo-fapi.binance.com",
    "demo": "https://demo-fapi.binance.com",
    "live": "https://fapi.binance.com",
}

# 保留舊名稱，避免既有外部工具匯入後立刻失效。
ENVIRONMENT_BASE_URLS = SPOT_ENVIRONMENT_BASE_URLS


@dataclass(frozen=True, slots=True)
class LiveTradingConfig:
    """限制可交易市場、環境、標的與單筆曝險。"""

    environment: TradingEnvironment = "testnet"
    market_type: TradingMarketType = "usd_m_futures"
    trading_enabled: bool = False
    allowed_symbols: tuple[str, ...] = ("BTC/USDT",)
    max_order_quote: float = 100.0
    max_balance_fraction: float = 0.25
    recv_window_ms: int = 5000
    estimated_fee_rate: float = 0.001
    estimated_slippage_rate: float = 0.0005
    exchange_protection_enabled: bool = True
    leverage: int = 2
    margin_type: Literal["ISOLATED"] = "ISOLATED"

    def __post_init__(self) -> None:
        if self.environment not in SPOT_ENVIRONMENT_BASE_URLS:
            raise ValueError("environment 必須是 testnet、demo 或 live")
        if self.market_type not in {"spot", "usd_m_futures"}:
            raise ValueError("market_type 必須是 spot 或 usd_m_futures")
        normalized = tuple(normalize_crypto_symbol(symbol) for symbol in self.allowed_symbols)
        if not normalized:
            raise ValueError("allowed_symbols 不可為空")
        object.__setattr__(self, "allowed_symbols", normalized)
        if self.max_order_quote <= 0:
            raise ValueError("max_order_quote 必須大於 0")
        if not 0 < self.max_balance_fraction <= 1:
            raise ValueError("max_balance_fraction 必須介於 0 與 1 之間")
        if not 1 <= self.recv_window_ms <= 60_000:
            raise ValueError("recv_window_ms 必須介於 1 與 60000")
        if not 0 <= self.estimated_fee_rate < 1:
            raise ValueError("estimated_fee_rate 必須介於 0 與 1 之間")
        if not 0 <= self.estimated_slippage_rate < 1:
            raise ValueError("estimated_slippage_rate 必須介於 0 與 1 之間")
        if not 1 <= self.leverage <= 3:
            raise ValueError("本系統安全上限只允許 1 到 3 倍槓桿")
        if self.margin_type != "ISOLATED":
            raise ValueError("目前只支援逐倉 ISOLATED，避免策略影響整個合約帳戶")

    @property
    def base_url(self) -> str:
        urls = (
            FUTURES_ENVIRONMENT_BASE_URLS
            if self.market_type == "usd_m_futures"
            else SPOT_ENVIRONMENT_BASE_URLS
        )
        return urls[self.environment]

    @property
    def confirmation_phrase(self) -> str:
        return f"EXECUTE {self.environment.upper()}"
