"""模擬交易的資金與成交設定。"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class PaperTradingConfig:
    """建立模擬帳戶時固定下來的成交參數。"""

    initial_capital: float = 1000.0
    fee_rate: float = 0.001
    slippage_rate: float = 0.0005
    position_fraction: float = 1.0
    allow_short: bool = False
    max_short_fraction: float = 0.0
    short_borrow_rate_annual: float = 0.0
    execution_mode: str = "spot"
    leverage: float = 1.0
    max_margin_fraction: float = 1.0
    daily_loss_limit: float = 1.0
    max_consecutive_losses: int = 0

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital 必須大於 0")
        if not 0 <= self.fee_rate < 1:
            raise ValueError("fee_rate 必須介於 0 與 1 之間")
        if not 0 <= self.slippage_rate < 1:
            raise ValueError("slippage_rate 必須介於 0 與 1 之間")
        if not 0 < self.position_fraction <= 1:
            raise ValueError("position_fraction 必須大於 0 且不超過 1")
        if not 0 <= self.max_short_fraction <= 1:
            raise ValueError("max_short_fraction 必須介於 0 與 1 之間")
        if self.allow_short and self.max_short_fraction <= 0:
            raise ValueError("啟用作空時 max_short_fraction 必須大於 0")
        if not self.allow_short and self.max_short_fraction != 0:
            raise ValueError("未啟用作空時 max_short_fraction 必須為 0")
        if self.short_borrow_rate_annual < 0:
            raise ValueError("short_borrow_rate_annual 不可小於 0")
        if self.execution_mode not in {"spot", "perpetual"}:
            raise ValueError("execution_mode 只支援 spot 或 perpetual")
        if self.leverage <= 0:
            raise ValueError("leverage 必須大於 0")
        if not 0 < self.max_margin_fraction <= 1:
            raise ValueError("max_margin_fraction 必須大於 0 且不超過 1")
        if not 0 < self.daily_loss_limit <= 1:
            raise ValueError("daily_loss_limit 必須大於 0 且不超過 1")
        if self.max_consecutive_losses < 0:
            raise ValueError("max_consecutive_losses 不可小於 0")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
