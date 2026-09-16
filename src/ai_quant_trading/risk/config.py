"""風險管理參數定義。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

from ai_quant_trading.risk.dynamic_leverage import DynamicLeverageConfig


StopLossMode = Literal["fixed", "atr"]


@dataclass(frozen=True, slots=True)
class RiskConfig:
    """單筆交易及整體帳戶使用的風控設定。"""

    max_risk_per_trade: float = 0.01
    stop_loss_mode: StopLossMode = "fixed"
    fixed_stop_loss_pct: float = 0.03
    atr_multiplier: float = 2.0
    take_profit_pct: float | None = 0.06
    max_drawdown_limit: float = 0.20
    max_position_fraction: float = 1.0
    dynamic_leverage: DynamicLeverageConfig = field(default_factory=DynamicLeverageConfig)

    def __post_init__(self) -> None:
        if isinstance(self.dynamic_leverage, dict):
            object.__setattr__(
                self,
                "dynamic_leverage",
                DynamicLeverageConfig(**self.dynamic_leverage),
            )
        if not 0 < self.max_risk_per_trade <= 1:
            raise ValueError("max_risk_per_trade 必須介於 0 與 1 之間")
        if self.stop_loss_mode not in {"fixed", "atr"}:
            raise ValueError("stop_loss_mode 必須是 fixed 或 atr")
        if not 0 < self.fixed_stop_loss_pct < 1:
            raise ValueError("fixed_stop_loss_pct 必須介於 0 與 1 之間")
        if self.atr_multiplier <= 0:
            raise ValueError("atr_multiplier 必須大於 0")
        if self.take_profit_pct is not None and self.take_profit_pct <= 0:
            raise ValueError("take_profit_pct 必須大於 0，或設為 None")
        if not 0 < self.max_drawdown_limit <= 1:
            raise ValueError("max_drawdown_limit 必須介於 0 與 1 之間")
        if not 0 < self.max_position_fraction <= 1:
            raise ValueError("max_position_fraction 必須介於 0 與 1 之間")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
