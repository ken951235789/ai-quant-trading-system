"""將 Transformer 多任務輸出轉成可執行的市場狀態。"""

from __future__ import annotations

from dataclasses import dataclass

from ai_quant_trading.trading.contracts import ModelForecast, RegimeDecision


@dataclass(frozen=True, slots=True)
class MarketRegimeConfig:
    """趨勢信心、不確定性與高波動風險倍率。"""

    minimum_probability: float = 0.45
    maximum_uncertainty: float = 0.62
    minimum_absolute_return_5: float = 0.0
    high_volatility_threshold: float = 0.03
    high_volatility_multiplier: float = 0.50
    neutral_multiplier: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("minimum_probability", self.minimum_probability),
            ("maximum_uncertainty", self.maximum_uncertainty),
            ("high_volatility_multiplier", self.high_volatility_multiplier),
            ("neutral_multiplier", self.neutral_multiplier),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} 必須介於 0 與 1")
        if self.minimum_absolute_return_5 < 0 or self.high_volatility_threshold <= 0:
            raise ValueError("報酬與高波動門檻必須為非負值，且波動門檻需大於 0")


class MarketRegimeDetector:
    """判斷偏多、偏空、震盪、高波動、不確定或不可用。"""

    def __init__(self, config: MarketRegimeConfig | None = None) -> None:
        self.config = config or MarketRegimeConfig()

    def classify(self, forecast: ModelForecast) -> RegimeDecision:
        config = self.config
        if not forecast.available:
            return RegimeDecision(
                "unavailable", False, False, 0.0, ("Transformer 預測不可用",)
            )
        if forecast.uncertainty > config.maximum_uncertainty:
            return RegimeDecision(
                "uncertain", False, False, 0.0, ("Transformer 不確定性過高",)
            )

        bullish = (
            forecast.bull_probability >= config.minimum_probability
            and forecast.bull_probability > forecast.bear_probability
            and forecast.return_5 >= config.minimum_absolute_return_5
        )
        bearish = (
            forecast.bear_probability >= config.minimum_probability
            and forecast.bear_probability > forecast.bull_probability
            and forecast.return_5 <= -config.minimum_absolute_return_5
        )
        high_volatility = forecast.volatility >= config.high_volatility_threshold
        if bullish:
            regime = "high_volatility_bull" if high_volatility else "bull_trend"
            multiplier = config.high_volatility_multiplier if high_volatility else 1.0
            reasons = ("高波動，降低多頭資金",) if high_volatility else ("多頭趨勢確認",)
            return RegimeDecision(regime, True, False, multiplier, reasons)
        if bearish:
            regime = "high_volatility_bear" if high_volatility else "bear_trend"
            multiplier = config.high_volatility_multiplier if high_volatility else 1.0
            reasons = ("高波動，降低空頭資金",) if high_volatility else ("空頭趨勢確認",)
            return RegimeDecision(regime, False, True, multiplier, reasons)
        return RegimeDecision(
            "range",
            config.neutral_multiplier > 0,
            config.neutral_multiplier > 0,
            config.neutral_multiplier,
            ("方向未獲一致確認",),
        )
