"""把 SAC 目標曝險轉成保守的實際資金配置。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ai_quant_trading.trading.contracts import (
    CapitalDecision,
    ModelForecast,
    RegimeDecision,
    TradeIntent,
)


@dataclass(frozen=True, slots=True)
class CapitalManagerConfig:
    """方向上限、波動目標與回撤縮倉設定。"""

    max_long_fraction: float = 0.50
    max_short_fraction: float = 0.50
    target_forecast_volatility: float | None = 0.015
    confidence_reference: float = 0.40
    soft_drawdown_limit: float = 0.05
    hard_drawdown_limit: float = 0.10
    soft_drawdown_multiplier: float = 0.50

    def __post_init__(self) -> None:
        if not 0 < self.max_long_fraction <= 1 or not 0 <= self.max_short_fraction <= 1:
            raise ValueError("多空部位上限必須介於 0 與 1")
        if self.target_forecast_volatility is not None and self.target_forecast_volatility <= 0:
            raise ValueError("target_forecast_volatility 必須大於 0 或設為 None")
        if not 0 < self.confidence_reference <= 1:
            raise ValueError("confidence_reference 必須介於 0 與 1")
        if not 0 < self.soft_drawdown_limit < self.hard_drawdown_limit <= 1:
            raise ValueError("回撤門檻必須符合 0 < soft < hard <= 1")
        if not 0 < self.soft_drawdown_multiplier <= 1:
            raise ValueError("soft_drawdown_multiplier 必須介於 0 與 1")


class CapitalManager:
    """不阻擋減倉，僅縮小新開倉、加碼與反手風險。"""

    def __init__(self, config: CapitalManagerConfig | None = None) -> None:
        self.config = config or CapitalManagerConfig()

    def allocate(
        self,
        intent: TradeIntent,
        forecast: ModelForecast,
        regime: RegimeDecision,
        *,
        drawdown: float = 0.0,
    ) -> CapitalDecision:
        config = self.config
        proposed = float(
            np.clip(intent.proposed_target, -config.max_short_fraction, config.max_long_fraction)
        )
        current = float(
            np.clip(intent.current_position, -config.max_short_fraction, config.max_long_fraction)
        )
        reasons: list[str] = []

        if current > 1e-12 and regime.allow_short and not regime.allow_long:
            return CapitalDecision(
                proposed,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                regime.risk_multiplier,
                ("市場狀態明確轉空，既有多頭先平倉",),
            )
        if current < -1e-12 and regime.allow_long and not regime.allow_short:
            return CapitalDecision(
                proposed,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                regime.risk_multiplier,
                ("市場狀態明確轉多，既有空頭先平倉",),
            )
        reducing = proposed * current >= 0 and abs(proposed) <= abs(current) + 1e-12
        if reducing:
            return CapitalDecision(proposed, proposed, 1.0, 1.0, 1.0, 1.0, 1.0, ())
        if proposed * current < 0 and abs(current) > 1e-12:
            return CapitalDecision(
                proposed, 0.0, 0.0, 0.0, 1.0, 1.0, regime.risk_multiplier,
                ("方向反轉先平倉，下一根再評估新方向",),
            )
        if proposed > 0 and not regime.allow_long:
            return CapitalDecision(
                proposed, current, 0.0, 0.0, 1.0, 1.0, regime.risk_multiplier,
                ("市場狀態不允許增加多頭風險",),
            )
        if proposed < 0 and not regime.allow_short:
            return CapitalDecision(
                proposed, current, 0.0, 0.0, 1.0, 1.0, regime.risk_multiplier,
                ("市場狀態不允許增加空頭風險",),
            )

        directional_probability = (
            forecast.bull_probability if proposed >= 0 else forecast.bear_probability
        )
        confidence_score = directional_probability * (1.0 - forecast.uncertainty)
        confidence_multiplier = float(
            np.clip(confidence_score / config.confidence_reference, 0.0, 1.0)
        )
        volatility_multiplier = 1.0
        if (
            config.target_forecast_volatility is not None
            and forecast.volatility > config.target_forecast_volatility
        ):
            volatility_multiplier = float(
                np.clip(config.target_forecast_volatility / forecast.volatility, 0.0, 1.0)
            )
            reasons.append("預測波動高於資金目標，縮小部位")
        drawdown_depth = max(-float(drawdown), 0.0)
        drawdown_multiplier = 1.0
        if drawdown_depth >= config.hard_drawdown_limit:
            drawdown_multiplier = 0.0
            reasons.append("達到資金管理硬性回撤，目標歸零")
        elif drawdown_depth >= config.soft_drawdown_limit:
            drawdown_multiplier = config.soft_drawdown_multiplier
            reasons.append("達到資金管理軟性回撤，縮小部位")
        total = float(
            np.clip(
                confidence_multiplier
                * volatility_multiplier
                * drawdown_multiplier
                * regime.risk_multiplier,
                0.0,
                1.0,
            )
        )
        if not math.isfinite(total):
            total = 0.0
            reasons.append("資金倍率無效，目標歸零")
        delta = proposed - current
        allocated = current + delta * total
        if total < 1.0:
            reasons.append("已依模型信心與市場風險調整資金")
        return CapitalDecision(
            proposed,
            float(np.clip(allocated, -config.max_short_fraction, config.max_long_fraction)),
            total,
            confidence_multiplier,
            volatility_multiplier,
            drawdown_multiplier,
            regime.risk_multiplier,
            tuple(dict.fromkeys(reasons)),
        )
