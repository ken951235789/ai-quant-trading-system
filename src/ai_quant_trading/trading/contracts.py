"""模型、資金管理與風控之間共用的不可變資料合約。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


def _finite_fraction(value: float, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} 必須是介於 0 與 1 的有限數值")
    return numeric


@dataclass(frozen=True, slots=True)
class ModelForecast:
    """Transformer 對未來報酬、波動與方向機率的標準輸出。"""

    available: bool
    return_1: float = 0.0
    return_5: float = 0.0
    return_20: float = 0.0
    volatility: float = 0.0
    bull_probability: float = 0.0
    bear_probability: float = 0.0
    uncertainty: float = 1.0
    probability_calibrated: bool = False
    long_success_probability: float | None = None
    short_success_probability: float | None = None
    success_probability_calibrated: bool = False

    def __post_init__(self) -> None:
        values = (self.return_1, self.return_5, self.return_20, self.volatility)
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("Transformer 報酬與波動輸出必須是有限數值")
        if self.volatility < 0:
            raise ValueError("Transformer 波動率不可小於 0")
        _finite_fraction(self.bull_probability, "bull_probability")
        _finite_fraction(self.bear_probability, "bear_probability")
        _finite_fraction(self.uncertainty, "uncertainty")
        if self.long_success_probability is not None:
            _finite_fraction(self.long_success_probability, "long_success_probability")
        if self.short_success_probability is not None:
            _finite_fraction(self.short_success_probability, "short_success_probability")

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "ModelForecast":
        """從即時特徵列建立預測；不可用時採安全的中性預設值。"""

        def number(name: str, default: float) -> float:
            try:
                numeric = float(values.get(name, default))
            except (TypeError, ValueError):
                return default
            return numeric if math.isfinite(numeric) else default

        def optional_number(name: str) -> float | None:
            if name not in values:
                return None
            try:
                numeric = float(values[name])
            except (TypeError, ValueError):
                return None
            return min(max(numeric, 0.0), 1.0) if math.isfinite(numeric) else None

        return cls(
            available=number("transformer_available", 0.0) >= 0.5,
            return_1=number("transformer_return_1", 0.0),
            return_5=number("transformer_return_5", 0.0),
            return_20=number("transformer_return_20", 0.0),
            volatility=max(number("transformer_volatility", 0.0), 0.0),
            bull_probability=min(max(number("transformer_bull_probability", 0.0), 0.0), 1.0),
            bear_probability=min(max(number("transformer_bear_probability", 0.0), 0.0), 1.0),
            uncertainty=min(max(number("transformer_uncertainty", 1.0), 0.0), 1.0),
            probability_calibrated=(
                number("transformer_probability_calibrated", 0.0) >= 0.5
            ),
            long_success_probability=optional_number(
                "transformer_long_success_probability"
            ),
            short_success_probability=optional_number(
                "transformer_short_success_probability"
            ),
            success_probability_calibrated=(
                number("transformer_success_probability_calibrated", 0.0) >= 0.5
            ),
        )

    @property
    def directional_confidence(self) -> float:
        """方向機率與不確定性合成的保守信心分數。"""
        return max(self.bull_probability, self.bear_probability) * (1.0 - self.uncertainty)


@dataclass(frozen=True, slots=True)
class TradeIntent:
    """SAC 提出的目標曝險，不代表已通過風控或已送單。"""

    decision_id: str
    timestamp: str
    symbol: str
    proposed_target: float
    current_position: float

    def __post_init__(self) -> None:
        if not self.decision_id.strip() or not self.symbol.strip():
            raise ValueError("decision_id 與 symbol 不可為空")
        for name, value in (
            ("proposed_target", self.proposed_target),
            ("current_position", self.current_position),
        ):
            if not math.isfinite(float(value)) or not -1.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} 必須介於 -1 與 1")


@dataclass(frozen=True, slots=True)
class RegimeDecision:
    """市場狀態分析師對方向與可承擔風險的判斷。"""

    regime: str
    allow_long: bool
    allow_short: bool
    risk_multiplier: float
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _finite_fraction(self.risk_multiplier, "risk_multiplier")


@dataclass(frozen=True, slots=True)
class CapitalDecision:
    """資金管理員套用信心、波動與回撤後的目標曝險。"""

    proposed_target: float
    allocated_target: float
    total_multiplier: float
    confidence_multiplier: float
    volatility_multiplier: float
    drawdown_multiplier: float
    regime_multiplier: float
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DecisionTrace:
    """單一決策經過市場狀態與資金配置後的完整可解釋紀錄。"""

    intent: TradeIntent
    forecast: ModelForecast
    regime: RegimeDecision
    capital: CapitalDecision

    def to_dict(self) -> dict[str, object]:
        """轉成可直接寫入 JSON 或 UI 狀態的扁平資料。"""
        trend = {
            "bull_trend": "偏多",
            "high_volatility_bull": "偏多",
            "bear_trend": "偏空",
            "high_volatility_bear": "偏空",
            "range": "中性",
            "uncertain": "中性",
            "unavailable": "不可用",
        }.get(self.regime.regime, self.regime.regime)
        return {
            "decision_id": self.intent.decision_id,
            "timestamp": self.intent.timestamp,
            "symbol": self.intent.symbol,
            "model_target_fraction": self.intent.proposed_target,
            "allocated_target_fraction": self.capital.allocated_target,
            "transformer_target_fraction": self.capital.allocated_target,
            "transformer_trend": trend,
            "transformer_return_1": self.forecast.return_1,
            "transformer_return_5": self.forecast.return_5,
            "transformer_return_20": self.forecast.return_20,
            "transformer_volatility": self.forecast.volatility,
            "transformer_bull_probability": self.forecast.bull_probability,
            "transformer_bear_probability": self.forecast.bear_probability,
            "transformer_uncertainty": self.forecast.uncertainty,
            "transformer_probability_calibrated": self.forecast.probability_calibrated,
            "market_regime": self.regime.regime,
            "regime_risk_multiplier": self.regime.risk_multiplier,
            "capital_multiplier": self.capital.total_multiplier,
            "confidence_multiplier": self.capital.confidence_multiplier,
            "volatility_multiplier": self.capital.volatility_multiplier,
            "drawdown_multiplier": self.capital.drawdown_multiplier,
            "governance_reasons": "；".join(
                dict.fromkeys((*self.regime.reasons, *self.capital.reasons))
            ),
        }
