"""依固定順序協調市場狀態分析與資金管理。"""

from __future__ import annotations

import math
from typing import Mapping

from ai_quant_trading.trading.capital import CapitalManagerConfig
from ai_quant_trading.trading.capital import CapitalManager
from ai_quant_trading.trading.contracts import DecisionTrace, ModelForecast, TradeIntent
from ai_quant_trading.trading.regime import MarketRegimeDetector


class TradingDecisionOrchestrator:
    """把 SAC 意圖依序交給 Regime Detector 與 Capital Manager。"""

    def __init__(
        self,
        *,
        regime_detector: MarketRegimeDetector | None = None,
        capital_manager: CapitalManager | None = None,
    ) -> None:
        self.regime_detector = regime_detector or MarketRegimeDetector()
        self.capital_manager = capital_manager or CapitalManager()

    def evaluate(
        self,
        intent: TradeIntent,
        forecast: ModelForecast,
        *,
        drawdown: float = 0.0,
    ) -> DecisionTrace:
        regime = self.regime_detector.classify(forecast)
        capital = self.capital_manager.allocate(
            intent,
            forecast,
            regime,
            drawdown=drawdown,
        )
        return DecisionTrace(intent, forecast, regime, capital)


def apply_target_thresholds(
    proposed_target: float,
    current_position: float,
    *,
    entry_threshold: float,
    exit_threshold: float,
) -> float:
    """套用共用進出場遲滯，避免弱訊號開倉或直接反手。"""
    if not 0 <= exit_threshold < entry_threshold <= 1:
        raise ValueError("目標門檻必須滿足 0 <= exit < entry <= 1")
    if abs(current_position) <= 1e-9:
        return proposed_target if abs(proposed_target) >= entry_threshold else 0.0
    if abs(proposed_target) <= exit_threshold:
        return 0.0
    if proposed_target * current_position < 0 and abs(proposed_target) < entry_threshold:
        return 0.0
    return proposed_target


def govern_model_target(
    *,
    proposed_target: float,
    current_position: float,
    current: Mapping[str, object],
    decision_id: str,
    timestamp: str,
    symbol: str,
    drawdown: float,
    max_long_fraction: float,
    max_short_fraction: float,
    hard_drawdown_limit: float,
    soft_drawdown_limit: float | None = None,
) -> tuple[float, dict[str, object]]:
    """讓 Paper 與 Live 共用相同的 Transformer、Regime 與資金治理。"""
    if "transformer_available" not in current:
        return proposed_target, {}
    probability_names = (
        "transformer_bull_probability",
        "transformer_bear_probability",
        "transformer_uncertainty",
        "transformer_available",
    )
    try:
        probabilities = tuple(float(current.get(name, float("nan"))) for name in probability_names)
    except (TypeError, ValueError):
        probabilities = (float("nan"),) * len(probability_names)
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
        return proposed_target, {
            "market_regime": "pre_normalized",
            "decision_guard_reason": "偵測到已標準化的 Transformer 欄位，略過二次趨勢解讀",
        }

    hard = max(float(hard_drawdown_limit), 1e-4)
    soft = hard * 0.5 if soft_drawdown_limit is None else float(soft_drawdown_limit)
    soft = min(max(soft, hard * 0.01), hard * 0.75)
    manager = CapitalManager(
        CapitalManagerConfig(
            max_long_fraction=max_long_fraction,
            max_short_fraction=max_short_fraction,
            soft_drawdown_limit=soft,
            hard_drawdown_limit=hard,
        )
    )
    intent = TradeIntent(
        decision_id=decision_id,
        timestamp=timestamp,
        symbol=symbol,
        proposed_target=float(max(min(proposed_target, 1.0), -1.0)),
        current_position=float(max(min(current_position, 1.0), -1.0)),
    )
    trace = TradingDecisionOrchestrator(capital_manager=manager).evaluate(
        intent,
        ModelForecast.from_mapping(current),
        drawdown=drawdown,
    )
    details = trace.to_dict()
    details["decision_guard_reason"] = details.get("governance_reasons") or (
        "SAC 與 Transformer 市場方向一致"
    )
    return trace.capital.allocated_target, details
