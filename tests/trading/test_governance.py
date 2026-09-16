"""市場狀態、資金管理與決策協調器測試。"""

from __future__ import annotations

import pytest

from ai_quant_trading.trading import (
    CapitalManager,
    CapitalManagerConfig,
    MarketRegimeConfig,
    MarketRegimeDetector,
    ModelForecast,
    TradeIntent,
    TradingDecisionOrchestrator,
)


def _intent(target: float, current: float = 0.0) -> TradeIntent:
    return TradeIntent(
        decision_id="BTCUSDT-15m-20260820T0000Z",
        timestamp="2026-08-20T00:00:00Z",
        symbol="BTC/USDT",
        proposed_target=target,
        current_position=current,
    )


def _bullish(**changes: float) -> ModelForecast:
    values = {
        "available": True,
        "return_1": 0.001,
        "return_5": 0.004,
        "return_20": 0.008,
        "volatility": 0.01,
        "bull_probability": 0.70,
        "bear_probability": 0.15,
        "uncertainty": 0.20,
    }
    values.update(changes)
    return ModelForecast(**values)


def test_bullish_regime_allows_only_long_risk() -> None:
    decision = MarketRegimeDetector().classify(_bullish())

    assert decision.regime == "bull_trend"
    assert decision.allow_long
    assert not decision.allow_short


def test_high_volatility_reduces_capital() -> None:
    orchestrator = TradingDecisionOrchestrator(
        regime_detector=MarketRegimeDetector(
            MarketRegimeConfig(high_volatility_threshold=0.02)
        ),
        capital_manager=CapitalManager(
            CapitalManagerConfig(target_forecast_volatility=0.015)
        ),
    )

    trace = orchestrator.evaluate(_intent(0.50), _bullish(volatility=0.03))

    assert trace.regime.regime == "high_volatility_bull"
    assert 0 < trace.capital.allocated_target < 0.50
    assert trace.capital.volatility_multiplier == pytest.approx(0.50)


def test_unconfirmed_direction_cannot_open_new_position() -> None:
    neutral = _bullish(
        bull_probability=0.30,
        bear_probability=0.30,
        return_5=0.0,
    )

    trace = TradingDecisionOrchestrator().evaluate(_intent(0.30), neutral)

    assert trace.regime.regime == "range"
    assert trace.capital.allocated_target == 0.0


def test_bearish_regime_can_force_long_position_flat() -> None:
    bearish = _bullish(
        return_5=-0.004,
        bull_probability=0.10,
        bear_probability=0.70,
    )

    trace = TradingDecisionOrchestrator().evaluate(_intent(0.10, 0.30), bearish)

    assert trace.capital.allocated_target == 0.0
    assert any("既有多頭先平倉" in reason for reason in trace.capital.reasons)


def test_reversal_flattens_before_opening_other_direction() -> None:
    trace = TradingDecisionOrchestrator().evaluate(_intent(-0.30, 0.20), _bullish())

    assert trace.capital.allocated_target == 0.0
    assert any("先平倉" in reason for reason in trace.capital.reasons)
