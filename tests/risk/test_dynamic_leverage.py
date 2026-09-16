from __future__ import annotations

import pytest

from ai_quant_trading.risk import (
    DynamicLeverageConfig,
    select_dynamic_leverage,
)
from ai_quant_trading.trading import ModelForecast


def _forecast(
    *,
    bull: float = 0.70,
    bear: float = 0.20,
    uncertainty: float = 0.20,
    volatility: float = 0.01,
    long_success: float | None = None,
    short_success: float | None = None,
) -> ModelForecast:
    return ModelForecast(
        available=True,
        bull_probability=bull,
        bear_probability=bear,
        uncertainty=uncertainty,
        volatility=volatility,
        probability_calibrated=True,
        long_success_probability=bull if long_success is None else long_success,
        short_success_probability=bear if short_success is None else short_success,
        success_probability_calibrated=True,
    )


def test_strong_calibrated_evidence_allows_three_times_leverage() -> None:
    decision = select_dynamic_leverage(
        proposed_target=0.60,
        current_position=0.0,
        current_leverage=1,
        max_margin_fraction=0.20,
        stop_distance_fraction=0.01,
        take_profit_fraction=0.03,
        fee_rate=0.0004,
        slippage_rate=0.0002,
        forecast=_forecast(),
        drawdown=0.01,
        spread_bps=2.0,
        probability_calibrated=True,
        config=DynamicLeverageConfig(enabled=True),
    )

    assert decision.selected_leverage == 3
    assert decision.approved_target == pytest.approx(0.60)
    assert decision.net_expectancy > 0


def test_uncalibrated_probability_can_only_use_one_times_leverage() -> None:
    decision = select_dynamic_leverage(
        proposed_target=-0.60,
        current_position=0.0,
        current_leverage=2,
        max_margin_fraction=0.20,
        stop_distance_fraction=0.01,
        take_profit_fraction=0.03,
        fee_rate=0.0,
        slippage_rate=0.0,
        forecast=_forecast(bull=0.05, bear=0.90),
        drawdown=0.0,
        probability_calibrated=False,
        config=DynamicLeverageConfig(enabled=True),
    )

    assert decision.selected_leverage == 1
    assert decision.evidence_cap == 1
    assert decision.approved_target == pytest.approx(-0.20)
    assert "未校準" in decision.reason_text


def test_negative_net_expectancy_rejects_new_risk() -> None:
    decision = select_dynamic_leverage(
        proposed_target=0.20,
        current_position=0.0,
        current_leverage=1,
        max_margin_fraction=0.20,
        stop_distance_fraction=0.02,
        take_profit_fraction=0.01,
        fee_rate=0.001,
        slippage_rate=0.001,
        forecast=_forecast(bull=0.55, uncertainty=0.10),
        drawdown=0.0,
        probability_calibrated=True,
        config=DynamicLeverageConfig(enabled=True),
    )

    assert decision.approved_target == 0.0
    assert decision.net_expectancy <= 0
    assert "期望值不為正" in decision.reason_text


def test_reduction_is_never_blocked_by_model_evidence() -> None:
    decision = select_dynamic_leverage(
        proposed_target=0.20,
        current_position=0.60,
        current_leverage=3,
        max_margin_fraction=0.20,
        stop_distance_fraction=0.02,
        take_profit_fraction=0.01,
        fee_rate=0.001,
        slippage_rate=0.001,
        forecast=None,
        drawdown=0.20,
        spread_bps=100.0,
        config=DynamicLeverageConfig(enabled=True),
    )

    assert decision.selected_leverage == 3
    assert decision.approved_target == pytest.approx(0.20)
    assert "不阻擋降低風險" in decision.reason_text


def test_disabled_mode_preserves_fixed_leverage_compatibility() -> None:
    decision = select_dynamic_leverage(
        proposed_target=0.40,
        current_position=0.0,
        current_leverage=2,
        max_margin_fraction=0.20,
        stop_distance_fraction=0.01,
        take_profit_fraction=0.02,
        fee_rate=0.0,
        slippage_rate=0.0,
        forecast=None,
        drawdown=0.0,
    )

    assert decision.enabled is False
    assert decision.selected_leverage == 2
    assert decision.approved_target == pytest.approx(0.40)


def test_invalid_threshold_order_is_rejected() -> None:
    with pytest.raises(ValueError, match="3x 的最低勝率"):
        DynamicLeverageConfig(
            leverage_2_min_probability=0.70,
            leverage_3_min_probability=0.60,
        )


def test_regime_probability_alone_cannot_raise_leverage() -> None:
    forecast = ModelForecast(
        available=True,
        bull_probability=0.99,
        bear_probability=0.005,
        uncertainty=0.05,
        volatility=0.005,
        probability_calibrated=True,
    )
    decision = select_dynamic_leverage(
        proposed_target=0.60,
        current_position=0.0,
        current_leverage=1,
        max_margin_fraction=0.20,
        stop_distance_fraction=0.01,
        take_profit_fraction=0.03,
        fee_rate=0.0004,
        slippage_rate=0.0002,
        forecast=forecast,
        drawdown=0.0,
        probability_calibrated=True,
        config=DynamicLeverageConfig(enabled=True),
    )

    assert decision.selected_leverage == 1
    assert decision.approved_target == pytest.approx(0.20)
    assert decision.net_expectancy == 0.0
    assert "停利先於停損" in decision.reason_text
