"""停損停利與部位 sizing 測試。"""

from __future__ import annotations

import pytest

from ai_quant_trading.risk import (
    RiskConfig,
    calculate_position_size,
    calculate_stop_loss,
    calculate_take_profit,
)


def test_fixed_stop_and_take_profit() -> None:
    config = RiskConfig(fixed_stop_loss_pct=0.03, take_profit_pct=0.06)
    assert calculate_stop_loss(100, config) == pytest.approx(97)
    assert calculate_take_profit(100, config) == pytest.approx(106)


def test_short_stop_and_take_profit_are_direction_aware() -> None:
    config = RiskConfig(fixed_stop_loss_pct=0.03, take_profit_pct=0.06)

    assert calculate_stop_loss(100, config, side="short") == pytest.approx(103)
    assert calculate_take_profit(100, config, side="short") == pytest.approx(94)


def test_atr_stop_uses_configured_multiplier() -> None:
    config = RiskConfig(stop_loss_mode="atr", atr_multiplier=2.5)
    assert calculate_stop_loss(100, config, atr=4) == pytest.approx(90)


def test_position_size_limits_loss_to_risk_budget() -> None:
    result = calculate_position_size(
        equity=1000,
        cash=1000,
        entry_price=100,
        stop_loss=95,
        fee_rate=0,
        max_risk_per_trade=0.01,
        max_position_fraction=1,
    )
    assert result.risk_budget == pytest.approx(10)
    assert result.quantity == pytest.approx(2)
    assert result.position_notional == pytest.approx(200)


def test_position_size_is_also_limited_by_available_cash() -> None:
    result = calculate_position_size(
        equity=1000,
        cash=1000,
        entry_price=100,
        stop_loss=99,
        fee_rate=0,
        max_risk_per_trade=0.5,
        max_position_fraction=0.25,
    )
    assert result.quantity == pytest.approx(2.5)
