"""雙專家共用 Risk Governor 測試。"""

import pytest

from ai_quant_trading.risk import ExpertCapitalAllocation, govern_target_position


def test_governor_applies_deadband_and_soft_drawdown() -> None:
    deadband = govern_target_position(
        proposed_target=0.12,
        current_position=0.10,
        drawdown=-0.01,
        holding_bars=10,
        max_position_fraction=0.20,
        hard_drawdown_limit=0.12,
        soft_drawdown_limit=0.08,
        rebalance_deadband=0.05,
    )
    reduced = govern_target_position(
        proposed_target=0.20,
        current_position=0.05,
        drawdown=-0.09,
        holding_bars=0,
        max_position_fraction=0.20,
        hard_drawdown_limit=0.12,
        soft_drawdown_limit=0.08,
        soft_drawdown_multiplier=0.5,
        minimum_holding_bars=20,
    )

    assert deadband.approved_target == pytest.approx(0.10)
    assert "rebalance_deadband" in deadband.reasons
    assert reduced.approved_target == pytest.approx(0.10)
    assert reduced.risk_multiplier == pytest.approx(0.5)


def test_governor_hard_drawdown_always_flattens() -> None:
    decision = govern_target_position(
        proposed_target=0.20,
        current_position=0.10,
        drawdown=-0.13,
        holding_bars=0,
        max_position_fraction=0.20,
        hard_drawdown_limit=0.12,
        minimum_holding_bars=20,
    )

    assert decision.halted
    assert decision.approved_target == 0
    assert decision.reasons == ("hard_drawdown",)


def test_governor_never_blocks_reduction_during_minimum_holding_period() -> None:
    decision = govern_target_position(
        proposed_target=0.02,
        current_position=0.10,
        drawdown=0.0,
        holding_bars=1,
        max_position_fraction=0.20,
        hard_drawdown_limit=0.12,
        rebalance_deadband=0.10,
        minimum_holding_bars=20,
    )

    assert decision.approved_target == pytest.approx(0.02)
    assert "minimum_holding_period" not in decision.reasons


def test_governor_flattens_instead_of_reversing_during_minimum_holding_period() -> None:
    decision = govern_target_position(
        proposed_target=-0.10,
        current_position=0.10,
        drawdown=0.0,
        holding_bars=1,
        max_position_fraction=0.20,
        allow_short=True,
        max_short_fraction=0.20,
        hard_drawdown_limit=0.12,
        minimum_holding_bars=20,
    )

    assert decision.approved_target == 0.0
    assert "minimum_holding_blocks_reversal" in decision.reasons


def test_governor_tapers_risk_between_soft_and_hard_drawdown() -> None:
    shallow = govern_target_position(
        proposed_target=0.20,
        current_position=0.0,
        drawdown=-0.09,
        holding_bars=0,
        max_position_fraction=0.20,
        hard_drawdown_limit=0.12,
        soft_drawdown_limit=0.08,
        soft_drawdown_multiplier=0.5,
        drawdown_curve_exponent=1.0,
    )
    deep = govern_target_position(
        proposed_target=0.20,
        current_position=0.0,
        drawdown=-0.11,
        holding_bars=0,
        max_position_fraction=0.20,
        hard_drawdown_limit=0.12,
        soft_drawdown_limit=0.08,
        soft_drawdown_multiplier=0.5,
        drawdown_curve_exponent=1.0,
    )

    assert shallow.risk_multiplier == pytest.approx(0.375)
    assert deep.risk_multiplier == pytest.approx(0.125)
    assert deep.approved_target < shallow.approved_target
    assert "adaptive_drawdown_reduction" in deep.reasons


def test_governor_allows_bounded_short_target() -> None:
    decision = govern_target_position(
        proposed_target=-0.20,
        current_position=0.0,
        drawdown=0.0,
        holding_bars=0,
        max_position_fraction=0.10,
        allow_short=True,
        max_short_fraction=0.05,
        hard_drawdown_limit=0.06,
    )

    assert decision.approved_target == pytest.approx(-0.05)


def test_expert_capital_allocation_keeps_reserve() -> None:
    allocation = ExpertCapitalAllocation().allocate(50_000)

    assert allocation == {
        "long_term": 35_000,
        "short_term": 10_000,
        "reserve": 5_000,
    }
