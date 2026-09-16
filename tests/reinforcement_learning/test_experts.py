"""長期與短期交易專家預設測試。"""

import pytest

from ai_quant_trading.reinforcement_learning import (
    build_expert_profile,
    validate_expert_interval,
)


def test_expert_profiles_use_different_risk_scales() -> None:
    long_term = build_expert_profile("long_term")
    short_term = build_expert_profile("short_term")

    assert long_term.environment.expert_kind == "long_term"
    assert short_term.environment.expert_kind == "short_term"
    assert long_term.environment.minimum_holding_bars > short_term.environment.minimum_holding_bars
    assert long_term.environment.max_drawdown_limit > short_term.environment.max_drawdown_limit
    assert long_term.paper_risk.atr_multiplier > short_term.paper_risk.atr_multiplier
    assert short_term.environment.allow_short
    assert short_term.environment.max_short_fraction == pytest.approx(0.50)
    assert short_term.environment.max_position_fraction == pytest.approx(0.50)
    assert short_term.environment.max_leverage == pytest.approx(3.0)
    assert short_term.environment.neutral_action_threshold == pytest.approx(0.10)
    assert short_term.environment.action_semantics == "hold_close_target"
    assert short_term.environment.turnover_penalty < 0.001
    assert short_term.environment.minimum_net_risk_reward == pytest.approx(0.75)
    assert short_term.environment.take_profit_distance == pytest.approx(0.015)
    assert short_term.environment.include_trade_plan_context
    assert short_term.environment.initial_capital_randomization > 0
    assert short_term.environment.slippage_randomization > 0
    assert not long_term.environment.allow_short
    assert (
        long_term.portfolio_risk.max_strategy_exposure
        > short_term.portfolio_risk.max_strategy_exposure
    )
    assert short_term.portfolio_risk.max_daily_loss < long_term.portfolio_risk.max_daily_loss


def test_expert_interval_validation_rejects_mixed_horizon() -> None:
    validate_expert_interval("long_term", "1d")
    validate_expert_interval("short_term", "15m")

    with pytest.raises(ValueError, match="長期交易專家不支援 1h"):
        validate_expert_interval("long_term", "1h")
    with pytest.raises(ValueError, match="短期交易專家不支援 1h"):
        validate_expert_interval("short_term", "1h")
