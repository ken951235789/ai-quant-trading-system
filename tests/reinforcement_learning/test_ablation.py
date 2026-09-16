"""SAC 單變因消融設定測試。"""

from __future__ import annotations

import pytest

from ai_quant_trading.reinforcement_learning import (
    PortfolioEnvConfig,
    build_sac_environment_ablations,
)


def _base() -> PortfolioEnvConfig:
    return PortfolioEnvConfig(
        fee_rate=0.0004,
        slippage_rate=0.0002,
        spread_rate=0.0001,
        max_position_fraction=0.5,
        allow_short=True,
        max_short_fraction=0.5,
        execution_mode="perpetual",
        normalized_action_space=True,
        neutral_action_threshold=0.10,
        turnover_penalty=0.01,
        minimum_gross_target_cost_multiple=4.0,
        minimum_net_risk_reward=1.5,
        rebalance_deadband=0.10,
    )


def test_sac_ablation_matrix_changes_one_research_question_at_a_time() -> None:
    specs = {item.name: item for item in build_sac_environment_ablations(_base())}

    assert specs["legacy_control"].environment.action_semantics == "legacy_target"
    assert specs["hold_close_semantics"].environment.action_semantics == "hold_close_target"
    assert specs["no_extra_turnover_penalty"].environment.turnover_penalty == 0.0
    assert specs["cost_aligned_turnover_penalty"].environment.turnover_penalty < 0.001
    assert specs["entry_exit_filters_off"].environment.minimum_net_risk_reward == 0.0
    assert (
        specs["relaxed_entry_exit_filters"].environment.minimum_gross_target_cost_multiple
        == pytest.approx(1.5)
    )


def test_sac_ablation_requires_normalized_long_short_environment() -> None:
    with pytest.raises(ValueError, match="SAC 多空消融"):
        build_sac_environment_ablations(PortfolioEnvConfig())
