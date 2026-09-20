"""Gymnasium Portfolio Environment 成交與 Reward 測試。"""

from math import log

from gymnasium.utils.env_checker import check_env
import numpy as np
import pandas as pd
import pytest

from ai_quant_trading.reinforcement_learning import (
    DiscretePortfolioActionWrapper,
    PortfolioEnvConfig,
    PortfolioTradingEnv,
    map_continuous_action,
    run_environment_diagnostic,
)


def make_environment_frame(closes: list[float], opens: list[float] | None = None) -> pd.DataFrame:
    opens = opens or closes
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=len(closes), freq="D", tz="UTC"),
            "open": opens,
            "high": np.maximum(opens, closes),
            "low": np.minimum(opens, closes),
            "close": closes,
            "feature_a": np.linspace(-1, 1, len(closes)),
            "feature_b": np.linspace(1, -1, len(closes)),
        }
    )


def test_environment_is_gymnasium_compatible() -> None:
    env = PortfolioTradingEnv(
        make_environment_frame([100, 101, 102, 103]),
        ["feature_a", "feature_b"],
    )

    check_env(env, skip_render_check=True)
    observation, info = env.reset(seed=42)

    assert observation.shape == (7,)
    assert env.observation_space.contains(observation)
    assert info["equity"] == 1000


def test_legacy_environment_keeps_original_observation_size() -> None:
    env = PortfolioTradingEnv(
        make_environment_frame([100, 101, 102]),
        ["feature_a", "feature_b"],
        PortfolioEnvConfig(include_position_context=False),
    )

    observation, _ = env.reset()

    assert observation.shape == (5,)
    assert env.observation_space.contains(observation)


def test_action_executes_at_next_open_and_marks_next_close() -> None:
    config = PortfolioEnvConfig(
        initial_capital=1000,
        fee_rate=0,
        slippage_rate=0,
        drawdown_penalty=0,
        turnover_penalty=0,
    )
    env = PortfolioTradingEnv(
        make_environment_frame([100, 110, 120], opens=[100, 100, 120]),
        ["feature_a", "feature_b"],
        config,
    )
    env.reset()

    observation, reward, terminated, truncated, bought = env.step(np.array([1.0]))

    assert bought["side"] == "BUY"
    assert bought["quantity"] == pytest.approx(10)
    assert bought["average_entry_price"] == pytest.approx(100)
    assert bought["unrealized_return"] == pytest.approx(0.10)
    assert bought["holding_bars"] == 1
    assert observation[-2] == pytest.approx(0.10)
    assert observation[-1] == pytest.approx(1 / 8)
    assert bought["equity"] == pytest.approx(1100)
    assert reward == pytest.approx(log(1.1))
    assert not terminated
    assert not truncated

    _, _, terminated, _, sold = env.step(np.array([0.0]))
    assert sold["side"] == "SELL"
    assert sold["equity"] == pytest.approx(1200)
    assert sold["average_entry_price"] == 0
    assert sold["holding_bars"] == 0
    assert terminated


def test_maximum_drawdown_terminates_episode() -> None:
    env = PortfolioTradingEnv(
        make_environment_frame([100, 50, 49], opens=[100, 100, 50]),
        ["feature_a"],
        PortfolioEnvConfig(
            fee_rate=0,
            slippage_rate=0,
            max_drawdown_limit=0.30,
        ),
    )
    env.reset()

    _, _, terminated, _, info = env.step(np.array([1.0]))

    assert terminated
    assert info["risk_terminated"]
    assert info["drawdown"] == pytest.approx(-0.5)
    with pytest.raises(RuntimeError, match="Episode 已結束"):
        env.step(np.array([0.0]))


def test_maximum_drawdown_adds_terminal_penalty() -> None:
    env = PortfolioTradingEnv(
        make_environment_frame([100, 50], opens=[100, 100]),
        ["feature_a"],
        PortfolioEnvConfig(
            fee_rate=0,
            slippage_rate=0,
            drawdown_penalty=0,
            turnover_penalty=0,
            downside_penalty=0,
            concentration_penalty=0,
            max_drawdown_limit=0.30,
            risk_termination_penalty=0.75,
        ),
    )
    env.reset()

    _, reward, terminated, _, info = env.step(np.array([1.0]))

    assert terminated
    assert info["risk_termination_penalty"] == pytest.approx(0.75)
    assert reward == pytest.approx(log(0.5) - 0.75)


def test_diagnostic_produces_finite_account_rows() -> None:
    env = PortfolioTradingEnv(
        make_environment_frame([100, 101, 102, 103, 104, 105]),
        ["feature_a", "feature_b"],
    )

    diagnostic = run_environment_diagnostic(env, max_steps=4)

    assert len(diagnostic) == 4
    assert set(["target_fraction", "equity", "reward", "drawdown"]).issubset(diagnostic.columns)
    assert np.isfinite(diagnostic[["equity", "reward", "drawdown"]].to_numpy()).all()


def test_rebalance_deadband_suppresses_small_position_changes() -> None:
    env = PortfolioTradingEnv(
        make_environment_frame([100, 100, 100], opens=[100, 100, 100]),
        ["feature_a"],
        PortfolioEnvConfig(
            fee_rate=0,
            slippage_rate=0,
            rebalance_deadband=0.05,
        ),
    )
    env.reset()
    env.step(np.array([0.20]))

    _, _, _, _, info = env.step(np.array([0.22]))

    assert info["side"] == "HOLD"
    assert info["target_fraction"] == pytest.approx(0.20)
    assert info["risk_reasons"] == "rebalance_deadband"


def test_short_action_profits_from_decline_and_covers_without_residue() -> None:
    env = PortfolioTradingEnv(
        make_environment_frame([100, 90, 90], opens=[100, 100, 90]),
        ["feature_a"],
        PortfolioEnvConfig(
            initial_capital=1000,
            fee_rate=0,
            slippage_rate=0,
            drawdown_penalty=0,
            turnover_penalty=0,
            allow_short=True,
            max_short_fraction=0.05,
        ),
    )
    env.reset()

    observation, reward, _, _, opened = env.step(np.array([-0.05]))

    assert env.action_space.low[0] == pytest.approx(-0.05)
    assert opened["side"] == "SELL"
    assert opened["quantity"] == pytest.approx(-0.5)
    assert opened["equity"] == pytest.approx(1005)
    assert opened["unrealized_return"] == pytest.approx(0.10)
    assert observation[-4] < 0
    assert reward == pytest.approx(log(1.005))

    _, _, terminated, _, covered = env.step(np.array([0.0]))

    assert covered["side"] == "BUY"
    assert covered["quantity"] == 0
    assert covered["average_entry_price"] == 0
    assert covered["equity"] == pytest.approx(1005)
    assert terminated


def test_short_position_pays_configured_carry_cost() -> None:
    env = PortfolioTradingEnv(
        make_environment_frame([100, 100, 100], opens=[100, 100, 100]),
        ["feature_a"],
        PortfolioEnvConfig(
            initial_capital=1000,
            fee_rate=0,
            slippage_rate=0,
            allow_short=True,
            max_short_fraction=0.05,
            short_borrow_rate_annual=0.365,
        ),
    )
    env.reset()

    _, _, _, _, info = env.step(np.array([-0.05]))

    expected_carry = 50 * 0.365 / 252
    assert info["short_carry_cost"] == pytest.approx(expected_carry)
    assert info["equity"] == pytest.approx(1000 - expected_carry)


def _perpetual_frame(rows: int = 6) -> pd.DataFrame:
    close = np.full(rows, 100.0)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="15min", tz="UTC"),
            "symbol": "BTC/USDT",
            "exchange": "binance_futures",
            "interval": "15m",
            "open": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "feature_a": np.linspace(-1, 1, rows),
            "funding_rate": 0.0001,
            "spread_bps": 1.0,
            "expected_return": 0.02,
            "event_blackout": 0.0,
        }
    )


def _perpetual_config(**changes: object) -> PortfolioEnvConfig:
    values = {
        "initial_capital": 1000.0,
        "fee_rate": 0.0,
        "slippage_rate": 0.0,
        "spread_rate": 0.0,
        "max_position_fraction": 0.5,
        "allow_short": True,
        "max_short_fraction": 0.5,
        "execution_mode": "perpetual",
        "normalized_action_space": True,
        "include_risk_context": True,
        "leverage": 2.0,
        "max_leverage": 3.0,
        "max_margin_fraction": 0.2,
        "risk_per_trade": 0.0025,
        "minimum_stop_distance": 0.004,
        "maximum_stop_distance": 0.015,
        "max_drawdown_limit": 0.10,
    }
    values.update(changes)
    return PortfolioEnvConfig(**values)


def test_perpetual_position_is_capped_by_isolated_margin_and_risk() -> None:
    env = PortfolioTradingEnv(_perpetual_frame(), ["feature_a"], _perpetual_config())
    observation, _ = env.reset()

    _, _, _, _, info = env.step(np.array([1.0], dtype=np.float32))

    assert observation.shape == (13,)
    assert info["proposed_target_fraction"] == pytest.approx(0.5)
    assert info["target_fraction"] == pytest.approx(0.4)
    assert info["margin_used"] == pytest.approx(200.0, abs=0.5)
    assert info["effective_leverage"] == pytest.approx(0.4, abs=0.01)
    assert "risk_position_cap" in info["risk_reasons"]


def test_perpetual_reverse_flattens_before_opening_other_direction() -> None:
    env = PortfolioTradingEnv(_perpetual_frame(), ["feature_a"], _perpetual_config())
    env.reset()
    env.step(np.array([1.0], dtype=np.float32))

    _, _, _, _, info = env.step(np.array([-1.0], dtype=np.float32))

    assert info["target_fraction"] == 0.0
    assert info["quantity"] == 0.0
    assert "reverse_flatten_first" in info["risk_reasons"]


def test_event_and_cost_filters_reject_new_risk() -> None:
    frame = _perpetual_frame()
    frame.loc[0, "event_blackout"] = 1.0
    env = PortfolioTradingEnv(frame, ["feature_a"], _perpetual_config())
    env.reset()

    _, _, _, _, event_info = env.step(np.array([1.0], dtype=np.float32))

    assert event_info["quantity"] == 0.0
    assert "event_blackout" in event_info["risk_reasons"]

    cheap_target = _perpetual_frame()
    cheap_target.loc[0, "expected_return"] = 0.001
    env = PortfolioTradingEnv(
        cheap_target,
        ["feature_a"],
        _perpetual_config(
            fee_rate=0.0005,
            slippage_rate=0.0005,
            minimum_gross_target_cost_multiple=4.0,
        ),
    )
    env.reset()
    _, _, _, _, cost_info = env.step(np.array([1.0], dtype=np.float32))

    assert cost_info["quantity"] == 0.0
    assert "insufficient_gross_target" in cost_info["risk_reasons"]


def test_sac_neutral_zone_and_net_risk_reward_reject_weak_entries() -> None:
    frame = _perpetual_frame()
    env = PortfolioTradingEnv(
        frame,
        ["feature_a"],
        _perpetual_config(neutral_action_threshold=0.10),
    )
    env.reset()

    _, _, _, _, neutral = env.step(np.array([0.08], dtype=np.float32))

    assert neutral["proposed_target_fraction"] == 0.0
    assert neutral["quantity"] == 0.0

    env = PortfolioTradingEnv(
        frame,
        ["feature_a"],
        _perpetual_config(
            minimum_net_risk_reward=1.5,
            take_profit_distance=0.005,
        ),
    )
    env.reset()
    _, _, _, _, rejected = env.step(np.array([1.0], dtype=np.float32))

    assert rejected["quantity"] == 0.0
    assert "insufficient_net_risk_reward" in rejected["risk_reasons"]


def test_sac_hold_close_semantics_distinguish_wait_hold_and_close() -> None:
    env = PortfolioTradingEnv(
        _perpetual_frame(rows=5),
        ["feature_a"],
        _perpetual_config(
            action_semantics="hold_close_target",
            hold_action_threshold=0.03,
            close_action_threshold=0.10,
        ),
    )
    env.reset()

    _, _, _, _, opened = env.step(np.array([1.0], dtype=np.float32))
    _, _, _, _, held = env.step(np.array([0.0], dtype=np.float32))
    _, _, _, _, closed = env.step(np.array([0.06], dtype=np.float32))

    assert opened["action_intent"] == "TARGET_LONG"
    assert held["action_intent"] == "HOLD_POSITION"
    assert held["side"] == "HOLD"
    assert held["quantity"] != 0.0
    assert closed["action_intent"] == "CLOSE_POSITION"
    assert closed["quantity"] == 0.0


def test_sac_hold_close_semantics_waits_when_already_flat() -> None:
    env = PortfolioTradingEnv(
        _perpetual_frame(rows=3),
        ["feature_a"],
        _perpetual_config(
            action_semantics="hold_close_target",
            hold_action_threshold=0.03,
            close_action_threshold=0.10,
        ),
    )
    env.reset()

    _, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))

    assert info["action_intent"] == "WAIT_FLAT"
    assert info["side"] == "HOLD"
    assert info["trade_notional"] == 0.0


def test_sac_continuous_target_keeps_small_deadband_without_large_flat_zone() -> None:
    config = _perpetual_config(
        action_semantics="continuous_target",
        rebalance_deadband=0.01,
        neutral_action_threshold=0.0,
    )

    ignored_target, ignored_intent = map_continuous_action(0.01, config)
    active_target, active_intent = map_continuous_action(0.04, config)
    short_target, short_intent = map_continuous_action(-0.04, config)

    assert ignored_target == pytest.approx(0.0)
    assert ignored_intent == "WAIT_FLAT"
    assert active_target == pytest.approx(0.02)
    assert active_intent == "TARGET_LONG"
    assert short_target == pytest.approx(-0.02)
    assert short_intent == "TARGET_SHORT"


def test_trade_plan_context_and_take_profit_are_simulated() -> None:
    frame = _perpetual_frame()
    frame.loc[1, "high"] = 102.0
    env = PortfolioTradingEnv(
        frame,
        ["feature_a"],
        _perpetual_config(
            include_trade_plan_context=True,
            take_profit_distance=0.01,
        ),
    )
    observation, _ = env.reset()

    observation, _, _, _, closed = env.step(np.array([1.0], dtype=np.float32))

    assert observation.shape == (15,)
    assert closed["take_profit_triggered"]
    assert closed["side"] == "TAKE_PROFIT"
    assert closed["quantity"] == 0.0
    assert closed["equity"] > 1000.0


def test_episode_domain_randomization_is_seeded_and_bounded() -> None:
    env = PortfolioTradingEnv(
        _perpetual_frame(),
        ["feature_a"],
        _perpetual_config(
            initial_capital_randomization=0.10,
            slippage_rate=0.0003,
            slippage_randomization=0.0001,
        ),
    )

    _, first = env.reset(seed=19)
    _, repeated = env.reset(seed=19)

    assert repeated["episode_initial_capital"] == pytest.approx(first["episode_initial_capital"])
    assert repeated["episode_slippage_rate"] == pytest.approx(first["episode_slippage_rate"])
    assert 900.0 <= float(first["episode_initial_capital"]) <= 1100.0
    assert 0.0002 <= float(first["episode_slippage_rate"]) <= 0.0004


def test_ppo_discrete_wrapper_uses_four_spec_actions() -> None:
    env = DiscretePortfolioActionWrapper(
        PortfolioTradingEnv(_perpetual_frame(), ["feature_a"], _perpetual_config())
    )
    env.reset()

    _, _, _, _, opened = env.step(1)
    _, _, _, _, held = env.step(0)
    _, _, _, _, closed = env.step(3)

    assert env.action_space.n == 4
    assert opened["quantity"] > 0
    assert held["quantity"] > 0
    assert closed["quantity"] == 0.0
