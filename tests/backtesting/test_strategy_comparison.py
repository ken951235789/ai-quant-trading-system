"""規則比較的因果性、動作語意及成本公平性測試。"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from ai_quant_trading.backtesting.strategy_comparison import (
    RULES, RulePolicy, cost_configs, indicators, rule_direction, summarize,
)
from ai_quant_trading.reinforcement_learning import PortfolioEnvConfig, PortfolioTradingEnv
from ai_quant_trading.reinforcement_learning.actions import map_continuous_action
from ai_quant_trading.reinforcement_learning.training import evaluate_rl_model


def bars(rows: int = 600) -> pd.DataFrame:
    price = 100 + np.sin(np.arange(rows) / 11) * 2 + np.arange(rows) * .005
    return pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=rows, freq="15min", tz="UTC"),
        "open": price, "high": price + .5, "low": price - .5, "close": price,
        "volume": 100., "symbol": "BTC/USDT", "exchange": "binance_futures", "interval": "15m",
        "expected_return": .003, "feature": 0.})


def test_indicators_are_causal_and_hourly_waits_until_close() -> None:
    frame = bars()
    before = indicators(frame)
    changed = frame.copy()
    changed.loc[450:, ["open", "high", "low", "close"]] *= 2
    after = indicators(changed)
    pd.testing.assert_frame_equal(before.iloc[:450], after.iloc[:450])
    assert np.isnan(before.hourly_direction.iloc[198])
    assert np.isfinite(before.hourly_direction.iloc[199])
    assert before.hourly_direction.iloc[199] == before.hourly_direction.iloc[202]


@pytest.mark.parametrize("semantics", ["hold_close_target", "continuous_target", "legacy_target"])
def test_rule_policy_flatten_hold_and_direction_mapping(semantics: str) -> None:
    config = PortfolioEnvConfig(allow_short=True, max_short_fraction=.5, max_position_fraction=.5,
        normalized_action_space=True, action_semantics=semantics, neutral_action_threshold=.1)
    frame = indicators(bars()).iloc[-5:].reset_index(drop=True)
    observation = np.array([0, .8, .25, 0])
    action, _ = RulePolicy("cash", frame, 1, config).predict(observation, deterministic=True)
    assert map_continuous_action(float(action[0]), config, current_position=.25)[0] == 0
    action, _ = RulePolicy("long_risk_managed", frame, 1, config).predict(observation, deterministic=True)
    assert map_continuous_action(float(action[0]), config, current_position=.25)[0] == pytest.approx(.25)


def test_rule_entries_and_exits_are_explicit() -> None:
    row = indicators(bars()).iloc[-1].copy()
    row.rsi = 25
    assert rule_direction("rsi_reversion", row, 0) == 1
    row.rsi = 75
    assert rule_direction("rsi_reversion", row, 0) == -1
    assert rule_direction("rsi_reversion", row, .2) == 0
    row.expected_return = .0016
    assert rule_direction("transformer_return_only", row, 0) == 0
    row.expected_return = -.002
    assert rule_direction("transformer_return_only", row, 0) == -1
    for name in RULES:
        assert rule_direction(name, row, 0) in (-1, 0, 1)


def test_shared_risk_configuration_and_cash_has_no_win_rate() -> None:
    config = PortfolioEnvConfig(allow_short=True, max_short_fraction=.5,
        normalized_action_space=True, action_semantics="hold_close_target")
    for current in cost_configs(config).values():
        assert current.risk_per_trade == config.risk_per_trade
        assert current.max_drawdown_limit == config.max_drawdown_limit
        assert current.max_position_fraction == config.max_position_fraction
    frame = bars(30)
    policy = RulePolicy("cash", indicators(frame), 1, config)
    evaluation, metrics = evaluate_rl_model(policy, frame, ["feature"], config, deterministic=True, seed=11)
    result = summarize(evaluation, metrics, config.initial_capital)
    assert result["total_return"] == 0
    assert result["win_rate"] is None
    assert result["profit_factor"] is None


@pytest.mark.parametrize("side", [1, -1])
@pytest.mark.parametrize("reason", ["stop", "target", "normal"])
def test_perpetual_closing_pnl_includes_entry_fee_and_exit_friction(side: int, reason: str) -> None:
    frame = bars(3)
    frame[["open", "high", "low", "close"]] = 100.
    config = PortfolioEnvConfig(initial_capital=1000, execution_mode="perpetual", allow_short=True,
        max_position_fraction=.5, max_short_fraction=.5, fee_rate=.001, slippage_rate=.002,
        spread_rate=.002, minimum_stop_distance=.01, maximum_stop_distance=.01,
        take_profit_distance=.02, max_drawdown_limit=.9)
    if reason == "stop":
        frame.loc[1, "low" if side > 0 else "high"] = 90 if side > 0 else 110
    elif reason == "target":
        frame.loc[1, "high" if side > 0 else "low"] = 110 if side > 0 else 90
    env = PortfolioTradingEnv(frame, ["feature"], config)
    env.reset(seed=11)
    _, _, _, _, info = env.step(np.array([side * .5]))
    if reason == "normal":
        first = info
        _, _, _, _, info = env.step(np.array([0.]))
        total_slippage = first["slippage_cost"] + info["slippage_cost"]
    else:
        total_slippage = info["slippage_cost"]
    assert info["trade_closed"]
    assert info["closed_trade_pnl"] == pytest.approx(info["equity"] - 1000)
    assert total_slippage > 2.9
    if reason != "normal":
        assert info["stop_triggered"] == (reason == "stop")
        assert info["take_profit_triggered"] == (reason == "target")
    zero = replace(config, fee_rate=0, slippage_rate=0, spread_rate=0)
    reference = PortfolioTradingEnv(frame, ["feature"], zero)
    reference.reset()
    _, _, _, _, no_cost = reference.step(np.array([side * .5]))
    if reason == "normal":
        _, _, _, _, no_cost = reference.step(np.array([0.]))
    assert info["equity"] < no_cost["equity"]


def test_gap_liquidation_separates_regular_fee_and_liquidation_fee() -> None:
    frame = bars(3)
    frame[["open", "high", "low", "close"]] = 100.
    frame.loc[2, ["open", "high", "low", "close"]] = 40.
    config = PortfolioEnvConfig(execution_mode="perpetual", allow_short=True,
        max_position_fraction=.5, max_short_fraction=.5, leverage=2, max_leverage=3,
        action_semantics="hold_close_target", fee_rate=.001, slippage_rate=.002,
        spread_rate=.002, minimum_stop_distance=.01, maximum_stop_distance=.01,
        max_drawdown_limit=.9)
    env = PortfolioTradingEnv(frame, ["feature"], config)
    env.reset()
    _, _, _, _, entry = env.step(np.array([.5]))
    _, _, terminated, _, result = env.step(np.array([0.]))
    assert terminated and result["liquidated"]
    exit_price = 40 * (1 - .003)
    assert result["fee"] == pytest.approx(entry["quantity"] * exit_price * .001)
    assert result["liquidation_fee"] == pytest.approx(entry["quantity"] * exit_price * config.liquidation_fee_rate)
    assert result["slippage_cost"] == pytest.approx(entry["quantity"] * .12)
    assert result["closed_trade_pnl"] == pytest.approx(result["equity"] - 1000)
