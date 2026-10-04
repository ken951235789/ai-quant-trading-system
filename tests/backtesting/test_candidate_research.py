"""候選研究的已知成交、因果性、成本與不重疊契約。"""

import numpy as np
import pandas as pd
import pytest

from ai_quant_trading.backtesting.candidate_research import (
    COSTS, RULES, CandidateRule, expectancy, replay_window, rule_frame,
)
from ai_quant_trading.transformer.strategy_events import StrategyEventConfig, prepare_event_frame


def bars(rows=5000):
    index = np.arange(rows)
    close = 100 + index * 0.001 + np.sin(index / 8) * 0.2
    opening = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=rows, freq="15min", tz="UTC"),
        "open": opening, "high": np.maximum(opening, close) + 0.1,
        "low": np.minimum(opening, close) - 0.1, "close": close, "volume": 10.,
        "symbol": "BTC/USDT", "exchange": "binance_futures", "interval": "15m"})


def test_all_candidates_are_causal_and_warmed_up():
    original = bars()
    changed = original.copy()
    changed.loc[4500:, ["open", "high", "low", "close"]] *= 3
    before = prepare_event_frame(original, StrategyEventConfig())
    after = prepare_event_frame(changed, StrategyEventConfig())
    for rule in RULES:
        left, right = rule_frame(before, rule), rule_frame(after, rule)
        pd.testing.assert_series_equal(left.candidate_side.iloc[:4500], right.candidate_side.iloc[:4500])
        assert left.candidate_side.iloc[:4000].eq(0).all()


def test_cost_filter_is_fixed_before_zero_base_or_stress_scenario():
    prepared = prepare_event_frame(bars(), StrategyEventConfig())
    prepared["candidate_side"] = 1
    prepared["event_atr"] = 0.01
    filtered = rule_frame(prepared, RULES[4])
    assert filtered.candidate_side.eq(0).all()
    prepared["event_atr"] = 1.
    assert rule_frame(prepared, RULES[4]).candidate_side.iloc[4000:].eq(1).all()


def test_same_inputs_future_pnl_is_not_used_to_schedule_next_entry():
    frame = bars(200).assign(candidate_side=1, event_regime=1, event_atr=10)
    rule = CandidateRule("test")
    zero = replay_window(frame, rule, COSTS["zero"], 0, len(frame))
    paid = replay_window(frame, rule, COSTS["base"], 0, len(frame))
    assert zero.endpoint.tolist() == paid.endpoint.tolist()
    assert (paid.net_return < zero.net_return).all()
    assert all(zero.endpoint.iloc[1:].to_numpy() >= zero.exit_endpoint.iloc[:-1].to_numpy() + 5)
    assert (zero.endpoint + 65 < len(frame)).all()
    assert (zero.exit_endpoint < len(frame)).all()
    empty = replay_window(frame.assign(candidate_side=0), rule, COSTS["base"], 0, len(frame))
    assert expectancy(empty)["mean"] is None


def test_expected_value_uses_realized_payoffs_not_nominal_target_stop_ratio():
    result = expectancy(pd.DataFrame({"net_return": [.01, -.005, .01, -.005]}))
    assert result["mean"] == pytest.approx(.0025)
    assert result["realized_payoff_ratio"] == 2
    assert result["descriptive_breakeven_win_rate"] == pytest.approx(1/3)
    assert result["profit_factor"] == 2


def test_pullback_and_range_reversal_emit_both_sides_from_known_bars():
    frame = bars().assign(open=100., high=100.2, low=99.8, close=100.,
        event_atr=1., event_regime=1, candidate_side=0, mtf_1h_adx_14=.1)
    frame.loc[4099, "close"] = 99.
    frame.loc[4100, "close"] = 101.
    frame.loc[4199, "close"] = 101.
    frame.loc[4200, ["close", "event_regime"]] = [99., -1]
    frame["mtf_15m_ema20_gap"] = frame.close / 100 - 1
    pullback = rule_frame(frame, RULES[5])
    assert pullback.loc[4100, "candidate_side"] == 1
    assert pullback.loc[4200, "candidate_side"] == -1
    frame.loc[4300, ["low", "close", "high"]] = [98., 100.1, 100.2]
    frame.loc[4400, ["high", "close", "low"]] = [102., 99.9, 99.8]
    reversal = rule_frame(frame, RULES[6])
    assert reversal.loc[4300, "candidate_side"] == 1
    assert reversal.loc[4400, "candidate_side"] == -1
