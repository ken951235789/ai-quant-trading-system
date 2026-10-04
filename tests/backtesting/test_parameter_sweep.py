"""大型網格的成交等價性、時間因果、資金統計與選擇隔離。"""

from dataclasses import replace
from itertools import product

import numpy as np
import pandas as pd
import pytest

from ai_quant_trading.backtesting.candidate_research import COSTS
from ai_quant_trading.backtesting.grid_execution import (
    GridParams, batch_events, holding_limit, parameter_grid, prepare_cache,
    reference_check, scheduled_indices, signal_frame,
)
from ai_quant_trading.backtesting.parameter_sweep import (
    joint_max_statistic, neighborhood_fraction, search_shortlist, summarize,
    validation_selection, windows_for,
)
from ai_quant_trading.transformer.strategy_events import StrategyEventConfig, prepare_event_frame


BASE = GridParams("15m", "breakout", 25, 2, 20, 2., 2., 32, 4, "loss")


def bars(n=200, random=False):
    rng = np.random.default_rng(81)
    close = 100 * np.exp(np.cumsum(rng.normal(.00005, .002, n))) if random else np.full(n, 100.)
    opening = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"timestamp": pd.date_range("2020-01-01 07:30", periods=n, freq="15min", tz="UTC"),
        "open": opening, "high": np.maximum(opening, close) + .05,
        "low": np.minimum(opening, close) - .05, "close": close, "volume": 10.,
        "symbol": "BTCUSDT", "exchange": "binance_futures", "interval": "15m",
        "candidate_side": 1, "event_regime": 1, "event_atr": .3})


def test_grid_is_complete_unique_and_windows_are_separated():
    grid = parameter_grid()
    assert len(grid) == len(set(grid)) == 11664 and BASE in grid
    w = windows_for(100000)
    assert w[0] == {"name": "search", "start": 0, "stop": 40000}
    assert w[1]["start"] == 40064 and w[-1]["stop"] == 90000
    with pytest.raises(ValueError):
        replace(BASE, adx=19)
    with pytest.raises(ValueError):
        windows_for(100)


@pytest.mark.parametrize("side", [1, -1])
@pytest.mark.parametrize("case", ["both", "gap", "regime", "target", "deadline", "time"])
def test_known_priority_gap_deadline_and_funding(side, case):
    frame = bars().assign(candidate_side=side, event_regime=side, event_atr=1.)
    p = replace(BASE, holding_bars=16)
    if case == "both":
        frame.loc[1, ["low", "high"]] = [90, 110]
    elif case == "gap":
        frame.loc[2, ["open", "low", "high"]] = [95, 94, 96] if side == 1 else [105, 104, 106]
    elif case == "regime":
        frame.loc[1, "event_regime"] = -side
        frame.loc[2, ["low", "high"]] = [90, 110]
    elif case == "target":
        frame.loc[3, "high" if side == 1 else "low"] = 105 if side == 1 else 95
    elif case == "deadline":
        frame.loc[17, ["low", "high"]] = [90, 110]
    for cost in COSTS.values():
        events = holding_limit(batch_events(frame, p, cost), frame, p.holding_bars, cost)
        reference_check(events, frame, p, cost, 0)
    if case in ("deadline", "time"):
        assert events["reason_code"][0] == 4 and events["exit_endpoint"][0] == 17
        assert events["funding_return"][0] == pytest.approx(.0001)
    elif case == "regime":
        assert events["reason_code"][0] == 3
    elif case in ("both", "gap"):
        assert events["reason_code"][0] == 1


@pytest.mark.parametrize("mode", ["loss", "opposite", "disabled"])
def test_random_event_parity_across_parameters(mode):
    frame = bars(260, random=True)
    rng = np.random.default_rng(29)
    frame["candidate_side"] = rng.choice([-1, 0, 1], len(frame))
    frame["event_regime"] = np.repeat(rng.choice([-1, 0, 1], 26), 10)
    for stop, rr, hold, scenario in product([1., 2.], [1., 3.], [16, 32, 64], COSTS):
        p = replace(BASE, stop_atr=stop, reward_r=rr, holding_bars=hold, regime_exit=mode)
        cost = COSTS[scenario]
        events = holding_limit(batch_events(frame, p, cost), frame, hold, cost)
        for i in [0, len(events["endpoint"]) // 2, len(events["endpoint"]) - 1]:
            reference_check(events, frame, p, cost, i)


def test_empty_candidates_schedule_and_common_purge():
    frame = bars().assign(candidate_side=0)
    events = batch_events(frame, BASE, COSTS["base"])
    assert len(scheduled_indices(events, 0, len(frame), 4)) == 0
    frame["candidate_side"] = 1
    events = holding_limit(batch_events(frame, BASE, COSTS["base"]), frame, 32, COSTS["base"])
    chosen = scheduled_indices(events, 0, 200, 4)
    assert events["endpoint"][chosen].tolist() == [0, 38, 76, 114]
    assert (events["endpoint"][chosen] + 65 < 200).all()


def test_signals_match_baseline_and_have_no_future_leak():
    frame = bars(5000, random=True)
    cache = prepare_cache(frame)
    original = signal_frame(cache, BASE)
    reference = prepare_event_frame(frame, StrategyEventConfig())
    for name in ("event_atr", "event_regime", "candidate_side"):
        np.testing.assert_allclose(original[name].iloc[4000:], reference[name].iloc[4000:], equal_nan=True)
    altered = frame.copy()
    altered.loc[4502:, ["open", "high", "low", "close"]] *= 1.5
    changed_cache = prepare_cache(altered)
    for tf, entry in product(["15m", "1h"], ["breakout", "pullback"]):
        p = replace(BASE, timeframe=tf, entry=entry, adx=15, confirmation=1)
        before, after = signal_frame(cache, p), signal_frame(changed_cache, p)
        pd.testing.assert_frame_equal(before.iloc[:4502], after.iloc[:4502])
        if tf == "1h":
            points = before.loc[before.candidate_side.ne(0)]
            assert len(points) > 0
            assert (points.timestamp + pd.Timedelta(minutes=15)).dt.minute.eq(0).all()


def test_summary_includes_idle_months_and_fixed_notional_compounding():
    events = {"net_return": np.array([.1, -.1]), "exit_endpoint": np.array([0, 2]),
        "entry_endpoint": np.array([0, 1]), "entry_price": np.array([100., 100.]),
        "stop_price": np.array([90., 90.]), "fee_return": np.zeros(2), "funding_return": np.zeros(2)}
    result, monthly = summarize(events, np.array([0, 1]), np.array([1, 2, 3]), np.array([1, 2, 3]),
                                {"start": 0, "stop": 3})
    assert monthly[1] == 0
    assert result["settled_return"] == pytest.approx(1.01 * .99 - 1)
    assert result["settled_drawdown"] == pytest.approx(.01)
    assert monthly.sum() == pytest.approx(result["log_growth"])


def test_max_stat_is_repeatable_conservative_and_handles_idle_strategies():
    x = np.random.default_rng(9).normal(0, .01, (15, 24))
    x[0] = 0
    a = joint_max_statistic(x, repetitions=100)
    b = joint_max_statistic(x, repetitions=100)
    np.testing.assert_array_equal(a["p_adjusted"], b["p_adjusted"])
    assert (a["p_adjusted"] >= a["p_unadjusted"]).all()
    assert a["p_adjusted"][0] == 1
    with pytest.raises(ValueError):
        joint_max_statistic(x[:, :3])


def test_selection_never_uses_forward_and_diagnostic_is_not_promotion():
    parameters = pd.DataFrame({"parameter_id": range(8), "timeframe": "15m", "entry": "breakout"})
    records = pd.DataFrame([{"parameter_id": i, "window": "search", "scenario": c,
        "trades": 200, "monthly_mean_log_return": .001 * i, "settled_drawdown": .01}
        for i, c in product(range(8), ["base", "stress"])])
    shortlist = search_shortlist(parameters, records)
    assert shortlist == [7, 6, 5, 4, 3]
    validation = records.assign(window="validation")
    chosen = validation_selection(shortlist, validation, dict.fromkeys(shortlist, False), dict.fromkeys(shortlist, .01))
    assert chosen["selected"] is None and chosen["diagnostic_leader"] == 7
    with pytest.raises(ValueError):
        search_shortlist(parameters, records.assign(window="forward_1"))
    with pytest.raises(ValueError):
        validation_selection(shortlist, records, {}, {})
    grid = parameter_grid()
    np.testing.assert_array_equal(neighborhood_fraction(grid, np.ones(len(grid), dtype=bool)), 1.)
