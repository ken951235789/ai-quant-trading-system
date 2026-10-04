"""複合策略必須雙向對稱、因果成立，失效回測價不能復活。"""

from dataclasses import replace
from itertools import product

import numpy as np
import pandas as pd
import pytest

from ai_quant_trading.backtesting.complex_research import complex_neighbors
from ai_quant_trading.backtesting.complex_strategies import (
    FAMILIES, complex_frame, complex_grid, complex_signals, prepare_complex_cache,
)
from ai_quant_trading.backtesting.grid_execution import signal_frame


def feature_fixture():
    values = {"open": 100., "high": 101., "low": 99., "close": 100.5, "volume": 10.,
        "atr": 1., "ema20": 100., "mid20": 100., "std20": .5, "upper": 101., "lower": 99.,
        "rsi": 55., "rvol": 1.5, "prior_high": 102., "prior_low": 98., "vwap": 100.,
        "lower_wick": .5, "upper_wick": .5, "1h_direction": 1., "4h_direction": 1.,
        "1h_adx": 35., "4h_slope": .001}
    return pd.DataFrame(values, index=pd.date_range("2020-01-01", periods=12, freq="1h", tz="UTC"))


def mirrored(f):
    changed = f.copy()
    for name in ("open", "close", "ema20", "mid20", "vwap"):
        changed[name] = 200 - f[name]
    for a, b in (("high", "low"), ("upper", "lower"), ("prior_high", "prior_low")):
        changed[a], changed[b] = 200 - f[b], 200 - f[a]
    changed["rsi"] = 100 - f.rsi
    changed["lower_wick"], changed["upper_wick"] = f.upper_wick, f.lower_wick
    changed[["1h_direction", "4h_direction", "4h_slope"]] *= -1
    return changed


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("strict", [False, True])
def test_known_entries_are_long_short_symmetric(family, strict):
    f = feature_fixture()
    at = 11
    if family in ("trend_pullback", "regime_switch"):
        f.iloc[at, f.columns.get_indexer(["close", "high", "low", "open"])] = [101.2, 101.5, 99.8, 100.1]
    elif family == "squeeze_breakout":
        f.iloc[at, f.columns.get_indexer(["close", "high", "low", "open"])] = [103., 103.1, 100.5, 101.]
    elif family == "vwap_reversion":
        f["1h_adx"], f["vwap"], f["rsi"] = 10., 101., 35.
        f.iloc[10, f.columns.get_indexer(["close", "low"])] = [98.5, 98.4]
        f.iloc[at, f.columns.get_indexer(["close", "high", "low", "open"])] = [99.4, 100., 98.9, 99.]
    elif family == "liquidity_reclaim":
        f["1h_adx"] = 10.
        f.iloc[at, f.columns.get_indexer(["close", "high", "low", "open", "lower_wick"])] = [98.2, 98.3, 97.5, 98.1, .75]
    else:
        f.iloc[7, f.columns.get_indexer(["close", "high", "low", "open"])] = [102.5, 102.6, 101.5, 102.]
        f.iloc[8, f.columns.get_indexer(["close", "high", "low", "open"])] = [102.3, 102.4, 102., 102.1]
        at = 8
        assert complex_signals(f, family, strict).iloc[7] == 0
    assert complex_signals(f, family, strict).iloc[at] == 1
    assert complex_signals(mirrored(f), family, strict).iloc[at] == -1


def test_retest_invalidated_level_never_reappears():
    f = feature_fixture()
    f.iloc[7, f.columns.get_indexer(["close", "high", "low", "open"])] = [102.5, 102.6, 101.5, 102.]
    f.iloc[8, f.columns.get_indexer(["close", "high", "low", "open"])] = [101., 102.4, 100.5, 101.2]
    f.iloc[9, f.columns.get_indexer(["close", "high", "low", "open"])] = [102.3, 102.4, 102., 102.1]
    assert complex_signals(f, "donchian_retest", False).iloc[9] == 0
    assert complex_signals(mirrored(f), "donchian_retest", False).iloc[9] == 0


def test_grid_size_baseline_and_neighbors():
    grid = complex_grid()
    assert len(grid) == len(set(grid)) == 1153
    assert grid[0].family == "original"
    fractions = complex_neighbors(grid, np.ones(len(grid), dtype=bool))
    assert fractions[0] == 0
    np.testing.assert_array_equal(fractions[1:], 1.)


def market(n=5100):
    close = 100 * np.exp(np.cumsum(np.random.default_rng(91).normal(.00002, .001, n)))
    opening = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"timestamp": pd.date_range("2020-01-01 07:30", periods=n, freq="15min", tz="UTC"),
        "open": opening, "high": np.maximum(opening, close) + .05,
        "low": np.minimum(opening, close) - .05, "close": close, "volume": 10.,
        "symbol": "BTCUSDT", "exchange": "binance_futures", "interval": "15m"})


def test_all_complex_features_and_signals_are_causal_at_partial_hour_boundaries():
    original = market()
    changed = original.copy()
    changed.loc[4502:, ["open", "high", "low", "close"]] *= 1.5
    changed.loc[4502:, "volume"] *= 10
    a, b = prepare_complex_cache(original), prepare_complex_cache(changed)
    for tf in ("15m", "1h"):
        end = original.timestamp.iloc[4502]
        before, after = a["complex_features"][tf], b["complex_features"][tf]
        pd.testing.assert_frame_equal(before.loc[before.index <= end], after.loc[after.index <= end])
    for family, timeframe, strict in product(FAMILIES, ("15m", "1h"), (False, True)):
        p = replace(complex_grid()[1], family=family, timeframe=timeframe, strict=strict)
        before, after = complex_frame(a, p), complex_frame(b, p)
        pd.testing.assert_frame_equal(before.iloc[:4502], after.iloc[:4502])
        assert before.candidate_side.iloc[:4000].eq(0).all()
        if timeframe == "1h":
            active = before.loc[before.candidate_side.ne(0), "timestamp"] + pd.Timedelta(minutes=15)
            assert active.dt.minute.eq(0).all()
    pd.testing.assert_frame_equal(complex_frame(a, complex_grid()[0]), signal_frame(a, complex_grid()[0].execution()))


def test_vwap_midnight_uses_bar_open_day_and_completed_bars_only():
    cache = prepare_complex_cache(market())
    for tf in ("15m", "1h"):
        f = cache["complex_features"][tf]
        end = pd.Timestamp("2020-01-04", tz="UTC")
        day = f.loc[(f.index > end - pd.Timedelta(days=1)) & (f.index <= end)]
        expected = (((day.high + day.low + day.close) / 3) * day.volume).sum() / day.volume.sum()
        assert f.loc[end, "vwap"] == pytest.approx(expected)
        next_bar = f.loc[end + pd.Timedelta(tf)]
        assert next_bar.vwap == pytest.approx((next_bar.high + next_bar.low + next_bar.close) / 3)


def test_regime_switch_is_flat_in_neutral_environment():
    f = feature_fixture()
    f["1h_adx"] = 22.
    assert complex_signals(f, "regime_switch", False).eq(0).all()
    with pytest.raises(ValueError):
        complex_signals(f, "guessed_strategy", False)
