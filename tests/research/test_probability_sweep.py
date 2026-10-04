"""門檻邊界、未來收益隔離、成本及無交易語意的回歸測試。"""

import numpy as np
import pandas as pd
import pytest

from ai_quant_trading.research.probability_sweep import (
    ThresholdReplay, aggregate_seeds, assert_trade_parity, trade_metrics, schedule_id,
)
from ai_quant_trading.transformer.strategy_events import StrategyEventConfig, select_event_trades


def predictions():
    return pd.DataFrame({"series_index": [0] * 5, "endpoint": [1, 2, 4, 8, 12],
        "event_exit_endpoint": [3, 5, 7, 11, 15], "tradeability_probability_32": [0, .55, .8, 1., .6],
        "predicted_return_32": [.001, .001, .001, .001, .0002], "actual_return_32": [-.01] * 5})


def test_boundaries_and_strict_edge():
    replay = ThresholdReplay(predictions(), 32, StrategyEventConfig(cooldown_bars=0), 2)
    assert replay.select(0).endpoint.tolist() == [1, 4, 8]
    assert replay.select(1).endpoint.tolist() == [8]
    assert replay.select(.55).endpoint.tolist() == [2, 8]
    assert replay.baseline.endpoint.tolist() == [1, 4, 8, 12]


def test_original_55_and_cooldown():
    config = StrategyEventConfig(cooldown_bars=4)
    expected, _ = select_event_trades(predictions(), 32, config, 2, filtered=True)
    actual = ThresholdReplay(predictions(), 32, config, 2).select(.55)
    pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True))
    assert actual.endpoint.tolist() == [2]


def test_future_returns_do_not_choose_trades():
    p = predictions()
    first = ThresholdReplay(p, 32, StrategyEventConfig(), 2).select(.4)
    p["actual_return_32"] = [99., -100., 5., 0., -20.]
    second = ThresholdReplay(p, 32, StrategyEventConfig(), 2).select(.4)
    assert first.endpoint.tolist() == second.endpoint.tolist()


@pytest.mark.parametrize("threshold", [-.01, 1.01, np.nan, np.inf])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        ThresholdReplay(predictions(), 32, StrategyEventConfig(), 2).select(threshold)


@pytest.mark.parametrize("column,value", [("actual_return_32", np.nan),
    ("tradeability_probability_32", -.1), ("predicted_return_32", np.inf), ("event_exit_endpoint", 0)])
def test_invalid_excluded_row_is_not_hidden(column, value):
    p = predictions()
    p.loc[0, column] = value
    with pytest.raises(ValueError):
        ThresholdReplay(p, 32, StrategyEventConfig(), 2)


def test_duplicate_fails():
    p = predictions()
    with pytest.raises(ValueError):
        ThresholdReplay(pd.concat([p, p.iloc[:1]]), 32, StrategyEventConfig(), 2)


def trades():
    return pd.DataFrame({"endpoint": [1, 10], "entry_endpoint": [2, 11],
        "exit_endpoint": [4, 13], "side": [1, -1], "gross_same_path_return": [.021, -.009],
        "spread_return": [0., 0.], "slippage_return": [0., 0.], "fee_return": [.001, .001],
        "funding_return": [0., 0.], "net_return": [.02, -.01]})


def test_accounting_and_settled_scope():
    metrics = trade_metrics(trades())
    assert metrics["mean_net"] == pytest.approx(.005)
    assert metrics["actual_win_rate"] == .5
    assert metrics["breakeven_win_rate"] == pytest.approx(1 / 3)
    assert metrics["profit_factor"] == 2
    assert metrics["settled_return_10pct"] == pytest.approx(1.002 * .999 - 1)
    assert metrics["settled_drawdown_10pct"] == pytest.approx(-.001)
    assert metrics["account_equity_validated"] is False
    assert metrics["ci_low"] is None and metrics["ci_high"] is None
    assert metrics["ci_status"] == "insufficient_sample"
    t = trades()
    t.loc[0, "fee_return"] = .002
    with pytest.raises(AssertionError):
        trade_metrics(t)


def test_empty_and_schedule_identity():
    empty = trade_metrics(trades().iloc[:0])
    assert empty["trades"] == 0
    assert empty["mean_net"] is None and empty["settled_drawdown_10pct"] is None
    assert schedule_id(trades()) == schedule_id(trades().copy())
    t = trades()
    t.loc[0, "net_return"] += .001
    assert schedule_id(t) != schedule_id(trades())


def test_empty_csv_parity_and_mismatch():
    expected = pd.DataFrame(columns=["endpoint", "exit_endpoint", "net_return"])
    assert_trade_parity(trades().iloc[:0], expected)
    assert_trade_parity(trades(), trades().astype(str))
    with pytest.raises(AssertionError):
        assert_trade_parity(trades(), trades().iloc[:1])


def test_seed_summary_does_not_count_empty_as_zero_return():
    identity = {"family": "original", "variant": "F_existing", "split": "test", "scenario": "base", "threshold_pct": 55}
    result = aggregate_seeds([{**identity, **trade_metrics(trades())},
                             {**identity, **trade_metrics(trades().iloc[:0])}]).iloc[0]
    assert result.active_seeds == 1 and result.seeds == 2
    assert result.mean_net_mean == pytest.approx(.005)
    assert result.trades_mean == 1


def test_descriptive_ci_requires_minimum_sample_but_does_not_qualify_model():
    t = pd.concat([trades()] * 4, ignore_index=True)
    metric = trade_metrics(t)
    assert metric["ci_low"] is not None
    assert metric["ci_status"] == "descriptive_unadjusted"
    assert metric["evidence"] == "INSUFFICIENT_EVIDENCE"
