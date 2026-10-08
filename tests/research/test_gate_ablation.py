"""四種篩選模式只改條件，不能改撮合或使用未來標籤。"""

import pandas as pd
import pytest

from ai_quant_trading.transformer.strategy_events import StrategyEventConfig, select_event_trades


def predictions():
    return pd.DataFrame({"series_index": [0]*4, "endpoint": [10, 20, 30, 40],
        "event_exit_endpoint": [12, 22, 32, 42], "tradeability_probability_32": [.7, .7, .3, .3],
        "predicted_return_32": [.001, -.001, .001, -.001], "actual_return_32": [-.1, .1, -.1, .1]})


@pytest.mark.parametrize("mode,expected", [("none", [10, 20, 30, 40]), ("probability", [10, 20]),
                                          ("return", [10, 30]), ("both", [10])])
def test_modes_and_future_perturbation(mode, expected):
    frame = predictions()
    config = StrategyEventConfig()
    selected, _ = select_event_trades(frame, 32, config, 2, filtered=True, filter_mode=mode)
    assert selected.endpoint.tolist() == expected
    frame.actual_return_32 *= -100
    other, _ = select_event_trades(frame, 32, config, 2, filtered=True, filter_mode=mode)
    assert other.endpoint.tolist() == expected


def test_legacy_mapping_and_invalid():
    frame, config = predictions(), StrategyEventConfig()
    for filtered, mode in ((False, "none"), (True, "both")):
        old, _ = select_event_trades(frame, 32, config, 2, filtered=filtered)
        new, _ = select_event_trades(frame, 32, config, 2, filtered=filtered, filter_mode=mode)
        pd.testing.assert_frame_equal(old, new)
    with pytest.raises(ValueError):
        select_event_trades(frame, 32, config, 2, filtered=True, filter_mode="optimal_future")
