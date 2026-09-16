"""共識、反轉衝突與比例化資料的因果契約測試。"""

import numpy as np
import pandas as pd
import pytest

from ai_quant_trading.features.market_context import (
    MARKET_CONTEXT_COLUMNS,
    add_model_market_features,
)
from ai_quant_trading.features.contracts import forbidden_model_feature_reason


def context_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0],
            "close": [102.0, 103.0, 104.0],
            "volume": [10.0, 11.0, 12.0],
            "macd": [1.0, 2.0, 3.0],
            "atr_14": [2.0, 2.0, 2.0],
        }
    )
    for interval, direction in (("5m", 1.0), ("15m", 1.0), ("4h", -1.0), ("1d", -1.0)):
        for name, value in {
            "ema_20_50_atr": direction * 3,
            "supertrend_direction_10_3": direction,
            "smc_structure_direction": direction,
            "available": 1.0,
            "age_ratio": 0.0,
            "bb_zscore_20": direction * 2,
            "rsi_14": 0.8 if direction > 0 else 0.2,
        }.items():
            frame[f"mtf_{interval}_{name}"] = value
    return frame


def test_cross_timeframe_conflict_and_entropy_are_preserved() -> None:
    frame = add_model_market_features(context_frame())
    row = frame.iloc[-1]
    assert row.market_trend_consensus == pytest.approx(0)
    assert row.market_fast_trend > 0.9
    assert row.market_slow_trend < -0.9
    assert row.market_fast_slow_conflict > 0.9
    assert row.market_cross_timeframe_disagreement > 0.9
    assert row.market_direction_entropy == pytest.approx(np.log(2) / np.log(3))
    assert row.market_timeframe_coverage == 1


def test_stale_and_missing_timeframes_do_not_vote_as_neutral() -> None:
    frame = context_frame()
    frame["mtf_4h_available"] = 0.0
    frame["mtf_1d_age_ratio"] = 2.0
    result = add_model_market_features(frame)
    assert result.iloc[-1].market_trend_consensus > 0.9
    assert result.iloc[-1].market_timeframe_coverage == 0.5
    assert result.iloc[-1].market_trend_reversal_conflict > 0.65


def test_normalized_features_are_price_scale_invariant() -> None:
    original = context_frame()
    scaled = original.copy()
    scaled[["open", "high", "low", "close", "macd", "atr_14"]] *= 100
    columns = [
        "macd_pct",
        "atr_pct",
        "price_body_ratio",
        "price_range_ratio",
        "price_close_location",
    ]
    pd.testing.assert_frame_equal(
        add_model_market_features(original)[columns],
        add_model_market_features(scaled)[columns],
    )


def test_future_changes_cannot_rewrite_context_and_reduced_frames_keep_context() -> None:
    frame = context_frame()
    expected = add_model_market_features(frame)
    changed = frame.copy()
    changed.loc[2, "mtf_5m_ema_20_50_atr"] = -100
    pd.testing.assert_frame_equal(
        expected.loc[:1, MARKET_CONTEXT_COLUMNS],
        add_model_market_features(changed).loc[:1, MARKET_CONTEXT_COLUMNS],
    )
    reduced = expected.drop(columns=["mtf_5m_ema_20_50_atr"])
    pd.testing.assert_frame_equal(
        expected[list(MARKET_CONTEXT_COLUMNS)],
        add_model_market_features(reduced)[list(MARKET_CONTEXT_COLUMNS)],
    )


def test_transformer_predictions_do_not_leak_into_market_context() -> None:
    frame = context_frame()
    expected = add_model_market_features(frame)
    frame["transformer_return_5"] = [-100, 100, -100]
    pd.testing.assert_frame_equal(
        expected[list(MARKET_CONTEXT_COLUMNS)],
        add_model_market_features(frame)[list(MARKET_CONTEXT_COLUMNS)],
    )


@pytest.mark.parametrize(
    "column", ["transformer_return_5", "u_transformer_signal_entropy", "expected_return"]
)
def test_transformer_refuses_its_own_outputs_as_inputs(column) -> None:
    assert forbidden_model_feature_reason(column, forbid_transformer_outputs=True) is not None
    assert forbidden_model_feature_reason(column) is None
