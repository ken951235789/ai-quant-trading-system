"""模型輸入完整性與漂移測試。"""

from __future__ import annotations

import pandas as pd

from ai_quant_trading.trading import assess_model_input_health


def test_normal_runtime_features_are_ready() -> None:
    frame = pd.DataFrame({"a": [0.1] * 20, "b": [-0.2] * 20})

    health = assess_model_input_health(
        frame,
        ["a", "b"],
        {"a": 0.0, "b": 0.0},
        {"a": 1.0, "b": 1.0},
    )

    assert health.ready
    assert not health.drifted
    assert health.risk_multiplier == 1.0


def test_latest_missing_feature_blocks_model_input() -> None:
    frame = pd.DataFrame({"a": [0.1, None], "b": [0.0, 0.0]})

    health = assess_model_input_health(
        frame,
        ["a", "b"],
        {"a": 0.0, "b": 0.0},
        {"a": 1.0, "b": 1.0},
    )

    assert not health.ready
    assert health.latest_missing_columns == ("a",)


def test_severe_distribution_shift_is_detected() -> None:
    frame = pd.DataFrame({"a": [100.0] * 20, "b": [100.0] * 20})

    health = assess_model_input_health(
        frame,
        ["a", "b"],
        {"a": 0.0, "b": 0.0},
        {"a": 1.0, "b": 1.0},
    )

    assert health.ready
    assert health.drifted
    assert health.severe_drift
    assert health.risk_multiplier == 0.0


def test_moderate_drift_produces_continuous_risk_reduction() -> None:
    frame = pd.DataFrame(
        {
            "a": [6.0] * 20,
            "b": [6.0] * 20,
            "c": [0.0] * 20,
            "d": [0.0] * 20,
        }
    )

    health = assess_model_input_health(
        frame,
        ["a", "b", "c", "d"],
        {column: 0.0 for column in frame},
        {column: 1.0 for column in frame},
        severe_fraction=0.75,
    )

    assert health.drifted
    assert not health.severe_drift
    assert 0.25 < health.risk_multiplier < 1.0
