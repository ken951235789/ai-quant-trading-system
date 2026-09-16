"""強化學習資料切分與標準化測試。"""

import numpy as np
import pandas as pd
import pytest

from ai_quant_trading.reinforcement_learning import RLSplitConfig, prepare_rl_dataset


def make_rl_frame(rows: int = 120) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    close = 100 + index
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="D", tz="UTC"),
            "symbol": "AAPL",
            "exchange": "yahoo_finance",
            "interval": "1d",
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1000 + index,
            "feature_a": index * 2,
            "feature_b": np.sin(index / 5),
        }
    )


def test_temporal_split_and_train_only_normalization() -> None:
    dataset = prepare_rl_dataset(
        make_rl_frame().sample(frac=1, random_state=42),
        ["feature_a", "feature_b"],
        RLSplitConfig(0.6, 0.2, min_rows_per_split=10),
    )

    assert len(dataset.train) == 72
    assert len(dataset.validation) == 24
    assert len(dataset.test) == 24
    assert dataset.observation_size == 7
    assert dataset.train["timestamp"].max() < dataset.validation["timestamp"].min()
    assert np.allclose(dataset.train[["feature_a", "feature_b"]].mean(), 0, atol=1e-7)
    assert not np.allclose(dataset.validation[["feature_a", "feature_b"]].mean(), 0)


def test_rejects_missing_feature_and_short_data() -> None:
    with pytest.raises(ValueError, match="缺少必要欄位"):
        prepare_rl_dataset(make_rl_frame(), ["missing_feature"])
    with pytest.raises(ValueError, match="資料量不足"):
        prepare_rl_dataset(
            make_rl_frame(12),
            ["feature_a"],
            RLSplitConfig(min_rows_per_split=5),
        )


def test_ai_coverage_is_measured_before_feature_standardization() -> None:
    frame = make_rl_frame()
    frame["transformer_available"] = 1.0
    frame.loc[:11, "transformer_available"] = 0.0

    dataset = prepare_rl_dataset(
        frame,
        ["feature_a", "transformer_available"],
        RLSplitConfig(min_rows_per_split=10),
    )

    assert dataset.availability_coverage["transformer_available"] == pytest.approx(0.90)
    assert dataset.frame["transformer_available"].mean() != pytest.approx(0.90)


def test_ai_availability_is_preserved_as_context_without_becoming_a_feature() -> None:
    frame = make_rl_frame()
    frame["transformer_available"] = 1.0
    frame.loc[:11, "transformer_available"] = 0.0

    dataset = prepare_rl_dataset(
        frame,
        ["feature_a"],
        RLSplitConfig(min_rows_per_split=10),
    )

    assert "transformer_available" in dataset.frame
    assert "transformer_available" not in dataset.feature_columns
    assert dataset.availability_coverage["transformer_available"] == pytest.approx(0.90)


@pytest.mark.parametrize(
    "column",
    ["target", "future_return", "actual_future_return", "take_profit_hit", "label_up"],
)
def test_rejects_hindsight_feature_columns(column: str) -> None:
    frame = make_rl_frame()
    frame[column] = 1.0

    with pytest.raises(ValueError, match="特徵包含禁止欄位"):
        prepare_rl_dataset(frame, ["feature_a", column])
