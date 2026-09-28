"""集成不得重造答案，並保留校準前後可比較的預測。"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from ai_quant_trading.transformer.evaluation import validate_ensemble_labels


def predictions():
    return pd.DataFrame({"series_index": [0, 0], "endpoint": [20, 21],
                         "source_key": [15, 15], "label_contract": ["fixture_v1"] * 2,
                         "timestamp_ns": [100, 200], "actual_direction_20": [1, 2],
                         "down_probability_20": [0.1, 0.1],
                         "neutral_probability_20": [0.8, 0.1],
                         "up_probability_20": [0.1, 0.8]})


def test_label_identity_and_probability_validation():
    frame = predictions()
    validate_ensemble_labels([frame, frame.copy()], (20,))
    for invalid in (frame.drop(columns="actual_direction_20"),
                    frame.assign(timestamp_ns=[200, 100]),
                    frame.assign(actual_direction_20=[0, 2]),
                    frame.assign(up_probability_20=[float("nan"), 0.8])):
        with pytest.raises(ValueError):
            validate_ensemble_labels([frame, invalid], (20,))


def test_ensemble_uses_persisted_labels_not_edge_threshold(tmp_path):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "ensemble_runner", root / "scripts/kaggle_transformer_v3_five_seed_runner.py",
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    frame = predictions()
    # 故意提供會重建出不同答案的 edge，評估仍只能使用已保存的實際標籤。
    frame["actual_long_edge_20"] = 1.0
    frame["actual_short_edge_20"] = -1.0
    frame.to_csv(tmp_path / "test_predictions.csv", index=False)
    result = runner._ensemble_diagnostic([{"run_dir": tmp_path}] * 2, (20,), 9999)
    assert result["aggregate"]["accuracy"] == 1.0
    assert result["label_source"] == "persisted_actual_direction"
