"""消融配對、向前時間邊界與選模隔離的測試。"""

import json
from pathlib import Path

import pytest

from ai_quant_trading.transformer.config import TransformerTrainingConfig
from ai_quant_trading.transformer.event_study import VARIANTS, choose_challenger, factorial_effects, study_windows, window_config


def records():
    result = []
    for seed in (42, 137):
        for variant, value in zip(VARIANTS, [1, 3, 4, 8], strict=True):
            result.append({"stage": "ablation", "seed": seed, "variant": variant,
                           "selection": {"metrics": {"event_prediction_skill_score": value}},
                           "test": {"metrics": {"event_prediction_skill_score": -1000 if value == 8 else 1000}}})
    return result


def test_selection_ignores_test_scores():
    assert choose_challenger(records()) == "D_context_mse"
    changed = records()
    for row in changed:
        row["test"]["metrics"]["event_prediction_skill_score"] *= -10
    assert choose_challenger(changed) == "D_context_mse"


def test_factorial_effects_are_paired_and_separate_interaction():
    effects = factorial_effects(records(), "selection", "event_prediction_skill_score")
    assert len(effects) == 2
    assert effects[0]["feature_effect"] == 4
    assert effects[0]["loss_effect"] == 3
    assert effects[0]["interaction"] == 2
    with pytest.raises(ValueError):
        factorial_effects(records()[:-1], "selection", "event_prediction_skill_score")


@pytest.mark.parametrize("rows", [20_000, 80_003, 176_729])
def test_registered_integer_boundaries_and_matched_settings(rows):
    root = Path(__file__).resolve().parents[2]
    config = TransformerTrainingConfig(**json.loads((root / "configs/transformer_strategy_event_v2.example.json").read_text())["training"])
    windows = study_windows(rows)
    assert len(windows) == 4
    assert windows[-1]["source_end"] == int(rows * 0.9)
    for previous, current in zip(windows, windows[1:]):
        assert previous["source_end"] == current["validation_end"]
    for window in windows:
        for variant in VARIANTS:
            training = window_config(config, window, variant, 42)
            assert int(window["source_end"] * training.train_fraction) == window["train_end"]
            assert int(window["source_end"] * (training.train_fraction + training.validation_fraction)) == window["validation_end"]
            assert training.strategy_event_config["minimum_probability"] == 0.55
            assert training.economic_minimum_edge_bps == 2
            assert training.checkpoint_metric == "event_prediction_skill_score"
            assert training.fee_bps_per_side == config.fee_bps_per_side
            assert training.seed == 42
