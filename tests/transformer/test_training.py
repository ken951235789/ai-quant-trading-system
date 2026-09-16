"""Transformer 時間切分、訓練與模型輸出的測試。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ai_quant_trading.transformer import (
    TemporalTransformerConfig,
    TransformerTrainingConfig,
    train_temporal_transformer,
)
from ai_quant_trading.features.multitimeframe import (
    BTC_MULTITIMEFRAME_INTERVALS,
    MULTITIMEFRAME_FEATURE_NAMES,
)
from ai_quant_trading.transformer.dataset import prepare_transformer_datasets


def _write_feature_csv(
    path: Path,
    *,
    rows: int = 220,
    interval: str = "1h",
) -> Path:
    random = np.random.default_rng(42)
    returns = random.normal(0.0002, 0.01, rows)
    close = 100 * np.cumprod(1 + returns)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-01-01",
                periods=rows,
                freq="h",
                tz="UTC",
            ),
            "symbol": "TEST",
            "exchange": "test",
            "interval": interval,
            "open": close * (1 + random.normal(0, 0.001, rows)),
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": random.lognormal(8, 0.2, rows),
            "return_1": returns,
            "rsi_14": random.uniform(20, 80, rows),
            "macd": random.normal(0, 1, rows),
            "atr_14": random.uniform(0.5, 2.0, rows),
            "historical_volatility_20": random.uniform(0.1, 0.5, rows),
            "target_timestamp": pd.date_range(
                "2025-01-01 01:00",
                periods=rows,
                freq="h",
                tz="UTC",
            ),
            "future_return": random.normal(0, 0.01, rows),
            "target": random.integers(0, 2, rows),
        }
    )
    frame.to_csv(path, index=False)
    return path


def _configs() -> tuple[TemporalTransformerConfig, TransformerTrainingConfig]:
    return (
        TemporalTransformerConfig(
            input_features=8,
            sequence_length=8,
            d_model=16,
            n_heads=4,
            n_layers=1,
            feedforward_dim=32,
            latent_dim=4,
            return_horizons=(1, 2),
            regime_classes=3,
            hierarchical_direction=True,
            horizon_adapter_dim=8,
        ),
        TransformerTrainingConfig(
            epochs=2,
            batch_size=32,
            learning_rate=0.001,
            early_stopping_patience=2,
            mixed_precision=False,
            device="cpu",
            num_workers=0,
            train_fraction=0.7,
            validation_fraction=0.15,
            seed=7,
            checkpoint_metric="hierarchical_skill_score",
        ),
    )


def test_dataset_excludes_future_targets_and_uses_time_splits(tmp_path: Path) -> None:
    source = _write_feature_csv(tmp_path / "features.csv")
    model_config, training_config = _configs()

    prepared = prepare_transformer_datasets(
        [source],
        model_config,
        training_config,
    )

    assert len(prepared.train) > len(prepared.validation) > 0
    assert len(prepared.test) > 0
    assert "future_return" not in prepared.scaler.feature_columns
    assert "target" not in prepared.scaler.feature_columns
    sample = prepared.train[0]
    assert sample["features"].shape == (8, prepared.resolved_model_config.input_features)
    assert sample["future_returns"].shape == (2,)
    assert sample["feature_mask"].shape == sample["features"].shape
    assert sample["edge_returns"].shape == (2, 2)
    assert sample["excursions"].shape == (2, 2)
    assert sample["tradeability"].shape == (2,)
    assert sample["movement_directions"].shape == (2,)
    assert sample["future_sides"].shape == (2,)
    assert prepared.scaler.label_mode == (
        "hierarchical_movement_side_cost_aware_tradeability"
    )
    assert len(prepared.scaler.feature_group_ids) == len(
        prepared.scaler.feature_columns
    )


def test_dataset_prioritizes_all_multitimeframe_scales(tmp_path: Path) -> None:
    source = _write_feature_csv(tmp_path / "multitimeframe.csv", rows=260)
    frame = pd.read_csv(source)
    for interval_index, interval in enumerate(["1m", "5m", "15m", "1h", "1d"]):
        frame[f"mtf_{interval}_return_1"] = 0.001 * (interval_index + 1)
        frame[f"mtf_{interval}_return_5"] = 0.002 * (interval_index + 1)
        frame[f"mtf_{interval}_available"] = 1.0
        frame[f"mtf_{interval}_age_ratio"] = 0.0
    frame["mtf_schema_version"] = 1
    frame.to_csv(source, index=False)
    model_config, training_config = _configs()
    model_config = TemporalTransformerConfig(
        input_features=16,
        sequence_length=model_config.sequence_length,
        d_model=model_config.d_model,
        n_heads=model_config.n_heads,
        n_layers=model_config.n_layers,
        feedforward_dim=model_config.feedforward_dim,
        latent_dim=model_config.latent_dim,
        return_horizons=model_config.return_horizons,
        regime_classes=model_config.regime_classes,
    )

    prepared = prepare_transformer_datasets([source], model_config, training_config)

    selected = prepared.scaler.feature_columns
    assert "mtf_schema_version" not in selected
    for interval in ["1m", "5m", "15m", "1h", "1d"]:
        assert any(column.startswith(f"mtf_{interval}_") for column in selected)


def test_nine_timeframe_dataset_keeps_every_financial_indicator(tmp_path: Path) -> None:
    source = _write_feature_csv(tmp_path / "btc_nine_timeframes.csv", rows=260)
    frame = pd.read_csv(source)
    mtf_values: dict[str, object] = {}
    for interval_index, interval in enumerate(BTC_MULTITIMEFRAME_INTERVALS):
        for feature_index, feature in enumerate(MULTITIMEFRAME_FEATURE_NAMES):
            mtf_values[f"mtf_{interval}_{feature}"] = (
                0.001 * (interval_index + 1) + feature_index * 0.00001
            )
        mtf_values[f"mtf_{interval}_available"] = 1.0
        mtf_values[f"mtf_{interval}_age_ratio"] = 0.0
    mtf_values["mtf_schema_version"] = 2
    mtf_values["mtf_source_intervals"] = "|".join(BTC_MULTITIMEFRAME_INTERVALS)
    frame = pd.concat(
        [frame, pd.DataFrame(mtf_values, index=frame.index)],
        axis=1,
    )
    frame.to_csv(source, index=False)
    _, training_config = _configs()
    mtf_feature_count = len(BTC_MULTITIMEFRAME_INTERVALS) * (
        len(MULTITIMEFRAME_FEATURE_NAMES) + 2
    )
    model_config = TemporalTransformerConfig(
        input_features=(mtf_feature_count * 4 + 2) // 3 + 16,
        sequence_length=8,
        d_model=16,
        n_heads=4,
        n_layers=1,
        feedforward_dim=32,
        latent_dim=8,
        return_horizons=(1, 2),
        regime_classes=3,
    )

    prepared = prepare_transformer_datasets([source], model_config, training_config)
    selected = set(prepared.scaler.feature_columns)

    for interval in BTC_MULTITIMEFRAME_INTERVALS:
        for feature in MULTITIMEFRAME_FEATURE_NAMES:
            assert f"mtf_{interval}_{feature}" in selected
        assert f"mtf_{interval}_available" in selected
        assert f"mtf_{interval}_age_ratio" in selected


def test_two_epoch_training_saves_complete_artifacts(tmp_path: Path) -> None:
    source = _write_feature_csv(tmp_path / "features.csv")
    model_config, training_config = _configs()
    updates: list[dict[str, object]] = []

    result = train_temporal_transformer(
        [source],
        model_config,
        training_config,
        tmp_path / "models",
        run_name="two_epoch_test",
        progress_callback=updates.append,
    )

    assert result.model_path.exists()
    assert result.history_csv.exists()
    assert result.summary_json.exists()
    assert (result.run_dir / "test_predictions.csv").exists()
    assert result.best_epoch in {1, 2}
    assert 0 <= result.metrics["direction_accuracy"] <= 1
    assert 0 <= result.metrics["cost_aware_direction_balanced_accuracy"] <= 1
    assert "direction_majority_baseline" in result.metrics
    assert "direction_skill_score" in result.metrics
    assert updates[-1]["status"] == "complete"
    history = pd.read_csv(result.history_csv)
    assert len(history) == 2
    assert "validation_direction_balanced_accuracy" in history
    assert "validation_direction_accuracy_lift" in history
    assert "validation_direction_skill_score" in history
    summary = json.loads(result.summary_json.read_text(encoding="utf-8"))
    assert summary["checkpoint_metric"] == "hierarchical_skill_score"
    assert len(summary["training_class_balance"]["direction_class_weights"]) == 2
    assert len(summary["training_class_balance"]["tradeability_pos_weights"]) == 2
    assert len(summary["training_class_balance"]["side_class_weights"]) == 2
    checkpoint = torch.load(
        result.model_path,
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint["schema_version"] == 3
    assert checkpoint["calibration"]["method"] == (
        "temperature_scaling_validation_grid_v1"
    )
    predictions = pd.read_csv(result.run_dir / "test_predictions.csv")
    assert "predicted_long_edge_1" in predictions
    assert "tradeability_probability_2" in predictions
    assert "movement_up_probability_2" in predictions
    assert "side_up_probability_2" in predictions
    assert "timeframe_fast_attention" in predictions
