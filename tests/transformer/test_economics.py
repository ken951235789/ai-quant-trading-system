"""驗證交易目標、時間隔離及經濟選模的關鍵不變條件。"""

from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest
import torch

from ai_quant_trading.transformer.config import TemporalTransformerConfig, TransformerTrainingConfig
from ai_quant_trading.transformer.dataset import _v3_targets, prepare_transformer_datasets
from ai_quant_trading.transformer.economics import (
    block_confidence_interval,
    fixed_hold_evaluation,
    score_signals,
    signal_diagnostics,
)
from ai_quant_trading.transformer.training import (
    _split_calibration,
    _task_loss,
    train_temporal_transformer,
)


def _frame(rows=500):
    rng = np.random.default_rng(51)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.008, rows)))
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="15min", tz="UTC"),
            "symbol": "BTC/USDT",
            "exchange": "binance_futures",
            "interval": "15m",
            "open": close,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": rng.uniform(1, 100, rows),
            "return_1": pd.Series(close).pct_change(),
            "rsi_14": rng.uniform(10, 90, rows),
            "atr_pct": 0.01,
        }
    )


def _configs():
    return TemporalTransformerConfig(
        input_features=4,
        sequence_length=8,
        d_model=16,
        n_heads=4,
        n_layers=1,
        feedforward_dim=32,
        latent_dim=4,
        return_horizons=(1, 2),
        hierarchical_direction=True,
        horizon_adapter_dim=8,
    ), TransformerTrainingConfig(
        epochs=2,
        batch_size=64,
        num_workers=0,
        device="cpu",
        cpu_threads=2,
        mixed_precision=False,
        trading_target_mode="terminal_net",
        checkpoint_metric="economic_selection_score",
        calibration_fraction=0.5,
        training_horizon_weights=(0.2, 0.8),
        embargo_bars=2,
        economic_minimum_trades=2,
        max_feature_correlation=None,
        drop_constant_features=False,
    )


def test_terminal_target_matches_edge_when_price_reverses():
    frame = _frame(6)
    frame["open"] = 100.0
    frame["close"] = [100, 103, 98, 100, 101, 100]
    frame["high"] = [100, 104, 100, 101, 102, 101]
    frame["low"] = [100, 100, 97, 99, 100, 99]
    model, training = _configs()
    terminal = _v3_targets(frame, model, training)
    legacy = _v3_targets(frame, model, replace(training, trading_target_mode="first_touch"))
    assert legacy["future_directions"][0, 1] == 2
    assert terminal["future_directions"][0, 1] == 0
    assert terminal["edge_returns"][0, 1, 1] == pytest.approx(0.02 - 0.0012)
    assert terminal["edge_returns"][0, 1, 0] == pytest.approx(-0.02 - 0.0012)


def test_calibration_purges_future_label_overlap(tmp_path):
    source = tmp_path / "features.csv"
    _frame().to_csv(source, index=False)
    model, training = _configs()
    data = prepare_transformer_datasets([source], model, training)
    cal, val, protocol = _split_calibration(data, training)
    assert max(point for _, point in cal.references) + 2 < min(point for _, point in val.references)
    assert not set(cal.references) & set(val.references)
    assert protocol["method"] == "chronological_disjoint_v1"
    assert data.scaler.label_mode == "terminal_net_cost_aware_tradeability"


def _predictions(rows=12):
    return pd.DataFrame(
        {
            "series_index": 0,
            "endpoint": np.arange(rows),
            "predicted_long_edge_2": 0.01,
            "predicted_short_edge_2": -0.01,
            "actual_long_edge_2": 0.005,
            "actual_short_edge_2": -0.007,
            "timestamp_ns": pd.date_range("2025-01-01", periods=rows, freq="15min", tz="UTC")
            .as_unit("ns")
            .asi8,
            "predicted_regime": 1,
        }
    )


def test_fixed_hold_does_not_overlap_and_cost_stress_reduces_return():
    frame = _predictions()
    metrics, trades = fixed_hold_evaluation(frame, 2, minimum_trades=2)
    stressed, _ = fixed_hold_evaluation(frame, 2, minimum_trades=2, extra_cost_bps=10)
    assert trades["endpoint"].tolist() == [0, 2, 4, 6, 8, 10]
    assert metrics["fixed_hold_total_return"] == pytest.approx(1.005**6 - 1)
    assert stressed["fixed_hold_total_return"] < metrics["fixed_hold_total_return"]
    assert metrics["fixed_hold_expectancy_ci_low"] == pytest.approx(0.005)


def test_negative_predicted_edges_abstain_and_future_cannot_change_score():
    frame = _predictions()
    original = score_signals(frame, 2)
    frame["actual_long_edge_2"] = -99.0
    pd.testing.assert_frame_equal(original, score_signals(frame, 2))
    frame["predicted_long_edge_2"] = -0.001
    metrics, trades = fixed_hold_evaluation(frame, 2)
    assert trades.empty
    assert metrics["economic_selection_score"] == -1.0


def test_horizon_weight_controls_training_loss():
    model, training = _configs()
    output = {
        "future_returns": torch.tensor([[0.0, 2.0]]),
        "volatility": torch.zeros(1, 1),
        "regime_logits": torch.zeros(1, 3),
    }
    batch = {
        "future_returns": torch.zeros(1, 2),
        "volatility": torch.zeros(1, 1),
        "regime": torch.zeros(1, dtype=torch.long),
    }
    _, loss = _task_loss(output, batch, replace(training, training_horizon_weights=(1, 0)), model)
    assert loss["return_loss"].item() == 0
    _, loss = _task_loss(output, batch, replace(training, training_horizon_weights=(0, 1)), model)
    assert loss["return_loss"].item() == pytest.approx(1.5)


def test_diagnostics_are_reproducible():
    diagnostics = signal_diagnostics(_predictions(), 2, 0.1)
    assert {"score_decile", "side", "predicted_regime", "utc_hour", "month"} <= set(
        diagnostics["group"]
    )
    values = np.random.default_rng(1).normal(0, 0.01, 80)
    assert block_confidence_interval(values) == block_confidence_interval(values)


def test_economic_training_saves_calibrated_evidence(tmp_path):
    source = tmp_path / "features.csv"
    _frame().to_csv(source, index=False)
    model, training = _configs()
    result = train_temporal_transformer([source], model, training, tmp_path / "models")
    summary = json.loads(result.summary_json.read_text(encoding="utf-8"))
    checkpoint = torch.load(result.model_path, weights_only=True)
    assert summary["calibration"] == checkpoint["calibration"]
    assert summary["calibration_protocol"]["method"] == "chronological_disjoint_v1"
    assert "fixed_hold_expectancy" in summary["validation_selection_metrics"]
    assert "fixed_hold_expectancy" in summary["test_metrics"]
    assert (result.run_dir / "test_signal_diagnostics.csv").is_file()
    assert (result.run_dir / "validation_predictions.csv").is_file()
    history = pd.read_csv(result.run_dir / "history.csv")
    assert summary["best_checkpoint_value"] == pytest.approx(history["checkpoint_value"].max())
