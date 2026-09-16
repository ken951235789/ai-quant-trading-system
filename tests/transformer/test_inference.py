"""Transformer checkpoint 批次推論測試。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from ai_quant_trading.transformer import (
    TemporalTransformerConfig,
    apply_transformer_checkpoint,
    transformer_oos_provenance,
)
from ai_quant_trading.transformer.model import MarketTemporalTransformer


def test_inference_writes_available_rows_and_latent_features(tmp_path: Path) -> None:
    config = TemporalTransformerConfig(
        input_features=2,
        sequence_length=4,
        d_model=8,
        n_heads=2,
        n_layers=1,
        feedforward_dim=16,
        latent_dim=3,
        return_horizons=(1, 5, 20),
        hierarchical_direction=True,
        horizon_adapter_dim=4,
    )
    model = MarketTemporalTransformer(config)
    checkpoint = tmp_path / "run" / "best_model.pt"
    checkpoint.parent.mkdir()
    torch.save(
        {
            "model_config": config.to_dict(),
            "feature_columns": ["feature_a", "feature_b"],
            "scaler": {
                "medians": [0.0, 0.0],
                "means": [0.0, 0.0],
                "scales": [1.0, 1.0],
                "return_means": [0.0, 0.0, 0.0],
                "return_scales": [1.0, 1.0, 1.0],
                "volatility_mean": 0.0,
                "volatility_scale": 1.0,
            },
            "state_dict": model.state_dict(),
        },
        checkpoint,
    )
    source = tmp_path / "features.csv"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
            "feature_a": range(10),
            "feature_b": range(10, 20),
        }
    ).to_csv(source, index=False)

    artifact = apply_transformer_checkpoint(
        checkpoint,
        source,
        device="cpu",
        batch_size=3,
    )
    saved = pd.read_csv(source)

    assert artifact.predicted_rows == 7
    assert saved["transformer_available"].tolist()[:3] == [0.0, 0.0, 0.0]
    assert saved["transformer_available"].tolist()[3:] == [1.0] * 7
    assert "transformer_latent_2" in saved
    assert "transformer_up_probability_5" in saved
    assert "transformer_movement_up_probability_5" in saved
    assert "transformer_side_up_probability_5" in saved
    assert "transformer_return_q10_20" in saved
    assert "transformer_long_edge_5" in saved
    assert "transformer_short_edge_5" in saved
    assert "transformer_tradeability_5" in saved
    assert "transformer_timeframe_fast_attention" in saved
    probabilities = saved.loc[
        saved["transformer_available"].eq(1),
        [
            "transformer_down_probability_5",
            "transformer_neutral_probability_5",
            "transformer_up_probability_5",
        ],
    ]
    assert probabilities.sum(axis=1).round(6).eq(1.0).all()
    assert saved["transformer_model_id"].iloc[-1] == "run"


def test_v3_inference_masks_a_missing_checkpoint_feature(tmp_path: Path) -> None:
    config = TemporalTransformerConfig(
        input_features=3,
        sequence_length=4,
        d_model=8,
        n_heads=2,
        n_layers=1,
        feedforward_dim=16,
        feature_group_ids=(0, 1, 2),
    )
    model = MarketTemporalTransformer(config)
    checkpoint = tmp_path / "v3" / "best_model.pt"
    checkpoint.parent.mkdir()
    torch.save(
        {
            "model_config": config.to_dict(),
            "feature_columns": ["fast", "medium", "slow"],
            "scaler": {
                "medians": [0.0, 0.0, 0.0],
                "means": [0.0, 0.0, 0.0],
                "scales": [1.0, 1.0, 1.0],
                "return_means": [0.0, 0.0, 0.0],
                "return_scales": [1.0, 1.0, 1.0],
                "volatility_mean": 0.0,
                "volatility_scale": 1.0,
            },
            "state_dict": model.state_dict(),
        },
        checkpoint,
    )
    source = tmp_path / "missing.csv"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=7, freq="5min", tz="UTC"),
            "fast": range(7),
            "slow": range(10, 17),
        }
    ).to_csv(source, index=False)

    artifact = apply_transformer_checkpoint(checkpoint, source, device="cpu")
    saved = pd.read_csv(artifact.output_path)

    assert artifact.predicted_rows == 4
    assert "medium" in saved
    assert saved["medium"].isna().all()
    assert saved["transformer_available"].iloc[-1] == 1.0
    assert saved["transformer_input_coverage"].iloc[-1] == pytest.approx(2 / 3)


def test_inference_marks_all_missing_core_features_unavailable(tmp_path: Path) -> None:
    config = TemporalTransformerConfig(
        input_features=2,
        sequence_length=4,
        d_model=8,
        n_heads=2,
        n_layers=1,
        feedforward_dim=16,
    )
    model = MarketTemporalTransformer(config)
    checkpoint = tmp_path / "core" / "best_model.pt"
    checkpoint.parent.mkdir()
    torch.save(
        {
            "model_config": config.to_dict(),
            "feature_columns": ["return_1", "atr_14"],
            "scaler": {
                "medians": [0.0, 0.0],
                "means": [0.0, 0.0],
                "scales": [1.0, 1.0],
                "return_means": [0.0, 0.0, 0.0],
                "return_scales": [1.0, 1.0, 1.0],
                "volatility_mean": 0.0,
                "volatility_scale": 1.0,
            },
            "state_dict": model.state_dict(),
        },
        checkpoint,
    )
    source = tmp_path / "empty_core.csv"
    pd.DataFrame(
        {"timestamp": pd.date_range("2026-01-01", periods=6, freq="5min", tz="UTC")}
    ).to_csv(source, index=False)

    artifact = apply_transformer_checkpoint(checkpoint, source, device="cpu")
    saved = pd.read_csv(artifact.output_path)

    assert artifact.predicted_rows == 0
    assert saved["transformer_available"].eq(0).all()
    assert saved["transformer_input_valid"].eq(0).all()


def test_oos_provenance_starts_after_validation_cutoff(tmp_path: Path) -> None:
    timestamps = pd.date_range("2026-01-01", periods=100, freq="15min", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "exchange": "binance_futures",
            "symbol": "BTC/USDT",
            "interval": "15m",
        }
    )
    checkpoint = tmp_path / "provenance.pt"
    torch.save(
        {
            "model_config": TemporalTransformerConfig(input_features=1).to_dict(),
            "feature_columns": ["return_1"],
            "scaler": {},
            "state_dict": {},
            "training_config": {"train_fraction": 0.6, "validation_fraction": 0.2},
            "sources": [
                {
                    "exchange": "binance_futures",
                    "symbol": "BTC/USDT",
                    "interval": "15m",
                    "rows": 100,
                    "start_at": timestamps[0].isoformat(),
                    "end_at": timestamps[-1].isoformat(),
                }
            ],
        },
        checkpoint,
    )

    provenance = transformer_oos_provenance(checkpoint, frame)

    assert pd.Timestamp(provenance["safe_rl_start_exclusive"]) == timestamps[79]
