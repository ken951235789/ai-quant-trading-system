"""Transformer 設定、輸出與安全預設值測試。"""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from ai_quant_trading.transformer import (
    TemporalTransformerConfig,
    ensure_transformer_context,
)
from ai_quant_trading.transformer.model import MarketTemporalTransformer


def test_configuration_rejects_invalid_attention_heads() -> None:
    with pytest.raises(ValueError, match="d_model"):
        TemporalTransformerConfig(d_model=127, n_heads=8)


def test_model_outputs_all_tasks() -> None:
    config = TemporalTransformerConfig(
        input_features=12,
        sequence_length=16,
        d_model=32,
        n_heads=4,
        n_layers=2,
        feedforward_dim=64,
        latent_dim=8,
    )
    model = MarketTemporalTransformer(config)

    output = model(torch.randn(3, 16, 12))

    assert output["future_returns"].shape == (3, 3)
    assert output["volatility"].shape == (3, 1)
    assert output["regime_probability"].shape == (3, 3)
    assert output["uncertainty"].shape == (3,)
    assert output["latent"].shape == (3, 8)
    assert output["direction_probability"].shape == (3, 3, 3)
    assert output["quantile_returns"].shape == (3, 3, 3)
    assert output["edge_returns"].shape == (3, 3, 2)
    assert output["excursions"].shape == (3, 3, 2)
    assert output["tradeability_probability"].shape == (3, 3)
    assert output["volatility_regime_probability"].shape == (3, 3)
    assert output["timeframe_attention"].shape == (3, 3)
    assert torch.allclose(
        output["direction_probability"].sum(dim=-1),
        torch.ones(3, 3),
        atol=1e-6,
    )
    assert model.trainable_parameters > 0


def test_v3_accepts_missing_feature_mask_and_normalizes_branch_attention() -> None:
    config = TemporalTransformerConfig(
        input_features=6,
        sequence_length=8,
        d_model=12,
        n_heads=3,
        n_layers=1,
        feedforward_dim=24,
        feature_group_ids=(0, 0, 1, 1, 2, 2),
    )
    features = torch.randn(2, 8, 6)
    feature_mask = torch.ones_like(features)
    feature_mask[:, :, :2] = 0

    output = MarketTemporalTransformer(config)(
        features,
        feature_mask=feature_mask,
    )

    assert torch.isfinite(output["latent"]).all()
    assert torch.allclose(
        output["timeframe_attention"].sum(dim=-1),
        torch.ones(2),
        atol=1e-6,
    )
    assert output["timeframe_attention"][:, 0].eq(0).all()


def test_hierarchical_model_combines_tradeability_and_side_probabilities() -> None:
    config = TemporalTransformerConfig(
        input_features=6,
        sequence_length=8,
        d_model=12,
        n_heads=3,
        n_layers=1,
        feedforward_dim=24,
        hierarchical_direction=True,
        horizon_adapter_dim=8,
    )

    output = MarketTemporalTransformer(config)(torch.randn(2, 8, 6))

    assert output["movement_probability"].shape == (2, 3, 3)
    assert output["side_probability"].shape == (2, 3, 2)
    assert torch.allclose(
        output["direction_probability"].sum(dim=-1),
        torch.ones(2, 3),
        atol=1e-6,
    )
    expected_neutral = 1.0 - output["tradeability_probability"]
    assert torch.allclose(
        output["direction_probability"][..., 1],
        expected_neutral,
        atol=1e-6,
    )


def test_legacy_architecture_is_rejected() -> None:
    with pytest.raises(ValueError, match="只支援 architecture_version=3"):
        TemporalTransformerConfig(architecture_version=2)


def test_untrained_transformer_context_defaults_to_unavailable() -> None:
    result = ensure_transformer_context(pd.DataFrame({"close": [100.0, 101.0]}))

    assert result["transformer_available"].eq(0).all()
    assert result["transformer_return_1"].eq(0).all()
