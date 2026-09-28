"""AI 管線設定保存與載入測試。"""

from __future__ import annotations

import json

from ai_quant_trading.ai_pipeline import (
    AIPipelineConfig,
    load_ai_pipeline_config,
    save_ai_pipeline_config,
)
from ai_quant_trading.sentiment import FinBERTConfig
from ai_quant_trading.transformer import (
    TemporalTransformerConfig,
    TransformerTrainingConfig,
)


def test_pipeline_configuration_round_trip(tmp_path) -> None:
    config = AIPipelineConfig(
        finbert=FinBERTConfig(batch_size=32),
        transformer=TemporalTransformerConfig(
            input_features=80,
            sequence_length=96,
            d_model=192,
            n_heads=8,
        ),
        transformer_training=TransformerTrainingConfig(
            embargo_bars=48,
            recency_half_life_days=365.0,
            recency_min_weight=0.10,
            checkpoint_horizon_weights=(0.15, 0.70, 0.15),
        ),
    )
    path = save_ai_pipeline_config(config, tmp_path / "pipeline.json")

    restored = load_ai_pipeline_config(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert restored.finbert.batch_size == 32
    assert restored.transformer.sequence_length == 96
    assert restored.transformer.return_horizons == (5, 20, 48)
    assert restored.transformer.timing_horizon == 5
    assert restored.transformer.primary_horizon == 20
    assert restored.transformer.regime_horizon == 48
    assert restored.transformer_training.embargo_bars == 48
    assert restored.transformer_training.recency_half_life_days == 365.0
    assert restored.transformer_training.checkpoint_horizon_weights == (0.15, 0.70, 0.15)
    assert "updated_at" in payload
