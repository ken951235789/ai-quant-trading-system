"""AI 管線設定保存與載入測試。"""

from __future__ import annotations

import json

from ai_quant_trading.ai_pipeline import (
    AIPipelineConfig,
    load_ai_pipeline_config,
    save_ai_pipeline_config,
)
from ai_quant_trading.sentiment import FinBERTConfig
from ai_quant_trading.transformer import TemporalTransformerConfig


def test_pipeline_configuration_round_trip(tmp_path) -> None:
    config = AIPipelineConfig(
        finbert=FinBERTConfig(batch_size=32),
        transformer=TemporalTransformerConfig(
            input_features=80,
            sequence_length=96,
            d_model=192,
            n_heads=8,
        ),
    )
    path = save_ai_pipeline_config(config, tmp_path / "pipeline.json")

    restored = load_ai_pipeline_config(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert restored.finbert.batch_size == 32
    assert restored.transformer.sequence_length == 96
    assert restored.transformer.return_horizons == (1, 5, 20)
    assert "updated_at" in payload
