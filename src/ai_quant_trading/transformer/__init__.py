"""金融時序 Transformer 架構與推論特徵契約。"""

from ai_quant_trading.transformer.config import (
    TemporalTransformerConfig,
    TransformerTrainingConfig,
)
from ai_quant_trading.transformer.features import (
    TRANSFORMER_CONTEXT_COLUMNS,
    ensure_transformer_context,
)
from ai_quant_trading.transformer.training import (
    TransformerBackendStatus,
    TransformerTrainingResult,
    train_temporal_transformer,
    transformer_backend_status,
)
from ai_quant_trading.transformer.inference import (
    TransformerInferenceArtifact,
    apply_transformer_checkpoint,
    infer_latest_transformer_context,
    infer_transformer_context_frame,
    transformer_checkpoint_sequence_length,
    transformer_oos_provenance,
)

__all__ = [
    "TRANSFORMER_CONTEXT_COLUMNS",
    "TemporalTransformerConfig",
    "TransformerTrainingConfig",
    "TransformerBackendStatus",
    "TransformerTrainingResult",
    "TransformerInferenceArtifact",
    "apply_transformer_checkpoint",
    "infer_latest_transformer_context",
    "infer_transformer_context_frame",
    "transformer_checkpoint_sequence_length",
    "transformer_oos_provenance",
    "ensure_transformer_context",
    "train_temporal_transformer",
    "transformer_backend_status",
]
