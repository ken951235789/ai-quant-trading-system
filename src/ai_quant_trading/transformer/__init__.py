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
from ai_quant_trading.transformer.crossfit import (
    TransformerCrossFitConfig,
    TransformerCrossFitFold,
    TransformerCrossFitPlan,
    TransformerCrossFitResult,
    assemble_transformer_crossfit_predictions,
    build_transformer_crossfit_plan,
    load_transformer_crossfit_predictions,
    run_transformer_crossfit,
)

__all__ = [
    "TRANSFORMER_CONTEXT_COLUMNS",
    "TemporalTransformerConfig",
    "TransformerTrainingConfig",
    "TransformerBackendStatus",
    "TransformerTrainingResult",
    "TransformerInferenceArtifact",
    "TransformerCrossFitConfig",
    "TransformerCrossFitFold",
    "TransformerCrossFitPlan",
    "TransformerCrossFitResult",
    "apply_transformer_checkpoint",
    "assemble_transformer_crossfit_predictions",
    "build_transformer_crossfit_plan",
    "load_transformer_crossfit_predictions",
    "infer_latest_transformer_context",
    "infer_transformer_context_frame",
    "transformer_checkpoint_sequence_length",
    "transformer_oos_provenance",
    "ensure_transformer_context",
    "train_temporal_transformer",
    "transformer_backend_status",
    "run_transformer_crossfit",
]
