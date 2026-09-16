"""多模態 AI 管線的版本化設定檔。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path

from ai_quant_trading.sentiment import FinBERTConfig
from ai_quant_trading.transformer import (
    TemporalTransformerConfig,
    TransformerTrainingConfig,
)


@dataclass(frozen=True, slots=True)
class AIPipelineConfig:
    """保存三個模型之間的功能開關與可重現參數。"""

    schema_version: int = 3
    finbert_enabled: bool = True
    transformer_enabled: bool = True
    ppo_use_finbert: bool = False
    ppo_use_transformer: bool = True
    finbert: FinBERTConfig = field(default_factory=FinBERTConfig)
    transformer: TemporalTransformerConfig = field(
        default_factory=TemporalTransformerConfig
    )
    transformer_training: TransformerTrainingConfig = field(
        default_factory=TransformerTrainingConfig
    )

    def __post_init__(self) -> None:
        if self.schema_version <= 0:
            raise ValueError("schema_version 必須大於 0")
        if self.ppo_use_finbert and not self.finbert_enabled:
            raise ValueError("SAC 要使用 FinBERT 時必須先啟用 FinBERT")
        if self.ppo_use_transformer and not self.transformer_enabled:
            raise ValueError("SAC 要使用 Transformer 時必須先啟用 Transformer")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def save_ai_pipeline_config(
    config: AIPipelineConfig,
    path: str | Path,
) -> Path:
    """以 UTF-8 JSON 保存，不包含任何 API Key。"""
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = config.to_dict()
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target


def load_ai_pipeline_config(path: str | Path) -> AIPipelineConfig:
    """讀取設定；若檔案不存在則回傳安全預設值。"""
    target = Path(path)
    if not target.exists():
        return AIPipelineConfig()
    payload = json.loads(target.read_text(encoding="utf-8"))
    transformer_values = dict(payload.get("transformer", {}))
    horizons = transformer_values.get("return_horizons")
    if horizons is not None:
        transformer_values["return_horizons"] = tuple(int(value) for value in horizons)
    quantiles = transformer_values.get("quantile_levels")
    if quantiles is not None:
        transformer_values["quantile_levels"] = tuple(
            float(value) for value in quantiles
        )
    return AIPipelineConfig(
        schema_version=int(payload.get("schema_version", 1)),
        finbert_enabled=bool(payload.get("finbert_enabled", True)),
        transformer_enabled=bool(payload.get("transformer_enabled", True)),
        ppo_use_finbert=bool(payload.get("ppo_use_finbert", False)),
        ppo_use_transformer=bool(payload.get("ppo_use_transformer", True)),
        finbert=FinBERTConfig(**dict(payload.get("finbert", {}))),
        transformer=TemporalTransformerConfig(**transformer_values),
        transformer_training=TransformerTrainingConfig(
            **dict(payload.get("transformer_training", {}))
        ),
    )
