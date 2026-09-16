"""FinBERT 新聞情緒模型與特徵聚合設定。"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class FinBERTConfig:
    """FinBERT 推論與時間衰減參數。"""

    model_name: str = "ProsusAI/finbert"
    # 固定官方模型版本，避免遠端模型更新後在未審查下改變行為。
    revision: str = "4556d13015211d73dccd3fdd39d39232506f3e43"
    device: str = "auto"
    batch_size: int = 16
    max_length: int = 192
    confidence_threshold: float = 0.55
    aggregation_window_hours: int = 24
    decay_half_life_hours: float = 8.0
    minimum_training_coverage: float = 0.30
    cache_dir: str = "data/models/finbert"

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name 不可為空")
        if not self.revision.strip():
            raise ValueError("revision 不可為空")
        if not self.device.strip():
            raise ValueError("device 不可為空")
        if self.batch_size <= 0:
            raise ValueError("batch_size 必須大於 0")
        if not 8 <= self.max_length <= 4096:
            raise ValueError("max_length 必須介於 8 與 4096 之間")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold 必須介於 0 與 1 之間")
        if self.aggregation_window_hours <= 0:
            raise ValueError("aggregation_window_hours 必須大於 0")
        if self.decay_half_life_hours <= 0:
            raise ValueError("decay_half_life_hours 必須大於 0")
        if not 0 < self.minimum_training_coverage <= 1:
            raise ValueError("minimum_training_coverage 必須介於 0 與 1")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
