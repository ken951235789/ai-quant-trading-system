"""金融時序 Transformer 的模型與訓練參數。"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class TemporalTransformerConfig:
    """多任務市場編碼器架構。"""

    input_features: int = 256
    sequence_length: int = 96
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 3
    feedforward_dim: int = 192
    dropout: float = 0.10
    latent_dim: int = 16
    return_horizons: tuple[int, ...] = (1, 5, 20)
    regime_classes: int = 3
    architecture_version: int = 3
    local_kernel_size: int = 3
    quantile_levels: tuple[float, ...] = (0.10, 0.50, 0.90)
    patch_size: int = 4
    patch_stride: int = 2
    feature_group_ids: tuple[int, ...] = ()
    feature_group_names: tuple[str, ...] = ("fast", "medium", "slow")
    volatility_regime_classes: int = 3
    hierarchical_direction: bool = False
    horizon_adapter_dim: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "return_horizons",
            tuple(int(value) for value in self.return_horizons),
        )
        object.__setattr__(
            self,
            "quantile_levels",
            tuple(float(value) for value in self.quantile_levels),
        )
        object.__setattr__(
            self,
            "feature_group_ids",
            tuple(int(value) for value in self.feature_group_ids),
        )
        object.__setattr__(
            self,
            "feature_group_names",
            tuple(str(value) for value in self.feature_group_names),
        )
        for name in [
            "input_features",
            "sequence_length",
            "d_model",
            "n_heads",
            "n_layers",
            "feedforward_dim",
            "latent_dim",
            "regime_classes",
            "architecture_version",
            "local_kernel_size",
            "patch_size",
            "patch_stride",
            "volatility_regime_classes",
        ]:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} 必須大於 0")
        if self.horizon_adapter_dim < 0:
            raise ValueError("horizon_adapter_dim 不可小於 0")
        if self.hierarchical_direction and self.horizon_adapter_dim <= 0:
            raise ValueError("階層式方向模型必須啟用 horizon_adapter_dim")
        if self.d_model % self.n_heads:
            raise ValueError("d_model 必須能被 n_heads 整除")
        if self.architecture_version != 3:
            raise ValueError("Transformer 主線只支援 architecture_version=3")
        if self.local_kernel_size % 2 == 0:
            raise ValueError("local_kernel_size 必須是奇數")
        if self.sequence_length < self.patch_size:
            raise ValueError("v3 的 sequence_length 不可小於 patch_size")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("v3 每個注意力頭的維度必須是偶數，才能套用 RoPE")
        if len(self.feature_group_names) != 3:
            raise ValueError("v3 必須定義 fast、medium、slow 三個特徵群組")
        if self.feature_group_ids and len(self.feature_group_ids) != self.input_features:
            raise ValueError("feature_group_ids 數量必須等於 input_features")
        if any(value not in {0, 1, 2} for value in self.feature_group_ids):
            raise ValueError("feature_group_ids 只能包含 0、1、2")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout 必須介於 0（含）與 1（不含）之間")
        if not self.return_horizons or any(value <= 0 for value in self.return_horizons):
            raise ValueError("return_horizons 必須包含正整數")
        if len(self.quantile_levels) < 3:
            raise ValueError("quantile_levels 至少需要三個分位數")
        if tuple(sorted(self.quantile_levels)) != self.quantile_levels:
            raise ValueError("quantile_levels 必須由小到大排列")
        if any(not 0 < value < 1 for value in self.quantile_levels):
            raise ValueError("quantile_levels 必須介於 0 與 1 之間")

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["return_horizons"] = list(self.return_horizons)
        result["quantile_levels"] = list(self.quantile_levels)
        result["feature_group_ids"] = list(self.feature_group_ids)
        result["feature_group_names"] = list(self.feature_group_names)
        return result


@dataclass(frozen=True, slots=True)
class TransformerTrainingConfig:
    """尚未執行訓練時也可保存的完整訓練方案。"""

    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.05
    gradient_clip: float = 1.0
    early_stopping_patience: int = 8
    mixed_precision: bool = True
    device: str = "auto"
    num_workers: int = 2
    cpu_threads: int = 4
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    seed: int = 42
    return_loss_weight: float = 1.0
    volatility_loss_weight: float = 0.5
    regime_loss_weight: float = 0.25
    direction_loss_weight: float = 0.50
    quantile_loss_weight: float = 0.25
    direction_threshold_bps: float = 12.0
    label_smoothing: float = 0.02
    max_rows_per_source: int | None = None
    fee_bps_per_side: float = 4.0
    slippage_bps_per_side: float = 2.0
    max_observed_spread_bps: float = 50.0
    edge_loss_weight: float = 0.50
    excursion_loss_weight: float = 0.25
    tradeability_loss_weight: float = 0.25
    volatility_regime_loss_weight: float = 0.20
    probability_calibration: bool = True
    direction_class_balance_power: float = 0.50
    direction_focal_gamma: float = 1.50
    tradeability_class_balance_power: float = 0.50
    checkpoint_metric: str = "direction_skill_score"
    checkpoint_loss_penalty: float = 0.02
    movement_threshold_bps: float = 2.0
    movement_atr_multiplier: float = 0.10
    side_loss_weight: float = 0.75
    side_class_balance_power: float = 0.25

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs 與 batch_size 必須大於 0")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate 必須大於 0，weight_decay 不可小於 0")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio 必須介於 0（含）與 1（不含）之間")
        if self.gradient_clip <= 0:
            raise ValueError("gradient_clip 必須大於 0")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience 必須大於 0")
        if self.num_workers < 0:
            raise ValueError("num_workers 不可小於 0")
        if self.cpu_threads <= 0:
            raise ValueError("cpu_threads 必須大於 0")
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction 必須介於 0 與 1 之間")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction 必須介於 0 與 1 之間")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("訓練與驗證比例總和必須小於 1")
        if not self.device.strip():
            raise ValueError("device 不可為空")
        if self.return_loss_weight <= 0:
            raise ValueError("return_loss_weight 必須大於 0")
        if self.volatility_loss_weight < 0:
            raise ValueError("volatility_loss_weight 不可小於 0")
        if self.regime_loss_weight < 0:
            raise ValueError("regime_loss_weight 不可小於 0")
        if self.direction_loss_weight < 0 or self.quantile_loss_weight < 0:
            raise ValueError("direction_loss_weight 與 quantile_loss_weight 不可小於 0")
        if self.direction_threshold_bps < 0:
            raise ValueError("direction_threshold_bps 不可小於 0")
        for name in [
            "fee_bps_per_side",
            "slippage_bps_per_side",
            "max_observed_spread_bps",
            "edge_loss_weight",
            "excursion_loss_weight",
            "tradeability_loss_weight",
            "volatility_regime_loss_weight",
            "direction_class_balance_power",
            "direction_focal_gamma",
            "tradeability_class_balance_power",
            "checkpoint_loss_penalty",
            "movement_threshold_bps",
            "movement_atr_multiplier",
            "side_loss_weight",
            "side_class_balance_power",
        ]:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} 不可小於 0")
        if self.direction_class_balance_power > 1:
            raise ValueError("direction_class_balance_power 不可大於 1")
        if self.tradeability_class_balance_power > 1:
            raise ValueError("tradeability_class_balance_power 不可大於 1")
        if self.side_class_balance_power > 1:
            raise ValueError("side_class_balance_power 不可大於 1")
        if self.checkpoint_metric not in {
            "validation_loss",
            "direction_skill_score",
            "hierarchical_skill_score",
            "cost_aware_direction_balanced_accuracy",
        }:
            raise ValueError("checkpoint_metric 不支援")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError("label_smoothing 必須介於 0（含）與 1（不含）之間")
        if self.max_rows_per_source is not None and self.max_rows_per_source < 100:
            raise ValueError("max_rows_per_source 至少需要 100")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
