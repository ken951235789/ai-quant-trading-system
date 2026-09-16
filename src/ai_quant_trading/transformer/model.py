"""PyTorch 多任務金融時序 Transformer。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ai_quant_trading.transformer.config import TemporalTransformerConfig


class _RotaryEncoderLayer(nn.Module):
    """使用 RoPE 的 Pre-Norm Transformer Encoder。"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attention_norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.attention_output = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)
        self.feedforward_norm = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
            nn.Dropout(dropout),
        )

    def _apply_rope(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(
            query.shape[-2],
            device=query.device,
            dtype=torch.float32,
        )
        dimensions = torch.arange(
            0,
            self.head_dim,
            2,
            device=query.device,
            dtype=torch.float32,
        )
        frequencies = torch.pow(10_000.0, -dimensions / self.head_dim)
        angles = torch.outer(positions, frequencies)
        cosine = angles.cos().to(dtype=query.dtype)[None, None, :, :]
        sine = angles.sin().to(dtype=query.dtype)[None, None, :, :]

        def rotate(value: torch.Tensor) -> torch.Tensor:
            even = value[..., 0::2]
            odd = value[..., 1::2]
            return torch.stack(
                (even * cosine - odd * sine, even * sine + odd * cosine),
                dim=-1,
            ).flatten(-2)

        return rotate(query), rotate(key)

    def forward(
        self,
        values: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized = self.attention_norm(values)
        batch_size, sequence_length, width = normalized.shape
        query, key, value = self.qkv(normalized).chunk(3, dim=-1)

        def heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(
                batch_size,
                sequence_length,
                self.n_heads,
                self.head_dim,
            ).transpose(1, 2)

        query, key, value = heads(query), heads(key), heads(value)
        query, key = self._apply_rope(query, key)
        attention_mask = None
        if padding_mask is not None:
            attention_mask = (~padding_mask)[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=(self.attention_dropout.p if self.training else 0.0),
        )
        attended = attended.transpose(1, 2).reshape(
            batch_size,
            sequence_length,
            width,
        )
        values = values + self.attention_dropout(self.attention_output(attended))
        return values + self.feedforward(self.feedforward_norm(values))


class _FeatureBranch(nn.Module):
    """將一組金融特徵投影並壓縮為較短的 Patch 序列。"""

    def __init__(
        self,
        feature_indices: tuple[int, ...],
        config: TemporalTransformerConfig,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "feature_indices",
            torch.tensor(feature_indices, dtype=torch.long),
            persistent=False,
        )
        self.feature_count = len(feature_indices)
        if self.feature_count:
            input_width = self.feature_count * 2
            self.input_norm = nn.LayerNorm(input_width)
            self.projection = nn.Linear(input_width, config.d_model)
            self.gate = nn.Sequential(
                nn.Linear(input_width, config.d_model),
                nn.Sigmoid(),
            )
        else:
            self.empty_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.local_mixer = nn.Conv1d(
            config.d_model,
            config.d_model,
            kernel_size=config.local_kernel_size,
            padding=config.local_kernel_size // 2,
            groups=config.d_model,
        )
        self.local_norm = nn.LayerNorm(config.d_model)
        self.patch_projection = nn.Conv1d(
            config.d_model,
            config.d_model,
            kernel_size=config.patch_size,
            stride=config.patch_stride,
        )
        self.patch_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        features: torch.Tensor,
        feature_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.feature_count:
            selected = torch.index_select(features, 2, self.feature_indices)
            selected_mask = torch.index_select(feature_mask, 2, self.feature_indices)
            combined = torch.cat([selected, selected_mask], dim=-1)
            normalized = self.input_norm(combined)
            encoded = self.projection(normalized) * (0.5 + self.gate(normalized))
            available = selected_mask.bool().any(dim=(1, 2))
        else:
            encoded = self.empty_token.expand(
                features.shape[0],
                features.shape[1],
                -1,
            )
            available = torch.zeros(
                features.shape[0],
                dtype=torch.bool,
                device=features.device,
            )
        local = self.local_mixer(encoded.transpose(1, 2)).transpose(1, 2)
        encoded = self.local_norm(encoded + F.gelu(local))
        patches = self.patch_projection(encoded.transpose(1, 2)).transpose(1, 2)
        return self.patch_norm(patches), available


class MarketTemporalTransformer(nn.Module):
    """把一段市場序列編碼成強化學習可使用的前瞻上下文。"""

    def __init__(self, config: TemporalTransformerConfig) -> None:
        super().__init__()
        self.config = config
        group_ids = config.feature_group_ids or tuple(
            index % 3 for index in range(config.input_features)
        )
        group_indices = tuple(
            tuple(index for index, value in enumerate(group_ids) if value == group)
            for group in range(3)
        )
        self.feature_branches = nn.ModuleList(
            [_FeatureBranch(indices, config) for indices in group_indices]
        )
        self.branch_embedding = nn.Parameter(torch.zeros(1, 3, config.d_model))
        self.v3_encoder = nn.ModuleList(
            [
                _RotaryEncoderLayer(
                    config.d_model,
                    config.n_heads,
                    config.feedforward_dim,
                    config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.branch_pooling_score = nn.Linear(config.d_model, 1)
        self.fusion_query = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.fusion_attention = nn.MultiheadAttention(
                config.d_model,
                config.n_heads,
                dropout=config.dropout,
                batch_first=True,
        )
        self.fusion_norm = nn.LayerNorm(config.d_model)

        self.output_norm = nn.LayerNorm(config.d_model)
        self.return_head = nn.Linear(config.d_model, len(config.return_horizons))
        self.volatility_head = nn.Linear(config.d_model, 1)
        self.regime_head = nn.Linear(config.d_model, config.regime_classes)
        self.latent_head = nn.Linear(config.d_model, config.latent_dim)
        horizon_count = len(config.return_horizons)
        if config.hierarchical_direction:
            self.horizon_embedding = nn.Parameter(
                torch.zeros(1, horizon_count, config.d_model)
            )
            self.horizon_adapters = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(config.d_model),
                        nn.Linear(config.d_model, config.horizon_adapter_dim),
                        nn.GELU(),
                        nn.Dropout(config.dropout),
                        nn.Linear(config.horizon_adapter_dim, config.d_model),
                    )
                    for _ in config.return_horizons
                ]
            )
            self.horizon_return_head = nn.Linear(config.d_model, 1)
            self.movement_head = nn.Linear(config.d_model, 3)
            self.side_head = nn.Linear(config.d_model, 2)
            self.horizon_quantile_head = nn.Linear(
                config.d_model,
                len(config.quantile_levels),
            )
            self.horizon_edge_head = nn.Linear(config.d_model, 2)
            self.horizon_excursion_head = nn.Sequential(
                nn.Linear(config.d_model, 2),
                nn.Softplus(),
            )
            self.horizon_tradeability_head = nn.Linear(config.d_model, 1)
            nn.init.normal_(self.horizon_embedding, std=0.02)
        else:
            self.direction_head = nn.Linear(config.d_model, horizon_count * 3)
            self.quantile_head = nn.Linear(
                config.d_model,
                horizon_count * len(config.quantile_levels),
            )
            self.edge_head = nn.Linear(config.d_model, horizon_count * 2)
            self.excursion_head = nn.Sequential(
                nn.Linear(config.d_model, horizon_count * 2),
                nn.Softplus(),
            )
            self.tradeability_head = nn.Linear(config.d_model, horizon_count)
        self.volatility_regime_head = nn.Linear(
            config.d_model,
            config.volatility_regime_classes,
        )

        nn.init.normal_(self.branch_embedding, std=0.02)
        nn.init.normal_(self.fusion_query, std=0.02)

    def _horizon_contexts(self, context: torch.Tensor) -> torch.Tensor:
        """為每個預測週期建立獨立殘差轉接層，降低多任務梯度衝突。"""
        contexts = []
        for index, adapter in enumerate(self.horizon_adapters):
            base = context + self.horizon_embedding[:, index]
            contexts.append(base + adapter(base))
        return torch.stack(contexts, dim=1)

    def _encode_v3(
        self,
        features: torch.Tensor,
        feature_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        branch_contexts: list[torch.Tensor] = []
        branch_availability: list[torch.Tensor] = []
        for branch_index, branch in enumerate(self.feature_branches):
            encoded, available = branch(features, feature_mask)
            encoded = encoded + self.branch_embedding[:, branch_index : branch_index + 1]
            for encoder_layer in self.v3_encoder:
                encoded = encoder_layer(encoded)
            scores = self.branch_pooling_score(encoded).squeeze(-1)
            weights = torch.softmax(scores, dim=1)
            branch_contexts.append(torch.sum(encoded * weights.unsqueeze(-1), dim=1))
            branch_availability.append(available)

        contexts = torch.stack(branch_contexts, dim=1)
        availability = torch.stack(branch_availability, dim=1)
        all_missing = ~availability.any(dim=1)
        if all_missing.any():
            availability = availability.clone()
            availability[all_missing, 1] = True
        query = self.fusion_query.expand(features.shape[0], -1, -1)
        fused, attention = self.fusion_attention(
            query,
            contexts,
            contexts,
            key_padding_mask=~availability,
            need_weights=True,
            average_attn_weights=True,
        )
        context = self.fusion_norm(query + fused).squeeze(1)
        display_attention = attention.squeeze(1) * availability.to(attention.dtype)
        display_attention = display_attention / display_attention.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-8)
        return context, display_attention

    def forward(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        feature_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """features 形狀為 [批次, 序列長度, 特徵數]。"""
        if features.ndim != 3:
            raise ValueError("features 必須是三維張量")
        if features.shape[1] > self.config.sequence_length:
            raise ValueError("輸入序列長度超過模型設定")
        if features.shape[2] != self.config.input_features:
            raise ValueError("輸入特徵數與模型設定不一致")

        if feature_mask is None:
            feature_mask = torch.isfinite(features).to(dtype=features.dtype)
        else:
            feature_mask = feature_mask.to(dtype=features.dtype)
        features = torch.nan_to_num(features)
        context, timeframe_attention = self._encode_v3(features, feature_mask)

        context = self.output_norm(context)
        regime_logits = self.regime_head(context)
        regime_probability = torch.softmax(regime_logits, dim=-1)
        result = {
            "volatility": self.volatility_head(context),
            "regime_logits": regime_logits,
            "regime_probability": regime_probability,
            "uncertainty": 1.0 - regime_probability.max(dim=-1).values,
            "latent": self.latent_head(context),
        }
        batch_size = features.shape[0]
        horizon_count = len(self.config.return_horizons)
        if self.config.hierarchical_direction:
            horizon_contexts = self._horizon_contexts(context)
            movement_logits = self.movement_head(horizon_contexts)
            movement_probability = torch.softmax(movement_logits, dim=-1)
            side_logits = self.side_head(horizon_contexts)
            side_probability = torch.softmax(side_logits, dim=-1)
            tradeability_logits = self.horizon_tradeability_head(
                horizon_contexts
            ).squeeze(-1)
            tradeability_probability = torch.sigmoid(tradeability_logits)
            direction_probability = torch.stack(
                (
                    tradeability_probability * side_probability[..., 0],
                    1.0 - tradeability_probability,
                    tradeability_probability * side_probability[..., 1],
                ),
                dim=-1,
            )
            direction_logits = torch.log(direction_probability.clamp_min(1e-8))
            result["future_returns"] = self.horizon_return_head(
                horizon_contexts
            ).squeeze(-1)
            quantile_returns = self.horizon_quantile_head(horizon_contexts)
            edge_returns = self.horizon_edge_head(horizon_contexts)
            excursions = self.horizon_excursion_head(horizon_contexts)
            result.update(
                {
                    "movement_logits": movement_logits,
                    "movement_probability": movement_probability,
                    "side_logits": side_logits,
                    "side_probability": side_probability,
                }
            )
        else:
            result["future_returns"] = self.return_head(context)
            direction_logits = self.direction_head(context).reshape(
                batch_size,
                horizon_count,
                3,
            )
            direction_probability = torch.softmax(direction_logits, dim=-1)
            quantile_returns = self.quantile_head(context).reshape(
                batch_size,
                horizon_count,
                len(self.config.quantile_levels),
            )
            edge_returns = self.edge_head(context).reshape(
                batch_size,
                horizon_count,
                2,
            )
            excursions = self.excursion_head(context).reshape(
                batch_size,
                horizon_count,
                2,
            )
            tradeability_logits = self.tradeability_head(context)
            tradeability_probability = torch.sigmoid(tradeability_logits)
        direction_entropy = -(
                direction_probability
                * torch.log(direction_probability.clamp_min(1e-8))
        ).sum(dim=-1).mean(dim=-1) / torch.log(
                torch.tensor(3.0, device=features.device)
        )
        regime_entropy = -(
                regime_probability
                * torch.log(regime_probability.clamp_min(1e-8))
        ).sum(dim=-1) / torch.log(
                torch.tensor(
                    float(self.config.regime_classes),
                    device=features.device,
                )
        )
        result.update(
            {
                "direction_logits": direction_logits,
                "direction_probability": direction_probability,
                "quantile_returns": quantile_returns,
                "uncertainty": (direction_entropy + regime_entropy) / 2,
            }
        )
        volatility_regime_logits = self.volatility_regime_head(context)
        volatility_regime_probability = torch.softmax(
                volatility_regime_logits,
                dim=-1,
        )
        volatility_entropy = -(
                volatility_regime_probability
                * torch.log(volatility_regime_probability.clamp_min(1e-8))
        ).sum(dim=-1) / torch.log(
                torch.tensor(
                    float(self.config.volatility_regime_classes),
                    device=features.device,
                )
        )
        trade_entropy = -(
                tradeability_probability
                * torch.log(tradeability_probability.clamp_min(1e-8))
                + (1 - tradeability_probability)
                * torch.log((1 - tradeability_probability).clamp_min(1e-8))
        ).mean(dim=-1) / torch.log(torch.tensor(2.0, device=features.device))
        result.update(
            {
                "edge_returns": edge_returns,
                "excursions": excursions,
                "tradeability_logits": tradeability_logits,
                "tradeability_probability": tradeability_probability,
                "volatility_regime_logits": volatility_regime_logits,
                "volatility_regime_probability": volatility_regime_probability,
                "timeframe_attention": timeframe_attention,
                "uncertainty": (
                    result["uncertainty"] + volatility_entropy + trade_entropy
                )
                / 3,
            }
        )
        return result

    @property
    def trainable_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
