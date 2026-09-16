"""金融時序 Transformer 的訓練、驗證與模型保存流程。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import sys
from time import monotonic
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from ai_quant_trading.performance import configure_torch_cpu_threads

from ai_quant_trading.transformer.config import (
    TemporalTransformerConfig,
    TransformerTrainingConfig,
)
from ai_quant_trading.transformer.dataset import (
    PreparedTransformerData,
    TransformerScaler,
    prepare_transformer_datasets,
)
from ai_quant_trading.transformer.model import MarketTemporalTransformer
from ai_quant_trading.transformer.data_contract import build_transformer_feature_contract


ProgressCallback = Callable[[dict[str, object]], None]


@dataclass(frozen=True, slots=True)
class TransformerBackendStatus:
    """目前 PyTorch 與 GPU 的可用狀態。"""

    available: bool
    torch_version: str
    cuda_available: bool
    cuda_version: str
    device_name: str
    total_memory_gb: float
    message: str


@dataclass(frozen=True, slots=True)
class TransformerTrainingResult:
    """完成一次訓練後的重要路徑與樣本外指標。"""

    run_dir: Path
    model_path: Path
    history_csv: Path
    summary_json: Path
    resolved_device: str
    duration_seconds: float
    best_epoch: int
    metrics: dict[str, float]


def transformer_backend_status() -> TransformerBackendStatus:
    """回傳介面可直接顯示的 PyTorch/CUDA 資訊。"""
    cuda_available = bool(torch.cuda.is_available())
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    total_memory = (
        torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if cuda_available
        else 0.0
    )
    return TransformerBackendStatus(
        available=True,
        torch_version=str(torch.__version__),
        cuda_available=cuda_available,
        cuda_version=str(torch.version.cuda or "-"),
        device_name=device_name,
        total_memory_gb=float(total_memory),
        message="CUDA 可用" if cuda_available else "目前將使用 CPU",
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _json_dump(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _resolved_device(requested: str) -> torch.device:
    normalized = requested.strip().lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定 CUDA 訓練，但 PyTorch 目前偵測不到可用 GPU")
    if normalized not in {"cpu", "cuda"}:
        raise ValueError("Transformer 裝置必須是 auto、cpu 或 cuda")
    return torch.device(normalized)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _worker_count(configured: int, cpu_threads: int) -> int:
    # PyInstaller 桌面版中的 DataLoader 子行程容易重複啟動 App，因此固定在主行程載入。
    if getattr(sys, "frozen", False):
        return 0
    # 至少保留一個執行緒給模型本身，避免 worker 與矩陣運算互相搶滿 CPU。
    return min(configured, max(cpu_threads - 1, 0))


def _loader(
    dataset: object,
    config: TransformerTrainingConfig,
    *,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    worker_count = _worker_count(config.num_workers, config.cpu_threads)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=worker_count,
        pin_memory=device.type == "cuda",
        persistent_workers=worker_count > 0,
    )


def _task_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: TransformerTrainingConfig,
    model_config: TemporalTransformerConfig,
    *,
    direction_class_weights: torch.Tensor | None = None,
    side_class_weights: torch.Tensor | None = None,
    tradeability_pos_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return_loss = nn.functional.smooth_l1_loss(
        output["future_returns"],
        batch["future_returns"],
    )
    volatility_loss = nn.functional.smooth_l1_loss(
        output["volatility"],
        batch["volatility"],
    )
    regime_loss = nn.functional.cross_entropy(
        output["regime_logits"],
        batch["regime"],
        label_smoothing=config.label_smoothing,
    )
    zero = return_loss.new_zeros(())
    direction_loss = zero
    side_loss = zero
    quantile_loss = zero
    crossing_loss = zero
    edge_loss = zero
    excursion_loss = zero
    tradeability_loss = zero
    volatility_regime_loss = zero
    classification_logits = output.get("movement_logits", output.get("direction_logits"))
    if classification_logits is not None:
        logits = classification_logits
        targets = (
            batch["movement_directions"]
            if "movement_logits" in output
            else batch["future_directions"]
        )
        element_loss = nn.functional.cross_entropy(
            logits.reshape(-1, 3),
            targets.reshape(-1),
            label_smoothing=config.label_smoothing,
            reduction="none",
        ).reshape_as(targets)
        true_probability = torch.softmax(logits, dim=-1).gather(
            -1,
            targets.unsqueeze(-1),
        ).squeeze(-1)
        focal_weight = (1.0 - true_probability).pow(
            config.direction_focal_gamma
        )
        sample_weight = torch.ones_like(element_loss)
        if direction_class_weights is not None:
            expanded_weights = direction_class_weights.unsqueeze(0).expand(
                targets.shape[0],
                -1,
                -1,
            )
            sample_weight = expanded_weights.gather(
                -1,
                targets.unsqueeze(-1),
            ).squeeze(-1)
        combined_weight = focal_weight * sample_weight
        direction_loss = (element_loss * combined_weight).sum() / combined_weight.sum().clamp_min(
            1e-8
        )
    if "side_logits" in output:
        side_logits = output["side_logits"]
        side_targets = batch["future_sides"]
        side_element_loss = nn.functional.cross_entropy(
            side_logits.reshape(-1, 2),
            side_targets.reshape(-1),
            label_smoothing=config.label_smoothing,
            reduction="none",
        ).reshape_as(side_targets)
        side_probability = torch.softmax(side_logits, dim=-1).gather(
            -1,
            side_targets.unsqueeze(-1),
        ).squeeze(-1)
        side_weight = (1.0 - side_probability).pow(config.direction_focal_gamma)
        if side_class_weights is not None:
            expanded_side_weights = side_class_weights.unsqueeze(0).expand(
                side_targets.shape[0],
                -1,
                -1,
            )
            side_weight = side_weight * expanded_side_weights.gather(
                -1,
                side_targets.unsqueeze(-1),
            ).squeeze(-1)
        # 多空頭只在扣除成本後具交易價值的樣本上學習，Hold 由獨立頭處理。
        side_weight = side_weight * batch["tradeability"]
        side_loss = (side_element_loss * side_weight).sum() / side_weight.sum().clamp_min(
            1e-8
        )
    if "quantile_returns" in output:
        levels = torch.tensor(
            model_config.quantile_levels,
            dtype=output["quantile_returns"].dtype,
            device=output["quantile_returns"].device,
        ).view(1, 1, -1)
        error = batch["future_returns"].unsqueeze(-1) - output["quantile_returns"]
        quantile_loss = torch.maximum((levels - 1) * error, levels * error).mean()
        crossing_loss = nn.functional.relu(
            output["quantile_returns"][..., :-1]
            - output["quantile_returns"][..., 1:]
        ).mean()
    if "edge_returns" in output and "edge_returns" in batch:
        edge_loss = nn.functional.smooth_l1_loss(
            output["edge_returns"],
            batch["edge_returns"],
        )
    if "excursions" in output and "excursions" in batch:
        excursion_loss = nn.functional.smooth_l1_loss(
            output["excursions"],
            batch["excursions"],
        )
    if "tradeability_logits" in output and "tradeability" in batch:
        tradeability_loss = nn.functional.binary_cross_entropy_with_logits(
            output["tradeability_logits"],
            batch["tradeability"],
            pos_weight=tradeability_pos_weights,
        )
    if "volatility_regime_logits" in output and "volatility_regime" in batch:
        volatility_regime_loss = nn.functional.cross_entropy(
            output["volatility_regime_logits"],
            batch["volatility_regime"],
            label_smoothing=config.label_smoothing,
        )
    total = (
        config.return_loss_weight * return_loss
        + config.volatility_loss_weight * volatility_loss
        + config.regime_loss_weight * regime_loss
        + config.direction_loss_weight * direction_loss
        + config.side_loss_weight * side_loss
        + config.quantile_loss_weight * (quantile_loss + 0.05 * crossing_loss)
        + config.edge_loss_weight * edge_loss
        + config.excursion_loss_weight * excursion_loss
        + config.tradeability_loss_weight * tradeability_loss
        + config.volatility_regime_loss_weight * volatility_regime_loss
    )
    return total, {
        "return_loss": return_loss,
        "volatility_loss": volatility_loss,
        "regime_loss": regime_loss,
        "direction_loss": direction_loss,
        "side_loss": side_loss,
        "quantile_loss": quantile_loss,
        "quantile_crossing_loss": crossing_loss,
        "edge_loss": edge_loss,
        "excursion_loss": excursion_loss,
        "tradeability_loss": tradeability_loss,
        "volatility_regime_loss": volatility_regime_loss,
    }


def _training_class_balance(
    data: PreparedTransformerData,
    config: TransformerTrainingConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """只使用 Train endpoints 建立每個預測週期的溫和類別權重。"""
    horizon_count = len(data.resolved_model_config.return_horizons)
    direction_counts = np.zeros((horizon_count, 3), dtype=np.float64)
    side_counts = np.zeros((horizon_count, 2), dtype=np.float64)
    tradeability_positive = np.zeros(horizon_count, dtype=np.float64)
    tradeability_total = np.zeros(horizon_count, dtype=np.float64)
    for series in data.train.series:
        endpoints = series.train_endpoints
        directions = (
            series.movement_directions[endpoints]
            if data.resolved_model_config.hierarchical_direction
            else series.future_directions[endpoints]
        )
        sides = series.future_sides[endpoints]
        trades = series.tradeability[endpoints]
        for horizon_index in range(horizon_count):
            direction_counts[horizon_index] += np.bincount(
                directions[:, horizon_index],
                minlength=3,
            )
            trade_mask = trades[:, horizon_index] > 0.5
            side_counts[horizon_index] += np.bincount(
                sides[trade_mask, horizon_index],
                minlength=2,
            )
        tradeability_positive += np.sum(trades, axis=0)
        tradeability_total += trades.shape[0]

    frequencies = direction_counts / np.maximum(
        direction_counts.sum(axis=1, keepdims=True),
        1.0,
    )
    direction_weights = np.power(
        np.maximum(frequencies, 1e-8),
        -config.direction_class_balance_power,
    )
    direction_weights /= np.maximum(
        direction_weights.mean(axis=1, keepdims=True),
        1e-8,
    )
    direction_weights = np.clip(direction_weights, 0.25, 4.0)
    direction_weights /= direction_weights.mean(axis=1, keepdims=True)

    side_frequencies = side_counts / np.maximum(
        side_counts.sum(axis=1, keepdims=True),
        1.0,
    )
    side_weights = np.power(
        np.maximum(side_frequencies, 1e-8),
        -config.side_class_balance_power,
    )
    side_weights = np.clip(side_weights, 0.5, 2.0)
    side_weights /= np.maximum(side_weights.mean(axis=1, keepdims=True), 1e-8)

    negative = np.maximum(tradeability_total - tradeability_positive, 0.0)
    positive_ratio = negative / np.maximum(tradeability_positive, 1.0)
    tradeability_weights = np.power(
        positive_ratio,
        config.tradeability_class_balance_power,
    )
    tradeability_weights = np.clip(tradeability_weights, 0.25, 4.0)
    details = {
        "direction_counts": direction_counts.astype(int).tolist(),
        "direction_frequencies": frequencies.tolist(),
        "direction_class_weights": direction_weights.tolist(),
        "direction_label": (
            "volatility_adaptive_movement"
            if data.resolved_model_config.hierarchical_direction
            else "cost_aware_action"
        ),
        "side_counts_on_tradeable_samples": side_counts.astype(int).tolist(),
        "side_frequencies_on_tradeable_samples": side_frequencies.tolist(),
        "side_class_weights": side_weights.tolist(),
        "tradeability_positive": tradeability_positive.astype(int).tolist(),
        "tradeability_total": tradeability_total.astype(int).tolist(),
        "tradeability_pos_weights": tradeability_weights.tolist(),
    }
    return direction_weights, side_weights, tradeability_weights, details


def _move_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        for key, value in batch.items()
    }


def _temperature_scale(
    output: dict[str, torch.Tensor],
    calibration: dict[str, float | str] | None,
) -> dict[str, torch.Tensor]:
    """套用驗證集溫度校準，保留原始 logits 供 Loss 與稽核使用。"""
    if not calibration:
        return output
    result = dict(output)
    probability_specs = [
        ("regime_logits", "regime_probability", "regime_temperature"),
        (
            "volatility_regime_logits",
            "volatility_regime_probability",
            "volatility_regime_temperature",
        ),
    ]
    if "movement_logits" in result:
        probability_specs.extend(
            [
                ("movement_logits", "movement_probability", "movement_temperature"),
                ("side_logits", "side_probability", "side_temperature"),
            ]
        )
    else:
        probability_specs.append(
            ("direction_logits", "direction_probability", "direction_temperature")
        )
    for logits_name, probability_name, temperature_name in probability_specs:
        if logits_name in result:
            temperature = max(float(calibration.get(temperature_name, 1.0)), 0.05)
            result[probability_name] = torch.softmax(
                result[logits_name] / temperature,
                dim=-1,
            )
    if "tradeability_logits" in result:
        temperature = max(
            float(calibration.get("tradeability_temperature", 1.0)),
            0.05,
        )
        result["tradeability_probability"] = torch.sigmoid(
            result["tradeability_logits"] / temperature
        )
    if "side_probability" in result and "tradeability_probability" in result:
        side = result["side_probability"]
        trade = result["tradeability_probability"]
        result["direction_probability"] = torch.stack(
            (trade * side[..., 0], 1.0 - trade, trade * side[..., 1]),
            dim=-1,
        )

    uncertainty_parts: list[torch.Tensor] = []
    for probability_name in (
        "regime_probability",
        "direction_probability",
        "movement_probability",
        "volatility_regime_probability",
    ):
        if probability_name not in result:
            continue
        probabilities = result[probability_name].clamp_min(1e-8)
        entropy = -(probabilities * probabilities.log()).sum(dim=-1)
        entropy = entropy / math.log(probabilities.shape[-1])
        if entropy.ndim > 1:
            entropy = entropy.mean(dim=-1)
        uncertainty_parts.append(entropy)
    if "tradeability_probability" in result:
        probabilities = result["tradeability_probability"].clamp(1e-8, 1 - 1e-8)
        entropy = -(
            probabilities * probabilities.log()
            + (1 - probabilities) * (1 - probabilities).log()
        ).mean(dim=-1) / math.log(2)
        uncertainty_parts.append(entropy)
    if uncertainty_parts:
        result["uncertainty"] = torch.stack(uncertainty_parts).mean(dim=0)
    return result


def _best_temperature(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    binary: bool = False,
) -> float:
    """以固定網格尋找最小驗證負對數概似，流程簡單且可重現。"""
    best_temperature = 1.0
    best_loss = math.inf
    for temperature in torch.linspace(0.50, 3.00, 51):
        if binary:
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits / temperature,
                targets,
            )
        else:
            loss = nn.functional.cross_entropy(
                logits / temperature,
                targets,
            )
        value = float(loss.item())
        if value < best_loss:
            best_loss = value
            best_temperature = float(temperature.item())
    return best_temperature


def _fit_probability_calibration(
    model: MarketTemporalTransformer,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float | str]:
    """只使用驗證集估計分類頭溫度，不接觸測試集。"""
    collected: dict[str, list[torch.Tensor]] = {
        "regime_logits": [],
        "regime": [],
        "direction_logits": [],
        "future_directions": [],
        "movement_logits": [],
        "movement_directions": [],
        "side_logits": [],
        "future_sides": [],
        "volatility_regime_logits": [],
        "volatility_regime": [],
        "tradeability_logits": [],
        "tradeability": [],
    }
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            moved = _move_batch(batch, device)
            output = model(
                moved["features"],
                feature_mask=moved.get("feature_mask"),
            )
            for name in (
                "regime_logits",
                "direction_logits",
                "movement_logits",
                "side_logits",
                "volatility_regime_logits",
                "tradeability_logits",
            ):
                if name in output:
                    collected[name].append(output[name].detach().float().cpu())
            for name in (
                "regime",
                "future_directions",
                "movement_directions",
                "future_sides",
                "volatility_regime",
                "tradeability",
            ):
                if name in moved:
                    collected[name].append(moved[name].detach().float().cpu())

    calibration = {"method": "temperature_scaling_validation_grid_v1"}
    if collected["regime_logits"]:
        calibration["regime_temperature"] = _best_temperature(
            torch.cat(collected["regime_logits"]),
            torch.cat(collected["regime"]).long(),
        )
    if collected["movement_logits"]:
        calibration["movement_temperature"] = _best_temperature(
            torch.cat(collected["movement_logits"]).reshape(-1, 3),
            torch.cat(collected["movement_directions"]).long().reshape(-1),
        )
        tradeable = torch.cat(collected["tradeability"]).reshape(-1) > 0.5
        side_logits = torch.cat(collected["side_logits"]).reshape(-1, 2)
        side_targets = torch.cat(collected["future_sides"]).long().reshape(-1)
        if tradeable.any():
            calibration["side_temperature"] = _best_temperature(
                side_logits[tradeable],
                side_targets[tradeable],
            )
    elif collected["direction_logits"]:
        calibration["direction_temperature"] = _best_temperature(
            torch.cat(collected["direction_logits"]).reshape(-1, 3),
            torch.cat(collected["future_directions"]).long().reshape(-1),
        )
    if collected["volatility_regime_logits"]:
        calibration["volatility_regime_temperature"] = _best_temperature(
            torch.cat(collected["volatility_regime_logits"]),
            torch.cat(collected["volatility_regime"]).long(),
        )
    if collected["tradeability_logits"]:
        calibration["tradeability_temperature"] = _best_temperature(
            torch.cat(collected["tradeability_logits"]).reshape(-1),
            torch.cat(collected["tradeability"]).reshape(-1),
            binary=True,
        )
    return calibration


def _confusion_scores(confusion: np.ndarray) -> dict[str, float]:
    """由 confusion matrix 計算 accuracy、類別平衡與多數類基準。"""
    support = confusion.sum(axis=1)
    predicted_support = confusion.sum(axis=0)
    total = max(int(confusion.sum()), 1)
    recall = np.divide(
        np.diag(confusion),
        support,
        out=np.zeros(len(support), dtype=np.float64),
        where=support > 0,
    )
    precision = np.divide(
        np.diag(confusion),
        predicted_support,
        out=np.zeros(len(support), dtype=np.float64),
        where=predicted_support > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(len(support), dtype=np.float64),
        where=(precision + recall) > 0,
    )
    accuracy = float(np.trace(confusion) / total)
    majority = float(support.max(initial=0) / total)
    populated = support > 0
    return {
        "accuracy": accuracy,
        "majority_baseline": majority,
        "accuracy_lift": accuracy - majority,
        "balanced_accuracy": (
            float(recall[populated].mean()) if populated.any() else 0.0
        ),
        "macro_f1": float(f1[populated].mean()) if populated.any() else 0.0,
    }


def _evaluate(
    model: MarketTemporalTransformer,
    loader: DataLoader,
    config: TransformerTrainingConfig,
    device: torch.device,
    *,
    scaler: TransformerScaler | None = None,
    horizons: tuple[int, ...] = (),
    collect_predictions: bool = False,
    calibration: dict[str, float | str] | None = None,
    direction_class_weights: torch.Tensor | None = None,
    side_class_weights: torch.Tensor | None = None,
    tradeability_pos_weights: torch.Tensor | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    correct_regimes = 0
    correct_directions = 0
    direction_items = 0
    horizon_count = len(model.config.return_horizons)
    direction_confusion = np.zeros((horizon_count, 3, 3), dtype=np.int64)
    movement_confusion = np.zeros((horizon_count, 3, 3), dtype=np.int64)
    side_confusion = np.zeros((horizon_count, 2, 2), dtype=np.int64)
    prediction_rows: list[dict[str, float | int]] = []
    with torch.inference_mode():
        for batch in loader:
            moved = _move_batch(batch, device)
            raw_output = model(
                moved["features"],
                feature_mask=moved.get("feature_mask"),
            )
            loss, losses = _task_loss(
                raw_output,
                moved,
                config,
                model.config,
                direction_class_weights=direction_class_weights,
                side_class_weights=side_class_weights,
                tradeability_pos_weights=tradeability_pos_weights,
            )
            output = _temperature_scale(raw_output, calibration)
            size = int(moved["features"].shape[0])
            total_items += size
            total_loss += float(loss.item()) * size
            predicted_regime = output["regime_logits"].argmax(dim=-1)
            correct_regimes += int(
                (predicted_regime == moved["regime"]).sum().item()
            )
            if "direction_probability" in output:
                predicted_direction = output["direction_probability"].argmax(dim=-1)
                correct_directions += int(
                    (predicted_direction == moved["future_directions"]).sum().item()
                )
                direction_items += int(moved["future_directions"].numel())
                actual_direction = moved["future_directions"]
                for horizon_index in range(horizon_count):
                    encoded = (
                        actual_direction[:, horizon_index] * 3
                        + predicted_direction[:, horizon_index]
                    )
                    direction_confusion[horizon_index] += (
                        torch.bincount(encoded, minlength=9)
                        .reshape(3, 3)
                        .detach()
                        .cpu()
                        .numpy()
                    )
            if "movement_probability" in output:
                predicted_movement = output["movement_probability"].argmax(dim=-1)
                actual_movement = moved["movement_directions"]
                predicted_side = output["side_probability"].argmax(dim=-1)
                actual_side = moved["future_sides"]
                tradeable = moved["tradeability"] > 0.5
                for horizon_index in range(horizon_count):
                    movement_encoded = (
                        actual_movement[:, horizon_index] * 3
                        + predicted_movement[:, horizon_index]
                    )
                    movement_confusion[horizon_index] += (
                        torch.bincount(movement_encoded, minlength=9)
                        .reshape(3, 3)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    mask = tradeable[:, horizon_index]
                    if mask.any():
                        side_encoded = (
                            actual_side[mask, horizon_index] * 2
                            + predicted_side[mask, horizon_index]
                        )
                        side_confusion[horizon_index] += (
                            torch.bincount(side_encoded, minlength=4)
                            .reshape(2, 2)
                            .detach()
                            .cpu()
                            .numpy()
                        )
            if collect_predictions:
                predicted_returns_z = (
                    output["future_returns"].detach().cpu().numpy()
                )
                actual_returns_z = moved["future_returns"].detach().cpu().numpy()
                predicted_volatility_z = (
                    output["volatility"].detach().cpu().numpy()
                )
                actual_volatility_z = moved["volatility"].detach().cpu().numpy()
                if scaler is not None:
                    return_means = np.asarray(scaler.return_means)
                    return_scales = np.asarray(scaler.return_scales)
                    predicted_returns = (
                        predicted_returns_z * return_scales + return_means
                    )
                    actual_returns = actual_returns_z * return_scales + return_means
                    predicted_volatility = (
                        predicted_volatility_z * scaler.volatility_scale
                        + scaler.volatility_mean
                    )
                    actual_volatility = (
                        actual_volatility_z * scaler.volatility_scale
                        + scaler.volatility_mean
                    )
                else:
                    predicted_returns = predicted_returns_z
                    actual_returns = actual_returns_z
                    predicted_volatility = predicted_volatility_z
                    actual_volatility = actual_volatility_z
                regimes = moved["regime"].detach().cpu().numpy()
                predicted_regimes = predicted_regime.detach().cpu().numpy()
                uncertainty = output["uncertainty"].detach().cpu().numpy()
                direction_probabilities = (
                    output["direction_probability"].detach().cpu().numpy()
                    if "direction_probability" in output
                    else None
                )
                movement_probabilities = (
                    output["movement_probability"].detach().cpu().numpy()
                    if "movement_probability" in output
                    else None
                )
                side_probabilities = (
                    output["side_probability"].detach().cpu().numpy()
                    if "side_probability" in output
                    else None
                )
                actual_movements = moved["movement_directions"].detach().cpu().numpy()
                actual_sides = moved["future_sides"].detach().cpu().numpy()
                quantile_returns = (
                    output["quantile_returns"].detach().cpu().numpy()
                    if "quantile_returns" in output
                    else None
                )
                if quantile_returns is not None and scaler is not None:
                    quantile_returns = (
                        quantile_returns
                        * np.asarray(scaler.return_scales)[None, :, None]
                        + np.asarray(scaler.return_means)[None, :, None]
                    )
                edge_returns = (
                    output["edge_returns"].detach().cpu().numpy()
                    if "edge_returns" in output
                    else None
                )
                actual_edges = (
                    moved["edge_returns"].detach().cpu().numpy()
                    if "edge_returns" in moved
                    else None
                )
                excursions = (
                    output["excursions"].detach().cpu().numpy()
                    if "excursions" in output
                    else None
                )
                actual_excursions = (
                    moved["excursions"].detach().cpu().numpy()
                    if "excursions" in moved
                    else None
                )
                if edge_returns is not None and scaler is not None:
                    edge_means = np.asarray(scaler.edge_means).reshape(-1, 2)
                    edge_scales = np.asarray(scaler.edge_scales).reshape(-1, 2)
                    edge_returns = edge_returns * edge_scales + edge_means
                    actual_edges = actual_edges * edge_scales + edge_means
                if excursions is not None and scaler is not None:
                    excursion_scales = np.asarray(scaler.excursion_scales).reshape(-1, 2)
                    excursions = excursions * excursion_scales
                    actual_excursions = actual_excursions * excursion_scales
                tradeability_probability = (
                    output["tradeability_probability"].detach().cpu().numpy()
                    if "tradeability_probability" in output
                    else None
                )
                actual_tradeability = (
                    moved["tradeability"].detach().cpu().numpy()
                    if "tradeability" in moved
                    else None
                )
                volatility_regime_probability = (
                    output["volatility_regime_probability"].detach().cpu().numpy()
                    if "volatility_regime_probability" in output
                    else None
                )
                actual_volatility_regime = (
                    moved["volatility_regime"].detach().cpu().numpy()
                    if "volatility_regime" in moved
                    else None
                )
                timeframe_attention = (
                    output["timeframe_attention"].detach().cpu().numpy()
                    if "timeframe_attention" in output
                    else None
                )
                for index in range(size):
                    row: dict[str, float | int] = {
                        "actual_regime": int(regimes[index]),
                        "predicted_regime": int(predicted_regimes[index]),
                        "predicted_volatility": float(
                            predicted_volatility[index, 0]
                        ),
                        "actual_volatility": float(actual_volatility[index, 0]),
                        "uncertainty": float(uncertainty[index]),
                    }
                    if volatility_regime_probability is not None:
                        row["actual_volatility_regime"] = int(
                            actual_volatility_regime[index]
                        )
                        for class_index, name in enumerate(("low", "normal", "high")):
                            row[f"volatility_regime_{name}_probability"] = float(
                                volatility_regime_probability[index, class_index]
                            )
                    if timeframe_attention is not None:
                        for group_index, name in enumerate(model.config.feature_group_names):
                            row[f"timeframe_{name}_attention"] = float(
                                timeframe_attention[index, group_index]
                            )
                    for horizon_index in range(predicted_returns.shape[1]):
                        horizon = (
                            horizons[horizon_index]
                            if horizon_index < len(horizons)
                            else horizon_index
                        )
                        row[f"predicted_return_{horizon}"] = float(
                            predicted_returns[index, horizon_index]
                        )
                        row[f"actual_return_{horizon}"] = float(
                            actual_returns[index, horizon_index]
                        )
                        if direction_probabilities is not None:
                            row[f"down_probability_{horizon}"] = float(
                                direction_probabilities[index, horizon_index, 0]
                            )
                            row[f"neutral_probability_{horizon}"] = float(
                                direction_probabilities[index, horizon_index, 1]
                            )
                            row[f"up_probability_{horizon}"] = float(
                                direction_probabilities[index, horizon_index, 2]
                            )
                        if movement_probabilities is not None:
                            row[f"actual_movement_{horizon}"] = int(
                                actual_movements[index, horizon_index]
                            )
                            for class_index, name in enumerate(
                                ("down", "neutral", "up")
                            ):
                                row[f"movement_{name}_probability_{horizon}"] = float(
                                    movement_probabilities[
                                        index,
                                        horizon_index,
                                        class_index,
                                    ]
                                )
                            row[f"actual_side_{horizon}"] = int(
                                actual_sides[index, horizon_index]
                            )
                            row[f"side_down_probability_{horizon}"] = float(
                                side_probabilities[index, horizon_index, 0]
                            )
                            row[f"side_up_probability_{horizon}"] = float(
                                side_probabilities[index, horizon_index, 1]
                            )
                        if quantile_returns is not None:
                            for quantile_index, level in enumerate(
                                model.config.quantile_levels
                            ):
                                token = int(round(level * 100))
                                row[f"predicted_return_q{token}_{horizon}"] = float(
                                    quantile_returns[
                                        index,
                                        horizon_index,
                                        quantile_index,
                                    ]
                                )
                        if edge_returns is not None:
                            for side_index, side in enumerate(("long", "short")):
                                row[f"predicted_{side}_edge_{horizon}"] = float(
                                    edge_returns[index, horizon_index, side_index]
                                )
                                row[f"actual_{side}_edge_{horizon}"] = float(
                                    actual_edges[index, horizon_index, side_index]
                                )
                        if excursions is not None:
                            for side_index, name in enumerate(("downside", "upside")):
                                row[f"predicted_{name}_excursion_{horizon}"] = float(
                                    excursions[index, horizon_index, side_index]
                                )
                                row[f"actual_{name}_excursion_{horizon}"] = float(
                                    actual_excursions[index, horizon_index, side_index]
                                )
                        if tradeability_probability is not None:
                            row[f"tradeability_probability_{horizon}"] = float(
                                tradeability_probability[index, horizon_index]
                            )
                            row[f"actual_tradeability_{horizon}"] = float(
                                actual_tradeability[index, horizon_index]
                            )
                    prediction_rows.append(row)
    if total_items == 0:
        raise ValueError("驗證或測試 Dataset 沒有樣本")
    prediction_frame = pd.DataFrame(prediction_rows)
    metrics = {
        "loss": total_loss / total_items,
        "regime_accuracy": correct_regimes / total_items,
    }
    if direction_items:
        metrics["cost_aware_direction_accuracy"] = (
            correct_directions / direction_items
        )
        balanced_values: list[float] = []
        macro_f1_values: list[float] = []
        majority_values: list[float] = []
        accuracy_values: list[float] = []
        for horizon_index, horizon in enumerate(model.config.return_horizons):
            confusion = direction_confusion[horizon_index]
            support = confusion.sum(axis=1)
            predicted_support = confusion.sum(axis=0)
            total = max(int(confusion.sum()), 1)
            recall = np.divide(
                np.diag(confusion),
                support,
                out=np.zeros(3, dtype=np.float64),
                where=support > 0,
            )
            precision = np.divide(
                np.diag(confusion),
                predicted_support,
                out=np.zeros(3, dtype=np.float64),
                where=predicted_support > 0,
            )
            f1 = np.divide(
                2 * precision * recall,
                precision + recall,
                out=np.zeros(3, dtype=np.float64),
                where=(precision + recall) > 0,
            )
            accuracy = float(np.trace(confusion) / total)
            majority = float(support.max(initial=0) / total)
            balanced = float(recall[support > 0].mean())
            macro_f1 = float(f1[support > 0].mean())
            lift = accuracy - majority
            metrics[f"cost_aware_direction_accuracy_{horizon}"] = accuracy
            metrics[f"direction_majority_baseline_{horizon}"] = majority
            metrics[f"cost_aware_direction_balanced_accuracy_{horizon}"] = balanced
            metrics[f"cost_aware_direction_macro_f1_{horizon}"] = macro_f1
            metrics[f"direction_accuracy_lift_{horizon}"] = lift
            balanced_values.append(balanced)
            macro_f1_values.append(macro_f1)
            majority_values.append(majority)
            accuracy_values.append(accuracy)
        balanced_accuracy = float(np.mean(balanced_values))
        majority_baseline = float(np.mean(majority_values))
        direction_accuracy = float(np.mean(accuracy_values))
        metrics["cost_aware_direction_balanced_accuracy"] = balanced_accuracy
        metrics["cost_aware_direction_macro_f1"] = float(np.mean(macro_f1_values))
        metrics["direction_majority_baseline"] = majority_baseline
        metrics["direction_accuracy_lift"] = direction_accuracy - majority_baseline
        metrics["direction_skill_score"] = (
            balanced_accuracy
            + 0.25 * metrics["direction_accuracy_lift"]
            - config.checkpoint_loss_penalty * metrics["loss"]
        )
        if model.config.hierarchical_direction:
            movement_balanced: list[float] = []
            movement_lifts: list[float] = []
            side_balanced: list[float] = []
            for horizon_index, horizon in enumerate(model.config.return_horizons):
                movement_scores = _confusion_scores(
                    movement_confusion[horizon_index]
                )
                side_scores = _confusion_scores(side_confusion[horizon_index])
                for name, value in movement_scores.items():
                    metrics[f"movement_{name}_{horizon}"] = value
                for name, value in side_scores.items():
                    metrics[f"conditional_side_{name}_{horizon}"] = value
                movement_balanced.append(movement_scores["balanced_accuracy"])
                movement_lifts.append(movement_scores["accuracy_lift"])
                side_balanced.append(side_scores["balanced_accuracy"])
            metrics["movement_balanced_accuracy"] = float(
                np.mean(movement_balanced)
            )
            metrics["movement_accuracy_lift"] = float(np.mean(movement_lifts))
            metrics["conditional_side_balanced_accuracy"] = float(
                np.mean(side_balanced)
            )
            metrics["hierarchical_skill_score"] = (
                0.60 * balanced_accuracy
                + 0.20 * metrics["movement_balanced_accuracy"]
                + 0.20 * metrics["conditional_side_balanced_accuracy"]
                + 0.25 * metrics["direction_accuracy_lift"]
                - config.checkpoint_loss_penalty * metrics["loss"]
            )
    if collect_predictions and not prediction_frame.empty:
        predicted_columns = [
            column
            for column in prediction_frame
            if column.startswith("predicted_return_")
            and "_q" not in column
        ]
        actual_columns = [
            column.replace("predicted_", "actual_")
            for column in predicted_columns
        ]
        predicted = prediction_frame[predicted_columns].to_numpy()
        actual = prediction_frame[actual_columns].to_numpy()
        metrics["return_mae"] = float(np.mean(np.abs(predicted - actual)))
        metrics["direction_accuracy"] = float(
            np.mean(np.sign(predicted) == np.sign(actual))
        )
        metrics["volatility_mae"] = float(
            np.mean(
                np.abs(
                    prediction_frame["predicted_volatility"]
                    - prediction_frame["actual_volatility"]
                )
            )
        )
        edge_columns = [
            column for column in prediction_frame if column.startswith("predicted_long_edge_") or column.startswith("predicted_short_edge_")
        ]
        if edge_columns:
            actual_edge_columns = [column.replace("predicted_", "actual_") for column in edge_columns]
            metrics["cost_aware_edge_mae"] = float(
                np.mean(
                    np.abs(
                        prediction_frame[edge_columns].to_numpy()
                        - prediction_frame[actual_edge_columns].to_numpy()
                    )
                )
            )
        trade_columns = [
            column for column in prediction_frame if column.startswith("tradeability_probability_")
        ]
        if trade_columns:
            actual_trade_columns = [column.replace("tradeability_probability_", "actual_tradeability_") for column in trade_columns]
            predicted_tradeability = prediction_frame[trade_columns].to_numpy()
            actual_tradeability_values = prediction_frame[actual_trade_columns].to_numpy()
            metrics["tradeability_brier"] = float(
                np.mean((predicted_tradeability - actual_tradeability_values) ** 2)
            )
            metrics["tradeability_accuracy"] = float(
                np.mean((predicted_tradeability >= 0.5) == actual_tradeability_values)
            )
    return metrics, prediction_frame


def _checkpoint_payload(
    model: MarketTemporalTransformer,
    data: PreparedTransformerData,
    model_config: TemporalTransformerConfig,
    training_config: TransformerTrainingConfig,
    epoch: int,
    validation_loss: float,
) -> dict[str, object]:
    return {
        "schema_version": 3,
        "created_at": _utc_now(),
        "epoch": epoch,
        "validation_loss": validation_loss,
        "model_config": model_config.to_dict(),
        "training_config": training_config.to_dict(),
        "feature_columns": list(data.scaler.feature_columns),
        "feature_contract": build_transformer_feature_contract(
            data.scaler.feature_columns
        ),
        "scaler": data.scaler.to_dict(),
        "sources": [source.to_dict() for source in data.sources],
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
    }


def train_temporal_transformer(
    source_paths: Sequence[str | Path],
    model_config: TemporalTransformerConfig,
    training_config: TransformerTrainingConfig,
    output_root: str | Path,
    *,
    run_name: str = "transformer",
    progress_callback: ProgressCallback | None = None,
) -> TransformerTrainingResult:
    """執行一次完整訓練，保存最佳模型並在最後評估時間外測試集。"""
    started = monotonic()
    configure_torch_cpu_threads(training_config.cpu_threads)
    _set_seed(training_config.seed)
    device = _resolved_device(training_config.device)
    data = prepare_transformer_datasets(
        source_paths,
        model_config,
        training_config,
    )
    resolved_config = data.resolved_model_config
    (
        direction_weights_array,
        side_weights_array,
        tradeability_weights_array,
        class_balance,
    ) = (
        _training_class_balance(data, training_config)
    )
    direction_class_weights = torch.tensor(
        direction_weights_array,
        dtype=torch.float32,
        device=device,
    )
    side_class_weights = torch.tensor(
        side_weights_array,
        dtype=torch.float32,
        device=device,
    )
    tradeability_pos_weights = torch.tensor(
        tradeability_weights_array,
        dtype=torch.float32,
        device=device,
    )
    safe_name = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in run_name.strip()
    ).strip("_") or "transformer"
    run_dir = Path(output_root).resolve() / f"{_utc_stamp()}_{safe_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    history_csv = run_dir / "history.csv"
    summary_json = run_dir / "training.json"
    model_path = run_dir / "best_model.pt"
    predictions_path = run_dir / "test_predictions.csv"

    train_loader = _loader(data.train, training_config, shuffle=True, device=device)
    validation_loader = _loader(
        data.validation,
        training_config,
        shuffle=False,
        device=device,
    )
    test_loader = _loader(data.test, training_config, shuffle=False, device=device)
    model = MarketTemporalTransformer(resolved_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    total_steps = max(1, training_config.epochs * len(train_loader))
    warmup_steps = int(total_steps * training_config.warmup_ratio)

    def learning_rate_multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        learning_rate_multiplier,
    )
    amp_enabled = training_config.mixed_precision and device.type == "cuda"
    grad_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_validation = math.inf
    selected_validation_loss = math.inf
    checkpoint_maximizes = training_config.checkpoint_metric != "validation_loss"
    best_checkpoint_value = -math.inf if checkpoint_maximizes else math.inf
    best_epoch = 0
    patience = 0
    history: list[dict[str, float | int]] = []
    completed_steps = 0

    if progress_callback:
        progress_callback(
            {
                "status": "preparing",
                "progress": 0.0,
                "epoch": 0,
                "epochs": training_config.epochs,
                "device": str(device),
                "sample_counts": data.sample_counts,
                "feature_count": resolved_config.input_features,
                "parameters": model.trainable_parameters,
                "elapsed_seconds": 0.0,
            }
        )

    try:
        for epoch_index in range(training_config.epochs):
            model.train()
            epoch_loss = 0.0
            epoch_items = 0
            report_every = max(1, len(train_loader) // 25)
            for batch_index, batch in enumerate(train_loader, start=1):
                moved = _move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    output = model(
                        moved["features"],
                        feature_mask=moved.get("feature_mask"),
                    )
                    loss, losses = _task_loss(
                        output,
                        moved,
                        training_config,
                        resolved_config,
                        direction_class_weights=direction_class_weights,
                        side_class_weights=side_class_weights,
                        tradeability_pos_weights=tradeability_pos_weights,
                    )
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    training_config.gradient_clip,
                )
                previous_scale = grad_scaler.get_scale()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                # AMP 偵測到非有限梯度時會略過 optimizer.step；此時不可推進學習率。
                if not amp_enabled or grad_scaler.get_scale() >= previous_scale:
                    scheduler.step()
                size = int(moved["features"].shape[0])
                epoch_items += size
                epoch_loss += float(loss.item()) * size
                completed_steps += 1
                if progress_callback and (
                    batch_index % report_every == 0
                    or batch_index == len(train_loader)
                ):
                    elapsed = monotonic() - started
                    fraction = completed_steps / total_steps
                    eta = elapsed / fraction - elapsed if fraction > 0 else None
                    progress_callback(
                        {
                            "status": "running",
                            "progress": fraction,
                            "epoch": epoch_index + 1,
                            "epochs": training_config.epochs,
                            "batch": batch_index,
                            "batches": len(train_loader),
                            "device": str(device),
                            "elapsed_seconds": elapsed,
                            "eta_seconds": eta,
                            "sample_counts": data.sample_counts,
                            "feature_count": resolved_config.input_features,
                            "parameters": model.trainable_parameters,
                            "metrics": {
                                "train_loss": epoch_loss / epoch_items,
                                "return_loss": float(losses["return_loss"].item()),
                                "volatility_loss": float(
                                    losses["volatility_loss"].item()
                                ),
                                "regime_loss": float(
                                    losses["regime_loss"].item()
                                ),
                                "direction_loss": float(
                                    losses["direction_loss"].item()
                                ),
                                "side_loss": float(losses["side_loss"].item()),
                                "quantile_loss": float(
                                    losses["quantile_loss"].item()
                                ),
                                "edge_loss": float(losses["edge_loss"].item()),
                                "excursion_loss": float(
                                    losses["excursion_loss"].item()
                                ),
                                "tradeability_loss": float(
                                    losses["tradeability_loss"].item()
                                ),
                                "volatility_regime_loss": float(
                                    losses["volatility_regime_loss"].item()
                                ),
                                "learning_rate": optimizer.param_groups[0]["lr"],
                                "gpu_memory_gb": (
                                    torch.cuda.memory_allocated(device) / (1024**3)
                                    if device.type == "cuda"
                                    else 0.0
                                ),
                            },
                        }
                    )

            validation_metrics, _ = _evaluate(
                model,
                validation_loader,
                training_config,
                device,
                direction_class_weights=direction_class_weights,
                side_class_weights=side_class_weights,
                tradeability_pos_weights=tradeability_pos_weights,
            )
            train_loss = epoch_loss / epoch_items
            validation_loss = float(validation_metrics["loss"])
            best_validation = min(best_validation, validation_loss)
            checkpoint_value = float(
                validation_metrics[
                    "loss"
                    if training_config.checkpoint_metric == "validation_loss"
                    else training_config.checkpoint_metric
                ]
            )
            history.append(
                {
                    "epoch": epoch_index + 1,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "validation_regime_accuracy": validation_metrics[
                        "regime_accuracy"
                    ],
                    "validation_cost_aware_direction_accuracy": validation_metrics.get(
                        "cost_aware_direction_accuracy",
                        0.0,
                    ),
                    "validation_direction_balanced_accuracy": validation_metrics.get(
                        "cost_aware_direction_balanced_accuracy",
                        0.0,
                    ),
                    "validation_direction_macro_f1": validation_metrics.get(
                        "cost_aware_direction_macro_f1",
                        0.0,
                    ),
                    "validation_direction_majority_baseline": validation_metrics.get(
                        "direction_majority_baseline",
                        0.0,
                    ),
                    "validation_direction_accuracy_lift": validation_metrics.get(
                        "direction_accuracy_lift",
                        0.0,
                    ),
                    "validation_direction_skill_score": validation_metrics.get(
                        "direction_skill_score",
                        0.0,
                    ),
                    "validation_movement_balanced_accuracy": validation_metrics.get(
                        "movement_balanced_accuracy",
                        0.0,
                    ),
                    "validation_conditional_side_balanced_accuracy": validation_metrics.get(
                        "conditional_side_balanced_accuracy",
                        0.0,
                    ),
                    "validation_hierarchical_skill_score": validation_metrics.get(
                        "hierarchical_skill_score",
                        0.0,
                    ),
                    "checkpoint_value": checkpoint_value,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
            pd.DataFrame(history).to_csv(history_csv, index=False, encoding="utf-8")
            improved = (
                checkpoint_value > best_checkpoint_value
                if checkpoint_maximizes
                else checkpoint_value < best_checkpoint_value
            )
            if improved:
                best_checkpoint_value = checkpoint_value
                selected_validation_loss = validation_loss
                best_epoch = epoch_index + 1
                patience = 0
                temporary_model = model_path.with_suffix(".tmp")
                torch.save(
                    _checkpoint_payload(
                        model,
                        data,
                        resolved_config,
                        training_config,
                        best_epoch,
                        selected_validation_loss,
                    ),
                    temporary_model,
                )
                temporary_model.replace(model_path)
            else:
                patience += 1
            if progress_callback:
                elapsed = monotonic() - started
                progress_callback(
                    {
                        "status": "validating",
                        "progress": completed_steps / total_steps,
                        "epoch": epoch_index + 1,
                        "epochs": training_config.epochs,
                        "device": str(device),
                        "elapsed_seconds": elapsed,
                        "eta_seconds": (
                            elapsed / completed_steps * (total_steps - completed_steps)
                            if completed_steps
                            else None
                        ),
                        "sample_counts": data.sample_counts,
                        "feature_count": resolved_config.input_features,
                        "parameters": model.trainable_parameters,
                        "metrics": {
                            "train_loss": train_loss,
                            "validation_loss": validation_loss,
                            "validation_regime_accuracy": validation_metrics[
                                "regime_accuracy"
                            ],
                            "validation_cost_aware_direction_accuracy": (
                                validation_metrics.get(
                                    "cost_aware_direction_accuracy",
                                    0.0,
                                )
                            ),
                            "validation_direction_balanced_accuracy": validation_metrics.get(
                                "cost_aware_direction_balanced_accuracy",
                                0.0,
                            ),
                            "validation_direction_accuracy_lift": validation_metrics.get(
                                "direction_accuracy_lift",
                                0.0,
                            ),
                            "validation_direction_skill_score": validation_metrics.get(
                                "direction_skill_score",
                                0.0,
                            ),
                            "checkpoint_value": checkpoint_value,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                        },
                    }
                )
            if patience >= training_config.early_stopping_patience:
                break

        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        calibration: dict[str, float | str] = {}
        if training_config.probability_calibration:
            calibration = _fit_probability_calibration(
                model,
                validation_loader,
                device,
            )
            checkpoint["calibration"] = calibration
            temporary_model = model_path.with_suffix(".tmp")
            torch.save(checkpoint, temporary_model)
            temporary_model.replace(model_path)
        test_metrics, predictions = _evaluate(
            model,
            test_loader,
            training_config,
            device,
            scaler=data.scaler,
            horizons=resolved_config.return_horizons,
            collect_predictions=True,
            calibration=calibration,
            direction_class_weights=direction_class_weights,
            side_class_weights=side_class_weights,
            tradeability_pos_weights=tradeability_pos_weights,
        )
        predictions.to_csv(predictions_path, index=False, encoding="utf-8")
        duration = monotonic() - started
        summary = {
            "schema_version": 3,
            "status": "complete",
            "created_at": _utc_now(),
            "duration_seconds": duration,
            "resolved_device": str(device),
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation,
            "selected_validation_loss": selected_validation_loss,
            "checkpoint_metric": training_config.checkpoint_metric,
            "best_checkpoint_value": best_checkpoint_value,
            "training_class_balance": class_balance,
            "test_metrics": test_metrics,
            "sample_counts": data.sample_counts,
            "model_config": resolved_config.to_dict(),
            "training_config": training_config.to_dict(),
            "trainable_parameters": model.trainable_parameters,
            "feature_columns": list(data.scaler.feature_columns),
            "scaler": data.scaler.to_dict(),
            "calibration": calibration,
            "sources": [source.to_dict() for source in data.sources],
            "artifacts": {
                "model": model_path.name,
                "history": history_csv.name,
                "test_predictions": predictions_path.name,
            },
        }
        _json_dump(summary_json, summary)
        from ai_quant_trading.operations.integrity import build_artifact_manifest

        build_artifact_manifest(run_dir)
        if progress_callback:
            progress_callback(
                {
                    "status": "complete",
                    "progress": 1.0,
                    "epoch": len(history),
                    "epochs": training_config.epochs,
                    "device": str(device),
                    "elapsed_seconds": duration,
                    "eta_seconds": 0.0,
                    "sample_counts": data.sample_counts,
                    "feature_count": resolved_config.input_features,
                    "parameters": model.trainable_parameters,
                    "metrics": {
                        "train_loss": float(history[-1]["train_loss"]),
                        "validation_loss": best_validation,
                        "validation_regime_accuracy": float(
                            history[-1]["validation_regime_accuracy"]
                        ),
                        "validation_cost_aware_direction_accuracy": float(
                            history[-1][
                                "validation_cost_aware_direction_accuracy"
                            ]
                        ),
                        "test_loss": test_metrics["loss"],
                        "test_regime_accuracy": test_metrics["regime_accuracy"],
                        "direction_accuracy": test_metrics.get(
                            "direction_accuracy",
                            0.0,
                        ),
                        "test_direction_balanced_accuracy": test_metrics.get(
                            "cost_aware_direction_balanced_accuracy",
                            0.0,
                        ),
                        "test_direction_accuracy_lift": test_metrics.get(
                            "direction_accuracy_lift",
                            0.0,
                        ),
                    },
                    "result_dir": str(run_dir),
                }
            )
        return TransformerTrainingResult(
            run_dir=run_dir,
            model_path=model_path,
            history_csv=history_csv,
            summary_json=summary_json,
            resolved_device=str(device),
            duration_seconds=duration,
            best_epoch=best_epoch,
            metrics={key: float(value) for key, value in test_metrics.items()},
        )
    except Exception:
        _json_dump(
            summary_json,
            {
                "schema_version": 1,
                "status": "failed",
                "created_at": _utc_now(),
                "model_config": resolved_config.to_dict(),
                "training_config": training_config.to_dict(),
                "sources": [source.to_dict() for source in data.sources],
            },
        )
        raise
