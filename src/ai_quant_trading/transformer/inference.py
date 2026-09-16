"""將已訓練 Transformer 批次推論寫回市場特徵 CSV。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from typing import Callable

import numpy as np
import pandas as pd
import torch

from ai_quant_trading.data_collection.csv_storage import save_canonical_context_csv
from ai_quant_trading.transformer.config import TemporalTransformerConfig
from ai_quant_trading.transformer.features import ensure_transformer_context
from ai_quant_trading.transformer.model import MarketTemporalTransformer
from ai_quant_trading.transformer.data_contract import build_transformer_feature_contract
from ai_quant_trading.features.market_context import add_model_market_features


InferenceProgressCallback = Callable[[dict[str, object]], None]


@dataclass(frozen=True, slots=True)
class TransformerInferenceArtifact:
    """單一特徵檔完成推論後的摘要。"""

    source_path: Path
    output_path: Path
    checkpoint_path: Path
    rows: int
    predicted_rows: int
    device: str
    duration_seconds: float
    return_horizons: tuple[int, ...]


def infer_transformer_context_frame(
    checkpoint_path: str | Path,
    frame: pd.DataFrame,
    *,
    device: str = "auto",
    mixed_precision: bool = True,
    batch_size: int = 512,
    progress_callback: InferenceProgressCallback | None = None,
) -> pd.DataFrame:
    """在暫存檔中推論 DataFrame，不覆寫使用者的原始資料。"""
    checkpoint = Path(checkpoint_path).resolve()
    with TemporaryDirectory(prefix="aiquant_transformer_") as temporary_dir:
        source = Path(temporary_dir) / "features.csv"
        frame.to_csv(source, index=False, encoding="utf-8")
        artifact = apply_transformer_checkpoint(
            checkpoint,
            source,
            device=device,
            batch_size=batch_size,
            mixed_precision=mixed_precision,
            progress_callback=progress_callback,
        )
        return pd.read_csv(artifact.output_path)


def transformer_checkpoint_sequence_length(checkpoint_path: str | Path) -> int:
    """讀取 checkpoint 的序列長度，供即時交易準備足夠的暖機資料。"""
    checkpoint = Path(checkpoint_path).resolve()
    payload = _load_checkpoint(checkpoint, torch.device("cpu"))
    return _model_config(payload).sequence_length


def transformer_oos_provenance(
    checkpoint_path: str | Path,
    frame: pd.DataFrame,
) -> dict[str, object]:
    """驗證來源身分並回傳模型選擇／校準區段之後的安全起點。"""
    checkpoint = Path(checkpoint_path).resolve()
    payload = _load_checkpoint(checkpoint, torch.device("cpu"))
    training_config = payload.get("training_config")
    sources = payload.get("sources")
    if not isinstance(training_config, dict) or not isinstance(sources, list) or not sources:
        raise ValueError("Transformer checkpoint 缺少資料切分來源，禁止建立 RL 串接資料")
    train_fraction = float(training_config.get("train_fraction", 0.0))
    validation_fraction = float(training_config.get("validation_fraction", 0.0))
    calibration_fraction = train_fraction + validation_fraction
    if not 0 < train_fraction < calibration_fraction < 1:
        raise ValueError("Transformer checkpoint 的時間切分設定不合法")

    ordered = frame.copy()
    if "timestamp" not in ordered:
        raise ValueError("RL 串接資料缺少 timestamp")
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], utc=True, errors="coerce")
    ordered = ordered.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    first = ordered.iloc[0]
    identity = (
        str(first.get("exchange", "")).lower(),
        str(first.get("symbol", "")).upper(),
        str(first.get("interval", "")).lower(),
    )
    source = next(
        (
            item
            for item in sources
            if isinstance(item, dict)
            and (
                str(item.get("exchange", "")).lower(),
                str(item.get("symbol", "")).upper(),
                str(item.get("interval", "")).lower(),
            )
            == identity
        ),
        None,
    )
    if source is None:
        raise ValueError("Transformer checkpoint 沒有相同市場的訓練來源")
    source_rows = int(source.get("rows", 0))
    source_start = pd.to_datetime(source.get("start_at"), utc=True, errors="coerce")
    source_end = pd.to_datetime(source.get("end_at"), utc=True, errors="coerce")
    if source_rows <= 0 or pd.isna(source_start) or pd.isna(source_end):
        raise ValueError("Transformer checkpoint 的來源範圍不完整")
    start_matches = ordered.index[ordered["timestamp"].eq(source_start)]
    if not len(start_matches):
        raise ValueError("RL 串接資料找不到 Transformer 訓練來源起點")
    source_start_index = int(start_matches[0])
    source_end_index = source_start_index + source_rows - 1
    if source_end_index >= len(ordered):
        raise ValueError("RL 串接資料短於 Transformer 原始訓練來源")
    if ordered.iloc[source_end_index]["timestamp"] != source_end:
        raise ValueError("RL 串接資料時間序列與 Transformer 訓練來源不一致")
    cutoff_index = source_start_index + int(source_rows * calibration_fraction) - 1
    cutoff = pd.Timestamp(ordered.iloc[cutoff_index]["timestamp"])
    return {
        "schema_version": 1,
        "policy": "after_transformer_validation_and_calibration",
        "checkpoint": str(checkpoint),
        "source_identity": {
            "exchange": identity[0],
            "symbol": identity[1],
            "interval": identity[2],
        },
        "transformer_source_start": source_start.isoformat(),
        "transformer_source_end": source_end.isoformat(),
        "transformer_fit_calibration_cutoff": cutoff.isoformat(),
        "safe_rl_start_exclusive": cutoff.isoformat(),
    }


def infer_latest_transformer_context(
    checkpoint_path: str | Path,
    frame: pd.DataFrame,
    *,
    device: str = "auto",
    mixed_precision: bool = True,
) -> pd.DataFrame:
    """只推論最新 observation 所需的序列，供模擬與實盤重建特徵。"""
    checkpoint = Path(checkpoint_path).resolve()
    payload = _load_checkpoint(checkpoint, torch.device("cpu"))
    config = _model_config(payload)
    if len(frame) < config.sequence_length:
        raise ValueError(
            f"最新資料只有 {len(frame)} 根 K 線，Transformer 至少需要 "
            f"{config.sequence_length} 根"
        )

    # 技術指標已由完整歷史資料計算完成，模型只需要最後一段序列即可產生最新預測。
    return infer_transformer_context_frame(
        checkpoint,
        frame.tail(config.sequence_length).copy(),
        device=device,
        mixed_precision=mixed_precision,
        batch_size=1,
    )


@lru_cache(maxsize=4)
def _load_checkpoint_cached(
    path_text: str,
    modified_ns: int,
    size: int,
) -> dict[str, object]:
    """依檔案身分快取 CPU checkpoint，避免即時推論重複讀取磁碟。"""
    del modified_ns, size
    payload = torch.load(Path(path_text), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Transformer checkpoint 格式不正確")
    return payload


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"找不到 Transformer 模型：{path}")
    del device
    stat = path.stat()
    payload = _load_checkpoint_cached(
        str(path.resolve()),
        stat.st_mtime_ns,
        stat.st_size,
    )
    required = {"model_config", "feature_columns", "scaler", "state_dict"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Transformer checkpoint 缺少欄位：{missing}")
    return payload


def _resolve_device(requested: str) -> torch.device:
    normalized = requested.strip().lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定 CUDA 推論，但目前偵測不到可用 GPU")
    if normalized not in {"cpu", "cuda"}:
        raise ValueError("推論裝置必須是 auto、cpu 或 cuda")
    return torch.device(normalized)


def _model_config(payload: dict[str, object]) -> TemporalTransformerConfig:
    raw = dict(payload["model_config"])
    raw["return_horizons"] = tuple(int(value) for value in raw["return_horizons"])
    if int(raw.get("architecture_version", 0)) != 3:
        raise ValueError("此 Checkpoint 不是 Transformer V3，請改用 V3 模型")
    if "quantile_levels" in raw:
        raw["quantile_levels"] = tuple(
            float(value) for value in raw["quantile_levels"]
        )
    if "feature_group_ids" in raw:
        raw["feature_group_ids"] = tuple(
            int(value) for value in raw["feature_group_ids"]
        )
    if "feature_group_names" in raw:
        raw["feature_group_names"] = tuple(
            str(value) for value in raw["feature_group_names"]
        )
    return TemporalTransformerConfig(**raw)


def _calibrate_output(
    output: dict[str, torch.Tensor],
    payload: dict[str, object],
) -> dict[str, torch.Tensor]:
    """將訓練時由驗證集估計的溫度套用至分類機率。"""
    raw_calibration = payload.get("calibration", {})
    calibration = dict(raw_calibration) if isinstance(raw_calibration, dict) else {}
    result = dict(output)
    specs = [
        ("regime_logits", "regime_probability", "regime_temperature"),
        (
            "volatility_regime_logits",
            "volatility_regime_probability",
            "volatility_regime_temperature",
        ),
    ]
    if "movement_logits" in result:
        specs.extend(
            [
                ("movement_logits", "movement_probability", "movement_temperature"),
                ("side_logits", "side_probability", "side_temperature"),
            ]
        )
    else:
        specs.append(
            ("direction_logits", "direction_probability", "direction_temperature")
        )
    for logits_name, probability_name, temperature_name in specs:
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
    for name in (
        "regime_probability",
        "direction_probability",
        "movement_probability",
        "volatility_regime_probability",
    ):
        if name not in result:
            continue
        probabilities = result[name].clamp_min(1e-8)
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


def apply_transformer_checkpoint(
    checkpoint_path: str | Path,
    source_path: str | Path,
    *,
    output_path: str | Path | None = None,
    device: str = "auto",
    batch_size: int = 512,
    mixed_precision: bool = True,
    progress_callback: InferenceProgressCallback | None = None,
) -> TransformerInferenceArtifact:
    """使用訓練期 scaler 與欄位順序推論，預設直接更新原特徵檔。"""
    if batch_size <= 0:
        raise ValueError("推論 batch_size 必須大於 0")
    started = monotonic()
    source = Path(source_path).resolve()
    target = Path(output_path).resolve() if output_path else source
    resolved_device = _resolve_device(device)
    checkpoint = Path(checkpoint_path).resolve()
    payload = _load_checkpoint(checkpoint, resolved_device)
    config = _model_config(payload)
    feature_columns = tuple(str(value) for value in payload["feature_columns"])
    scaler = dict(payload["scaler"])

    frame = add_model_market_features(pd.read_csv(source))
    if "timestamp" not in frame:
        raise ValueError(f"{source.name} 缺少 timestamp")
    provided_columns = set(frame.columns)
    sources = payload.get("sources", [])
    if isinstance(sources, list) and sources and {
        "exchange",
        "symbol",
        "interval",
    }.issubset(frame.columns):
        expected_identities = {
            (
                str(item.get("exchange", "")).lower(),
                str(item.get("symbol", "")).upper(),
                str(item.get("interval", "")).lower(),
            )
            for item in sources
            if isinstance(item, dict)
        }
        actual_identities = {
            (
                str(row.exchange).lower(),
                str(row.symbol).upper(),
                str(row.interval).lower(),
            )
            for row in frame[["exchange", "symbol", "interval"]]
            .drop_duplicates()
            .itertuples(index=False)
        }
        if expected_identities and not actual_identities.issubset(expected_identities):
            raise ValueError("Transformer checkpoint 與輸入市場身分不一致")
    missing = [column for column in feature_columns if column not in frame]
    if missing:
        for column in missing:
            frame[column] = np.nan
    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"], utc=True, errors="coerce", format="mixed"
    )
    frame = (
        frame.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )
    if len(frame) < config.sequence_length:
        raise ValueError(
            f"資料只有 {len(frame)} 列，模型至少需要 {config.sequence_length} 列"
        )
    raw_features = (
        frame.loc[:, feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy(dtype=np.float64)
    )
    medians = np.asarray(scaler["medians"], dtype=np.float64)
    means = np.asarray(scaler["means"], dtype=np.float64)
    scales = np.asarray(scaler["scales"], dtype=np.float64)
    if not (
        raw_features.shape[1] == len(medians) == len(means) == len(scales)
    ):
        raise ValueError("Transformer scaler 維度與特徵欄位不一致")
    scales = np.where(np.isfinite(scales) & (np.abs(scales) > 1e-12), scales, 1.0)
    filled = np.where(np.isfinite(raw_features), raw_features, medians)
    standardized = ((filled - means) / scales).astype(np.float32)
    feature_mask = np.isfinite(raw_features).astype(np.float32)

    raw_contract = payload.get("feature_contract")
    contract = (
        dict(raw_contract)
        if isinstance(raw_contract, dict)
        else build_transformer_feature_contract(feature_columns)
    )
    core_columns = tuple(
        column
        for column in contract.get("core_feature_columns", feature_columns)
        if column in feature_columns
    )
    core_indices = [feature_columns.index(column) for column in core_columns]
    minimum_coverage = float(contract.get("minimum_core_coverage", 0.25))
    required_timeframes = tuple(str(value) for value in contract.get("required_timeframes", ()))

    model = MarketTemporalTransformer(config).to(resolved_device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    amp_enabled = bool(mixed_precision and resolved_device.type == "cuda")
    endpoint_indices = np.arange(config.sequence_length - 1, len(frame))
    endpoint_coverage = np.zeros(len(endpoint_indices), dtype=np.float64)
    endpoint_valid = np.ones(len(endpoint_indices), dtype=bool)
    if not core_indices or not any(column in provided_columns for column in core_columns):
        endpoint_valid[:] = False
    else:
        for output_index, endpoint in enumerate(endpoint_indices):
            start = endpoint - config.sequence_length + 1
            core_window = feature_mask[start : endpoint + 1, core_indices]
            endpoint_coverage[output_index] = float(core_window.mean())
            if endpoint_coverage[output_index] < minimum_coverage:
                endpoint_valid[output_index] = False
                continue
            for timeframe in required_timeframes:
                timeframe_indices = [
                    index
                    for index, column in enumerate(feature_columns)
                    if column.lower().startswith(f"mtf_{timeframe.lower()}_")
                ]
                if timeframe_indices and not feature_mask[
                    start : endpoint + 1, timeframe_indices
                ].any():
                    endpoint_valid[output_index] = False
                    break
    predicted_returns: list[np.ndarray] = []
    predicted_volatility: list[np.ndarray] = []
    regime_probabilities: list[np.ndarray] = []
    uncertainties: list[np.ndarray] = []
    latents: list[np.ndarray] = []
    direction_probabilities: list[np.ndarray] = []
    movement_probabilities: list[np.ndarray] = []
    side_probabilities: list[np.ndarray] = []
    quantile_predictions: list[np.ndarray] = []
    edge_predictions: list[np.ndarray] = []
    excursion_predictions: list[np.ndarray] = []
    tradeability_probabilities: list[np.ndarray] = []
    volatility_regime_probabilities: list[np.ndarray] = []
    timeframe_attentions: list[np.ndarray] = []
    total_batches = int(np.ceil(len(endpoint_indices) / batch_size))

    with torch.inference_mode():
        for batch_index, start_index in enumerate(
            range(0, len(endpoint_indices), batch_size),
            start=1,
        ):
            endpoints = endpoint_indices[start_index : start_index + batch_size]
            sequences = np.stack(
                [
                    standardized[
                        endpoint - config.sequence_length + 1 : endpoint + 1
                    ]
                    for endpoint in endpoints
                ]
            )
            sequence_masks = np.stack(
                [
                    feature_mask[
                        endpoint - config.sequence_length + 1 : endpoint + 1
                    ]
                    for endpoint in endpoints
                ]
            )
            tensor = torch.from_numpy(sequences).to(
                resolved_device,
                non_blocking=resolved_device.type == "cuda",
            )
            mask_tensor = torch.from_numpy(sequence_masks).to(
                resolved_device,
                non_blocking=resolved_device.type == "cuda",
            )
            with torch.autocast(
                device_type=resolved_device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                output = _calibrate_output(
                    model(tensor, feature_mask=mask_tensor),
                    payload,
                )
            predicted_returns.append(output["future_returns"].float().cpu().numpy())
            predicted_volatility.append(output["volatility"].float().cpu().numpy())
            regime_probabilities.append(
                output["regime_probability"].float().cpu().numpy()
            )
            uncertainties.append(output["uncertainty"].float().cpu().numpy())
            latents.append(output["latent"].float().cpu().numpy())
            if "direction_probability" in output:
                direction_probabilities.append(
                    output["direction_probability"].float().cpu().numpy()
                )
            if "movement_probability" in output:
                movement_probabilities.append(
                    output["movement_probability"].float().cpu().numpy()
                )
                side_probabilities.append(
                    output["side_probability"].float().cpu().numpy()
                )
            if "quantile_returns" in output:
                quantile_predictions.append(
                    output["quantile_returns"].float().cpu().numpy()
                )
            if "edge_returns" in output:
                edge_predictions.append(output["edge_returns"].float().cpu().numpy())
                excursion_predictions.append(
                    output["excursions"].float().cpu().numpy()
                )
                tradeability_probabilities.append(
                    output["tradeability_probability"].float().cpu().numpy()
                )
                volatility_regime_probabilities.append(
                    output["volatility_regime_probability"].float().cpu().numpy()
                )
                timeframe_attentions.append(
                    output["timeframe_attention"].float().cpu().numpy()
                )
            if progress_callback:
                progress_callback(
                    {
                        "progress": batch_index / total_batches,
                        "batch": batch_index,
                        "batches": total_batches,
                        "predicted_rows": min(
                            start_index + batch_size, len(endpoint_indices)
                        ),
                        "total_rows": len(endpoint_indices),
                        "device": str(resolved_device),
                    }
                )

    return_z = np.concatenate(predicted_returns)
    volatility_z = np.concatenate(predicted_volatility).reshape(-1)
    probabilities = np.concatenate(regime_probabilities)
    uncertainty = np.concatenate(uncertainties)
    latent = np.concatenate(latents)
    direction_probability = (
        np.concatenate(direction_probabilities)
        if direction_probabilities
        else None
    )
    movement_probability = (
        np.concatenate(movement_probabilities) if movement_probabilities else None
    )
    side_probability = (
        np.concatenate(side_probabilities) if side_probabilities else None
    )
    quantile_z = (
        np.concatenate(quantile_predictions)
        if quantile_predictions
        else None
    )
    edge_z = np.concatenate(edge_predictions) if edge_predictions else None
    excursion_z = (
        np.concatenate(excursion_predictions) if excursion_predictions else None
    )
    tradeability_probability = (
        np.concatenate(tradeability_probabilities)
        if tradeability_probabilities
        else None
    )
    volatility_regime_probability = (
        np.concatenate(volatility_regime_probabilities)
        if volatility_regime_probabilities
        else None
    )
    timeframe_attention = (
        np.concatenate(timeframe_attentions) if timeframe_attentions else None
    )
    return_means = np.asarray(scaler["return_means"], dtype=np.float64)
    return_scales = np.asarray(scaler["return_scales"], dtype=np.float64)
    returns = return_z * return_scales + return_means
    quantile_returns = (
        quantile_z * return_scales[None, :, None] + return_means[None, :, None]
        if quantile_z is not None
        else None
    )
    volatility = (
        volatility_z * float(scaler["volatility_scale"])
        + float(scaler["volatility_mean"])
    )
    edge_returns = None
    if edge_z is not None:
        edge_means = np.asarray(
            scaler.get("edge_means", np.zeros(edge_z.shape[1:])),
            dtype=np.float64,
        ).reshape(-1, 2)
        edge_scales = np.asarray(
            scaler.get("edge_scales", np.ones(edge_z.shape[1:])),
            dtype=np.float64,
        ).reshape(-1, 2)
        edge_returns = edge_z * edge_scales + edge_means
    excursions = None
    if excursion_z is not None:
        excursion_scales = np.asarray(
            scaler.get("excursion_scales", np.ones(excursion_z.shape[1:])),
            dtype=np.float64,
        ).reshape(-1, 2)
        excursions = excursion_z * excursion_scales

    result = ensure_transformer_context(frame)
    result.loc[:, "transformer_available"] = 0.0
    result.loc[:, "transformer_probability_calibrated"] = 0.0
    result.loc[:, "transformer_input_coverage"] = 0.0
    result.loc[:, "transformer_input_valid"] = 0.0
    result.loc[endpoint_indices, "transformer_input_coverage"] = endpoint_coverage
    valid_endpoint_indices = endpoint_indices[endpoint_valid]
    result.loc[valid_endpoint_indices, "transformer_input_valid"] = 1.0
    for horizon_index, horizon in enumerate(config.return_horizons):
        column = f"transformer_return_{horizon}"
        if column not in result:
            result[column] = 0.0
        result.loc[endpoint_indices, column] = returns[:, horizon_index]
        if direction_probability is not None:
            for class_index, name in enumerate(("down", "neutral", "up")):
                probability_column = f"transformer_{name}_probability_{horizon}"
                if probability_column not in result:
                    result[probability_column] = 0.0
                result.loc[endpoint_indices, probability_column] = (
                    direction_probability[:, horizon_index, class_index]
                )
        if movement_probability is not None:
            for class_index, name in enumerate(("down", "neutral", "up")):
                column = f"transformer_movement_{name}_probability_{horizon}"
                result[column] = 0.0
                result.loc[endpoint_indices, column] = movement_probability[
                    :, horizon_index, class_index
                ]
            for class_index, name in enumerate(("down", "up")):
                column = f"transformer_side_{name}_probability_{horizon}"
                result[column] = 0.0
                result.loc[endpoint_indices, column] = side_probability[
                    :, horizon_index, class_index
                ]
        if quantile_returns is not None:
            for quantile_index, level in enumerate(config.quantile_levels):
                token = int(round(level * 100))
                quantile_column = f"transformer_return_q{token}_{horizon}"
                if quantile_column not in result:
                    result[quantile_column] = 0.0
                result.loc[endpoint_indices, quantile_column] = quantile_returns[
                    :, horizon_index, quantile_index
                ]
        if edge_returns is not None:
            for side_index, side in enumerate(("long", "short")):
                edge_column = f"transformer_{side}_edge_{horizon}"
                result[edge_column] = 0.0
                result.loc[endpoint_indices, edge_column] = edge_returns[
                    :, horizon_index, side_index
                ]
            for excursion_index, name in enumerate(("downside", "upside")):
                excursion_column = f"transformer_{name}_excursion_{horizon}"
                result[excursion_column] = 0.0
                result.loc[endpoint_indices, excursion_column] = excursions[
                    :, horizon_index, excursion_index
                ]
            tradeability_column = f"transformer_tradeability_{horizon}"
            result[tradeability_column] = 0.0
            result.loc[endpoint_indices, tradeability_column] = (
                tradeability_probability[:, horizon_index]
            )
    result.loc[endpoint_indices, "transformer_volatility"] = np.maximum(
        volatility, 0.0
    )
    result.loc[endpoint_indices, "transformer_bear_probability"] = probabilities[:, 0]
    result.loc[endpoint_indices, "transformer_bull_probability"] = probabilities[:, -1]
    result.loc[endpoint_indices, "transformer_uncertainty"] = uncertainty
    if volatility_regime_probability is not None:
        for class_index, name in enumerate(("low", "normal", "high")):
            column = f"transformer_volatility_regime_{name}_probability"
            result[column] = 0.0
            result.loc[endpoint_indices, column] = volatility_regime_probability[
                :, class_index
            ]
    if timeframe_attention is not None:
        for group_index, name in enumerate(config.feature_group_names):
            column = f"transformer_timeframe_{name}_attention"
            result[column] = 0.0
            result.loc[endpoint_indices, column] = timeframe_attention[:, group_index]
    result.loc[valid_endpoint_indices, "transformer_available"] = 1.0
    raw_calibration = payload.get("calibration", {})
    probability_calibrated = bool(
        isinstance(raw_calibration, dict) and raw_calibration.get("method")
    )
    result.loc[valid_endpoint_indices, "transformer_probability_calibrated"] = float(
        probability_calibrated
    )
    for latent_index in range(latent.shape[1]):
        column = f"transformer_latent_{latent_index}"
        result[column] = 0.0
        result.loc[endpoint_indices, column] = latent[:, latent_index]
    result["transformer_model_id"] = checkpoint.parent.name
    result["transformer_inferred_at"] = datetime.now(timezone.utc).isoformat()
    result["timestamp"] = result["timestamp"].map(lambda value: value.isoformat())
    save_canonical_context_csv(
        result,
        target,
        key_columns=("timestamp",),
        merge_existing=False,
    )
    return TransformerInferenceArtifact(
        source_path=source,
        output_path=target,
        checkpoint_path=checkpoint,
        rows=len(result),
        predicted_rows=int(endpoint_valid.sum()),
        device=str(resolved_device),
        duration_seconds=monotonic() - started,
        return_horizons=config.return_horizons,
    )
