"""為 SAC 產生時間因果正確的 Transformer cross-fit 樣本外預測。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Literal

import pandas as pd

from ai_quant_trading.operations.integrity import build_artifact_manifest, sha256_file
from ai_quant_trading.transformer.config import (
    TemporalTransformerConfig,
    TransformerTrainingConfig,
)
from ai_quant_trading.transformer.inference import infer_transformer_context_frame
from ai_quant_trading.transformer.training import train_temporal_transformer


CrossFitMode = Literal["expanding", "rolling"]


@dataclass(frozen=True, slots=True)
class TransformerCrossFitConfig:
    """Cross-fit 時間切分與 final holdout 封存設定。"""

    folds: int = 4
    mode: CrossFitMode = "expanding"
    initial_train_fraction: float = 0.40
    final_holdout_fraction: float = 0.10
    minimum_fit_rows: int = 4_000
    minimum_oos_rows: int = 1_000
    minimum_final_holdout_rows: int = 1_000
    rolling_train_rows: int | None = None
    seed_stride: int = 1_000
    expected_return_horizon: int = 5
    keep_fold_sources: bool = False

    def __post_init__(self) -> None:
        if not 2 <= self.folds <= 12:
            raise ValueError("Transformer cross-fit folds 必須介於 2 與 12")
        if self.mode not in {"expanding", "rolling"}:
            raise ValueError("cross-fit mode 只支援 expanding 或 rolling")
        if not 0.20 <= self.initial_train_fraction < 0.80:
            raise ValueError("initial_train_fraction 必須介於 0.20（含）與 0.80（不含）")
        if not 0.05 <= self.final_holdout_fraction <= 0.30:
            raise ValueError("final_holdout_fraction 必須介於 0.05 與 0.30")
        for name in (
            "minimum_fit_rows",
            "minimum_oos_rows",
            "minimum_final_holdout_rows",
            "seed_stride",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} 必須大於 0")
        if self.mode == "rolling" and (
            self.rolling_train_rows is None
            or self.rolling_train_rows < self.minimum_fit_rows
        ):
            raise ValueError("rolling 模式必須提供不小於 minimum_fit_rows 的 rolling_train_rows")
        if self.expected_return_horizon <= 0:
            raise ValueError("expected_return_horizon 必須大於 0")


@dataclass(frozen=True, slots=True)
class TransformerCrossFitFold:
    """單一因果訓練窗與其後方非重疊 OOS 區塊。"""

    fold: int
    fit_start: int
    fit_end: int
    oos_start: int
    oos_end: int
    fit_start_at: str
    fit_end_at: str
    oos_start_at: str
    oos_end_at: str
    label_purge_bars: int


@dataclass(frozen=True, slots=True)
class TransformerCrossFitPlan:
    """不接觸 final holdout 的完整 cross-fit 時間契約。"""

    folds: tuple[TransformerCrossFitFold, ...]
    selection_rows: int
    final_holdout_start: int
    final_holdout_rows: int
    final_holdout_start_at: str
    final_holdout_end_at: str


@dataclass(frozen=True, slots=True)
class TransformerCrossFitResult:
    """Cross-fit OOS 合併檔與稽核資料。"""

    run_dir: Path
    predictions_csv: Path
    summary_json: Path
    plan: TransformerCrossFitPlan
    predicted_rows: int


CrossFitProgressCallback = Callable[[dict[str, object]], None]
CrossFitTrainer = Callable[
    [
        Path,
        TransformerCrossFitFold,
        Path,
        TemporalTransformerConfig,
        TransformerTrainingConfig,
    ],
    Path,
]
CrossFitInferencer = Callable[
    [Path, pd.DataFrame, TransformerCrossFitFold],
    pd.DataFrame,
]


def _canonical_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in frame:
        raise ValueError("Transformer cross-fit 資料缺少 timestamp")
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(
        result["timestamp"], utc=True, errors="coerce", format="mixed"
    )
    result = (
        result.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )
    if len(result) < 2:
        raise ValueError("Transformer cross-fit 至少需要兩列資料")
    return result


def build_transformer_crossfit_plan(
    frame: pd.DataFrame,
    config: TransformerCrossFitConfig,
    *,
    max_horizon_bars: int,
) -> TransformerCrossFitPlan:
    """建立 expanding／rolling 計畫；OOS 區塊互斥且尾端 holdout 不可見。"""
    if max_horizon_bars <= 0:
        raise ValueError("max_horizon_bars 必須大於 0")
    ordered = _canonical_frame(frame)
    holdout_rows = max(
        config.minimum_final_holdout_rows,
        math.ceil(len(ordered) * config.final_holdout_fraction),
    )
    selection_rows = len(ordered) - holdout_rows
    initial_fit_rows = max(
        config.minimum_fit_rows,
        math.floor(selection_rows * config.initial_train_fraction),
    )
    remaining = selection_rows - initial_fit_rows
    if remaining < config.folds * config.minimum_oos_rows:
        raise ValueError(
            "資料不足以建立 Transformer cross-fit："
            f"選模區 {selection_rows} 列、初始訓練 {initial_fit_rows} 列、"
            f"剩餘 OOS {remaining} 列"
        )
    block = remaining // config.folds
    folds: list[TransformerCrossFitFold] = []
    for index in range(config.folds):
        fit_end = initial_fit_rows + block * index
        fit_start = 0
        if config.mode == "rolling":
            window = int(config.rolling_train_rows or initial_fit_rows)
            fit_start = max(0, fit_end - window)
        oos_start = fit_end
        oos_end = selection_rows if index == config.folds - 1 else fit_end + block
        if fit_end - fit_start < config.minimum_fit_rows:
            raise ValueError(f"第 {index + 1} fold 訓練列數不足")
        if oos_end - oos_start < config.minimum_oos_rows:
            raise ValueError(f"第 {index + 1} fold OOS 列數不足")
        folds.append(
            TransformerCrossFitFold(
                fold=index + 1,
                fit_start=fit_start,
                fit_end=fit_end,
                oos_start=oos_start,
                oos_end=oos_end,
                fit_start_at=ordered.iloc[fit_start]["timestamp"].isoformat(),
                fit_end_at=ordered.iloc[fit_end - 1]["timestamp"].isoformat(),
                oos_start_at=ordered.iloc[oos_start]["timestamp"].isoformat(),
                oos_end_at=ordered.iloc[oos_end - 1]["timestamp"].isoformat(),
                label_purge_bars=max_horizon_bars,
            )
        )
    holdout = ordered.iloc[selection_rows:]
    return TransformerCrossFitPlan(
        folds=tuple(folds),
        selection_rows=selection_rows,
        final_holdout_start=selection_rows,
        final_holdout_rows=len(holdout),
        final_holdout_start_at=holdout.iloc[0]["timestamp"].isoformat(),
        final_holdout_end_at=holdout.iloc[-1]["timestamp"].isoformat(),
    )


def assemble_transformer_crossfit_predictions(
    fold_predictions: list[tuple[TransformerCrossFitFold, pd.DataFrame, str]],
) -> pd.DataFrame:
    """驗證每列都晚於該 fold fit cutoff，再合併成無重疊 OOS 資料。"""
    if not fold_predictions:
        raise ValueError("沒有可合併的 Transformer cross-fit 預測")
    parts: list[pd.DataFrame] = []
    for fold, raw, checkpoint_sha256 in fold_predictions:
        part = _canonical_frame(raw)
        start = pd.Timestamp(fold.oos_start_at)
        end = pd.Timestamp(fold.oos_end_at)
        fit_end = pd.Timestamp(fold.fit_end_at)
        if (part["timestamp"] <= fit_end).any():
            raise ValueError(f"第 {fold.fold} fold 含有 fit cutoff 以前的非 OOS 預測")
        if (part["timestamp"] < start).any() or (part["timestamp"] > end).any():
            raise ValueError(f"第 {fold.fold} fold 預測超出宣告的 OOS 邊界")
        if "transformer_available" not in part:
            raise ValueError(f"第 {fold.fold} fold 缺少 transformer_available")
        part["transformer_oos"] = 1
        part["transformer_oos_fold"] = fold.fold
        part["transformer_fit_end_at"] = fold.fit_end_at
        part["transformer_checkpoint_sha256"] = checkpoint_sha256
        parts.append(part)
    merged = pd.concat(parts, ignore_index=True).sort_values("timestamp")
    if merged["timestamp"].duplicated().any():
        raise ValueError("Transformer cross-fit OOS 區塊有重疊 timestamp")
    return merged.reset_index(drop=True)


def _default_trainer(
    source_path: Path,
    fold: TransformerCrossFitFold,
    output_root: Path,
    model_config: TemporalTransformerConfig,
    training_config: TransformerTrainingConfig,
) -> Path:
    result = train_temporal_transformer(
        [source_path],
        model_config,
        training_config,
        output_root,
        run_name=f"crossfit_fold_{fold.fold:02d}",
    )
    return result.model_path


def _default_inferencer(
    checkpoint: Path,
    context: pd.DataFrame,
    _fold: TransformerCrossFitFold,
) -> pd.DataFrame:
    return infer_transformer_context_frame(checkpoint, context)


def run_transformer_crossfit(
    source_path: str | Path,
    model_config: TemporalTransformerConfig,
    training_config: TransformerTrainingConfig,
    crossfit_config: TransformerCrossFitConfig,
    output_root: str | Path,
    *,
    run_name: str = "transformer_crossfit",
    trainer: CrossFitTrainer | None = None,
    inferencer: CrossFitInferencer | None = None,
    progress_callback: CrossFitProgressCallback | None = None,
) -> TransformerCrossFitResult:
    """依 fold 訓練不同 checkpoint，只保存下一段 OOS 預測供 SAC 使用。"""
    source = Path(source_path).resolve()
    frame = _canonical_frame(pd.read_csv(source))
    if crossfit_config.expected_return_horizon not in model_config.return_horizons:
        raise ValueError("expected_return_horizon 不在 Transformer return_horizons 內")
    plan = build_transformer_crossfit_plan(
        frame,
        crossfit_config,
        max_horizon_bars=max(model_config.return_horizons),
    )
    safe_name = "".join(
        value if value.isalnum() or value in {"-", "_"} else "_"
        for value in run_name.strip()
    ).strip("_") or "transformer_crossfit"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = Path(output_root).resolve() / f"{stamp}_{safe_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    folds_root = run_dir / "folds"
    folds_root.mkdir()
    selected_trainer = trainer or _default_trainer
    selected_inferencer = inferencer or _default_inferencer
    source_sha256 = sha256_file(source)
    predictions: list[tuple[TransformerCrossFitFold, pd.DataFrame, str]] = []
    fold_summaries: list[dict[str, object]] = []

    with TemporaryDirectory(prefix="aiquant_crossfit_") as temporary:
        temporary_root = Path(temporary)
        for fold in plan.folds:
            fold_root = folds_root / f"fold_{fold.fold:02d}"
            fold_root.mkdir()
            fit = frame.iloc[fold.fit_start : fold.fit_end].reset_index(drop=True)
            if crossfit_config.keep_fold_sources:
                fit_path = fold_root / "fit_source.csv"
            else:
                fit_path = temporary_root / f"fold_{fold.fold:02d}_fit.csv"
            fit.to_csv(fit_path, index=False, encoding="utf-8")
            fold_training_config = replace(
                training_config,
                seed=training_config.seed + (fold.fold - 1) * crossfit_config.seed_stride,
                max_rows_per_source=None,
            )
            if progress_callback:
                progress_callback(
                    {
                        "status": "training_fold",
                        "fold": fold.fold,
                        "folds": len(plan.folds),
                        "progress": (fold.fold - 1) / len(plan.folds),
                    }
                )
            checkpoint = Path(
                selected_trainer(
                    fit_path,
                    fold,
                    fold_root / "models",
                    model_config,
                    fold_training_config,
                )
            ).resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(f"第 {fold.fold} fold 沒有產生 checkpoint")
            context_start = max(fold.fit_start, fold.oos_start - model_config.sequence_length + 1)
            context = frame.iloc[context_start : fold.oos_end].reset_index(drop=True)
            inferred = _canonical_frame(selected_inferencer(checkpoint, context, fold))
            oos_start = pd.Timestamp(fold.oos_start_at)
            oos_end = pd.Timestamp(fold.oos_end_at)
            inferred = inferred.loc[
                (inferred["timestamp"] >= oos_start)
                & (inferred["timestamp"] <= oos_end)
            ].reset_index(drop=True)
            if "transformer_available" not in inferred:
                raise ValueError(f"第 {fold.fold} fold 推論缺少 transformer_available")
            available = pd.to_numeric(
                inferred["transformer_available"], errors="coerce"
            ).fillna(0.0)
            inferred = inferred.loc[available >= 0.5].reset_index(drop=True)
            checkpoint_hash = sha256_file(checkpoint)
            predictions.append((fold, inferred, checkpoint_hash))
            fold_summaries.append(
                {
                    **asdict(fold),
                    "fit_rows": len(fit),
                    "predicted_rows": len(inferred),
                    "checkpoint": str(checkpoint.relative_to(run_dir)),
                    "checkpoint_sha256": checkpoint_hash,
                    "seed": fold_training_config.seed,
                }
            )

    merged = assemble_transformer_crossfit_predictions(predictions)
    merged["transformer_crossfit_mode"] = crossfit_config.mode
    merged["transformer_source_sha256"] = source_sha256
    expected_source = f"transformer_return_{crossfit_config.expected_return_horizon}"
    if expected_source not in merged:
        raise ValueError(f"Cross-fit OOS 預測缺少預期報酬來源：{expected_source}")
    merged["expected_return"] = pd.to_numeric(merged[expected_source], errors="coerce")
    predictions_csv = run_dir / "oos_predictions.csv"
    merged.to_csv(predictions_csv, index=False, encoding="utf-8")
    expected_return_contract = {
        "schema_version": 1,
        "source_column": expected_source,
        "horizon_bars": crossfit_config.expected_return_horizon,
        "units": "fractional_gross_return",
        "direction": "positive_long_negative_short",
    }
    summary_json = run_dir / "crossfit.json"
    summary = {
        "schema_version": 1,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "causal_policy": "fold checkpoint only predicts its immediately following non-overlapping OOS block",
        "source": str(source),
        "source_sha256": source_sha256,
        "model_config": model_config.to_dict(),
        "training_config": training_config.to_dict(),
        "crossfit_config": asdict(crossfit_config),
        "plan": {
            "selection_rows": plan.selection_rows,
            "final_holdout_start": plan.final_holdout_start,
            "final_holdout_rows": plan.final_holdout_rows,
            "final_holdout_start_at": plan.final_holdout_start_at,
            "final_holdout_end_at": plan.final_holdout_end_at,
            "final_holdout_sealed": True,
        },
        "folds": fold_summaries,
        "predicted_rows": len(merged),
        "expected_return_contract": expected_return_contract,
        "predictions_sha256": sha256_file(predictions_csv),
        "artifacts": {"oos_predictions": predictions_csv.name},
    }
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    build_artifact_manifest(run_dir)
    if progress_callback:
        progress_callback(
            {
                "status": "complete",
                "fold": len(plan.folds),
                "folds": len(plan.folds),
                "progress": 1.0,
                "predicted_rows": len(merged),
                "result_dir": str(run_dir),
            }
        )
    return TransformerCrossFitResult(
        run_dir,
        predictions_csv,
        summary_json,
        plan,
        len(merged),
    )


def load_transformer_crossfit_predictions(
    run_dir_or_summary: str | Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """驗證 cross-fit 成品雜湊，並恢復 SAC 必需的 expected_return 契約。"""
    source = Path(run_dir_or_summary).resolve()
    summary_path = source / "crossfit.json" if source.is_dir() else source
    if not summary_path.is_file():
        raise FileNotFoundError(f"找不到 Transformer cross-fit 摘要：{summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    plan = dict(payload.get("plan", {}))
    if payload.get("status") != "complete" or not bool(
        plan.get("final_holdout_sealed", False)
    ):
        raise ValueError("Transformer cross-fit 尚未完成或 final holdout 未封存")
    relative = str(dict(payload.get("artifacts", {})).get("oos_predictions", ""))
    predictions_path = summary_path.parent / relative
    if not relative or not predictions_path.is_file():
        raise FileNotFoundError("Transformer cross-fit 找不到 OOS 預測檔")
    expected_hash = str(payload.get("predictions_sha256", ""))
    if not expected_hash or sha256_file(predictions_path) != expected_hash:
        raise ValueError("Transformer cross-fit OOS 預測檔雜湊不一致")
    frame = _canonical_frame(pd.read_csv(predictions_path))
    if "transformer_oos" not in frame or not pd.to_numeric(
        frame["transformer_oos"], errors="coerce"
    ).fillna(0.0).eq(1.0).all():
        raise ValueError("Transformer cross-fit 預測含有非 OOS 資料")
    contract = dict(payload.get("expected_return_contract", {}))
    required = {
        "schema_version",
        "source_column",
        "horizon_bars",
        "units",
        "direction",
    }
    if not required.issubset(contract):
        raise ValueError("Transformer cross-fit 缺少 expected_return 契約")
    frame.attrs["expected_return_contract"] = contract
    return frame, payload
