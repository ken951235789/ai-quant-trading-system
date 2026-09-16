"""在 Kaggle T4 依序訓練五個 Transformer V3 seeds 並以驗證集選模。"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from time import monotonic
import traceback
from zipfile import ZipFile

import numpy as np
import pandas as pd


WORKING_ROOT = Path("/kaggle/working")
RESULT_ROOT = WORKING_ROOT / "transformer_v3_five_seed"
PROGRESS_PATH = WORKING_ROOT / "five_seed_progress.json"
SUMMARY_PATH = WORKING_ROOT / "five_seed_summary.json"
SEEDS = (11, 23, 42, 67, 101)


def _json_write(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare_bundle_root(destination: Path) -> Path:
    archives = list(Path("/kaggle/input").rglob("transformer_v3_formal_bundle.zip"))
    if archives:
        destination.mkdir(parents=True, exist_ok=True)
        with ZipFile(archives[0]) as handle:
            handle.extractall(destination)
        return destination
    manifests = list(Path("/kaggle/input").rglob("formal_manifest.json"))
    if not manifests:
        raise FileNotFoundError("Kaggle Input 找不到 Transformer V3 正式訓練包")
    return manifests[0].parent


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def _classification_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, object]:
    """計算不受類別失衡掩蓋的方向分類指標。"""
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (actual, predicted), 1)
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
    return {
        "accuracy": accuracy,
        "majority_baseline": majority,
        "accuracy_lift": accuracy - majority,
        "balanced_accuracy": float(recall[support > 0].mean()),
        "macro_f1": float(f1[support > 0].mean()),
        "confusion": confusion.tolist(),
    }


def _ensemble_diagnostic(
    runs: list[dict[str, object]],
    horizons: tuple[int, ...],
    direction_threshold_bps: float,
) -> dict[str, object]:
    """等權平均五個 seed 的機率；只做測試診斷，不參與選模。"""
    frames = [
        pd.read_csv(Path(str(run["run_dir"])) / "test_predictions.csv")
        for run in runs
    ]
    if len({len(frame) for frame in frames}) != 1:
        raise RuntimeError("五個 seed 的測試預測列數不一致")

    per_horizon: dict[str, object] = {}
    aggregate_values: dict[str, list[float]] = {
        "accuracy": [],
        "majority_baseline": [],
        "balanced_accuracy": [],
        "macro_f1": [],
    }
    threshold = direction_threshold_bps / 10_000
    for horizon in horizons:
        reference = frames[0]
        long_edge = reference[f"actual_long_edge_{horizon}"].to_numpy()
        short_edge = reference[f"actual_short_edge_{horizon}"].to_numpy()
        for frame in frames[1:]:
            if not np.allclose(
                frame[f"actual_long_edge_{horizon}"].to_numpy(),
                long_edge,
                rtol=0,
                atol=1e-10,
            ) or not np.allclose(
                frame[f"actual_short_edge_{horizon}"].to_numpy(),
                short_edge,
                rtol=0,
                atol=1e-10,
            ):
                raise RuntimeError("五個 seed 的測試標籤沒有完全對齊")
        actual = np.where(
            (long_edge > threshold) & (long_edge >= short_edge),
            2,
            np.where(short_edge > threshold, 0, 1),
        )
        probability_columns = [
            f"down_probability_{horizon}",
            f"neutral_probability_{horizon}",
            f"up_probability_{horizon}",
        ]
        probabilities = np.mean(
            [frame[probability_columns].to_numpy() for frame in frames],
            axis=0,
        )
        metrics = _classification_metrics(actual, probabilities.argmax(axis=1))
        per_horizon[str(horizon)] = metrics
        for name in aggregate_values:
            aggregate_values[name].append(float(metrics[name]))

    aggregate = {
        name: float(np.mean(values))
        for name, values in aggregate_values.items()
    }
    aggregate["accuracy_lift"] = (
        aggregate["accuracy"] - aggregate["majority_baseline"]
    )
    return {
        "method": "equal_probability_average",
        "selection_uses_test": False,
        "per_horizon": per_horizon,
        "aggregate": aggregate,
    }


def main() -> int:
    started = monotonic()
    try:
        root = _prepare_bundle_root(Path("/tmp/ai_quant_transformer_v3_five_seed"))
        sys.path.insert(0, str(root / "src"))

        import torch

        from ai_quant_trading.transformer.config import (
            TemporalTransformerConfig,
            TransformerTrainingConfig,
        )
        from ai_quant_trading.transformer.training import train_temporal_transformer

        if not torch.cuda.is_available():
            raise RuntimeError("Kaggle Session 沒有可用 GPU")
        gpu_name = torch.cuda.get_device_name(0)
        if "T4" not in gpu_name.upper():
            raise RuntimeError(f"五 seed 正式訓練要求 Tesla T4，目前取得：{gpu_name}")

        manifest = json.loads((root / "formal_manifest.json").read_text(encoding="utf-8"))
        source = root / "data" / "transformer_v3_formal.csv"
        expected_hash = str(dict(manifest["data"])["sha256"])
        if _sha256(source) != expected_hash:
            raise RuntimeError("正式訓練資料 SHA-256 驗證失敗")

        hardware = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": gpu_name,
            "gpu_count": torch.cuda.device_count(),
            "cpu_count": os.cpu_count(),
        }
        print("五 seed 訓練硬體：", json.dumps(hardware, ensure_ascii=False), flush=True)
        model_config = TemporalTransformerConfig(
            input_features=256,
            sequence_length=192,
            d_model=96,
            n_heads=4,
            n_layers=3,
            feedforward_dim=192,
            dropout=0.20,
            latent_dim=16,
            return_horizons=(1, 5, 20),
            regime_classes=3,
            architecture_version=3,
            local_kernel_size=3,
            quantile_levels=(0.10, 0.50, 0.90),
            patch_size=4,
            patch_stride=2,
            volatility_regime_classes=3,
        )

        RESULT_ROOT.mkdir(parents=True, exist_ok=True)
        runs: list[dict[str, object]] = []
        for seed_index, seed in enumerate(SEEDS, start=1):
            training_config = TransformerTrainingConfig(
                epochs=60,
                batch_size=128,
                learning_rate=1.5e-4,
                weight_decay=5e-4,
                warmup_ratio=0.10,
                gradient_clip=0.75,
                early_stopping_patience=10,
                mixed_precision=True,
                device="cuda",
                num_workers=2,
                cpu_threads=4,
                train_fraction=0.60,
                validation_fraction=0.20,
                seed=seed,
                return_loss_weight=0.75,
                volatility_loss_weight=0.25,
                regime_loss_weight=0.10,
                direction_loss_weight=1.00,
                quantile_loss_weight=0.15,
                direction_threshold_bps=12.0,
                label_smoothing=0.01,
                fee_bps_per_side=4.0,
                slippage_bps_per_side=2.0,
                max_observed_spread_bps=50.0,
                edge_loss_weight=0.75,
                excursion_loss_weight=0.20,
                tradeability_loss_weight=0.40,
                volatility_regime_loss_weight=0.10,
                probability_calibration=True,
                direction_class_balance_power=0.50,
                direction_focal_gamma=1.50,
                tradeability_class_balance_power=0.50,
                checkpoint_metric="direction_skill_score",
                checkpoint_loss_penalty=0.02,
            )
            last_printed_percent = -1

            def progress(payload: dict[str, object]) -> None:
                nonlocal last_printed_percent
                seed_progress = float(payload.get("progress", 0.0))
                global_progress = ((seed_index - 1) + seed_progress) / len(SEEDS)
                event = {
                    **payload,
                    "seed": seed,
                    "seed_index": seed_index,
                    "seed_count": len(SEEDS),
                    "global_progress": global_progress,
                    "updated_elapsed_seconds": monotonic() - started,
                    "hardware": hardware,
                }
                _json_write(PROGRESS_PATH, event)
                status = str(payload.get("status", ""))
                percent = int(seed_progress * 100)
                if (
                    status in {"preparing", "validating", "complete"}
                    or percent >= last_printed_percent + 10
                ):
                    last_printed_percent = max(last_printed_percent, percent)
                    metrics = dict(payload.get("metrics", {}))
                    print(
                        f"seed {seed} [{seed_index}/{len(SEEDS)}] {status} {percent}% | "
                        f"epoch {payload.get('epoch', 0)}/{payload.get('epochs', 60)} | "
                        f"balanced={float(metrics.get('validation_direction_balanced_accuracy', 0.0)):.4f} | "
                        f"lift={float(metrics.get('validation_direction_accuracy_lift', 0.0)):.4f} | "
                        f"skill={float(metrics.get('validation_direction_skill_score', 0.0)):.4f}",
                        flush=True,
                    )

            result = train_temporal_transformer(
                [source],
                model_config,
                training_config,
                RESULT_ROOT,
                run_name=f"btc_15m_mtf_v3_seed{seed}",
                progress_callback=progress,
            )
            summary = json.loads(result.summary_json.read_text(encoding="utf-8"))
            test_metrics = dict(summary["test_metrics"])
            runs.append(
                {
                    "seed": seed,
                    "run_dir": str(result.run_dir),
                    "best_epoch": int(summary["best_epoch"]),
                    "checkpoint_metric": summary["checkpoint_metric"],
                    "validation_score": float(summary["best_checkpoint_value"]),
                    "selected_validation_loss": float(summary["selected_validation_loss"]),
                    "test_metrics": test_metrics,
                }
            )
            gc.collect()
            torch.cuda.empty_cache()

        selected = max(runs, key=lambda item: float(item["validation_score"]))
        balanced_values = [
            float(dict(item["test_metrics"])["cost_aware_direction_balanced_accuracy"])
            for item in runs
        ]
        lift_values = [
            float(dict(item["test_metrics"])["direction_accuracy_lift"])
            for item in runs
        ]
        selected_metrics = dict(selected["test_metrics"])
        per_horizon = {}
        for horizon in model_config.return_horizons:
            per_horizon[str(horizon)] = {
                "accuracy": float(
                    selected_metrics[f"cost_aware_direction_accuracy_{horizon}"]
                ),
                "majority_baseline": float(
                    selected_metrics[f"direction_majority_baseline_{horizon}"]
                ),
                "accuracy_lift": float(
                    selected_metrics[f"direction_accuracy_lift_{horizon}"]
                ),
                "balanced_accuracy": float(
                    selected_metrics[
                        f"cost_aware_direction_balanced_accuracy_{horizon}"
                    ]
                ),
                "macro_f1": float(
                    selected_metrics[f"cost_aware_direction_macro_f1_{horizon}"]
                ),
            }
        robust_balanced = _median(balanced_values)
        robust_lift = _median(lift_values)
        ensemble = _ensemble_diagnostic(
            runs,
            model_config.return_horizons,
            direction_threshold_bps=12.0,
        )
        quality_gate = {
            "five_seeds_completed": len(runs) == len(SEEDS),
            "median_balanced_accuracy_above_random_by_3pct": robust_balanced >= (1 / 3 + 0.03),
            "median_raw_accuracy_lift_nonnegative": robust_lift >= 0.0,
            "selected_all_horizons_balanced_above_random": all(
                item["balanced_accuracy"] > 1 / 3 for item in per_horizon.values()
            ),
        }
        quality_gate["passed"] = all(quality_gate.values())
        deployment_gate = {
            "selected_beats_majority_on_every_horizon": all(
                item["accuracy_lift"] > 0 for item in per_horizon.values()
            ),
            "ensemble_beats_majority_on_every_horizon": all(
                float(item["accuracy_lift"]) > 0
                for item in dict(ensemble["per_horizon"]).values()
            ),
            "cost_backtest_completed": False,
            "paper_trading_completed": False,
        }
        deployment_gate["passed"] = all(deployment_gate.values())
        report = {
            "schema_version": 1,
            "status": "complete",
            "candidate_only": True,
            "selection_uses_validation_only": True,
            "test_not_used_for_seed_selection": True,
            "duration_seconds": monotonic() - started,
            "hardware": hardware,
            "data_manifest": manifest,
            "model_config": model_config.to_dict(),
            "seeds": list(SEEDS),
            "runs": runs,
            "selected_seed": selected["seed"],
            "selected_run_dir": selected["run_dir"],
            "selected_per_horizon": per_horizon,
            "robustness": {
                "test_balanced_accuracy_median": robust_balanced,
                "test_balanced_accuracy_min": min(balanced_values),
                "test_balanced_accuracy_max": max(balanced_values),
                "test_accuracy_lift_median": robust_lift,
            },
            "ensemble_test_diagnostic": ensemble,
            "quality_gate": quality_gate,
            "deployment_gate": deployment_gate,
            "note": "研究閘門通過不等於可營利；實盤前仍須完成 SAC、成本回測與紙上交易。",
        }
        _json_write(SUMMARY_PATH, report)
        print("FIVE_SEED_TRANSFORMER_V3_COMPLETE", flush=True)
        print(json.dumps(report["quality_gate"], ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "status": "failed",
            "duration_seconds": monotonic() - started,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _json_write(SUMMARY_PATH, failure)
        print(failure["traceback"], flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
