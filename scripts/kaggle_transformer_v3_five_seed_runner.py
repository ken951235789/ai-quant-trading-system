"""在 Kaggle T4 依序訓練五個 Transformer V3 seeds 並以驗證集選模。"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from time import monotonic
import traceback
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd


WORKING_ROOT = Path("/kaggle/working")
RESULT_ROOT = WORKING_ROOT / "transformer_v3_five_seed"
PROGRESS_PATH = WORKING_ROOT / "five_seed_progress.json"
SUMMARY_PATH = WORKING_ROOT / "five_seed_summary.json"
SELECTED_ROOT = WORKING_ROOT / "transformer_v31_selected"
PORTABLE_PATH = WORKING_ROOT / "transformer_v34_five_seed_candidate.zip"
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


def _build_selected_archive(selected_run: Path) -> None:
    """保存後續 SAC 所需的唯一候選 checkpoint 與研究摘要。"""
    if SELECTED_ROOT.exists():
        shutil.rmtree(SELECTED_ROOT)
    SELECTED_ROOT.mkdir(parents=True)
    # 完整保留 manifest 引用的檔案，避免只帶權重卻無法驗證成品完整性。
    for source in selected_run.iterdir():
        if source.is_file():
            shutil.copy2(source, SELECTED_ROOT / source.name)
    evidence = WORKING_ROOT / "seed_evidence"
    evidence.mkdir(exist_ok=True)
    for run in RESULT_ROOT.iterdir():
        if not run.is_dir():
            continue
        target = evidence / run.name
        target.mkdir(exist_ok=True)
        for source in run.iterdir():
            if source.name in {"training.json", "history.csv"} or source.name.endswith(("_signal_diagnostics.csv", "_fixed_hold_trades.csv")):
                shutil.copy2(source, target / source.name)
    shutil.copy2(SUMMARY_PATH, SELECTED_ROOT / SUMMARY_PATH.name)
    with ZipFile(PORTABLE_PATH, "w", ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(SELECTED_ROOT.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(SELECTED_ROOT))


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
    from ai_quant_trading.transformer.evaluation import validate_ensemble_labels

    del direction_threshold_bps  # 保留舊呼叫介面，但不再用固定門檻重建答案。
    frames = [
        pd.read_csv(Path(str(run["run_dir"])) / "test_predictions.csv")
        for run in runs
    ]
    validate_ensemble_labels(frames, horizons)

    per_horizon: dict[str, object] = {}
    aggregate_values: dict[str, list[float]] = {
        "accuracy": [],
        "majority_baseline": [],
        "balanced_accuracy": [],
        "macro_f1": [],
    }
    for horizon in horizons:
        reference = frames[0]
        actual = reference[f"actual_direction_{horizon}"].to_numpy(dtype=int)
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
        "label_source": "persisted_actual_direction",
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

        config_fields = set(TemporalTransformerConfig.__dataclass_fields__)
        required_config_fields = {"hierarchical_direction", "horizon_adapter_dim"}
        print(
            "Transformer 設定來源：",
            sys.modules[TemporalTransformerConfig.__module__].__file__,
            flush=True,
        )
        if not required_config_fields.issubset(config_fields):
            raise RuntimeError(
                "Kaggle 掛載到舊版 Transformer 設定；請等待最新 Dataset 完成後再提交"
            )

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
            sequence_length=256,
            d_model=128,
            n_heads=8,
            n_layers=4,
            feedforward_dim=384,
            dropout=0.15,
            latent_dim=24,
            return_horizons=(5, 20, 48),
            regime_classes=3,
            architecture_version=3,
            local_kernel_size=3,
            quantile_levels=(0.10, 0.50, 0.90),
            patch_size=8,
            patch_stride=4,
            volatility_regime_classes=3,
            hierarchical_direction=True,
            horizon_adapter_dim=64,
        )

        RESULT_ROOT.mkdir(parents=True, exist_ok=True)
        runs: list[dict[str, object]] = []
        for seed_index, seed in enumerate(SEEDS, start=1):
            training_config = TransformerTrainingConfig(
                epochs=50,
                batch_size=96,
                learning_rate=1.2e-4,
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
                return_loss_weight=0.50,
                volatility_loss_weight=0.20,
                regime_loss_weight=0.10,
                direction_loss_weight=0.75,
                side_loss_weight=0.90,
                quantile_loss_weight=0.15,
                direction_threshold_bps=12.0,
                movement_threshold_bps=2.0,
                movement_atr_multiplier=0.50,
                label_smoothing=0.01,
                fee_bps_per_side=4.0,
                slippage_bps_per_side=2.0,
                max_observed_spread_bps=50.0,
                edge_loss_weight=0.90,
                excursion_loss_weight=0.20,
                tradeability_loss_weight=0.75,
                volatility_regime_loss_weight=0.10,
                probability_calibration=True,
                direction_class_balance_power=0.25,
                direction_focal_gamma=1.00,
                side_class_balance_power=0.25,
                tradeability_class_balance_power=0.50,
                checkpoint_metric="economic_selection_score",
                checkpoint_horizon_weights=(0.15, 0.70, 0.15),
                training_horizon_weights=(0.15, 0.70, 0.15),
                trading_target_mode="terminal_net",
                calibration_fraction=0.50,
                economic_minimum_trades=50,
                economic_minimum_edge_bps=2.0,
                economic_downside_penalty=0.10,
                checkpoint_loss_penalty=0.02,
                embargo_bars=48,
                recency_half_life_days=365.0,
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
                        f"epoch {payload.get('epoch', 0)}/{payload.get('epochs', 50)} | "
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
            validation_selection_metrics = dict(
                summary.get("validation_selection_metrics", {})
            )
            runs.append(
                {
                    "seed": seed,
                    "run_dir": str(result.run_dir),
                    "best_epoch": int(summary["best_epoch"]),
                    "checkpoint_metric": summary["checkpoint_metric"],
                    "validation_score": float(summary["best_checkpoint_value"]),
                    "selected_validation_loss": float(summary["selected_validation_loss"]),
                    "validation_selection_metrics": validation_selection_metrics,
                    "test_metrics": test_metrics,
                    "data_diagnostics": dict(summary.get("data_diagnostics", {})),
                    "economic_evaluation": summary.get("economic_evaluation", {}),
                    "calibration_protocol": summary.get("calibration_protocol", {}),
                }
            )
            gc.collect()
            torch.cuda.empty_cache()

        primary_horizon = int(model_config.primary_horizon)
        expectancy_key = "fixed_hold_expectancy"
        profit_factor_key = "fixed_hold_profit_factor"
        trades_key = "fixed_hold_trades"

        def validation_selection_key(item: dict[str, object]) -> tuple[float, ...]:
            metrics = dict(item.get("validation_selection_metrics", {}))
            expectancy = float(metrics.get(expectancy_key, -1_000_000_000.0))
            profit_factor = float(metrics.get(profit_factor_key, 0.0))
            trades = float(metrics.get(trades_key, 0.0))
            economic_pass = float(expectancy > 0.0 and profit_factor > 1.0 and trades >= 100)
            return (
                economic_pass,
                float(metrics.get("economic_selection_score", -1.0)),
                expectancy,
                profit_factor,
                float(item["validation_score"]),
            )

        selected = max(runs, key=validation_selection_key)
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
        timing_horizon = int(model_config.timing_horizon)
        regime_horizon = int(model_config.regime_horizon)
        primary_expectancies = [
            float(dict(item["test_metrics"]).get(expectancy_key, -1_000_000_000.0))
            for item in runs
        ]
        primary_profit_factors = [
            float(dict(item["test_metrics"]).get(profit_factor_key, 0.0))
            for item in runs
        ]
        positive_expectancy_ratio = sum(value > 0.0 for value in primary_expectancies) / len(
            primary_expectancies
        )
        profitable_factor_ratio = sum(value > 1.0 for value in primary_profit_factors) / len(
            primary_profit_factors
        )
        selected_validation_metrics = dict(selected["validation_selection_metrics"])
        selected_validation_expectancy = float(
            selected_validation_metrics.get(expectancy_key, -1_000_000_000.0)
        )
        selected_validation_profit_factor = float(
            selected_validation_metrics.get(profit_factor_key, 0.0)
        )
        ensemble = _ensemble_diagnostic(
            runs,
            model_config.return_horizons,
            direction_threshold_bps=12.0,
        )
        quality_gate = {
            "five_seeds_completed": len(runs) == len(SEEDS),
            "all_data_diagnostics_passed": all(
                bool(dict(item.get("data_diagnostics", {})).get("passed", False))
                for item in runs
            ),
            "selected_validation_expectancy_positive": (
                selected_validation_expectancy > 0.0
            ),
            "selected_validation_profit_factor_above_one": (
                selected_validation_profit_factor > 1.0
            ),
            "median_fixed_hold_expectancy_positive": _median(primary_expectancies) > 0.0,
            "median_fixed_hold_profit_factor_above_one": _median(primary_profit_factors) > 1.0,
            "primary_positive_expectancy_seed_ratio_at_least_60pct": (
                positive_expectancy_ratio >= 0.60
            ),
            "primary_profit_factor_seed_ratio_at_least_60pct": (
                profitable_factor_ratio >= 0.60
            ),
            "selected_primary_selective_expectancy_positive": (
                float(selected_metrics.get(expectancy_key, -1_000_000_000.0)) > 0.0
            ),
            "selected_primary_selective_profit_factor_above_one": (
                float(selected_metrics.get(profit_factor_key, 0.0)) > 1.0
            ),
        }
        quality_gate["passed"] = all(quality_gate.values())
        validation_gate = {
            "at_least_60pct_positive_validation_seeds": sum(
                float(dict(run["validation_selection_metrics"]).get(expectancy_key, 0)) > 0
                for run in runs
            ) / len(runs) >= 0.60,
            "selected_validation_ci_positive": float(selected_validation_metrics.get("fixed_hold_expectancy_ci_low", -1)) > 0,
            "selected_validation_enough_trades": float(selected_validation_metrics.get(trades_key, 0)) >= 100,
            "selected_validation_stress_positive": float(dict(dict(selected.get("economic_evaluation", {})).get("validation", {})).get("extra_one_way_cost", {}).get("fixed_hold_expectancy", -1)) > 0,
        }
        validation_gate["passed"] = all(validation_gate.values())
        deployment_gate = {
            "selected_primary_high_confidence_expectancy_positive": (
                float(selected_metrics.get(expectancy_key, -1_000_000_000.0)) > 0.0
            ),
            "selected_primary_high_confidence_profit_factor_above_one": (
                float(selected_metrics.get(profit_factor_key, 0.0)) > 1.0
            ),
            "full_execution_backtest_completed": False,
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
            "selection_basis": {
                "source": "validation_only",
                "primary_horizon": primary_horizon,
                "method": "fixed_hold_net_edge_lower_confidence_bound",
                "minimum_edge_bps": 2.0,
                "downside_penalty": 0.10,
                "expectancy": selected_validation_expectancy,
                "profit_factor": selected_validation_profit_factor,
                "minimum_trades": 100,
                "final_holdout_used": False,
            },
            "selected_per_horizon": per_horizon,
            "robustness": {
                "test_balanced_accuracy_median": robust_balanced,
                "test_balanced_accuracy_min": min(balanced_values),
                "test_balanced_accuracy_max": max(balanced_values),
                "test_accuracy_lift_median": robust_lift,
                "fixed_hold_expectancy_median": _median(primary_expectancies),
                "fixed_hold_expectancy_mean": float(np.mean(primary_expectancies)),
                "fixed_hold_expectancy_std": float(np.std(primary_expectancies)),
                "fixed_hold_expectancy_worst": min(primary_expectancies),
                "fixed_hold_profit_factor_median": _median(
                    primary_profit_factors
                ),
                "primary_positive_expectancy_seed_ratio": positive_expectancy_ratio,
                "primary_profit_factor_above_one_seed_ratio": profitable_factor_ratio,
                "timing_balanced_accuracy_selected": per_horizon[str(timing_horizon)][
                    "balanced_accuracy"
                ],
                "regime_balanced_accuracy_selected": per_horizon[str(regime_horizon)][
                    "balanced_accuracy"
                ],
            },
            "ensemble_test_diagnostic": ensemble,
            "evaluation_protocol": {
                "target": "terminal_net",
                "split": "train_60_calibration_10_selection_10_research_test_20_with_purge_embargo",
                "test_role": "previously_inspected_research_test_not_pristine_final_holdout",
                "final_holdout_required": "new_unseen_data_after_research_freeze",
                "fixed_hold_scope": "1x non-overlapping terminal close; closed-equity drawdown only; estimated funding",
            },
            "quality_gate": quality_gate,
            "validation_gate": validation_gate,
            "deployment_gate": deployment_gate,
            "note": "研究閘門通過不等於可營利；實盤前仍須完成 SAC、成本回測與紙上交易。",
        }
        _json_write(SUMMARY_PATH, report)
        _build_selected_archive(Path(str(selected["run_dir"])))
        # Kaggle 輸出保留單一候選模型；各 seed 指標已完整寫入 summary。
        shutil.rmtree(RESULT_ROOT, ignore_errors=True)
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
