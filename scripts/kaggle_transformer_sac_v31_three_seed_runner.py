"""在 Kaggle T4 完成三 seed Transformer V3.1 與 SAC 樣本外研究。"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import monotonic
import traceback
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd


WORKING_ROOT = Path("/kaggle/working")
TEMP_ROOT = Path("/tmp/ai_quant_transformer_sac_v31")
TRANSFORMER_ROOT = WORKING_ROOT / "transformer_v31_three_seed"
RL_ROOT = WORKING_ROOT / "sac_v31_research"
PROGRESS_PATH = WORKING_ROOT / "transformer_sac_v31_progress.json"
SUMMARY_PATH = WORKING_ROOT / "transformer_sac_v31_summary.json"
PORTABLE_PATH = WORKING_ROOT / "transformer_sac_v31_models.zip"
SEEDS = (11, 42, 101)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _install_dependencies(root: Path) -> None:
    """Kaggle 映像通常已有套件；固定版本可避免環境漂移。"""
    wheelhouse = root / "wheelhouse"
    if not wheelhouse.is_dir():
        raise FileNotFoundError("正式資料包缺少離線 Python wheelhouse")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-index",
            "--find-links",
            str(wheelhouse),
            "stable-baselines3==2.7.1",
            "gymnasium==1.2.3",
        ],
        check=True,
    )


def _prepare_bundle_root(destination: Path) -> Path:
    archives = list(Path("/kaggle/input").rglob("transformer_v3_formal_bundle.zip"))
    if archives:
        destination.mkdir(parents=True, exist_ok=True)
        with ZipFile(archives[0]) as handle:
            handle.extractall(destination)
        return destination
    manifests = list(Path("/kaggle/input").rglob("formal_manifest.json"))
    if not manifests:
        raise FileNotFoundError("Kaggle Input 找不到正式訓練資料包")
    return manifests[0].parent


def _copy_to_archive(
    archive: ZipFile,
    path: Path,
    archive_root: str,
) -> None:
    if path.is_file():
        archive.write(path, f"{archive_root}/{path.name}")
        return
    for child in sorted(path.rglob("*")):
        if child.is_file():
            relative = child.relative_to(path).as_posix()
            archive.write(child, f"{archive_root}/{relative}")


def _build_portable_archive(
    selected_transformer: Path,
    transformer_report: Path,
    research_result: object,
    combined_report: Path,
) -> None:
    """只保存模型與評估，不重複打包數百 MB 的訓練 CSV。"""
    with ZipFile(PORTABLE_PATH, "w", ZIP_DEFLATED, compresslevel=6) as archive:
        _copy_to_archive(
            archive,
            selected_transformer,
            "transformer_selected",
        )
        _copy_to_archive(archive, transformer_report, "reports")
        _copy_to_archive(archive, combined_report, "reports")
        summary_json = Path(str(research_result.summary_json))
        runs_csv = Path(str(research_result.runs_csv))
        _copy_to_archive(archive, summary_json, "sac_research")
        _copy_to_archive(archive, runs_csv, "sac_research")
        holdout_evaluation = Path(str(research_result.experiment_dir)) / (
            "final_holdout_evaluation.csv"
        )
        if holdout_evaluation.is_file():
            _copy_to_archive(archive, holdout_evaluation, "sac_research")
        for index, run_dir in enumerate(research_result.run_dirs, start=1):
            run = Path(str(run_dir))
            for name in (
                "training.json",
                "progress.json",
                "best_model/best_model.zip",
                "best_model.zip",
                "final_model.zip",
                "validation_evaluation.csv",
                "test_evaluation.csv",
            ):
                candidate = run / name
                if candidate.is_file():
                    _copy_to_archive(archive, candidate, f"sac_runs/run_{index:02d}")


def main() -> int:
    started = monotonic()
    try:
        root = _prepare_bundle_root(TEMP_ROOT / "bundle")
        _install_dependencies(root)
        sys.path.insert(0, str(root / "src"))

        import stable_baselines3
        import torch

        from ai_quant_trading.reinforcement_learning.config import (
            PortfolioEnvConfig,
            RLSplitConfig,
        )
        from ai_quant_trading.reinforcement_learning.dataset import prepare_rl_dataset
        from ai_quant_trading.reinforcement_learning.experiments import (
            RLResearchConfig,
            run_rl_research_experiment,
        )
        from ai_quant_trading.reinforcement_learning.storage import save_rl_environment
        from ai_quant_trading.reinforcement_learning.training_config import RLTrainingConfig
        from ai_quant_trading.reinforcement_learning.universal import (
            build_compact_short_term_frame,
        )
        from ai_quant_trading.transformer.config import (
            TemporalTransformerConfig,
            TransformerTrainingConfig,
        )
        from ai_quant_trading.transformer.inference import (
            apply_transformer_checkpoint,
            transformer_oos_provenance,
        )
        from ai_quant_trading.transformer.training import train_temporal_transformer

        if not torch.cuda.is_available():
            raise RuntimeError("Kaggle Session 沒有可用 GPU")
        hardware = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
            "cpu_count": os.cpu_count(),
        }
        print("訓練硬體：", json.dumps(hardware, ensure_ascii=False), flush=True)

        manifest = json.loads((root / "formal_manifest.json").read_text(encoding="utf-8"))
        source = root / "data" / "transformer_v3_formal.csv"
        expected_hash = str(dict(manifest["data"])["sha256"])
        if _sha256(source) != expected_hash:
            raise RuntimeError("正式訓練資料 SHA-256 驗證失敗")

        model_config = TemporalTransformerConfig(
            input_features=256,
            sequence_length=256,
            d_model=128,
            n_heads=8,
            n_layers=4,
            feedforward_dim=384,
            dropout=0.15,
            latent_dim=24,
            return_horizons=(1, 5, 20),
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
        TRANSFORMER_ROOT.mkdir(parents=True, exist_ok=True)
        transformer_runs: list[dict[str, object]] = []
        for seed_index, seed in enumerate(SEEDS, start=1):
            training_config = TransformerTrainingConfig(
                epochs=50,
                batch_size=96,
                learning_rate=1.2e-4,
                weight_decay=5e-4,
                warmup_ratio=0.10,
                gradient_clip=0.75,
                early_stopping_patience=8,
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
                movement_atr_multiplier=0.10,
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
                direction_focal_gamma=1.0,
                side_class_balance_power=0.25,
                tradeability_class_balance_power=0.50,
                checkpoint_metric="hierarchical_skill_score",
                checkpoint_loss_penalty=0.02,
            )
            last_percent = -1

            def transformer_progress(payload: dict[str, object]) -> None:
                nonlocal last_percent
                local_progress = float(payload.get("progress", 0.0))
                global_progress = ((seed_index - 1) + local_progress) / len(SEEDS)
                event = {
                    **payload,
                    "stage": "transformer",
                    "seed": seed,
                    "seed_index": seed_index,
                    "seed_count": len(SEEDS),
                    "pipeline_progress": global_progress * 0.45,
                    "elapsed_seconds_total": monotonic() - started,
                }
                _write_json(PROGRESS_PATH, event)
                percent = int(local_progress * 100)
                if percent >= last_percent + 10 or payload.get("status") in {
                    "preparing",
                    "validating",
                    "complete",
                }:
                    last_percent = max(last_percent, percent)
                    metrics = dict(payload.get("metrics", {}))
                    print(
                        f"Transformer seed {seed} [{seed_index}/3] {percent}% | "
                        f"hierarchical_skill="
                        f"{float(metrics.get('validation_hierarchical_skill_score', 0.0)):.4f}",
                        flush=True,
                    )

            result = train_temporal_transformer(
                [source],
                model_config,
                training_config,
                TRANSFORMER_ROOT,
                run_name=f"btc_15m_mtf_v31_seed{seed}",
                progress_callback=transformer_progress,
            )
            training_summary = json.loads(result.summary_json.read_text(encoding="utf-8"))
            transformer_runs.append(
                {
                    "seed": seed,
                    "run_dir": str(result.run_dir),
                    "model_path": str(result.model_path),
                    "best_epoch": int(training_summary["best_epoch"]),
                    "validation_score": float(training_summary["best_checkpoint_value"]),
                    "selected_validation_loss": float(
                        training_summary["selected_validation_loss"]
                    ),
                    "test_metrics": dict(training_summary["test_metrics"]),
                }
            )
            gc.collect()
            torch.cuda.empty_cache()

        selected = max(
            transformer_runs,
            key=lambda item: float(item["validation_score"]),
        )
        selected_checkpoint = Path(str(selected["model_path"]))
        transformer_report_path = WORKING_ROOT / "transformer_v31_three_seed_summary.json"
        transformer_report = {
            "schema_version": 2,
            "status": "complete",
            "candidate_only": True,
            "selection_uses_validation_only": True,
            "test_not_used_for_seed_selection": True,
            "model_config": model_config.to_dict(),
            "seeds": list(SEEDS),
            "runs": transformer_runs,
            "selected_seed": int(selected["seed"]),
            "selected_model": str(selected_checkpoint),
        }
        _write_json(transformer_report_path, transformer_report)

        context_path = TEMP_ROOT / "selected_transformer_context.csv"
        inference = apply_transformer_checkpoint(
            selected_checkpoint,
            source,
            output_path=context_path,
            device="cuda",
            batch_size=512,
            mixed_precision=True,
        )
        inferred = pd.read_csv(inference.output_path, low_memory=False)
        provenance = transformer_oos_provenance(selected_checkpoint, inferred)
        enriched, feature_columns = build_compact_short_term_frame(inferred)
        timestamps = pd.to_datetime(enriched["timestamp"], utc=True, errors="coerce")
        safe_start = pd.Timestamp(provenance["safe_rl_start_exclusive"])
        oos = enriched.loc[
            timestamps.gt(safe_start)
            & pd.to_numeric(
                enriched["transformer_available"], errors="coerce"
            ).fillna(0.0).eq(1.0)
        ].reset_index(drop=True)
        if len(oos) < 10_000:
            raise RuntimeError(f"Transformer 樣本外資料僅 {len(oos):,} 列，不足以訓練 SAC")
        # 風控閘門需要可交易方向的預期毛報酬，使用 Transformer 的 5 根預測值。
        from ai_quant_trading.reinforcement_learning.feature_contract import (
            attach_expected_return, build_expected_return_contract,
        )
        oos = attach_expected_return(oos, build_expected_return_contract(5))
        if oos["expected_return"].notna().mean() < 0.99:
            raise RuntimeError("Transformer 5 根預期報酬覆蓋率不足，禁止建立 SAC 環境")

        missing_features = [column for column in feature_columns if column not in oos]
        if missing_features:
            raise RuntimeError(f"SAC 特徵契約缺少欄位：{missing_features}")
        rl_dataset = prepare_rl_dataset(
            oos,
            feature_columns,
            RLSplitConfig(0.60, 0.20, min_rows_per_split=1_000),
        )
        env_config = PortfolioEnvConfig(
            initial_capital=1_000.0,
            fee_rate=0.0004,
            slippage_rate=0.0002,
            max_position_fraction=0.50,
            allow_short=True,
            max_short_fraction=0.50,
            drawdown_penalty=2.0,
            turnover_penalty=0.01,
            max_drawdown_limit=0.10,
            episode_length=2_016,
            random_start=True,
            holding_period_reference=5,
            include_position_context=True,
            expert_kind="short_term",
            # deadband 使用實際持倉比例；2% 可抑制小調倉，又不會封死 10% 以上訊號。
            rebalance_deadband=0.02,
            minimum_holding_bars=4,
            soft_drawdown_limit=0.06,
            soft_drawdown_multiplier=0.50,
            drawdown_curve_exponent=2.0,
            downside_penalty=0.75,
            concentration_penalty=0.005,
            risk_termination_penalty=2.0,
            execution_mode="perpetual",
            normalized_action_space=True,
            include_risk_context=True,
            include_trade_plan_context=True,
            neutral_action_threshold=0.10,
            leverage=2.0,
            max_leverage=3.0,
            max_margin_fraction=0.20,
            risk_per_trade=0.005,
            disaster_stop_atr_multiplier=1.5,
            minimum_stop_distance=0.003,
            maximum_stop_distance=0.0075,
            take_profit_distance=0.015,
            daily_loss_limit=0.015,
            max_consecutive_losses=3,
            max_spread_bps=15.0,
            max_slippage_bps=10.0,
            max_atr_fraction=0.05,
            # SAC 必須在訓練期實際承擔成本，不能先由分析師訊號把探索全部封死。
            minimum_gross_target_cost_multiple=0.0,
            minimum_net_risk_reward=1.50,
            initial_capital_randomization=0.10,
            slippage_randomization=0.0001,
        )
        artifact = save_rl_environment(
            rl_dataset,
            env_config,
            RL_ROOT,
            source_path=source,
            transformer_checkpoint=selected_checkpoint,
            transformer_provenance=provenance,
        )
        print(
            f"SAC 樣本外資料 {len(oos):,} 列、特徵 {len(feature_columns)} 欄",
            flush=True,
        )
        sac_config = RLTrainingConfig(
            algorithm="sac",
            total_timesteps=200_000,
            device="cuda",
            seed=SEEDS[0],
            n_envs=1,
            cpu_threads=4,
            learning_rate=1e-4,
            gamma=0.995,
            batch_size=512,
            net_arch=(256, 256, 128),
            activation_fn="relu",
            checkpoint_freq=100_000,
            evaluation_freq=50_000,
            n_eval_episodes=1,
            deterministic_eval=True,
            save_replay_buffer=False,
            sac_buffer_size=400_000,
            sac_learning_starts=10_000,
            sac_tau=0.005,
            sac_train_freq=1,
            sac_gradient_steps=1,
            sac_ent_coef="auto",
            sac_action_noise="normal",
            sac_action_noise_sigma=0.03,
        )
        last_sac_percent = -1

        def sac_progress(payload: dict[str, object]) -> None:
            nonlocal last_sac_percent
            progress = float(payload.get("progress", 0.0))
            event = {
                **payload,
                "stage": "sac",
                "pipeline_progress": 0.45 + progress * 0.55,
                "elapsed_seconds_total": monotonic() - started,
            }
            _write_json(PROGRESS_PATH, event)
            percent = int(progress * 100)
            if percent >= last_sac_percent + 5 or payload.get("status") != "running":
                last_sac_percent = max(last_sac_percent, percent)
                metrics = dict(payload.get("metrics", {}))
                print(
                    f"SAC run {payload.get('experiment_run', 0)}/"
                    f"{payload.get('experiment_runs', 0)} {percent}% | "
                    f"{float(metrics.get('steps_per_second', 0.0)):.1f} steps/s",
                    flush=True,
                )

        research = run_rl_research_experiment(
            artifact.run_dir,
            sac_config,
            RLResearchConfig(
                seeds=SEEDS,
                walk_forward_folds=2,
                min_rows_per_split=1_000,
                final_holdout_fraction=0.10,
                min_final_holdout_rows=2_000,
            ),
            progress_callback=sac_progress,
        )
        sac_summary = json.loads(research.summary_json.read_text(encoding="utf-8"))
        combined = {
            "schema_version": 2,
            "status": "complete",
            "candidate_only": True,
            "duration_seconds": monotonic() - started,
            "hardware": hardware,
            "data_manifest": manifest,
            "transformer": transformer_report,
            "transformer_oos": {
                "rows": len(oos),
                "feature_count_for_sac": len(feature_columns),
                "safe_start_exclusive": provenance["safe_rl_start_exclusive"],
                "end": str(oos.iloc[-1]["timestamp"]),
            },
            "sac": sac_summary,
            "deployment_allowed": bool(sac_summary.get("eligible", False)),
            "note": "候選模型只有通過 final holdout 仍不足以直接實盤，尚需紙上交易與 Testnet。",
        }
        _write_json(SUMMARY_PATH, combined)
        _build_portable_archive(
            Path(str(selected["run_dir"])),
            transformer_report_path,
            research,
            SUMMARY_PATH,
        )
        # Kaggle 輸出只留可攜模型與摘要，避免重複下載大型 CSV。
        shutil.rmtree(RL_ROOT, ignore_errors=True)
        for item in transformer_runs:
            run_dir = Path(str(item["run_dir"]))
            if run_dir != Path(str(selected["run_dir"])):
                shutil.rmtree(run_dir, ignore_errors=True)
        context_path.unlink(missing_ok=True)
        print("TRANSFORMER_SAC_V31_THREE_SEED_COMPLETE", flush=True)
        return 0
    except Exception as exc:
        failure = {
            "schema_version": 2,
            "status": "failed",
            "duration_seconds": monotonic() - started,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(SUMMARY_PATH, failure)
        print(failure["traceback"], flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
