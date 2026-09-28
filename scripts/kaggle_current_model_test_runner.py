"""在 Kaggle T4 驗證目前九週期 Transformer V3 + SAC 完整管線。"""

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
TEMP_ROOT = Path("/tmp/ai_quant_current_model_test")
PROGRESS_PATH = WORKING_ROOT / "current_model_test_progress.json"
SUMMARY_PATH = WORKING_ROOT / "current_model_test_summary.json"
PORTABLE_PATH = WORKING_ROOT / "current_model_test_models.zip"
SEED = 11


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare_bundle(destination: Path) -> Path:
    archives = list(Path("/kaggle/input").rglob("current_model_test_bundle.zip"))
    if len(archives) == 1:
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True)
        with ZipFile(archives[0]) as handle:
            handle.extractall(destination)
        return destination
    if len(archives) > 1:
        raise FileNotFoundError(f"找到多個測試包：{len(archives)} 個")

    # Kaggle Dataset 會自動展開 ZIP；此時直接使用 manifest 所在目錄。
    manifests = list(
        Path("/kaggle/input").rglob("current_model_test_manifest.json")
    )
    if len(manifests) != 1:
        raise FileNotFoundError(
            f"預期一個已展開測試資料集，實際找到 {len(manifests)} 個"
        )
    return manifests[0].parent


def _install_dependencies(root: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-index",
            "--find-links",
            str(root / "wheelhouse"),
            "stable-baselines3==2.7.1",
            "gymnasium==1.2.3",
        ],
        check=True,
    )


def _verify_inputs(root: Path, manifest: dict[str, object]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for interval, raw in dict(manifest["files"]).items():
        item = dict(raw)
        path = root / str(item["file"])
        if not path.is_file() or _sha256(path) != str(item["sha256"]):
            raise RuntimeError(f"{interval} 行情檔 SHA-256 驗證失敗")
        paths[str(interval)] = path
    return paths


def _archive_outputs(transformer_dir: Path, research: object) -> None:
    with ZipFile(PORTABLE_PATH, "w", ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(transformer_dir.rglob("*")):
            if path.is_file() and path.name in {
                "best_model.pt", "training.json", "history.csv",
                "test_predictions.csv", "artifact_manifest.json",
            }:
                archive.write(path, f"transformer/{path.name}")
        for path in (
            Path(str(research.summary_json)),
            Path(str(research.runs_csv)),
            SUMMARY_PATH,
        ):
            if path.is_file():
                archive.write(path, f"reports/{path.name}")
        for index, run_dir in enumerate(research.run_dirs, start=1):
            run = Path(str(run_dir))
            for name in (
                "training.json", "final_model.zip", "best_model.zip",
                "validation_evaluation.csv", "test_evaluation.csv",
            ):
                path = run / name
                if path.is_file():
                    archive.write(path, f"sac/run_{index:02d}/{name}")


def main() -> int:
    started = monotonic()
    try:
        TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        root = _prepare_bundle(TEMP_ROOT / "bundle")
        _install_dependencies(root)
        sys.path.insert(0, str(root / "src"))

        import stable_baselines3
        import torch

        from ai_quant_trading.features.multitimeframe import (
            build_multitimeframe_frame,
        )
        from ai_quant_trading.reinforcement_learning.config import (
            PortfolioEnvConfig,
            RLSplitConfig,
        )
        from ai_quant_trading.reinforcement_learning.dataset import (
            prepare_rl_dataset,
        )
        from ai_quant_trading.reinforcement_learning.experiments import (
            RLResearchConfig,
            run_rl_research_experiment,
        )
        from ai_quant_trading.reinforcement_learning.feature_contract import (
            attach_expected_return,
            build_expected_return_contract,
        )
        from ai_quant_trading.reinforcement_learning.storage import (
            save_rl_environment,
        )
        from ai_quant_trading.reinforcement_learning.training_config import (
            RLTrainingConfig,
        )
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
        from ai_quant_trading.transformer.training import (
            train_temporal_transformer,
        )

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
        print("測試硬體：", json.dumps(hardware, ensure_ascii=False), flush=True)

        manifest = json.loads(
            (root / "current_model_test_manifest.json").read_text(encoding="utf-8")
        )
        paths = _verify_inputs(root, manifest)
        _write_json(
            PROGRESS_PATH,
            {"status": "running", "stage": "nine_timeframe_features", "progress": 0.03},
        )
        frames = {
            interval: pd.read_csv(path, low_memory=False)
            for interval, path in paths.items()
        }
        fused, coverage = build_multitimeframe_frame(frames, "15m")
        del frames
        gc.collect()
        timestamps = pd.to_datetime(fused["timestamp"], utc=True, errors="coerce")
        research_start = pd.Timestamp(str(manifest["research_start_at"]))
        fused = fused.loc[timestamps >= research_start].reset_index(drop=True)
        if len(fused) < 25_000:
            raise RuntimeError(f"九週期融合後只有 {len(fused):,} 根 15m K 線")
        if any(value < 0.95 for value in coverage.values()):
            raise RuntimeError(f"九週期覆蓋率不足：{coverage}")
        source = TEMP_ROOT / "btc_15m_nine_timeframe.csv"
        fused.to_csv(source, index=False, encoding="utf-8")
        print(
            f"九週期資料：{len(fused):,} 根，欄位 {len(fused.columns):,}，"
            f"覆蓋率最低 {min(coverage.values()):.2%}",
            flush=True,
        )
        del fused
        gc.collect()

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
        training_config = TransformerTrainingConfig(
            epochs=8,
            batch_size=96,
            learning_rate=1.2e-4,
            weight_decay=5e-4,
            warmup_ratio=0.10,
            gradient_clip=0.75,
            early_stopping_patience=4,
            mixed_precision=True,
            device="cuda",
            num_workers=2,
            cpu_threads=4,
            train_fraction=0.60,
            validation_fraction=0.20,
            seed=SEED,
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
            direction_focal_gamma=1.0,
            side_class_balance_power=0.25,
            tradeability_class_balance_power=0.50,
            checkpoint_metric="deployment_horizon_skill_score",
            checkpoint_horizon_weights=(0.15, 0.70, 0.15),
            embargo_bars=48,
            recency_half_life_days=365.0,
            checkpoint_loss_penalty=0.02,
        )
        transformer_root = WORKING_ROOT / "current_transformer_v3_test"

        def transformer_progress(payload: dict[str, object]) -> None:
            local = float(payload.get("progress", 0.0))
            _write_json(
                PROGRESS_PATH,
                {
                    **payload,
                    "stage": "transformer",
                    "pipeline_progress": 0.08 + local * 0.42,
                    "elapsed_seconds_total": monotonic() - started,
                },
            )
            if int(local * 100) % 10 == 0:
                print(f"Transformer {local:.0%}", flush=True)

        transformer = train_temporal_transformer(
            [source],
            model_config,
            training_config,
            transformer_root,
            run_name="btc_15m_nine_timeframe_current_test",
            progress_callback=transformer_progress,
        )
        transformer_summary = json.loads(
            transformer.summary_json.read_text(encoding="utf-8")
        )
        gc.collect()
        torch.cuda.empty_cache()

        context_path = TEMP_ROOT / "transformer_context.csv"
        inference = apply_transformer_checkpoint(
            transformer.model_path,
            source,
            output_path=context_path,
            device="cuda",
            batch_size=512,
            mixed_precision=True,
        )
        inferred = pd.read_csv(inference.output_path, low_memory=False)
        provenance = transformer_oos_provenance(transformer.model_path, inferred)
        enriched, feature_columns = build_compact_short_term_frame(
            inferred, maximum_features=112
        )
        if not 80 <= len(feature_columns) <= 120:
            raise RuntimeError(f"SAC 特徵數不符合契約：{len(feature_columns)}")
        if any(name in feature_columns for name in ("open", "high", "low", "close", "volume")):
            raise RuntimeError("SAC 輸入錯誤包含原始 OHLCV")

        timestamps = pd.to_datetime(enriched["timestamp"], utc=True, errors="coerce")
        safe_start = pd.Timestamp(str(provenance["safe_rl_start_exclusive"]))
        oos = enriched.loc[
            timestamps.gt(safe_start)
            & pd.to_numeric(
                enriched["transformer_available"], errors="coerce"
            ).fillna(0.0).eq(1.0)
        ].reset_index(drop=True)
        oos = attach_expected_return(oos, build_expected_return_contract(20))
        if len(oos) < 5_000:
            raise RuntimeError(f"Transformer 樣本外資料只有 {len(oos):,} 列")
        if float(oos["expected_return"].notna().mean()) < 0.99:
            raise RuntimeError("20 根 expected_return 覆蓋率不足")

        rl_dataset = prepare_rl_dataset(
            oos,
            feature_columns,
            RLSplitConfig(0.60, 0.20, min_rows_per_split=500),
        )
        env_config = PortfolioEnvConfig(
            initial_capital=1_000.0,
            fee_rate=0.0004,
            slippage_rate=0.0002,
            max_position_fraction=0.50,
            allow_short=True,
            max_short_fraction=0.50,
            drawdown_penalty=2.0,
            turnover_penalty=0.0002,
            max_drawdown_limit=0.10,
            episode_length=2_016,
            random_start=True,
            holding_period_reference=20,
            include_position_context=True,
            expert_kind="short_term",
            rebalance_deadband=0.01,
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
            neutral_action_threshold=0.0,
            action_semantics="continuous_target",
            hold_action_threshold=0.03,
            close_action_threshold=0.10,
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
            minimum_gross_target_cost_multiple=0.0,
            minimum_net_risk_reward=1.50,
            initial_capital_randomization=0.10,
            slippage_randomization=0.0001,
        )
        environment = save_rl_environment(
            rl_dataset,
            env_config,
            WORKING_ROOT / "current_sac_environment",
            source_path=source,
            transformer_checkpoint=transformer.model_path,
            transformer_provenance=provenance,
        )
        sac_config = RLTrainingConfig(
            algorithm="sac",
            total_timesteps=100_000,
            device="cuda",
            seed=SEED,
            n_envs=1,
            cpu_threads=4,
            learning_rate=1e-4,
            gamma=0.995,
            batch_size=512,
            net_arch=(256, 256, 128),
            activation_fn="relu",
            checkpoint_freq=50_000,
            evaluation_freq=25_000,
            n_eval_episodes=1,
            deterministic_eval=True,
            save_replay_buffer=False,
            sac_buffer_size=400_000,
            sac_learning_starts=20_000,
            sac_tau=0.005,
            sac_train_freq=1,
            sac_gradient_steps=1,
            sac_ent_coef="auto_0.01",
            sac_action_noise="normal",
            sac_action_noise_sigma=0.05,
        )

        def sac_progress(payload: dict[str, object]) -> None:
            local = float(payload.get("progress", 0.0))
            _write_json(
                PROGRESS_PATH,
                {
                    **payload,
                    "stage": "sac",
                    "pipeline_progress": 0.50 + local * 0.50,
                    "elapsed_seconds_total": monotonic() - started,
                },
            )
            percent = int(local * 100)
            if percent % 10 == 0 or payload.get("status") != "running":
                print(f"SAC {percent}%", flush=True)

        research = run_rl_research_experiment(
            environment.run_dir,
            sac_config,
            RLResearchConfig(
                seeds=(SEED,),
                walk_forward_folds=1,
                min_rows_per_split=500,
                final_holdout_fraction=0.10,
                min_final_holdout_rows=300,
            ),
            progress_callback=sac_progress,
        )
        sac_summary = json.loads(research.summary_json.read_text(encoding="utf-8"))
        combined = {
            "schema_version": 1,
            "status": "complete",
            "candidate_only": True,
            "deployment_allowed": False,
            "duration_seconds": monotonic() - started,
            "hardware": hardware,
            "data": {
                "intervals": manifest["intervals"],
                "research_start_at": manifest["research_start_at"],
                "snapshot_end_at": manifest["snapshot_end_at"],
                "coverage": coverage,
                "transformer_rows": int(transformer_summary["sample_counts"]["train"])
                + int(transformer_summary["sample_counts"]["validation"])
                + int(transformer_summary["sample_counts"]["test"]),
                "sac_oos_rows": len(oos),
                "sac_feature_count": len(feature_columns),
                "expected_return_horizon": 20,
            },
            "transformer": transformer_summary,
            "sac": sac_summary,
            "note": "此為 1 seed/1 fold/100k steps 整合測試，不是可部署正式模型。",
        }
        _write_json(SUMMARY_PATH, combined)
        _archive_outputs(transformer.run_dir, research)
        _write_json(
            PROGRESS_PATH,
            {
                "status": "complete",
                "stage": "complete",
                "progress": 1.0,
                "duration_seconds": monotonic() - started,
            },
        )
        print("目前模型端到端測試完成", flush=True)
        return 0
    except Exception as exc:
        _write_json(
            SUMMARY_PATH,
            {
                "schema_version": 1,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "duration_seconds": monotonic() - started,
                "traceback": traceback.format_exc(),
            },
        )
        _write_json(
            PROGRESS_PATH,
            {"status": "failed", "stage": "failed", "error": str(exc)},
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
