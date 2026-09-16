"""在 Kaggle 使用既有 Transformer V3.1 重跑修正版 SAC 三 seed 研究。"""

from __future__ import annotations

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
TEMP_ROOT = Path("/tmp/ai_quant_sac_v31_corrected")
RL_ROOT = WORKING_ROOT / "sac_v31_corrected_research"
PROGRESS_PATH = WORKING_ROOT / "sac_v31_corrected_progress.json"
SUMMARY_PATH = WORKING_ROOT / "sac_v31_corrected_summary.json"
PORTABLE_PATH = WORKING_ROOT / "sac_v31_corrected_models.zip"
DEFAULT_SEEDS = (11, 42, 101)
DEFAULT_TOTAL_TIMESTEPS = 200_000
JOB_CONFIG_NAME = "sac_job_config.json"


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


def _prepare_bundle_root(destination: Path) -> Path:
    archives = list(Path("/kaggle/input").rglob("transformer_v3_formal_bundle.zip"))
    if archives:
        destination.mkdir(parents=True, exist_ok=True)
        with ZipFile(archives[0]) as handle:
            handle.extractall(destination)
        return destination
    # Kaggle 以資料夾模式掛載 Dataset 時，直接使用 manifest 所在根目錄。
    manifests = list(Path("/kaggle/input").rglob("formal_manifest.json"))
    if not manifests:
        raise FileNotFoundError("Kaggle Input 找不到正式訓練資料包")
    return manifests[0].parent


def _find_transformer_checkpoint() -> Path:
    checkpoints = list(Path("/kaggle/input").rglob("best_model.pt"))
    if len(checkpoints) != 1:
        raise RuntimeError(
            f"預期恰好一個 Transformer checkpoint，實際找到 {len(checkpoints)} 個"
        )
    checkpoint = checkpoints[0]
    for required in ("training.json", "artifact_manifest.json"):
        if not (checkpoint.parent / required).is_file():
            raise FileNotFoundError(f"Transformer 輸入缺少 {required}")
    return checkpoint


def _load_job_config() -> dict[str, object]:
    """讀取可選的 Kaggle 工作設定；沒有設定檔時維持原始三 seed 篩選。"""
    paths = list(Path("/kaggle/input").rglob(JOB_CONFIG_NAME))
    if len(paths) > 1:
        raise RuntimeError(f"找到多個 {JOB_CONFIG_NAME}，無法判斷應使用哪一份")
    payload: dict[str, object] = {}
    if paths:
        payload = json.loads(paths[0].read_text(encoding="utf-8"))
    seeds = tuple(int(value) for value in payload.get("seeds", DEFAULT_SEEDS))
    total_timesteps = int(payload.get("total_timesteps", DEFAULT_TOTAL_TIMESTEPS))
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("SAC 工作設定的 seeds 必須是不可重複的非負整數")
    if total_timesteps < 100_000:
        raise ValueError("正式 SAC 工作的 total_timesteps 不可低於 100,000")
    return {
        "profile": str(payload.get("profile", "screening_200k")),
        "seeds": seeds,
        "total_timesteps": total_timesteps,
        "walk_forward_folds": int(payload.get("walk_forward_folds", 2)),
    }


def _install_dependencies(root: Path) -> None:
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


def _archive_path(archive: ZipFile, path: Path, root: str) -> None:
    if path.is_file():
        archive.write(path, f"{root}/{path.name}")
        return
    for child in sorted(path.rglob("*")):
        if child.is_file():
            archive.write(child, f"{root}/{child.relative_to(path).as_posix()}")


def _build_portable_archive(
    checkpoint: Path,
    research: object,
    combined_report: Path,
) -> None:
    """保存可重現模型、評估與設定，不重複打包大型市場資料。"""
    with ZipFile(PORTABLE_PATH, "w", ZIP_DEFLATED, compresslevel=6) as archive:
        for name in ("best_model.pt", "training.json", "artifact_manifest.json"):
            _archive_path(archive, checkpoint.parent / name, "transformer_selected")
        _archive_path(archive, combined_report, "reports")
        _archive_path(archive, Path(str(research.summary_json)), "sac_research")
        _archive_path(archive, Path(str(research.runs_csv)), "sac_research")
        holdout = Path(str(research.experiment_dir)) / "final_holdout_evaluation.csv"
        if holdout.is_file():
            _archive_path(archive, holdout, "sac_research")
        for index, run_dir in enumerate(research.run_dirs, start=1):
            run = Path(str(run_dir))
            for name in (
                "training.json",
                "progress.json",
                "artifact_manifest.json",
                "best_model/best_model.zip",
                "best_model.zip",
                "final_model.zip",
                "validation_evaluation.csv",
                "test_evaluation.csv",
            ):
                candidate = run / name
                if candidate.is_file():
                    _archive_path(archive, candidate, f"sac_runs/run_{index:02d}")


def main() -> int:
    started = monotonic()
    try:
        root = _prepare_bundle_root(TEMP_ROOT / "bundle")
        checkpoint = _find_transformer_checkpoint()
        job_config = _load_job_config()
        seeds = tuple(int(value) for value in job_config["seeds"])
        total_timesteps = int(job_config["total_timesteps"])
        walk_forward_folds = int(job_config["walk_forward_folds"])
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
        from ai_quant_trading.transformer.inference import (
            apply_transformer_checkpoint,
            transformer_oos_provenance,
        )
        from ai_quant_trading.reinforcement_learning.feature_contract import (
            attach_expected_return, build_expected_return_contract,
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
        print("訓練硬體：", json.dumps(hardware, ensure_ascii=False), flush=True)
        print("SAC 工作設定：", json.dumps(job_config, ensure_ascii=False), flush=True)

        manifest = json.loads((root / "formal_manifest.json").read_text(encoding="utf-8"))
        source = root / "data" / "transformer_v3_formal.csv"
        expected_hash = str(dict(manifest["data"])["sha256"])
        if _sha256(source) != expected_hash:
            raise RuntimeError("正式訓練資料 SHA-256 驗證失敗")

        _write_json(
            PROGRESS_PATH,
            {
                "status": "running",
                "stage": "transformer_inference",
                "progress": 0.0,
            },
        )
        context_path = TEMP_ROOT / "selected_transformer_context.csv"
        inference = apply_transformer_checkpoint(
            checkpoint,
            source,
            output_path=context_path,
            device="cuda",
            batch_size=512,
            mixed_precision=True,
        )
        inferred = pd.read_csv(inference.output_path, low_memory=False)
        provenance = transformer_oos_provenance(checkpoint, inferred)
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

        # 用 5 根預測報酬評估方向是否足以支付來回成本；此欄只給撮合風控使用。
        oos = attach_expected_return(oos, build_expected_return_contract(5))
        expected_return_coverage = float(oos["expected_return"].notna().mean())
        if expected_return_coverage < 0.99:
            raise RuntimeError(
                f"Transformer 5 根預期報酬覆蓋率僅 {expected_return_coverage:.2%}"
            )

        missing = [column for column in feature_columns if column not in oos]
        if missing:
            raise RuntimeError(f"SAC 特徵契約缺少欄位：{missing}")
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
            # 訓練期必須讓 SAC 實際承受成本並取得 reward；分析師訊號保留為特徵，
            # 不在環境層硬擋探索。正式下單仍由執行期品質閘門保護。
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
            transformer_checkpoint=checkpoint,
            transformer_provenance=provenance,
        )
        print(
            f"SAC 樣本外資料 {len(oos):,} 列、特徵 {len(feature_columns)} 欄、"
            f"預期報酬覆蓋率 {expected_return_coverage:.2%}",
            flush=True,
        )

        sac_config = RLTrainingConfig(
            algorithm="sac",
            total_timesteps=total_timesteps,
            device="cuda",
            seed=seeds[0],
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
        last_percent = -1

        def progress(payload: dict[str, object]) -> None:
            nonlocal last_percent
            value = float(payload.get("progress", 0.0))
            event = {
                **payload,
                "stage": "sac",
                "elapsed_seconds_total": monotonic() - started,
            }
            _write_json(PROGRESS_PATH, event)
            percent = int(value * 100)
            if percent >= last_percent + 5 or payload.get("status") != "running":
                last_percent = max(last_percent, percent)
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
                seeds=seeds,
                walk_forward_folds=walk_forward_folds,
                min_rows_per_split=1_000,
                final_holdout_fraction=0.10,
                min_final_holdout_rows=2_000,
            ),
            progress_callback=progress,
        )
        sac_summary = json.loads(research.summary_json.read_text(encoding="utf-8"))
        combined = {
            "schema_version": 2,
            "status": "complete",
            "candidate_only": True,
            "duration_seconds": monotonic() - started,
            "hardware": hardware,
            "job_config": job_config,
            "data_manifest": manifest,
            "transformer": {
                "checkpoint": str(checkpoint),
                "seed": 11,
                "reused_completed_checkpoint": True,
                "provenance": provenance,
            },
            "transformer_oos": {
                "rows": len(oos),
                "feature_count_for_sac": len(feature_columns),
                "expected_return_coverage": expected_return_coverage,
                "safe_start_exclusive": provenance["safe_rl_start_exclusive"],
                "end": str(oos.iloc[-1]["timestamp"]),
            },
            "corrected_contract": {
                "expected_return": "transformer_return_5（分析情境，不作訓練期硬閘門）",
                "minimum_gross_target_cost_multiple": 0.0,
                "take_profit_distance": 0.015,
                "maximum_stop_distance": 0.0075,
                "neutral_action_threshold": 0.10,
                "rebalance_deadband": 0.02,
                "turnover_penalty": 0.01,
                "previous_failure": (
                    "風控啟用 minimum_net_risk_reward，卻沒有 expected_return 或固定停利，"
                    "導致所有風險增加動作被拒絕。"
                ),
            },
            "sac": sac_summary,
            "deployment_allowed": bool(sac_summary.get("eligible", False)),
            "note": "即使通過 final holdout，仍需紙上交易與 Testnet 才可考慮實盤。",
        }
        _write_json(SUMMARY_PATH, combined)
        _build_portable_archive(checkpoint, research, SUMMARY_PATH)
        shutil.rmtree(RL_ROOT, ignore_errors=True)
        context_path.unlink(missing_ok=True)
        _write_json(
            PROGRESS_PATH,
            {
                "status": "complete",
                "stage": "complete",
                "progress": 1.0,
                "duration_seconds": monotonic() - started,
                "deployment_allowed": combined["deployment_allowed"],
            },
        )
        print("SAC_V31_CORRECTED_THREE_SEED_COMPLETE", flush=True)
        return 0
    except Exception as exc:
        _write_json(
            PROGRESS_PATH,
            {
                "status": "error",
                "stage": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "elapsed_seconds_total": monotonic() - started,
            },
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
