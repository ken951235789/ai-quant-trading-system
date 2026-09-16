"""在 Kaggle GPU Notebook 量測本專案 Transformer V3 與 SAC 訓練速度。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter
from zipfile import ZipFile


def _install_dependencies() -> float:
    started = perf_counter()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "stable-baselines3==2.7.1",
            "gymnasium>=1.1,<2.0",
        ],
        check=True,
    )
    return perf_counter() - started


def _prepare_bundle_root(destination: Path) -> Path:
    candidates = list(Path("/kaggle/input").rglob("benchmark_bundle.zip"))
    if not candidates:
        candidates = list(Path.cwd().rglob("benchmark_bundle.zip"))
    if candidates:
        with ZipFile(candidates[0]) as handle:
            handle.extractall(destination)
        return destination

    # Kaggle 有時會在建立 Dataset 時自動解開 ZIP，因此也支援已展開的目錄。
    manifests = list(Path("/kaggle/input").rglob("benchmark_manifest.json"))
    if not manifests:
        raise FileNotFoundError("Kaggle Input 找不到 benchmark_bundle.zip 或 benchmark_manifest.json")
    return manifests[0].parent


def _run_transformer(root: Path, output: Path) -> dict[str, object]:
    from ai_quant_trading.transformer.config import (
        TemporalTransformerConfig,
        TransformerTrainingConfig,
    )
    from ai_quant_trading.transformer.training import train_temporal_transformer

    events: list[dict[str, object]] = []

    def progress(payload: dict[str, object]) -> None:
        events.append(dict(payload))
        progress_value = float(payload.get("progress", 0.0))
        if payload.get("status") in {"preparing", "validating", "complete"}:
            print(
                f"Transformer {payload.get('status')} "
                f"{progress_value:.0%}，耗時 {float(payload.get('elapsed_seconds', 0)):.1f}s"
            )

    model_config = TemporalTransformerConfig(
        input_features=256,
        sequence_length=96,
        d_model=96,
        n_heads=4,
        n_layers=3,
        feedforward_dim=192,
        dropout=0.10,
        latent_dim=16,
        return_horizons=(1, 5, 20),
        architecture_version=3,
        patch_size=4,
        patch_stride=2,
    )
    training_config = TransformerTrainingConfig(
        epochs=1,
        batch_size=128,
        learning_rate=3e-4,
        weight_decay=1e-4,
        early_stopping_patience=1,
        mixed_precision=True,
        device="cuda",
        num_workers=2,
        cpu_threads=4,
        probability_calibration=True,
    )
    result = train_temporal_transformer(
        [root / "data" / "transformer_benchmark.csv"],
        model_config,
        training_config,
        output / "transformer",
        run_name="kaggle_speed",
        progress_callback=progress,
    )
    validation_events = [event for event in events if event.get("status") == "validating"]
    epoch_seconds = (
        float(validation_events[-1].get("elapsed_seconds", result.duration_seconds))
        if validation_events
        else result.duration_seconds
    )
    summary = json.loads(result.summary_json.read_text(encoding="utf-8"))
    return {
        "duration_seconds": result.duration_seconds,
        "epoch_seconds_including_prepare_validation": epoch_seconds,
        "device": result.resolved_device,
        "parameters": int(summary.get("trainable_parameters", 0)),
        "sample_counts": summary.get("sample_counts", {}),
        "feature_count": len(summary.get("feature_columns", [])),
        "peak_gpu_memory_gb": max(
            (
                float(dict(event.get("metrics", {})).get("gpu_memory_gb", 0.0))
                for event in events
            ),
            default=0.0,
        ),
    }


def _run_sac(environment: Path, device: str, steps: int) -> dict[str, object]:
    from ai_quant_trading.reinforcement_learning.training import train_rl_agent
    from ai_quant_trading.reinforcement_learning.training_config import RLTrainingConfig

    events: list[dict[str, object]] = []

    def progress(payload: dict[str, object]) -> None:
        events.append(dict(payload))
        value = float(payload.get("progress", 0.0))
        if value >= 1.0 or len(events) % 20 == 0:
            metrics = dict(payload.get("metrics", {}))
            print(
                f"SAC {device} {value:.0%}，"
                f"{float(metrics.get('steps_per_second', 0.0)):.1f} steps/s"
            )

    config = RLTrainingConfig(
        algorithm="sac",
        total_timesteps=steps,
        device=device,
        seed=42,
        n_envs=1,
        cpu_threads=4,
        learning_rate=1e-4,
        gamma=0.995,
        batch_size=256,
        net_arch=(256, 256),
        checkpoint_freq=0,
        evaluation_freq=0,
        save_replay_buffer=False,
        sac_buffer_size=max(20_000, steps + 1),
        sac_learning_starts=1_000,
        sac_train_freq=1,
        sac_gradient_steps=1,
        sac_ent_coef="auto",
        sac_action_noise="normal",
        sac_action_noise_sigma=0.05,
    )
    result = train_rl_agent(
        environment,
        config,
        progress_callback=progress,
    )
    running = [event for event in events if event.get("status") == "running"]
    training_seconds = (
        float(running[-1].get("elapsed_seconds", result.duration_seconds))
        if running
        else result.duration_seconds
    )
    return {
        "device": result.resolved_device,
        "steps": steps,
        "training_seconds": training_seconds,
        "total_seconds_with_evaluation": result.duration_seconds,
        "steps_per_second": steps / max(training_seconds, 1e-9),
        "test_total_return": float(result.metrics["test"]["total_return"]),
    }


def _hour(seconds: float) -> float:
    return seconds / 3_600.0


def main() -> int:
    output = Path("/tmp/ai_quant_benchmark_output")
    output.mkdir(parents=True, exist_ok=True)
    install_seconds = _install_dependencies()
    extracted = Path("/tmp/ai_quant_benchmark_bundle")
    extracted = _prepare_bundle_root(extracted)
    sys.path.insert(0, str(extracted / "src"))

    import stable_baselines3
    import torch

    manifest = json.loads(
        (extracted / "benchmark_manifest.json").read_text(encoding="utf-8")
    )
    hardware = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
        "cpu_count": os.cpu_count(),
    }
    print("硬體：", json.dumps(hardware, ensure_ascii=False))
    if not torch.cuda.is_available():
        raise RuntimeError("此 Kaggle Session 沒有 GPU，請確認 kernel-metadata enable_gpu=true")

    transformer = _run_transformer(extracted, output)
    rl_environment = Path("/tmp/ai_quant_rl_environment")
    shutil.copytree(extracted / "rl_environment", rl_environment, dirs_exist_ok=True)
    sac_cuda = _run_sac(rl_environment, "cuda", 10_000)
    sac_cpu = _run_sac(rl_environment, "cpu", 10_000)

    row_ratio = float(manifest["transformer_formal_rows"]) / float(
        manifest["transformer_benchmark_rows"]
    )
    epoch_seconds = float(transformer["epoch_seconds_including_prepare_validation"])
    transformer_hours_20 = _hour(epoch_seconds * row_ratio * 20)
    transformer_hours_50 = _hour(epoch_seconds * row_ratio * 50)
    fastest_sac = max(
        (sac_cuda, sac_cpu),
        key=lambda item: float(item["steps_per_second"]),
    )
    sac_one_run_hours = _hour(
        1_000_000 / max(float(fastest_sac["steps_per_second"]), 1e-9)
    )
    estimates = {
        "transformer_20_epochs_hours": transformer_hours_20,
        "transformer_50_epochs_hours": transformer_hours_50,
        "sac_fastest_device": fastest_sac["device"],
        "sac_1m_steps_one_run_hours": sac_one_run_hours,
        "sac_1m_steps_15_runs_hours": sac_one_run_hours * 15,
        "fits_single_30h_week_20epoch_plus_one_sac": (
            transformer_hours_20 + sac_one_run_hours <= 30.0
        ),
        "fits_single_30h_week_50epoch_plus_15_sac": (
            transformer_hours_50 + sac_one_run_hours * 15 <= 30.0
        ),
        "note": "線性外推僅供排程；正式時間仍受 early stopping、I/O 與 Kaggle GPU 型號影響。",
    }
    report = {
        "schema_version": 1,
        "install_seconds": install_seconds,
        "hardware": hardware,
        "manifest": manifest,
        "transformer": transformer,
        "sac_cuda": sac_cuda,
        "sac_cpu": sac_cpu,
        "estimates": estimates,
    }
    report_path = Path("/kaggle/working/benchmark_result.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("BENCHMARK_RESULT_START")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("BENCHMARK_RESULT_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
