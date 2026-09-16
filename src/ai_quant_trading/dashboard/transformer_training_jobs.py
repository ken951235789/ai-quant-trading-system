"""在 Dashboard 背景執行 Transformer，並把進度持久保存到磁碟。"""

from __future__ import annotations

import csv
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from threading import RLock
from time import sleep
import traceback
from typing import Any, Callable, Sequence
from uuid import uuid4

from ai_quant_trading.transformer import (
    TemporalTransformerConfig,
    TransformerTrainingConfig,
    TransformerTrainingResult,
    train_temporal_transformer,
)
from ai_quant_trading.dashboard.model_job_lock import MODEL_JOB_LOCK


ACTIVE_TRANSFORMER_JOB_STATUSES = {"queued", "running"}
HISTORY_COLUMNS = (
    "updated_at",
    "progress",
    "epoch",
    "epochs",
    "batch",
    "batches",
    "train_loss",
    "return_loss",
    "volatility_loss",
    "regime_loss",
    "direction_loss",
    "quantile_loss",
    "validation_loss",
    "validation_regime_accuracy",
    "validation_cost_aware_direction_accuracy",
    "test_loss",
    "test_regime_accuracy",
    "direction_accuracy",
    "learning_rate",
    "gpu_memory_gb",
)


@dataclass(frozen=True, slots=True)
class TransformerTrainingJob:
    """可在 Streamlit 重跑後重新接回的 Transformer 工作。"""

    job_dir: Path
    metadata_json: Path
    progress_json: Path
    history_csv: Path
    payload: dict[str, Any]

    @property
    def job_id(self) -> str:
        return self.job_dir.name

    @property
    def status(self) -> str:
        return str(self.payload.get("status", "failed"))


TransformerRunner = Callable[..., TransformerTrainingResult]

_LOCK = RLock()
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="transformer-training")
_FUTURES: dict[str, Future[None]] = {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _jobs_root(transformer_root: str | Path) -> Path:
    return Path(transformer_root).resolve() / "training_jobs"


def _models_root(transformer_root: str | Path) -> Path:
    return Path(transformer_root).resolve() / "models"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """以原子替換避免 UI 與 callback 同時讀寫到半份 JSON。"""
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            for attempt in range(10):
                try:
                    temporary.replace(path)
                    break
                except PermissionError:
                    if attempt == 9:
                        raise
                    sleep(0.01 * (attempt + 1))
        finally:
            temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _job_from_dir(job_dir: Path) -> TransformerTrainingJob:
    metadata_json = job_dir / "job.json"
    return TransformerTrainingJob(
        job_dir=job_dir,
        metadata_json=metadata_json,
        progress_json=job_dir / "progress.json",
        history_csv=job_dir / "history.csv",
        payload=_read_json(metadata_json),
    )


def _process_is_alive(process_id: int) -> bool:
    if process_id <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(0x1000, False, process_id)
        if not handle:
            return False
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == 259
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(process_id, 0)
    except (OSError, ValueError):
        return False
    return True


def _mark_stale_job(job: TransformerTrainingJob) -> TransformerTrainingJob:
    if job.status not in ACTIVE_TRANSFORMER_JOB_STATUSES:
        return job
    if _process_is_alive(int(job.payload.get("process_id", -1))):
        return job
    payload = {
        **job.payload,
        "status": "interrupted",
        "completed_at": _utc_now(),
        "error": "應用程式已重新啟動，上一個 Transformer 訓練已中斷。",
    }
    _write_json(job.metadata_json, payload)
    return TransformerTrainingJob(
        job.job_dir,
        job.metadata_json,
        job.progress_json,
        job.history_csv,
        payload,
    )


def list_transformer_training_jobs(
    transformer_root: str | Path,
) -> list[TransformerTrainingJob]:
    """列出工作紀錄，執行中工作優先由 UI 顯示。"""
    root = _jobs_root(transformer_root)
    if not root.exists():
        return []
    jobs: list[TransformerTrainingJob] = []
    for metadata_path in root.glob("*/job.json"):
        try:
            jobs.append(_mark_stale_job(_job_from_dir(metadata_path.parent)))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return sorted(
        jobs,
        key=lambda item: str(item.payload.get("created_at", "")),
        reverse=True,
    )


def latest_transformer_training_job(
    transformer_root: str | Path,
) -> TransformerTrainingJob | None:
    jobs = list_transformer_training_jobs(transformer_root)
    active = next(
        (job for job in jobs if job.status in ACTIVE_TRANSFORMER_JOB_STATUSES),
        None,
    )
    return active or (jobs[0] if jobs else None)


def read_transformer_training_progress(
    job: TransformerTrainingJob,
) -> dict[str, Any]:
    if not job.progress_json.exists():
        return {}
    try:
        return _read_json(job.progress_json)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}


def _append_history(path: Path, payload: dict[str, object]) -> None:
    raw_metrics = payload.get("metrics", {})
    metrics = raw_metrics if isinstance(raw_metrics, dict) else {}
    row = {
        "updated_at": payload.get("updated_at", _utc_now()),
        "progress": payload.get("progress", 0),
        "epoch": payload.get("epoch", 0),
        "epochs": payload.get("epochs", 0),
        "batch": payload.get("batch", ""),
        "batches": payload.get("batches", ""),
    }
    for column in HISTORY_COLUMNS[6:]:
        row[column] = metrics.get(column, "")
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _update_job(job_dir: Path, **updates: object) -> dict[str, Any]:
    with _LOCK:
        metadata_path = job_dir / "job.json"
        payload = _read_json(metadata_path)
        payload.update(updates)
        _write_json(metadata_path, payload)
        return payload


def _run_job(
    job_dir: Path,
    source_paths: tuple[Path, ...],
    model_config: TemporalTransformerConfig,
    training_config: TransformerTrainingConfig,
    output_root: Path,
    run_name: str,
    runner: TransformerRunner,
) -> None:
    progress_path = job_dir / "progress.json"
    history_path = job_dir / "history.csv"
    _update_job(job_dir, status="running", started_at=_utc_now())

    def report(payload: dict[str, object]) -> None:
        durable = {**payload, "job_id": job_dir.name, "updated_at": _utc_now()}
        _write_json(progress_path, durable)
        _append_history(history_path, durable)

    report(
        {
            "status": "preparing",
            "progress": 0.0,
            "epoch": 0,
            "epochs": training_config.epochs,
            "device": training_config.device,
            "cpu_threads": training_config.cpu_threads,
        }
    )
    try:
        result = runner(
            source_paths,
            model_config,
            training_config,
            output_root,
            run_name=run_name,
            progress_callback=report,
        )
        current = read_transformer_training_progress(_job_from_dir(job_dir))
        _write_json(
            progress_path,
            {
                **current,
                "status": "complete",
                "progress": 1.0,
                "result_dir": str(result.run_dir.resolve()),
                "updated_at": _utc_now(),
            },
        )
        _update_job(
            job_dir,
            status="complete",
            completed_at=_utc_now(),
            result_dir=str(result.run_dir.resolve()),
            model_path=str(result.model_path.resolve()),
            resolved_device=result.resolved_device,
            duration_seconds=result.duration_seconds,
            best_epoch=result.best_epoch,
            metrics=result.metrics,
        )
    except Exception as exc:
        current = read_transformer_training_progress(_job_from_dir(job_dir))
        _write_json(
            progress_path,
            {
                **current,
                "status": "failed",
                "error": str(exc),
                "updated_at": _utc_now(),
            },
        )
        _update_job(
            job_dir,
            status="failed",
            completed_at=_utc_now(),
            error=str(exc),
            traceback=traceback.format_exc(),
        )


def submit_transformer_training_job(
    transformer_root: str | Path,
    source_paths: Sequence[str | Path],
    model_config: TemporalTransformerConfig,
    training_config: TransformerTrainingConfig,
    *,
    run_name: str = "transformer",
    runner: TransformerRunner = train_temporal_transformer,
) -> TransformerTrainingJob:
    """送出背景工作；同時間只允許一個 Transformer 訓練。"""
    root = Path(transformer_root).resolve()
    with MODEL_JOB_LOCK, _LOCK:
        from ai_quant_trading.dashboard.rl_training_jobs import (
            ACTIVE_JOB_STATUSES,
            list_rl_training_jobs,
        )

        rl_dir = root.parent / "rl" / "environments"
        rl_active = next(
            (
                job
                for job in list_rl_training_jobs(rl_dir)
                if job.status in ACTIVE_JOB_STATUSES
            ),
            None,
        )
        if rl_active is not None:
            raise RuntimeError(
                "強化學習正在執行，完成後才能啟動 Transformer 訓練。"
            )
        active = next(
            (
                job
                for job in list_transformer_training_jobs(root)
                if job.status in ACTIVE_TRANSFORMER_JOB_STATUSES
            ),
            None,
        )
        if active is not None:
            raise RuntimeError("已有 Transformer 訓練工作正在執行")
        paths = tuple(Path(path).resolve() for path in source_paths)
        if not paths:
            raise ValueError("至少選擇一份特徵資料")
        job_dir = _jobs_root(root) / _utc_stamp()
        job_dir.mkdir(parents=True, exist_ok=False)
        payload: dict[str, object] = {
            "job_id": job_dir.name,
            "status": "queued",
            "created_at": _utc_now(),
            "process_id": os.getpid(),
            "source_paths": [str(path) for path in paths],
            "model_config": model_config.to_dict(),
            "training_config": training_config.to_dict(),
            "run_name": run_name,
        }
        _write_json(job_dir / "job.json", payload)
        job = _job_from_dir(job_dir)
        future = _EXECUTOR.submit(
            _run_job,
            job_dir,
            paths,
            model_config,
            training_config,
            _models_root(root),
            run_name,
            runner,
        )
        _FUTURES[job.job_id] = future
        return job
