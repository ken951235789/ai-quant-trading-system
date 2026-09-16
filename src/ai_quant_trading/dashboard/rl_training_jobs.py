"""在 Dashboard 背景執行強化學習，並持久保存工作進度。"""

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
from typing import Any, Callable
from uuid import uuid4

from ai_quant_trading.reinforcement_learning import (
    RLResearchConfig,
    RLResearchResult,
    RLTrainingConfig,
    RLTrainingResult,
    run_rl_research_experiment,
    train_rl_agent,
)
from ai_quant_trading.dashboard.model_job_lock import MODEL_JOB_LOCK


ACTIVE_JOB_STATUSES = {"queued", "running"}
HISTORY_COLUMNS = (
    "updated_at",
    "progress",
    "completed_timesteps",
    "total_timesteps",
    "experiment_run",
    "experiment_runs",
    "experiment_seed",
    "experiment_fold",
    "recent_reward_mean",
    "episode_reward_mean",
    "evaluation_reward",
    "value_loss",
    "policy_loss",
    "actor_loss",
    "critic_loss",
    "entropy_loss",
    "approx_kl",
    "explained_variance",
)


@dataclass(frozen=True, slots=True)
class RLTrainingJob:
    """可跨 Streamlit 重跑重新讀取的背景訓練工作。"""

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


SingleRunner = Callable[..., RLTrainingResult]
ResearchRunner = Callable[..., RLResearchResult]

_LOCK = RLock()
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rl-training")
_FUTURES: dict[str, Future[None]] = {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _jobs_root(rl_dir: str | Path) -> Path:
    """RL_DIR 指向 environments，因此工作紀錄放在同層目錄。"""
    return Path(rl_dir).resolve().parent / "training_jobs"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    # Streamlit 重跑與訓練 callback 可能同時更新進度；同程序先序列化，
    # 跨程序再靠原子 replace 與短暫退避處理 Windows 檔案占用。
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


def _job_from_dir(job_dir: Path) -> RLTrainingJob:
    metadata_json = job_dir / "job.json"
    return RLTrainingJob(
        job_dir=job_dir,
        metadata_json=metadata_json,
        progress_json=job_dir / "progress.json",
        history_csv=job_dir / "history.csv",
        payload=_read_json(metadata_json),
    )


def _process_is_alive(process_id: int) -> bool:
    """跨 App 行程檢查訓練擁有者，避免多開視窗時誤判中斷。"""
    if process_id <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(
            process_query_limited_information,
            False,
            process_id,
        )
        if not handle:
            return False
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(process_id, 0)
    except (OSError, ValueError):
        return False
    return True


def _mark_stale_job(job: RLTrainingJob) -> RLTrainingJob:
    """應用程式重啟後，舊行程中的執行中工作不應永久占用訓練鎖。"""
    if job.status not in ACTIVE_JOB_STATUSES:
        return job
    if _process_is_alive(int(job.payload.get("process_id", -1))):
        return job
    payload = {
        **job.payload,
        "status": "interrupted",
        "completed_at": _utc_now(),
        "error": "應用程式已重新啟動，上一個背景訓練工作已中斷。",
    }
    _write_json(job.metadata_json, payload)
    return RLTrainingJob(
        job.job_dir,
        job.metadata_json,
        job.progress_json,
        job.history_csv,
        payload,
    )


def list_rl_training_jobs(rl_dir: str | Path) -> list[RLTrainingJob]:
    """列出工作紀錄；目前行程以外的未完成工作會標記為中斷。"""
    root = _jobs_root(rl_dir)
    if not root.exists():
        return []
    jobs: list[RLTrainingJob] = []
    for metadata_json in root.glob("*/job.json"):
        try:
            jobs.append(_mark_stale_job(_job_from_dir(metadata_json.parent)))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return sorted(
        jobs,
        key=lambda item: str(item.payload.get("created_at", "")),
        reverse=True,
    )


def latest_rl_training_job(rl_dir: str | Path) -> RLTrainingJob | None:
    """優先回傳執行中的工作，否則回傳最近一次工作。"""
    jobs = list_rl_training_jobs(rl_dir)
    active = next((job for job in jobs if job.status in ACTIVE_JOB_STATUSES), None)
    return active or (jobs[0] if jobs else None)


def read_rl_training_progress(job: RLTrainingJob) -> dict[str, Any]:
    """讀取最新進度；原子寫入可避免讀到半份 JSON。"""
    if not job.progress_json.exists():
        return {}
    try:
        return _read_json(job.progress_json)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}


def _history_record(payload: dict[str, object]) -> dict[str, object]:
    raw_metrics = payload.get("metrics", {})
    metrics = raw_metrics if isinstance(raw_metrics, dict) else {}
    record: dict[str, object] = {
        "updated_at": str(payload.get("updated_at", _utc_now())),
        "progress": payload.get("progress", 0),
        "completed_timesteps": payload.get("completed_timesteps", 0),
        "total_timesteps": payload.get("total_timesteps", 0),
        "experiment_run": payload.get("experiment_run", ""),
        "experiment_runs": payload.get("experiment_runs", ""),
        "experiment_seed": payload.get("experiment_seed", ""),
        "experiment_fold": payload.get("experiment_fold", ""),
    }
    for column in HISTORY_COLUMNS[8:]:
        record[column] = metrics.get(column, "")
    return record


def _append_history(path: Path, payload: dict[str, object]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(_history_record(payload))


def _update_job(job_dir: Path, **updates: object) -> dict[str, Any]:
    with _LOCK:
        metadata_json = job_dir / "job.json"
        payload = _read_json(metadata_json)
        payload.update(updates)
        _write_json(metadata_json, payload)
        return payload


def _run_training_job(
    job_dir: Path,
    environment_dir: Path,
    config: RLTrainingConfig,
    research_config: RLResearchConfig | None,
    resume_from: Path | None,
    backend_status: Any | None,
    single_runner: SingleRunner,
    research_runner: ResearchRunner,
) -> None:
    progress_json = job_dir / "progress.json"
    history_csv = job_dir / "history.csv"
    _update_job(job_dir, status="running", started_at=_utc_now())
    _write_json(
        progress_json,
        {
            "status": "preparing",
            "progress": 0.0,
            "completed_timesteps": 0,
            "total_timesteps": config.total_timesteps,
            "device": config.device,
            "cpu_threads": config.cpu_threads,
            "updated_at": _utc_now(),
        },
    )

    def report(payload: dict[str, object]) -> None:
        durable_payload = {**payload, "job_id": job_dir.name, "updated_at": _utc_now()}
        _write_json(progress_json, durable_payload)
        _append_history(history_csv, durable_payload)

    try:
        if research_config is not None:
            research_kwargs: dict[str, object] = {"progress_callback": report}
            if backend_status is not None:
                research_kwargs["backend_status"] = backend_status
            result = research_runner(
                environment_dir,
                config,
                research_config,
                **research_kwargs,
            )
            result_updates: dict[str, object] = {
                "result_type": "research",
                "result_dir": str(result.experiment_dir.resolve()),
                "eligible": result.eligible,
                "run_count": len(result.run_dirs),
            }
        else:
            training_kwargs: dict[str, object] = {
                "progress_callback": report,
                "resume_from": resume_from,
            }
            if backend_status is not None:
                training_kwargs["backend_status"] = backend_status
            result = single_runner(environment_dir, config, **training_kwargs)
            result_updates = {
                "result_type": "training",
                "result_dir": str(result.paths.run_dir.resolve()),
                "resolved_device": result.resolved_device,
                "duration_seconds": result.duration_seconds,
            }
        completed_progress = read_rl_training_progress(_job_from_dir(job_dir))
        _write_json(
            progress_json,
            {
                **completed_progress,
                "job_id": job_dir.name,
                "status": "complete",
                "progress": 1.0,
                "updated_at": _utc_now(),
            },
        )
        _update_job(
            job_dir,
            status="complete",
            completed_at=_utc_now(),
            **result_updates,
        )
    except Exception as exc:
        failed_progress = read_rl_training_progress(_job_from_dir(job_dir))
        _write_json(
            progress_json,
            {
                **failed_progress,
                "job_id": job_dir.name,
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


def submit_rl_training_job(
    rl_dir: str | Path,
    environment_dir: str | Path,
    config: RLTrainingConfig,
    *,
    research_config: RLResearchConfig | None = None,
    resume_from: str | Path | None = None,
    backend_status: Any | None = None,
    single_runner: SingleRunner = train_rl_agent,
    research_runner: ResearchRunner = run_rl_research_experiment,
) -> RLTrainingJob:
    """送出背景訓練；避免同一張顯示卡同時執行兩個工作。"""
    with MODEL_JOB_LOCK, _LOCK:
        from ai_quant_trading.dashboard.transformer_training_jobs import (
            ACTIVE_TRANSFORMER_JOB_STATUSES,
            list_transformer_training_jobs,
        )

        transformer_root = Path(rl_dir).resolve().parent.parent / "transformer"
        transformer_active = next(
            (
                job
                for job in list_transformer_training_jobs(transformer_root)
                if job.status in ACTIVE_TRANSFORMER_JOB_STATUSES
            ),
            None,
        )
        if transformer_active is not None:
            raise RuntimeError(
                "Transformer 訓練正在執行，完成後才能啟動 PPO 訓練。"
            )
        active = next(
            (
                job
                for job in list_rl_training_jobs(rl_dir)
                if job.status in ACTIVE_JOB_STATUSES
            ),
            None,
        )
        if active is not None:
            raise RuntimeError(f"已有訓練工作正在執行：{active.job_id}")
        jobs_root = _jobs_root(rl_dir)
        jobs_root.mkdir(parents=True, exist_ok=True)
        kind = "research" if research_config is not None else "training"
        job_dir = jobs_root / f"{_utc_stamp()}_{config.algorithm}_{kind}"
        job_dir.mkdir(parents=False, exist_ok=False)
        research_payload = None
        if research_config is not None:
            research_payload = {
                "seeds": list(research_config.seeds),
                "walk_forward_folds": research_config.walk_forward_folds,
                "min_rows_per_split": research_config.min_rows_per_split,
            }
        payload: dict[str, object] = {
            "job_id": job_dir.name,
            "status": "queued",
            "kind": kind,
            "created_at": _utc_now(),
            "process_id": os.getpid(),
            "environment_dir": str(Path(environment_dir).resolve()),
            "training_config": config.to_dict(),
            "research_config": research_payload,
            "resume_from": str(Path(resume_from).resolve()) if resume_from else None,
        }
        _write_json(job_dir / "job.json", payload)
        _write_json(
            job_dir / "progress.json",
            {
                "status": "queued",
                "progress": 0.0,
                "completed_timesteps": 0,
                "total_timesteps": config.total_timesteps,
                "device": config.device,
                "cpu_threads": config.cpu_threads,
                "updated_at": _utc_now(),
            },
        )
        future = _EXECUTOR.submit(
            _run_training_job,
            job_dir,
            Path(environment_dir).resolve(),
            config,
            research_config,
            Path(resume_from).resolve() if resume_from else None,
            backend_status,
            single_runner,
            research_runner,
        )
        _FUTURES[job_dir.name] = future
        future.add_done_callback(lambda _: _FUTURES.pop(job_dir.name, None))
        return _job_from_dir(job_dir)
