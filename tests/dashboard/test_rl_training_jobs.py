"""Dashboard 背景強化學習工作測試。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Event
from time import monotonic, sleep
from types import SimpleNamespace

import pytest

from ai_quant_trading.dashboard.rl_training_jobs import (
    _read_json,
    _write_json,
    latest_rl_training_job,
    list_rl_training_jobs,
    read_rl_training_progress,
    submit_rl_training_job,
)
from ai_quant_trading.reinforcement_learning import RLTrainingConfig


def _wait_for_status(rl_dir: Path, expected: str, timeout: float = 3.0):
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        job = latest_rl_training_job(rl_dir)
        if job is not None and job.status == expected:
            return job
        sleep(0.01)
    raise AssertionError(f"背景工作未在期限內進入 {expected}")


def test_background_job_keeps_progress_across_repeated_reads(tmp_path: Path) -> None:
    rl_dir = tmp_path / "processed" / "rl" / "environments"
    environment_dir = rl_dir / "demo_environment"
    environment_dir.mkdir(parents=True)
    started = Event()
    release = Event()

    def fake_runner(environment, config, *, progress_callback, resume_from):
        progress_callback(
            {
                "status": "running",
                "progress": 0.25,
                "completed_timesteps": 250,
                "total_timesteps": 1_000,
                "elapsed_seconds": 2.0,
                "device": "cpu",
                "metrics": {"recent_reward_mean": 0.12, "steps_per_second": 125},
            }
        )
        started.set()
        assert release.wait(2)
        run_dir = Path(environment) / "training" / "fake_run"
        run_dir.mkdir(parents=True)
        return SimpleNamespace(
            paths=SimpleNamespace(run_dir=run_dir),
            resolved_device="cpu",
            duration_seconds=2.5,
        )

    job = submit_rl_training_job(
        rl_dir,
        environment_dir,
        RLTrainingConfig(total_timesteps=1_000, batch_size=256, ppo_n_steps=1_024),
        single_runner=fake_runner,
    )
    assert started.wait(2)

    # 模擬 Streamlit 因點擊其他按鈕而多次重跑，進度仍由磁碟接回。
    first_read = latest_rl_training_job(rl_dir)
    second_read = latest_rl_training_job(rl_dir)
    assert first_read is not None and second_read is not None
    assert first_read.job_id == second_read.job_id == job.job_id
    assert read_rl_training_progress(second_read)["progress"] == pytest.approx(0.25)
    assert second_read.history_csv.exists()

    with pytest.raises(RuntimeError, match="已有訓練工作"):
        submit_rl_training_job(
            rl_dir,
            environment_dir,
            RLTrainingConfig(total_timesteps=1_000, batch_size=256, ppo_n_steps=1_024),
            single_runner=fake_runner,
        )

    release.set()
    completed = _wait_for_status(rl_dir, "complete")
    assert completed.payload["result_type"] == "training"
    assert read_rl_training_progress(completed)["progress"] == pytest.approx(1.0)


def test_background_job_records_failure_instead_of_disappearing(tmp_path: Path) -> None:
    rl_dir = tmp_path / "processed" / "rl" / "environments"
    environment_dir = rl_dir / "demo_environment"
    environment_dir.mkdir(parents=True)

    def failing_runner(environment, config, *, progress_callback, resume_from):
        raise RuntimeError("測試訓練失敗")

    submit_rl_training_job(
        rl_dir,
        environment_dir,
        RLTrainingConfig(total_timesteps=1_000, batch_size=256, ppo_n_steps=1_024),
        single_runner=failing_runner,
    )

    failed = _wait_for_status(rl_dir, "failed")
    assert failed.payload["error"] == "測試訓練失敗"
    assert read_rl_training_progress(failed)["status"] == "failed"


def test_job_from_previous_process_is_marked_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "ai_quant_trading.dashboard.rl_training_jobs._process_is_alive",
        lambda process_id: False,
    )
    rl_dir = tmp_path / "processed" / "rl" / "environments"
    job_dir = rl_dir.parent / "training_jobs" / "old_job"
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "job_id": "old_job",
                "status": "running",
                "created_at": "2026-01-01T00:00:00+00:00",
                "process_id": os.getpid() + 100_000,
            }
        ),
        encoding="utf-8",
    )

    jobs = list_rl_training_jobs(rl_dir)

    assert jobs[0].status == "interrupted"
    assert "重新啟動" in str(jobs[0].payload["error"])


def test_job_owned_by_another_live_process_stays_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "ai_quant_trading.dashboard.rl_training_jobs._process_is_alive",
        lambda process_id: True,
    )
    rl_dir = tmp_path / "processed" / "rl" / "environments"
    job_dir = rl_dir.parent / "training_jobs" / "live_job"
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "job_id": "live_job",
                "status": "running",
                "created_at": "2026-01-01T00:00:00+00:00",
                "process_id": os.getpid() + 1,
            }
        ),
        encoding="utf-8",
    )

    jobs = list_rl_training_jobs(rl_dir)

    assert jobs[0].status == "running"


def test_concurrent_job_metadata_writes_remain_valid(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    path = tmp_path / "job.json"
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(_write_json, path, {"sequence": index})
            for index in range(40)
        ]
        for future in futures:
            future.result()

    payload = _read_json(path)
    assert 0 <= int(payload["sequence"]) < 40
    assert not list(tmp_path.glob("*.tmp"))


def test_rl_job_is_blocked_while_transformer_is_training(tmp_path: Path) -> None:
    rl_dir = tmp_path / "processed" / "rl" / "environments"
    environment_dir = rl_dir / "demo_environment"
    environment_dir.mkdir(parents=True)
    transformer_job_dir = (
        tmp_path / "processed" / "transformer" / "training_jobs" / "active_tf"
    )
    transformer_job_dir.mkdir(parents=True)
    (transformer_job_dir / "job.json").write_text(
        json.dumps(
            {
                "job_id": "active_tf",
                "status": "running",
                "created_at": "2026-08-15T00:00:00+00:00",
                "process_id": os.getpid(),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Transformer 訓練正在執行"):
        submit_rl_training_job(
            rl_dir,
            environment_dir,
            RLTrainingConfig(total_timesteps=1_000, batch_size=256, ppo_n_steps=1_024),
        )
