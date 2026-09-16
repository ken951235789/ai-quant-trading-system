"""Transformer 背景工作與持久進度測試。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Event
from time import monotonic, sleep
from types import SimpleNamespace

import pytest

from ai_quant_trading.dashboard.transformer_training_jobs import (
    latest_transformer_training_job,
    read_transformer_training_progress,
    submit_transformer_training_job,
)
from ai_quant_trading.transformer import (
    TemporalTransformerConfig,
    TransformerTrainingConfig,
)


def _wait_for_status(root: Path, expected: str, timeout: float = 3.0):
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        job = latest_transformer_training_job(root)
        if job is not None and job.status == expected:
            return job
        sleep(0.01)
    raise AssertionError(f"背景工作未在期限內進入 {expected}")


def test_transformer_job_progress_survives_repeated_reads(tmp_path: Path) -> None:
    source = tmp_path / "features.csv"
    source.write_text("timestamp,close\n2026-01-01,1\n", encoding="utf-8")
    started = Event()
    release = Event()

    def fake_runner(
        source_paths,
        model_config,
        training_config,
        output_root,
        *,
        run_name,
        progress_callback,
    ):
        del source_paths, model_config, training_config, run_name
        progress_callback(
            {
                "status": "running",
                "progress": 0.5,
                "epoch": 1,
                "epochs": 2,
                "metrics": {"train_loss": 0.25},
            }
        )
        started.set()
        assert release.wait(2)
        run_dir = Path(output_root) / "fake_run"
        run_dir.mkdir(parents=True)
        return SimpleNamespace(
            run_dir=run_dir,
            model_path=run_dir / "best_model.pt",
            resolved_device="cpu",
            duration_seconds=1.0,
            best_epoch=1,
            metrics={"loss": 0.2},
        )

    job = submit_transformer_training_job(
        tmp_path / "transformer",
        [source],
        TemporalTransformerConfig(),
        TransformerTrainingConfig(epochs=2),
        runner=fake_runner,
    )
    assert started.wait(2)

    first = latest_transformer_training_job(tmp_path / "transformer")
    second = latest_transformer_training_job(tmp_path / "transformer")
    assert first is not None and second is not None
    assert first.job_id == second.job_id == job.job_id
    assert read_transformer_training_progress(second)["progress"] == 0.5

    release.set()
    completed = _wait_for_status(tmp_path / "transformer", "complete")
    assert completed.payload["best_epoch"] == 1
    assert read_transformer_training_progress(completed)["progress"] == 1.0


def test_transformer_job_is_blocked_while_rl_is_training(tmp_path: Path) -> None:
    root = tmp_path / "processed" / "transformer"
    source = tmp_path / "features.csv"
    source.write_text("timestamp,close\n2026-01-01,1\n", encoding="utf-8")
    rl_job_dir = tmp_path / "processed" / "rl" / "training_jobs" / "active_rl"
    rl_job_dir.mkdir(parents=True)
    (rl_job_dir / "job.json").write_text(
        json.dumps(
            {
                "job_id": "active_rl",
                "status": "running",
                "created_at": "2026-08-15T00:00:00+00:00",
                "process_id": os.getpid(),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="強化學習正在執行"):
        submit_transformer_training_job(
            root,
            [source],
            TemporalTransformerConfig(),
            TransformerTrainingConfig(epochs=2),
        )
