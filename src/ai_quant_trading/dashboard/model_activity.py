"""集中查詢目前正在執行的模型訓練工作。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ai_quant_trading.dashboard.rl_training_jobs import (
    ACTIVE_JOB_STATUSES,
    latest_rl_training_job,
)
from ai_quant_trading.dashboard.transformer_training_jobs import (
    ACTIVE_TRANSFORMER_JOB_STATUSES,
    latest_transformer_training_job,
)


@dataclass(frozen=True, slots=True)
class ModelActivity:
    """提供各頁面一致的模型工作鎖定資訊。"""

    kind: str
    job_id: str

    @property
    def label(self) -> str:
        return {
            "ppo": "PPO 強化學習",
            "sac": "SAC 強化學習",
            "rl": "強化學習",
            "transformer": "Transformer",
        }.get(self.kind, "模型訓練")


def active_model_training(
    rl_dir: str | Path,
    transformer_root: str | Path,
) -> ModelActivity | None:
    """回傳目前唯一的模型訓練工作，沒有工作時回傳 None。"""
    rl_job = latest_rl_training_job(rl_dir)
    if rl_job is not None and rl_job.status in ACTIVE_JOB_STATUSES:
        training_config = rl_job.payload.get("training_config", {})
        algorithm = (
            str(training_config.get("algorithm", "rl")).lower()
            if isinstance(training_config, dict)
            else "rl"
        )
        return ModelActivity(algorithm, rl_job.job_id)

    transformer_job = latest_transformer_training_job(transformer_root)
    if (
        transformer_job is not None
        and transformer_job.status in ACTIVE_TRANSFORMER_JOB_STATUSES
    ):
        return ModelActivity("transformer", transformer_job.job_id)
    return None
