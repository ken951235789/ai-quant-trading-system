"""PPO／SAC 訓練設定與 GPU 裝置解析測試。"""

import pytest

from ai_quant_trading.reinforcement_learning import (
    RLTrainingConfig,
    TrainingBackendStatus,
    resolve_training_device,
)


def test_ppo_rollout_must_match_batch_size() -> None:
    with pytest.raises(ValueError, match="可被 batch_size 整除"):
        RLTrainingConfig(ppo_n_steps=100, n_envs=1, batch_size=64)


def test_sac_episode_frequency_rejects_parallel_environments() -> None:
    with pytest.raises(ValueError, match="單一環境"):
        RLTrainingConfig(
            algorithm="sac",
            n_envs=2,
            batch_size=64,
            sac_train_freq_unit="episode",
        )


def test_sac_does_not_apply_irrelevant_ppo_batch_rule() -> None:
    config = RLTrainingConfig(algorithm="sac", batch_size=100)

    assert config.batch_size == 100


def test_cpu_threads_must_be_positive() -> None:
    with pytest.raises(ValueError, match="cpu_threads"):
        RLTrainingConfig(cpu_threads=0)


def test_sac_target_entropy_rejects_auto_initial_value() -> None:
    """SB3 只允許 ent_coef 使用 auto_N，target_entropy 不支援此格式。"""
    with pytest.raises(ValueError, match="SAC target_entropy"):
        RLTrainingConfig(algorithm="sac", sac_target_entropy="auto_0.1")

    assert RLTrainingConfig(algorithm="sac", sac_ent_coef="auto_0.1").sac_ent_coef == "auto_0.1"


def test_resolves_cpu_and_cuda_without_silent_fallback() -> None:
    cpu_status = TrainingBackendStatus(True, "2", "2", False, ())
    gpu_status = TrainingBackendStatus(
        True,
        "2",
        "2",
        True,
        ("GPU A", "GPU B"),
        (0, 1),
    )

    assert resolve_training_device("auto", cpu_status) == "cpu"
    assert resolve_training_device("auto", gpu_status) == "cuda:0"
    assert resolve_training_device("cuda:1", gpu_status) == "cuda:1"
    with pytest.raises(RuntimeError, match="不能使用 GPU"):
        resolve_training_device("cuda", cpu_status)
    with pytest.raises(RuntimeError, match="不可用"):
        resolve_training_device("cuda:2", gpu_status)

    second_gpu_only = TrainingBackendStatus(True, "2", "2", True, ("GPU B",), (1,))
    assert resolve_training_device("auto", second_gpu_only) == "cuda:1"
