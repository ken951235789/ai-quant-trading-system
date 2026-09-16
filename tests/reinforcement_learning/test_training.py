"""PPO／SAC 短訓練、模型保存與樣本外評估測試。"""

import json
from pathlib import Path
import tempfile

import pandas as pd
import pytest

from ai_quant_trading.reinforcement_learning import (
    PortfolioEnvConfig,
    PortfolioTradingEnv,
    RLSplitConfig,
    RLTrainingConfig,
    build_algorithm_parameters,
    list_rl_checkpoints,
    list_rl_training_runs,
    prepare_rl_dataset,
    save_rl_environment,
    prepare_universal_rl_dataset,
    save_universal_rl_environment,
    train_rl_agent,
)
from ai_quant_trading.reinforcement_learning.training import _load_environment_artifact
from tests.reinforcement_learning.test_dataset import make_rl_frame
from tests.reinforcement_learning.test_universal import make_universal_frames


def _save_test_environment(root: str) -> Path:
    dataset = prepare_rl_dataset(
        make_rl_frame(),
        ["feature_a", "feature_b"],
        RLSplitConfig(min_rows_per_split=10),
    )
    return save_rl_environment(
        dataset,
        PortfolioEnvConfig(episode_length=32, random_start=True),
        root,
    ).run_dir


def test_builds_sac_action_noise_and_parameters() -> None:
    config = RLTrainingConfig(
        algorithm="sac",
        sac_action_noise="normal",
        sac_action_noise_sigma=0.2,
    )

    parameters, noise = build_algorithm_parameters(config, 1)

    assert parameters["buffer_size"] == 200_000
    assert parameters["train_freq"] == (1, "step")
    assert noise is parameters["action_noise"]


def test_loads_legacy_environment_without_changing_observation_shape() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        environment_dir = _save_test_environment(temp_dir)
        metadata_path = environment_dir / "environment.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["environment_config"].pop("include_position_context")
        metadata["observation_size"] = len(metadata["feature_columns"]) + 3
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

        _, features, config, train, _, _ = _load_environment_artifact(environment_dir)

        assert config.include_position_context is False
        assert PortfolioTradingEnv(train, features, config).observation_space.shape == (
            len(features) + 3,
        )


def test_trains_and_evaluates_one_policy_across_markets(tmp_path) -> None:
    dataset = prepare_universal_rl_dataset(
        make_universal_frames(),
        RLSplitConfig(0.6, 0.2, min_rows_per_split=20),
    )
    environment_dir = save_universal_rl_environment(
        dataset,
        PortfolioEnvConfig(episode_length=32, random_start=True),
        tmp_path,
    ).run_dir

    updates: list[dict[str, object]] = []
    result = train_rl_agent(
        environment_dir,
        RLTrainingConfig(
            algorithm="ppo",
            total_timesteps=16,
            batch_size=4,
            net_arch=(16, 16),
            checkpoint_freq=8,
            evaluation_freq=8,
            ppo_n_steps=8,
            ppo_n_epochs=1,
        ),
        progress_callback=updates.append,
    )

    assert result.metrics["test"]["market_count"] == 2
    assert set(result.metrics["test_markets"]) == {"AAPL", "BTC/USDT"}
    assert "market" in pd.read_csv(result.paths.test_csv).columns
    assert updates
    assert float(updates[-1]["progress"]) == 1.0
    assert updates[-1]["status"] == "evaluating"
    progress_metrics = updates[-1]["metrics"]
    assert isinstance(progress_metrics, dict)
    assert float(progress_metrics["steps_per_second"]) > 0
    assert "recent_reward_mean" in progress_metrics
    assert updates[-1]["snapshot"]


@pytest.mark.parametrize("algorithm", ["ppo", "sac"])
def test_trains_saves_and_evaluates_algorithms(algorithm: str) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        environment_dir = _save_test_environment(temp_dir)
        common = {
            "algorithm": algorithm,
            "total_timesteps": 16,
            "batch_size": 4,
            "net_arch": (16, 16),
            "checkpoint_freq": 8,
            "evaluation_freq": 8,
        }
        if algorithm == "ppo":
            config = RLTrainingConfig(**common, ppo_n_steps=8, ppo_n_epochs=1)
        else:
            config = RLTrainingConfig(
                **common,
                sac_buffer_size=100,
                sac_learning_starts=4,
                save_replay_buffer=True,
            )

        result = train_rl_agent(environment_dir, config)
        metadata = json.loads(result.paths.metadata_json.read_text(encoding="utf-8"))
        environment = json.loads((environment_dir / "environment.json").read_text(encoding="utf-8"))

        assert result.paths.final_model.exists()
        assert result.paths.best_model.exists()
        assert result.paths.validation_csv.exists()
        assert result.paths.test_csv.exists()
        assert result.metrics["test"]["steps"] == 23
        assert metadata["status"] == "complete"
        assert environment["training_started"] is True
        assert environment["last_training"]["algorithm"] == algorithm
        assert list_rl_training_runs(temp_dir) == [result.paths.run_dir]
        assert result.paths.final_model in list_rl_checkpoints(result.paths.run_dir)
        if algorithm == "sac":
            assert (result.paths.run_dir / "final_replay_buffer.pkl").exists()
