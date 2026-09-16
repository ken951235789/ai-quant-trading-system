"""強化學習環境成品保存測試。"""

import json
from pathlib import Path
import tempfile

import pandas as pd
import pytest

from ai_quant_trading.reinforcement_learning import (
    PortfolioEnvConfig,
    RLSplitConfig,
    list_rl_environments,
    prepare_rl_dataset,
    save_rl_environment,
)
from tests.reinforcement_learning.test_dataset import make_rl_frame


def test_saves_environment_without_starting_training() -> None:
    dataset = prepare_rl_dataset(
        make_rl_frame(),
        ["feature_a", "feature_b"],
        RLSplitConfig(min_rows_per_split=10),
    )
    diagnostic = pd.DataFrame([{"timestamp": "2025-01-01", "reward": 0.0}])

    with tempfile.TemporaryDirectory() as temp_dir:
        paths = save_rl_environment(
            dataset,
            PortfolioEnvConfig(
                include_risk_context=True,
                include_trade_plan_context=True,
            ),
            temp_dir,
            source_path="features.csv",
            diagnostic=diagnostic,
        )
        metadata = json.loads(paths.metadata_json.read_text(encoding="utf-8"))

        assert paths.train_csv.exists()
        assert paths.validation_csv.exists()
        assert paths.test_csv.exists()
        assert paths.diagnostic_csv.exists()
        assert metadata["status"] == "environment_ready"
        assert metadata["training_started"] is False
        assert metadata["normalization"]["fit_on"] == "train_only"
        assert metadata["observation_size"] == len(dataset.feature_columns) + 14
        assert metadata["ai_context"]["coverage"]["aggregate"] == {
            "finbert": 0.0,
            "transformer": 0.0,
        }
        assert list_rl_environments(Path(temp_dir)) == [paths.run_dir]


def test_environment_keeps_reproducible_and_runtime_finbert_paths(tmp_path) -> None:
    dataset = prepare_rl_dataset(
        make_rl_frame(),
        ["feature_a", "feature_b"],
        RLSplitConfig(min_rows_per_split=10),
    )
    news = tmp_path / "finbert_news_latest.csv"
    news.write_text("news_id,published_at\n", encoding="utf-8")

    paths = save_rl_environment(
        dataset,
        PortfolioEnvConfig(),
        tmp_path / "environments",
        finbert_scored_news_path=news,
    )
    metadata = json.loads(paths.metadata_json.read_text(encoding="utf-8"))

    assert metadata["ai_context"]["finbert_scored_news_path"] == "ai\\finbert_news.csv"
    assert metadata["ai_context"]["finbert_runtime_news_path"] == str(news.resolve())
    assert (paths.run_dir / "ai" / "finbert_news.csv").is_file()


@pytest.mark.parametrize(
    "environment",
    [
        PortfolioEnvConfig(minimum_gross_target_cost_multiple=1.5),
        PortfolioEnvConfig(minimum_net_risk_reward=1.2),
    ],
)
def test_environment_rejects_trade_gate_without_expected_return(
    tmp_path: Path,
    environment: PortfolioEnvConfig,
) -> None:
    dataset = prepare_rl_dataset(
        make_rl_frame(),
        ["feature_a", "feature_b"],
        RLSplitConfig(min_rows_per_split=10),
    )

    with pytest.raises(ValueError, match="expected_return"):
        save_rl_environment(dataset, environment, tmp_path / "environments")


def test_environment_accepts_transformer_expected_return(tmp_path: Path) -> None:
    frame = make_rl_frame()
    frame["expected_return"] = 0.01
    dataset = prepare_rl_dataset(
        frame,
        ["feature_a", "feature_b"],
        RLSplitConfig(min_rows_per_split=10),
    )

    paths = save_rl_environment(
        dataset,
        PortfolioEnvConfig(
            minimum_gross_target_cost_multiple=1.5,
            minimum_net_risk_reward=1.2,
        ),
        tmp_path / "environments",
    )

    assert paths.run_dir.is_dir()
