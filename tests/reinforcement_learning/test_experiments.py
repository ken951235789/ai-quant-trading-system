"""多 seed 與 Walk-forward 實驗測試。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from ai_quant_trading.reinforcement_learning import (
    PortfolioEnvConfig,
    RLResearchConfig,
    RLSplitConfig,
    RLTrainingConfig,
    aggregate_research_runs,
    create_walk_forward_environments,
    evaluate_research_final_holdout,
    parse_seed_list,
    prepare_rl_dataset,
    run_rl_research_experiment,
    save_rl_environment,
    train_rl_agent,
)
from tests.reinforcement_learning.test_dataset import make_rl_frame


def _write_single_environment(root: Path) -> Path:
    environment = root / "environment"
    environment.mkdir()
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC"),
            "close": range(100),
        }
    )
    frame.iloc[:60].to_csv(environment / "train.csv", index=False)
    frame.iloc[60:80].to_csv(environment / "validation.csv", index=False)
    frame.iloc[80:].to_csv(environment / "test.csv", index=False)
    (environment / "environment.json").write_text(
        json.dumps(
            {
                "status": "environment_ready",
                "source": {"symbol": "TEST"},
                "split_rows": {"train": 60, "validation": 20, "test": 20},
                "normalization": {"fit_on": "train_only"},
            }
        ),
        encoding="utf-8",
    )
    return environment


def test_walk_forward_uses_expanding_train_and_distinct_test_windows(tmp_path: Path) -> None:
    environment = _write_single_environment(tmp_path)
    (environment / "ai").mkdir()
    (environment / "ai" / "transformer_model.pt").write_bytes(b"checkpoint")

    folds = create_walk_forward_environments(
        environment,
        tmp_path / "folds",
        folds=3,
        min_rows_per_split=5,
        min_final_holdout_rows=5,
    )

    assert len(folds) == 3
    assert all((path / "ai" / "transformer_model.pt").is_file() for path in folds)
    assert [len(pd.read_csv(path / "train.csv")) for path in folds] == [60, 67, 74]
    assert [len(pd.read_csv(path / "validation.csv")) for path in folds] == [7, 7, 7]
    test_starts = [pd.read_csv(path / "test.csv").iloc[0]["timestamp"] for path in folds]
    assert len(set(test_starts)) == 3
    metadata = json.loads((folds[-1] / "environment.json").read_text(encoding="utf-8"))
    assert metadata["walk_forward"]["method"] == "expanding_train_non_overlapping_test"
    assert metadata["walk_forward"]["final_holdout_excluded"] is True
    holdout = pd.read_csv(tmp_path / "final_holdout" / "test.csv")
    assert len(holdout) == 10
    assert pd.Timestamp(test_starts[-1]) < pd.Timestamp(holdout.iloc[0]["timestamp"])


def test_aggregate_research_runs_selects_latest_fold_of_robust_seed(tmp_path: Path) -> None:
    runs: list[Path] = []
    for seed in (11, 23, 42):
        for fold in (1, 2):
            environment = tmp_path / f"env_{seed}_{fold}"
            run = environment / "training" / f"run_{seed}_{fold}"
            run.mkdir(parents=True)
            (environment / "environment.json").write_text(
                json.dumps({"walk_forward": {"fold": fold}}), encoding="utf-8"
            )
            test_return = 0.04 if seed == 23 else 0.02
            payload = {
                "status": "complete",
                "environment_dir": str(environment),
                "environment_expert": {"kind": "short_term"},
                "training_config": {"seed": seed},
                "metrics": {
                    "validation": {"total_return": 0.01},
                    "test": {
                        "total_return": test_return,
                        "max_drawdown": 0.04,
                        "steps": 200,
                        "trades": 10,
                        "positive_market_ratio": 0.7,
                        "sharpe_ratio": 0.8,
                        "fee_to_initial_capital": 0.005,
                        "turnover_to_initial_capital": 1.2,
                        "profit_factor": 1.5,
                        "expectancy": 1.0,
                        "liquidation_count": 0,
                    },
                },
            }
            (run / "training.json").write_text(json.dumps(payload), encoding="utf-8")
            runs.append(run)

    frame, summary = aggregate_research_runs(runs)

    assert len(frame) == 6
    assert summary["eligible"] is True
    assert summary["candidate_seed"] == 23
    assert str(summary["candidate_run"]).endswith("run_23_2")
    assert summary["max_drawdown_limit"] == pytest.approx(0.06)


def test_parse_seed_list_rejects_duplicates() -> None:
    assert parse_seed_list("11, 23,42") == (11, 23, 42)
    with pytest.raises(ValueError, match="不可重複"):
        parse_seed_list("11,11")


def test_research_selects_before_opening_final_holdout(tmp_path: Path) -> None:
    environment = _write_single_environment(tmp_path)
    evaluated: list[Path] = []

    def fake_trainer(environment_dir, config, **_kwargs):
        fold = int(
            json.loads((Path(environment_dir) / "environment.json").read_text(encoding="utf-8"))[
                "walk_forward"
            ]["fold"]
        )
        run = Path(environment_dir) / "training" / f"run_{config.seed}"
        run.mkdir(parents=True)
        payload = {
            "status": "complete",
            "environment_dir": str(environment_dir),
            "environment_expert": {"kind": "short_term"},
            "training_config": {"seed": config.seed},
            "metrics": {
                "validation": {"total_return": 0.01},
                "test": {
                    "total_return": 0.02 + config.seed / 100_000 + fold / 100_000,
                    "max_drawdown": 0.02,
                    "steps": 200,
                    "trades": 10,
                    "positive_market_ratio": 1.0,
                    "sharpe_ratio": 0.8,
                    "fee_to_initial_capital": 0.005,
                    "turnover_to_initial_capital": 1.0,
                    "profit_factor": 1.5,
                    "expectancy": 1.0,
                    "liquidation_count": 0,
                },
            },
        }
        (run / "training.json").write_text(json.dumps(payload), encoding="utf-8")
        return SimpleNamespace(paths=SimpleNamespace(run_dir=run))

    def fake_holdout_evaluator(candidate, holdout, output, **_kwargs):
        holdout = Path(holdout)
        evaluated.append(Path(candidate))
        assert holdout.name == "final_holdout"
        assert (holdout / "artifact_manifest.json").is_file()
        Path(output).write_text("timestamp,equity\n2026-01-01,101\n", encoding="utf-8")
        return {
            "status": "complete",
            "eligible": True,
            "reasons": [],
            "selection_use": False,
            "candidate_run": str(Path(candidate).resolve()),
            "candidate_manifest_sha256": "a" * 64,
            "holdout_manifest_sha256": "b" * 64,
            "evaluation_csv_sha256": "c" * 64,
            "metrics": {"total_return": 0.01},
        }

    result = run_rl_research_experiment(
        environment,
        RLTrainingConfig(
            algorithm="ppo",
            total_timesteps=8,
            batch_size=4,
            ppo_n_steps=4,
            ppo_n_epochs=1,
        ),
        RLResearchConfig(
            seeds=(11, 23, 42),
            walk_forward_folds=2,
            min_rows_per_split=5,
            min_final_holdout_rows=5,
        ),
        trainer=fake_trainer,
        holdout_evaluator=fake_holdout_evaluator,
    )

    summary = json.loads(result.summary_json.read_text(encoding="utf-8"))
    assert result.eligible
    assert len(evaluated) == 1
    assert summary["selection_eligible"] is True
    assert summary["final_holdout"]["selection_use"] is False
    assert summary["research_protocol_version"] == 2


def test_actual_ppo_can_run_once_on_sealed_final_holdout(tmp_path: Path) -> None:
    dataset = prepare_rl_dataset(
        make_rl_frame(240),
        ["feature_a", "feature_b"],
        RLSplitConfig(0.6, 0.2, min_rows_per_split=20),
    )
    environment = save_rl_environment(
        dataset,
        PortfolioEnvConfig(episode_length=32, random_start=True),
        tmp_path / "environments",
    ).run_dir
    folds = create_walk_forward_environments(
        environment,
        tmp_path / "experiment" / "environments",
        folds=2,
        min_rows_per_split=20,
        min_final_holdout_rows=20,
    )
    trained = train_rl_agent(
        folds[-1],
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
    )
    output = tmp_path / "experiment" / "final_holdout_evaluation.csv"

    evidence = evaluate_research_final_holdout(
        trained.paths.run_dir,
        tmp_path / "experiment" / "final_holdout",
        output,
    )

    assert evidence["status"] == "complete"
    assert evidence["selection_use"] is False
    assert len(str(evidence["candidate_manifest_sha256"])) == 64
    assert len(str(evidence["holdout_manifest_sha256"])) == 64
    assert len(str(evidence["evaluation_csv_sha256"])) == 64
    assert output.is_file()
    with pytest.raises(FileExistsError, match="禁止覆寫"):
        evaluate_research_final_holdout(
            trained.paths.run_dir,
            tmp_path / "experiment" / "final_holdout",
            output,
        )
