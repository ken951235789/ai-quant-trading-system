"""Champion／Challenger 登錄與安全門檻測試。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from ai_quant_trading.operations.integrity import (
    MANIFEST_NAME,
    build_artifact_manifest,
    sha256_file,
)
from ai_quant_trading.reinforcement_learning import (
    ensure_registered_champion,
    load_model_registry,
    promote_model_challenger,
    register_model_challenger,
)
from ai_quant_trading.reinforcement_learning.registry import assess_paper_model_evidence


def _write_candidate(root: Path) -> tuple[Path, Path]:
    experiment = root / "research" / "experiment"
    run = experiment / "environments" / "fold_02" / "training" / "run"
    run.mkdir(parents=True)
    training = {
        "status": "complete",
        "environment_expert": {"kind": "short_term"},
        "training_config": {"seed": 23},
        "metrics": {
            "validation": {"total_return": 0.02},
            "test": {
                "total_return": 0.03,
                "max_drawdown": 0.04,
                "steps": 200,
                "trades": 10,
                "positive_market_ratio": 0.7,
                "sharpe_ratio": 0.8,
                "fee_to_initial_capital": 0.005,
                "profit_factor": 1.5,
                "expectancy": 1.0,
                "liquidation_count": 0,
            },
        },
    }
    (run / "training.json").write_text(json.dumps(training), encoding="utf-8")
    (run / "final_model.zip").write_bytes(b"trusted-test-model")
    build_artifact_manifest(run)
    holdout = experiment / "final_holdout"
    holdout.mkdir()
    (holdout / "environment.json").write_text(
        json.dumps(
            {
                "status": "final_holdout_sealed",
                "final_holdout": {"sealed": True, "selection_use": False},
            }
        ),
        encoding="utf-8",
    )
    (holdout / "test.csv").write_text("timestamp,close\n2026-01-01,100\n", encoding="utf-8")
    build_artifact_manifest(
        holdout,
        files=[path for path in holdout.rglob("*") if path.is_file()],
    )
    (experiment / "final_holdout_evaluation.csv").write_text(
        "timestamp,equity\n2026-01-01,103\n",
        encoding="utf-8",
    )
    summary = experiment / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "research_protocol_version": 2,
                "eligible": True,
                "selection_eligible": True,
                "candidate_run": str(run.resolve()),
                "candidate_run_relative": str(run.relative_to(experiment)),
                "final_holdout": {
                    "status": "complete",
                    "eligible": True,
                    "reasons": [],
                    "selection_use": False,
                    "candidate_run": str(run.resolve()),
                    "candidate_run_relative": str(run.relative_to(experiment)),
                    "candidate_manifest_sha256": sha256_file(run / MANIFEST_NAME),
                    "holdout_manifest_sha256": sha256_file(holdout / MANIFEST_NAME),
                    "holdout_dir_relative": "final_holdout",
                    "evaluation_csv_relative": "final_holdout_evaluation.csv",
                    "evaluation_csv_sha256": sha256_file(
                        experiment / "final_holdout_evaluation.csv"
                    ),
                    "metrics": {"total_return": 0.03},
                },
            }
        ),
        encoding="utf-8",
    )
    return run, summary


def test_register_challenger_does_not_create_champion(tmp_path: Path) -> None:
    run, summary = _write_candidate(tmp_path)
    registry_path = tmp_path / "model_registry.json"

    register_model_challenger(registry_path, run, research_summary=summary)
    registry = load_model_registry(registry_path)

    expert = registry["experts"]["short_term"]
    assert expert["champion"] is None
    assert expert["challengers"][0]["run_dir"] == str(run.resolve())


def test_execution_gate_only_accepts_registered_champion(tmp_path: Path) -> None:
    approved, summary = _write_candidate(tmp_path)
    rejected = tmp_path / "rejected"
    rejected.mkdir()
    registry_path = tmp_path / "model_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "version": 1,
                "experts": {
                    "short_term": {
                        "champion": {
                            "run_dir": str(approved.resolve()),
                            "research_summary": str(summary.resolve()),
                            "research_summary_sha256": sha256_file(summary),
                            "candidate_manifest_sha256": sha256_file(approved / MANIFEST_NAME),
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    ensure_registered_champion(registry_path, approved, "short_term")
    with pytest.raises(ValueError, match="不是 Registry"):
        ensure_registered_champion(registry_path, rejected, "short_term")
    summary.write_text(summary.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="研究摘要已變更"):
        ensure_registered_champion(registry_path, approved, "short_term")


def test_promotion_requires_real_time_paper_evidence(tmp_path: Path) -> None:
    run, summary = _write_candidate(tmp_path)
    registry_path = tmp_path / "model_registry.json"
    register_model_challenger(registry_path, run, research_summary=summary)

    with pytest.raises(ValueError, match="模擬倉證據不足"):
        promote_model_challenger(
            registry_path,
            run,
            paper_root=tmp_path / "paper",
        )


def test_promotion_requires_registered_challenger(tmp_path: Path) -> None:
    run, _summary = _write_candidate(tmp_path)

    with pytest.raises(ValueError, match="尚未登錄為 Challenger"):
        promote_model_challenger(
            tmp_path / "model_registry.json",
            run,
            paper_root=tmp_path / "paper",
        )


def test_promotion_rejects_evidence_path_outside_experiment(tmp_path: Path) -> None:
    run, summary = _write_candidate(tmp_path)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["final_holdout"]["holdout_dir_relative"] = "../../outside"
    summary.write_text(json.dumps(payload), encoding="utf-8")
    registry_path = tmp_path / "model_registry.json"
    register_model_challenger(registry_path, run, research_summary=summary)

    with pytest.raises(ValueError, match="超出研究實驗目錄"):
        promote_model_challenger(
            registry_path,
            run,
            paper_root=tmp_path / "paper",
        )


def test_paper_evidence_uses_each_accounts_terminal_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = tmp_path / "training" / "candidate"
    run.mkdir(parents=True)
    paths = SimpleNamespace(performance_csv=tmp_path / "performance.csv")
    timestamps = pd.date_range("2026-01-01", periods=201, freq="8h", tz="UTC")
    performance = pd.DataFrame(
        {
            "timestamp": timestamps,
            "total_return": [0.10] * 200 + [-0.05],
            "drawdown": [-0.01] * 201,
        }
    )
    state = SimpleNamespace(
        model_kind="rl",
        model_dir=run.name,
        risk_halted=False,
        account_id="paper-a",
    )
    monkeypatch.setattr(
        "ai_quant_trading.reinforcement_learning.registry.list_paper_accounts",
        lambda _root: [paths],
    )
    monkeypatch.setattr(
        "ai_quant_trading.reinforcement_learning.registry.load_account_state",
        lambda _paths: state,
    )
    monkeypatch.setattr(
        "ai_quant_trading.reinforcement_learning.registry.read_account_csv",
        lambda _path: performance,
    )

    evidence = assess_paper_model_evidence(tmp_path, run, expert_kind="short_term")

    assert evidence.total_return == pytest.approx(-0.05)
    assert not evidence.eligible
    assert any("期末報酬" in reason for reason in evidence.reasons)
