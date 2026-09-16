"""RL Policy 最新 observation 與實盤品質門檻測試。"""

from __future__ import annotations

from pathlib import Path

from gymnasium import spaces
import numpy as np
import pandas as pd
import pytest

from ai_quant_trading.reinforcement_learning import (
    LoadedRLPolicy,
    PortfolioEnvConfig,
    assess_rl_training_quality,
    ensure_rl_execution_eligible,
    latest_rl_target,
    load_rl_policy,
)
from ai_quant_trading.reinforcement_learning.policy import (
    _attach_runtime_ai_context,
    _runtime_finbert_news_path,
    _runtime_inference_window,
    prepare_rl_policy_market_frame,
)
from ai_quant_trading.reinforcement_learning.feature_contract import build_expected_return_contract


class FixedPolicyModel:
    def __init__(self, observation_size: int, target: float = 0.25) -> None:
        self.observation_space = spaces.Box(-10, 10, shape=(observation_size,))
        self.target = target
        self.last_observation = None

    def predict(self, observation, deterministic=True):
        self.last_observation = np.asarray(observation)
        return np.array([self.target], dtype=np.float32), None


def make_policy() -> LoadedRLPolicy:
    model = FixedPolicyModel(6)
    return LoadedRLPolicy(
        model=model,
        training_dir=Path("training"),
        environment_dir=Path("environment"),
        training_metadata={},
        environment_metadata={
            "feature_columns": ["return_1"],
            "normalization": {
                "mean": {"return_1": 0.01},
                "std": {"return_1": 0.02},
            },
        },
        feature_columns=["return_1"],
        env_config=PortfolioEnvConfig(holding_period_reference=8),
    )


def make_policy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": ["2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"],
            "symbol": ["BTC/USDT", "BTC/USDT"],
            "exchange": ["binance", "binance"],
            "interval": ["1d", "1d"],
            "open": [100, 101],
            "high": [102, 103],
            "low": [99, 100],
            "close": [101, 102],
            "volume": [10, 11],
            "return_1": [0.01, 0.03],
            "atr_14": [2.0, 2.1],
        }
    )


def test_latest_target_uses_saved_normalization_and_portfolio_context() -> None:
    policy = make_policy()

    signal = latest_rl_target(
        make_policy_frame(),
        policy,
        cash_ratio=0.75,
        position_ratio=0.25,
        drawdown=-0.05,
        unrealized_return=0.10,
        holding_bars=4,
    )
    assert signal.target_fraction == pytest.approx(0.25)
    assert signal.symbol == "BTC/USDT"
    np.testing.assert_allclose(
        policy.model.last_observation,
        [1.0, 0.75, 0.25, -0.05, 0.10, 0.50],
    )


@pytest.mark.parametrize("horizon", [5, 20])
def test_runtime_expected_return_uses_metadata_not_hardcoded_horizon(tmp_path, monkeypatch, horizon) -> None:
    policy = make_policy()
    checkpoint = tmp_path / "transformer.pt"
    checkpoint.write_bytes(b"fixture")
    policy.environment_metadata["ai_context"] = {
        "transformer_enabled": True, "transformer_checkpoint": str(checkpoint),
        "expected_return_contract": build_expected_return_contract(horizon),
    }
    frame = make_policy_frame()
    frame["transformer_return_5"] = .003
    frame["transformer_return_20"] = -.02
    frame["transformer_available"] = 1.
    monkeypatch.setattr(
        "ai_quant_trading.reinforcement_learning.policy.infer_latest_transformer_context",
        lambda checkpoint, frame, **kwargs: frame.copy(),
    )
    result = _attach_runtime_ai_context(frame, policy)
    assert result.expected_return.eq(.003 if horizon == 5 else -.02).all()


def test_single_market_u_features_enable_runtime_ratio_transformation() -> None:
    policy = make_policy()
    policy.feature_columns = ["u_mtf_15m_rsi_14", "u_transformer_return_5"]
    assert policy.universal


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_prepared_runtime_refuses_invalid_expected_return(invalid) -> None:
    policy = make_policy()
    policy.environment_metadata["ai_context"] = {
        "transformer_enabled": True,
        "expected_return_contract": build_expected_return_contract(5),
    }
    frame = make_policy_frame()
    frame["transformer_return_5"] = invalid
    frame["transformer_return_20"] = .02
    with pytest.raises(ValueError, match="最新 expected_return 無效"):
        prepare_rl_policy_market_frame(frame, policy)


def test_prepared_runtime_refuses_unverifiable_expected_return_source() -> None:
    policy = make_policy()
    policy.environment_metadata["ai_context"] = {
        "transformer_enabled": True,
        "expected_return_contract": build_expected_return_contract(5),
    }
    frame = make_policy_frame()
    frame["expected_return"] = .02
    with pytest.raises(ValueError, match="無法驗證 expected_return 來源"):
        prepare_rl_policy_market_frame(frame, policy)

def test_latest_target_rejects_incomplete_latest_features() -> None:
    frame = make_policy_frame()
    frame.loc[1, "return_1"] = np.nan

    with pytest.raises(ValueError, match="禁止沿用舊訊號"):
        latest_rl_target(
            frame,
            make_policy(),
            cash_ratio=1,
            position_ratio=0,
        )


def test_latest_target_preserves_negative_short_action() -> None:
    policy = make_policy()
    policy.model.target = -0.04
    policy.env_config = PortfolioEnvConfig(
        holding_period_reference=8,
        allow_short=True,
        max_short_fraction=0.05,
    )

    signal = latest_rl_target(
        make_policy_frame(),
        policy,
        cash_ratio=1.04,
        position_ratio=-0.04,
    )

    assert signal.target_fraction == pytest.approx(-0.04)
    assert policy.model.last_observation[2] == pytest.approx(-0.04)


def test_latest_target_adds_trade_plan_context_and_applies_neutral_zone() -> None:
    policy = make_policy()
    policy.model = FixedPolicyModel(8, target=0.08)
    policy.env_config = PortfolioEnvConfig(
        holding_period_reference=8,
        normalized_action_space=True,
        include_trade_plan_context=True,
        neutral_action_threshold=0.10,
    )

    signal = latest_rl_target(
        make_policy_frame(),
        policy,
        cash_ratio=1.0,
        position_ratio=0.0,
        entry_price_distance=-0.02,
        take_profit_distance=0.015,
    )

    assert signal.target_fraction == 0.0
    assert policy.model.last_observation[-2] == pytest.approx(-0.02)
    assert policy.model.last_observation[-1] == pytest.approx(0.015)


def test_runtime_inference_window_keeps_warmup_when_multiple_bars_were_missed() -> None:
    frame = pd.DataFrame(
        {"timestamp": pd.date_range("2026-01-01", periods=400, freq="4h", tz="UTC")}
    )
    since = frame.iloc[349]["timestamp"]

    selected, mode = _runtime_inference_window(frame, since, sequence_length=64)

    assert mode == "full"
    assert len(selected) == 300
    assert selected.iloc[250]["timestamp"] == frame.iloc[350]["timestamp"]


def test_runtime_inference_window_uses_fast_latest_mode_for_one_new_bar() -> None:
    frame = pd.DataFrame(
        {"timestamp": pd.date_range("2026-01-01", periods=80, freq="4h", tz="UTC")}
    )

    selected, mode = _runtime_inference_window(
        frame,
        frame.iloc[-2]["timestamp"],
        sequence_length=64,
    )

    assert mode == "latest"
    assert len(selected) == len(frame)


def test_runtime_finbert_prefers_project_latest_over_environment_snapshot(tmp_path) -> None:
    project = tmp_path / "project"
    environment = project / "data" / "processed" / "rl" / "environments" / "demo"
    snapshot = environment / "ai" / "finbert_news.csv"
    latest = project / "data" / "processed" / "sentiment" / "finbert_news_latest.csv"
    snapshot.parent.mkdir(parents=True)
    latest.parent.mkdir(parents=True)
    snapshot.write_text("snapshot", encoding="utf-8")
    latest.write_text("latest", encoding="utf-8")
    policy = make_policy()
    policy.environment_dir = environment

    selected = _runtime_finbert_news_path(
        policy,
        {"finbert_scored_news_path": "ai/finbert_news.csv"},
    )

    assert selected == latest.resolve()


def test_policy_loader_rejects_model_without_integrity_manifest(tmp_path) -> None:
    run = tmp_path / "training" / "untrusted"
    run.mkdir(parents=True)
    (run / "training.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="完整性清單"):
        load_rl_policy(run)


def test_rl_quality_requires_positive_validation_test_and_risk_control() -> None:
    metadata = {
        "status": "complete",
        "metrics": {
            "validation": {"total_return": 0.02},
            "test": {
                "total_return": 0.03,
                "max_drawdown": 0.08,
                "steps": 300,
                "trades": 12,
                "positive_market_ratio": 0.75,
                "profit_factor": 1.5,
                "expectancy": 1.0,
                "liquidation_count": 0,
            },
        },
    }

    assert assess_rl_training_quality(metadata).eligible
    metadata["metrics"]["test"]["positive_market_ratio"] = 0.40
    report = assess_rl_training_quality(metadata)
    assert not report.eligible
    with pytest.raises(ValueError, match="未達實盤品質門檻"):
        ensure_rl_execution_eligible(metadata)


def test_rl_quality_rejects_sparse_finbert_history() -> None:
    metadata = {
        "status": "complete",
        "environment_ai_context": {
            "runtime_ready": True,
            "finbert_enabled": True,
            "transformer_enabled": True,
            "coverage": {"aggregate": {"finbert": 0.0008, "transformer": 0.99}},
        },
        "metrics": {
            "validation": {"total_return": 0.02},
            "test": {
                "total_return": 0.03,
                "max_drawdown": 0.02,
                "steps": 300,
                "trades": 12,
                "positive_market_ratio": 0.75,
                "sharpe_ratio": 1.0,
            },
        },
    }

    report = assess_rl_training_quality(metadata)

    assert not report.eligible
    assert any("FinBERT 歷史覆蓋" in reason for reason in report.reasons)


def test_rl_quality_rejects_policy_stuck_at_position_limit() -> None:
    metadata = {
        "status": "complete",
        "metrics": {
            "validation": {"total_return": 0.02},
            "test": {
                "total_return": 0.03,
                "max_drawdown": 0.02,
                "steps": 300,
                "trades": 12,
                "positive_market_ratio": 0.75,
                "sharpe_ratio": 1.0,
                "action_saturation_ratio": 1.0,
            },
        },
    }

    report = assess_rl_training_quality(metadata)

    assert not report.eligible
    assert any("策略可能已退化" in reason for reason in report.reasons)
