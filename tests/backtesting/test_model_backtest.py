"""PPO 與 Transformer 歷史回測測試。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai_quant_trading.backtesting import (
    TransformerSignalConfig,
    build_transformer_actions,
    filter_history,
    list_ppo_backtest_runs,
    list_transformer_backtest_runs,
    run_transformer_backtest,
    save_latest_model_backtest,
    transformer_split,
)
from ai_quant_trading.backtesting.model_backtest import (
    _ActionSequencePolicy,
    _rl_backtest_benchmarks,
)
from ai_quant_trading.reinforcement_learning import (
    PortfolioEnvConfig,
    evaluate_rl_model,
)


def _transformer_frame(rows: int = 80) -> pd.DataFrame:
    close = 100 * np.cumprod(np.full(rows, 1.001))
    predicted = np.where(np.arange(rows) < rows // 2, 0.002, -0.002)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="5min", tz="UTC"),
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": 10.0,
            "transformer_available": 1.0,
            "transformer_return_5": predicted,
            "transformer_volatility": 0.001,
            "transformer_bull_probability": np.where(predicted > 0, 0.8, 0.1),
            "transformer_bear_probability": np.where(predicted < 0, 0.8, 0.1),
            "transformer_uncertainty": 0.1,
        }
    )


def test_transformer_actions_support_long_short_and_filters() -> None:
    frame = _transformer_frame(20)
    config = TransformerSignalConfig(
        horizon=5,
        minimum_return=0.0005,
        minimum_regime_probability=0.5,
        maximum_uncertainty=0.5,
        max_long_fraction=0.8,
        allow_short=True,
        max_short_fraction=0.6,
    )

    actions = build_transformer_actions(frame, config)

    assert actions.iloc[0] > 0
    assert actions.iloc[-1] < 0
    assert actions.max() <= 0.8
    assert actions.min() >= -0.6


def test_transformer_backtest_uses_next_open_and_writes_latest(tmp_path: Path) -> None:
    result = run_transformer_backtest(
        _transformer_frame(),
        TransformerSignalConfig(horizon=5, minimum_return=0.0005),
        initial_capital=1000,
        fee_rate=0.001,
        slippage_rate=0.0005,
        short_borrow_rate_annual=0.01,
        rebalance_deadband=0.01,
        maximum_drawdown=0.5,
    )
    output = save_latest_model_backtest(result, tmp_path)

    assert len(result.evaluation) == 79
    assert "benchmark_equity" in result.evaluation
    assert "predicted_return" in result.evaluation
    assert result.metrics["prediction_samples"] == 75
    assert (output / "evaluation.csv").exists()
    assert (
        json.loads((output / "summary.json").read_text(encoding="utf-8"))["model_kind"]
        == "transformer"
    )


def test_split_and_date_filter_are_time_ordered() -> None:
    frame = _transformer_frame(100)
    metadata = {"training_config": {"train_fraction": 0.7, "validation_fraction": 0.15}}

    assert len(transformer_split(frame, metadata, "train")) == 70
    assert len(transformer_split(frame, metadata, "validation")) == 15
    assert len(transformer_split(frame, metadata, "test")) == 15
    filtered = filter_history(
        frame,
        start=frame.iloc[10]["timestamp"],
        end=frame.iloc[19]["timestamp"],
    )
    assert len(filtered) == 10


def test_model_lists_only_complete_btc_artifacts(tmp_path: Path) -> None:
    ppo = tmp_path / "rl" / "btc" / "training" / "run"
    ppo.mkdir(parents=True)
    (ppo / "final_model.zip").write_bytes(b"model")
    (ppo / "training.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "environment_source": {"symbol": "BTC/USDT"},
                "training_config": {"algorithm": "ppo"},
            }
        ),
        encoding="utf-8",
    )
    transformer = tmp_path / "transformer" / "models" / "run"
    transformer.mkdir(parents=True)
    (transformer / "best_model.pt").write_bytes(b"model")
    (transformer / "training.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "sources": [{"symbol": "BTC/USDT"}],
                "artifacts": {"model": "best_model.pt"},
            }
        ),
        encoding="utf-8",
    )

    assert list_ppo_backtest_runs(tmp_path / "rl") == [ppo]
    assert list_transformer_backtest_runs(tmp_path / "transformer") == [transformer]


def test_rl_benchmarks_compare_costs_rules_random_and_buy_hold() -> None:
    frame = _transformer_frame(240)
    features = ["transformer_available"]
    config = PortfolioEnvConfig(
        initial_capital=1000.0,
        fee_rate=0.0004,
        slippage_rate=0.0001,
        max_position_fraction=0.5,
        allow_short=True,
        max_short_fraction=0.5,
        execution_mode="perpetual",
        normalized_action_space=True,
        include_risk_context=True,
        max_drawdown_limit=0.5,
    )
    actions = np.full(len(frame), 0.5, dtype=np.float32)
    evaluation, metrics = evaluate_rl_model(
        _ActionSequencePolicy(actions),
        frame,
        features,
        config,
        deterministic=True,
        seed=42,
    )

    benchmarks = _rl_backtest_benchmarks(
        evaluation,
        frame,
        features,
        config,
        metrics,
        seed=42,
    )

    assert set(benchmarks) == {
        "model_with_costs",
        "model_without_costs",
        "fixed_ema",
        "random",
        "buy_and_hold",
    }
    assert (
        benchmarks["model_without_costs"]["final_equity"]
        >= benchmarks["model_with_costs"]["final_equity"]
    )
    assert benchmarks["buy_and_hold"]["trades"] == 1
