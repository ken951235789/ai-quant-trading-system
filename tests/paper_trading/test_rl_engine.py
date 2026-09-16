"""PPO／SAC 模型模擬交易時序與持倉測試。"""

from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ai_quant_trading.paper_trading import (
    PaperTradingConfig,
    run_rl_paper_cycle,
)
from ai_quant_trading.paper_trading.rl_engine import (
    _apply_default_transformer_governance,
    _apply_rl_target_thresholds,
    _effective_soft_drawdown_limit,
    _portable_model_reference,
    download_latest_rl_paper_market_data,
)
from ai_quant_trading.paper_trading.storage import read_account_csv
from ai_quant_trading.reinforcement_learning import LoadedRLPolicy, PortfolioEnvConfig
from ai_quant_trading.risk import DynamicLeverageConfig, RiskConfig


def test_stricter_hard_drawdown_moves_soft_limit_earlier() -> None:
    assert _effective_soft_drawdown_limit(0.15, 0.15) == pytest.approx(0.1125)
    assert _effective_soft_drawdown_limit(0.05, 0.20) == pytest.approx(0.05)
    assert _effective_soft_drawdown_limit(None, 0.20) is None


class FeatureActionModel:
    """把第一個標準化市場特徵直接當成目標持倉。"""

    class ObservationSpace:
        shape = (6,)

    observation_space = ObservationSpace()

    def predict(self, observation, deterministic=True):
        del deterministic
        return np.asarray([observation[0]], dtype=float), None


def make_policy() -> LoadedRLPolicy:
    environment = {
        "environment_kind": "single",
        "source": {"exchange": "yahoo_finance", "symbol": "AAPL", "interval": "1d"},
        "normalization": {"mean": {"model_feature": 0.0}, "std": {"model_feature": 1.0}},
    }
    return LoadedRLPolicy(
        FeatureActionModel(),
        Path("training/ppo"),
        Path("environment"),
        {"training_config": {"algorithm": "ppo"}},
        environment,
        ["model_feature"],
        PortfolioEnvConfig(max_position_fraction=0.25),
    )


def make_short_policy() -> LoadedRLPolicy:
    """建立允許 5% 空頭目標的測試策略。"""
    policy = make_policy()
    return LoadedRLPolicy(
        policy.model,
        policy.training_dir,
        policy.environment_dir,
        policy.training_metadata,
        policy.environment_metadata,
        policy.feature_columns,
        PortfolioEnvConfig(
            max_position_fraction=0.10,
            allow_short=True,
            max_short_fraction=0.05,
            short_borrow_rate_annual=0.10,
        ),
    )


def make_market(actions: list[float]) -> pd.DataFrame:
    rows = []
    for offset, action in enumerate(actions):
        rows.append(
            {
                "timestamp": (pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=offset)),
                "exchange": "yahoo_finance",
                "symbol": "AAPL",
                "interval": "1d",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1000.0,
                "atr_14": 2.0,
                "model_feature": action,
            }
        )
    return pd.DataFrame(rows)


def test_rl_target_executes_at_next_open_and_limits_position_size() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        common = {
            "policy": make_policy(),
            "model_dir": "training/ppo",
            "root_dir": temporary_directory,
            "account_id": "rl_demo",
            "config": PaperTradingConfig(
                initial_capital=1000,
                fee_rate=0,
                slippage_rate=0,
                position_fraction=1,
            ),
            "risk_config": RiskConfig(
                max_risk_per_trade=1,
                fixed_stop_loss_pct=0.5,
                take_profit_pct=None,
                max_position_fraction=1,
            ),
        }
        initialized = run_rl_paper_cycle(make_market([0.20]), **common)
        assert initialized.state.model_kind == "rl"
        assert initialized.state.pending_signal == 1
        assert initialized.state.pending_target_fraction == pytest.approx(0.20)

        bought = run_rl_paper_cycle(make_market([0.20, 0.0]), **common)
        assert bought.executed_signal == 1
        assert bought.state.quantity == pytest.approx(2.0)
        assert bought.state.pending_signal == -1

        sold = run_rl_paper_cycle(make_market([0.20, 0.0, 0.0]), **common)
        assert sold.executed_signal == -1
        assert sold.state.quantity == 0
        assert len(read_account_csv(sold.paths.trades_csv)) == 1


def test_perpetual_paper_account_uses_margin_accounting() -> None:
    class PerpetualModel(FeatureActionModel):
        class ObservationSpace:
            shape = (13,)

        observation_space = ObservationSpace()

    environment = {
        "environment_kind": "single",
        "source": {
            "exchange": "binance_futures",
            "symbol": "BTC/USDT",
            "interval": "15m",
        },
        "normalization": {
            "mean": {"model_feature": 0.0},
            "std": {"model_feature": 1.0},
        },
    }
    policy = LoadedRLPolicy(
        PerpetualModel(),
        Path("training/sac"),
        Path("environment"),
        {"training_config": {"algorithm": "sac"}},
        environment,
        ["model_feature"],
        PortfolioEnvConfig(
            max_position_fraction=0.5,
            allow_short=True,
            max_short_fraction=0.5,
            execution_mode="perpetual",
            expert_kind="short_term",
            include_risk_context=True,
            leverage=2.0,
            max_leverage=3.0,
            max_margin_fraction=0.2,
        ),
    )
    market = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01", periods=3, freq="15min", tz="UTC"
            ),
            "exchange": "binance_futures",
            "symbol": "BTC/USDT",
            "interval": "15m",
            "open": [100.0, 100.0, 110.0],
            "high": [101.0, 111.0, 111.0],
            "low": [99.0, 99.0, 109.0],
            "close": [100.0, 110.0, 110.0],
            "volume": 1000.0,
            "atr_14": 2.0,
            "model_feature": [0.4, 0.0, 0.0],
            "funding_rate": 0.0,
        }
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        common = {
            "policy": policy,
            "model_dir": "training/sac",
            "root_dir": temporary_directory,
            "account_id": "perpetual_demo",
            "config": PaperTradingConfig(
                initial_capital=1000,
                fee_rate=0,
                slippage_rate=0,
                position_fraction=0.5,
                allow_short=True,
                max_short_fraction=0.5,
            ),
            "risk_config": RiskConfig(
                max_risk_per_trade=1,
                fixed_stop_loss_pct=0.5,
                take_profit_pct=None,
                max_position_fraction=0.5,
            ),
        }
        run_rl_paper_cycle(market.iloc[:1], **common)
        opened = run_rl_paper_cycle(market.iloc[:2], **common)

        assert opened.state.config.execution_mode == "perpetual"
        assert opened.state.quantity == pytest.approx(4.0)
        assert opened.state.cash == pytest.approx(1000.0)

        closed = run_rl_paper_cycle(market, **common)
        performance = read_account_csv(closed.paths.performance_csv)

        assert closed.state.quantity == 0.0
        assert closed.state.cash == pytest.approx(1040.0)
        assert performance.iloc[1]["margin_used"] == pytest.approx(220.0)


def test_perpetual_paper_account_records_dynamic_leverage_decision() -> None:
    class PerpetualModel(FeatureActionModel):
        class ObservationSpace:
            shape = (6,)

        observation_space = ObservationSpace()

    environment = {
        "environment_kind": "single",
        "source": {
            "exchange": "binance_futures",
            "symbol": "BTC/USDT",
            "interval": "15m",
        },
        "normalization": {
            "mean": {"model_feature": 0.0},
            "std": {"model_feature": 1.0},
        },
    }
    policy = LoadedRLPolicy(
        PerpetualModel(),
        Path("training/sac"),
        Path("environment"),
        {"training_config": {"algorithm": "sac"}},
        environment,
        ["model_feature"],
        PortfolioEnvConfig(
            max_position_fraction=0.8,
            allow_short=True,
            max_short_fraction=0.8,
            execution_mode="perpetual",
            leverage=3.0,
            max_leverage=3.0,
            max_margin_fraction=0.1,
        ),
    )
    market = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=2, freq="15min", tz="UTC"),
            "exchange": "binance_futures",
            "symbol": "BTC/USDT",
            "interval": "15m",
            "open": [100.0, 100.0],
            "high": [100.5, 100.5],
            "low": [99.5, 99.5],
            "close": [100.0, 100.0],
            "volume": 1000.0,
            "atr_14": 1.0,
            "model_feature": [0.8, 0.8],
            "funding_rate": 0.0,
            "transformer_available": 1.0,
            "transformer_return_1": 0.01,
            "transformer_return_5": 0.02,
            "transformer_return_20": 0.03,
            "transformer_volatility": 0.01,
            "transformer_bull_probability": 0.80,
            "transformer_bear_probability": 0.10,
            "transformer_uncertainty": 0.10,
            "transformer_probability_calibrated": 1.0,
            "transformer_long_success_probability": 0.80,
            "transformer_short_success_probability": 0.65,
            "transformer_success_probability_calibrated": 1.0,
        }
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        common = {
            "policy": policy,
            "model_dir": "training/sac",
            "root_dir": temporary_directory,
            "account_id": "dynamic_leverage_demo",
            "config": PaperTradingConfig(
                initial_capital=1000,
                fee_rate=0,
                slippage_rate=0,
                position_fraction=0.8,
                allow_short=True,
                max_short_fraction=0.8,
                execution_mode="perpetual",
                leverage=3,
                max_margin_fraction=0.1,
            ),
            "risk_config": RiskConfig(
                max_risk_per_trade=1,
                fixed_stop_loss_pct=0.01,
                take_profit_pct=0.03,
                max_position_fraction=0.8,
                dynamic_leverage=DynamicLeverageConfig(enabled=True),
            ),
        }
        initialized = run_rl_paper_cycle(market.iloc[:1], **common)
        predictions = read_account_csv(initialized.paths.predictions_csv)

        assert initialized.state.pending_leverage in {2, 3}
        assert predictions.iloc[-1]["selected_leverage"] == initialized.state.pending_leverage
        assert predictions.iloc[-1]["transformer_probability_calibrated"] in {True, 1, 1.0}
        assert predictions.iloc[-1]["leverage_net_expectancy"] > 0

        opened = run_rl_paper_cycle(market, **common)
        assert opened.state.position_leverage == initialized.state.pending_leverage
        assert read_account_csv(opened.paths.orders_csv).iloc[-1]["leverage"] == (
            opened.state.position_leverage
        )


def test_model_reference_is_portable_inside_project(tmp_path, monkeypatch) -> None:
    model_dir = tmp_path / "data" / "processed" / "rl" / "training" / "ppo_run"
    model_dir.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    reference = _portable_model_reference(model_dir)

    assert Path(reference) == Path("data/processed/rl/training/ppo_run")


def test_rl_target_thresholds_block_weak_entries_and_weak_reversals() -> None:
    assert _apply_rl_target_thresholds(
        0.09, 0.0, entry_threshold=0.10, exit_threshold=0.02
    ) == 0.0
    assert _apply_rl_target_thresholds(
        0.10, 0.0, entry_threshold=0.10, exit_threshold=0.02
    ) == pytest.approx(0.10)
    assert _apply_rl_target_thresholds(
        0.01, 0.20, entry_threshold=0.10, exit_threshold=0.02
    ) == 0.0
    assert _apply_rl_target_thresholds(
        -0.05, 0.20, entry_threshold=0.10, exit_threshold=0.02
    ) == 0.0
    assert _apply_rl_target_thresholds(
        -0.20, 0.20, entry_threshold=0.10, exit_threshold=0.02
    ) == pytest.approx(-0.20)


def test_default_governance_does_not_reinterpret_standardized_probabilities() -> None:
    account = SimpleNamespace(
        symbol="BTC/USDT",
        interval="15m",
        config=PaperTradingConfig(
            position_fraction=0.5,
            allow_short=True,
            max_short_fraction=0.5,
        ),
        risk_config=RiskConfig(max_drawdown_limit=0.10),
    )
    standardized = pd.Series(
        {
            "timestamp": "2026-08-20T00:00:00Z",
            "transformer_available": 0.15,
            "transformer_bull_probability": -1.7,
            "transformer_bear_probability": 2.1,
            "transformer_uncertainty": -1.2,
        }
    )

    target, details = _apply_default_transformer_governance(
        0.25,
        0.0,
        standardized,
        account,
        drawdown=0.0,
    )

    assert target == 0.25
    assert details["market_regime"] == "pre_normalized"


def test_multitimeframe_policy_downloads_and_rebuilds_required_bundle(
    tmp_path,
    monkeypatch,
) -> None:
    policy = make_policy()
    policy.environment_metadata["source"] = {
        "exchange": "binance",
        "symbol": "BTC/USDT",
        "interval": "15m",
    }
    policy.feature_columns = [
        "return_1",
        "mtf_5m_return_1",
        "mtf_15m_return_1",
    ]
    policy.environment_metadata["source"]["exchange"] = "binance_futures"
    raw_paths = [tmp_path / "btc_5m.csv", tmp_path / "btc_15m.csv"]
    feature_paths = [tmp_path / "features_5m.csv", tmp_path / "features_15m.csv"]
    fused_path = tmp_path / "features_mtf.csv"
    calls = {}

    def fake_bundle(symbols, raw_dir, **kwargs):
        calls["symbols"] = symbols
        calls["raw_dir"] = raw_dir
        calls["intervals"] = kwargs["intervals"]
        calls["limit"] = kwargs["limit"]
        return SimpleNamespace(ohlcv_files=raw_paths)

    def fake_features(paths, **kwargs):
        calls["feature_sources"] = tuple(paths)
        calls["feature_output"] = kwargs["output_dir"]
        return tuple(SimpleNamespace(output_path=path) for path in feature_paths)

    def fake_fusion(paths, decision_interval, output_dir):
        calls["fusion_sources"] = tuple(paths)
        calls["decision_interval"] = decision_interval
        calls["fusion_output"] = output_dir
        return (SimpleNamespace(output_path=fused_path, symbol="BTC/USDT"),)

    monkeypatch.setattr(
        "ai_quant_trading.paper_trading.rl_engine.collect_binance_futures_timeframe_bundle",
        fake_bundle,
    )
    monkeypatch.setattr(
        "ai_quant_trading.paper_trading.rl_engine.build_features_from_csv_batch",
        fake_features,
    )
    monkeypatch.setattr(
        "ai_quant_trading.paper_trading.rl_engine.build_multitimeframe_feature_datasets",
        fake_fusion,
    )

    result = download_latest_rl_paper_market_data(
        policy,
        tmp_path / "raw",
        limit=220,
        runtime_feature_dir=tmp_path / "cache",
    )

    assert result == fused_path
    assert calls["symbols"] == ["BTC/USDT"]
    assert calls["intervals"] == ("5m", "15m")
    assert calls["limit"] == 250
    assert calls["feature_sources"] == tuple(raw_paths)
    assert calls["fusion_sources"] == tuple(feature_paths)
    assert calls["decision_interval"] == "15m"
    assert calls["fusion_output"] == tmp_path / "cache"


def test_rl_cycle_rejects_duplicate_bar() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        arguments = {
            "frame": make_market([0.20]),
            "policy": make_policy(),
            "model_dir": "training/ppo",
            "root_dir": temporary_directory,
            "account_id": "rl_duplicate",
        }
        run_rl_paper_cycle(**arguments)
        duplicate = run_rl_paper_cycle(**arguments)

        assert not duplicate.processed
        assert duplicate.message == "最新 K 線已處理，沒有重複下單。"


def test_rl_cycle_rebalances_partial_target_instead_of_full_exit() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        common = {
            "policy": make_policy(),
            "model_dir": "training/ppo",
            "root_dir": temporary_directory,
            "account_id": "rl_rebalance",
            "config": PaperTradingConfig(
                initial_capital=1000,
                fee_rate=0,
                slippage_rate=0,
                position_fraction=1,
            ),
            "risk_config": RiskConfig(
                max_risk_per_trade=1,
                fixed_stop_loss_pct=0.5,
                take_profit_pct=None,
                max_position_fraction=1,
            ),
        }
        run_rl_paper_cycle(make_market([0.20]), **common)
        bought = run_rl_paper_cycle(make_market([0.20, 0.10]), **common)
        reduced = run_rl_paper_cycle(make_market([0.20, 0.10, 0.10]), **common)

        assert bought.state.quantity == pytest.approx(2.0)
        assert bought.state.pending_target_fraction == pytest.approx(0.10)
        assert reduced.state.quantity == pytest.approx(1.0)
        assert reduced.state.cash == pytest.approx(900.0)
        assert len(read_account_csv(reduced.paths.trades_csv)) == 1


def test_rl_short_target_profits_from_decline_and_buys_to_cover() -> None:
    """負目標應於下一根開空，價格下跌後回補並實現獲利。"""
    market = make_market([-0.05, 0.0, 0.0])
    market.loc[1, ["high", "low", "close"]] = [100.0, 89.0, 90.0]
    market.loc[2, ["open", "high", "low", "close"]] = [90.0, 91.0, 89.0, 90.0]
    with tempfile.TemporaryDirectory() as temporary_directory:
        common = {
            "policy": make_short_policy(),
            "model_dir": "training/ppo_short",
            "root_dir": temporary_directory,
            "account_id": "rl_short",
            "config": PaperTradingConfig(
                initial_capital=1000,
                fee_rate=0,
                slippage_rate=0,
                position_fraction=0.10,
                allow_short=True,
                max_short_fraction=0.05,
                short_borrow_rate_annual=0,
            ),
            "risk_config": RiskConfig(
                max_risk_per_trade=1,
                fixed_stop_loss_pct=0.5,
                take_profit_pct=None,
                max_position_fraction=0.10,
            ),
            "entry_threshold": 0.04,
            "exit_threshold": 0.01,
        }
        run_rl_paper_cycle(market.iloc[:1], **common)
        opened = run_rl_paper_cycle(market.iloc[:2], **common)
        assert opened.state.quantity == pytest.approx(-0.5)
        assert opened.state.cash == pytest.approx(1050.0)
        assert opened.state.pending_target_fraction == 0.0

        covered = run_rl_paper_cycle(market, **common)
        assert covered.state.quantity == 0.0
        assert covered.state.cash == pytest.approx(1005.0)
        orders = read_account_csv(covered.paths.orders_csv)
        assert list(orders["side"]) == ["SELL_SHORT", "BUY_TO_COVER"]
        trades = read_account_csv(covered.paths.trades_csv)
        assert float(trades.iloc[-1]["net_pnl"]) == pytest.approx(5.0)


def test_rl_short_position_charges_periodic_carry_cost() -> None:
    """空單持有期間應依市場週期扣除年化成本。"""
    with tempfile.TemporaryDirectory() as temporary_directory:
        market = make_market([-0.05, -0.05])
        common = {
            "policy": make_short_policy(),
            "model_dir": "training/ppo_short",
            "root_dir": temporary_directory,
            "account_id": "rl_short_carry",
            "config": PaperTradingConfig(
                initial_capital=1000,
                fee_rate=0,
                slippage_rate=0,
                position_fraction=0.10,
                allow_short=True,
                max_short_fraction=0.05,
                short_borrow_rate_annual=0.252,
            ),
            "risk_config": RiskConfig(
                max_risk_per_trade=1,
                fixed_stop_loss_pct=0.5,
                take_profit_pct=None,
                max_position_fraction=0.10,
            ),
            "entry_threshold": 0.04,
            "exit_threshold": 0.01,
        }
        run_rl_paper_cycle(market.iloc[:1], **common)
        result = run_rl_paper_cycle(market, **common)

        assert result.state.quantity == pytest.approx(-0.5)
        assert result.state.short_carry_paid == pytest.approx(0.05)
        performance = read_account_csv(result.paths.performance_csv)
        assert float(performance.iloc[-1]["short_carry_cost"]) == pytest.approx(0.05)


def test_rl_short_stop_loss_is_above_entry_price() -> None:
    """空單價格上漲碰到停損時，應以買進回補結束交易。"""
    market = make_market([-0.05, -0.05])
    market.loc[1, ["high", "low", "close"]] = [104.0, 99.0, 103.0]
    with tempfile.TemporaryDirectory() as temporary_directory:
        common = {
            "policy": make_short_policy(),
            "model_dir": "training/ppo_short",
            "root_dir": temporary_directory,
            "account_id": "rl_short_stop",
            "config": PaperTradingConfig(
                initial_capital=1000,
                fee_rate=0,
                slippage_rate=0,
                position_fraction=0.10,
                allow_short=True,
                max_short_fraction=0.05,
            ),
            "risk_config": RiskConfig(
                max_risk_per_trade=1,
                fixed_stop_loss_pct=0.03,
                take_profit_pct=None,
                max_position_fraction=0.10,
            ),
            "entry_threshold": 0.04,
            "exit_threshold": 0.01,
        }
        run_rl_paper_cycle(market.iloc[:1], **common)
        result = run_rl_paper_cycle(market, **common)

        assert result.state.quantity == 0.0
        orders = read_account_csv(result.paths.orders_csv)
        assert list(orders["side"]) == ["SELL_SHORT", "BUY_TO_COVER"]
        trades = read_account_csv(result.paths.trades_csv)
        assert trades.iloc[-1]["exit_reason"] == "stop_loss"
        assert float(trades.iloc[-1]["net_pnl"]) == pytest.approx(-1.5)
