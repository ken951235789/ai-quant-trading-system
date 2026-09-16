"""RL signed target 到 USD-M 多空調倉流程測試。"""

from dataclasses import replace
from pathlib import Path

from gymnasium import spaces
import numpy as np
import pandas as pd

from ai_quant_trading.live_trading.config import LiveTradingConfig
from ai_quant_trading.live_trading.futures_gateway import FuturesTradingGateway
from ai_quant_trading.live_trading.futures_service import run_futures_live_target_cycle
from ai_quant_trading.live_trading.rl_service import run_live_rl_cycle
from ai_quant_trading.live_trading.service import LiveSignal
from ai_quant_trading.live_trading.storage import (
    LivePosition,
    activate_emergency_halt,
    emergency_halt_reason,
    live_trading_paths,
    load_positions,
    save_positions,
)
from ai_quant_trading.reinforcement_learning import LoadedRLPolicy, PortfolioEnvConfig
from ai_quant_trading.risk import DynamicLeverageConfig, RiskConfig
from tests.live_trading.test_futures_gateway import FakeFuturesClient


class FixedTargetModel:
    observation_space = spaces.Box(-10, 10, shape=(6,))

    def __init__(self, target: float):
        self.target = target
        self.last_observation = None

    def predict(self, observation, deterministic=True):
        self.last_observation = np.asarray(observation).copy()
        return np.array([self.target], dtype=np.float32), None


def make_policy(target: float) -> LoadedRLPolicy:
    return LoadedRLPolicy(
        model=FixedTargetModel(target),
        training_dir=Path("models/rl/futures-test"),
        environment_dir=Path("models/rl/environment"),
        training_metadata={},
        environment_metadata={
            "source": {
                "exchange": "binance_futures",
                "symbol": "BTC/USDT",
                "interval": "15m",
            },
            "feature_columns": ["return_1"],
            "normalization": {
                "mean": {"return_1": 0.01},
                "std": {"return_1": 0.02},
            },
        },
        feature_columns=["return_1"],
        env_config=PortfolioEnvConfig(
            max_position_fraction=0.5,
            allow_short=True,
            max_short_fraction=0.5,
            execution_mode="perpetual",
            leverage=2,
            max_leverage=3,
            max_margin_fraction=0.5,
        ),
    )


def make_frame(timestamp: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [timestamp, pd.Timestamp(timestamp) + pd.Timedelta(minutes=15)],
            "symbol": ["BTC/USDT", "BTC/USDT"],
            "exchange": ["binance_futures", "binance_futures"],
            "interval": ["15m", "15m"],
            "open": [50000, 50010],
            "high": [50100, 50110],
            "low": [49900, 49910],
            "close": [50010, 50020],
            "volume": [10, 11],
            "return_1": [0.01, 0.03],
            "atr_14": [200.0, 210.0],
        }
    )


def make_calibrated_frame(timestamp: str) -> pd.DataFrame:
    frame = make_frame(timestamp)
    frame["transformer_available"] = 1.0
    frame["transformer_return_1"] = 0.01
    frame["transformer_return_5"] = 0.02
    frame["transformer_return_20"] = 0.03
    frame["transformer_volatility"] = 0.01
    frame["transformer_bull_probability"] = 0.80
    frame["transformer_bear_probability"] = 0.10
    frame["transformer_uncertainty"] = 0.10
    frame["transformer_probability_calibrated"] = 1.0
    frame["transformer_long_success_probability"] = 0.80
    frame["transformer_short_success_probability"] = 0.65
    frame["transformer_success_probability_calibrated"] = 1.0
    frame["spread_bps"] = 2.0
    return frame


def make_gateway(client, *, enabled=False):
    return FuturesTradingGateway(
        client,
        LiveTradingConfig(
            market_type="usd_m_futures",
            trading_enabled=enabled,
            max_order_quote=1000,
            max_balance_fraction=1.0,
            exchange_protection_enabled=False,
            leverage=2,
        ),
    )


def test_negative_rl_target_validates_short_open(tmp_path) -> None:
    client = FakeFuturesClient()

    result = run_live_rl_cycle(
        make_frame("2026-01-01T00:00:00Z"),
        make_policy(-0.25),
        gateway=make_gateway(client),
        storage_root=tmp_path,
        risk_config=RiskConfig(
            max_risk_per_trade=0.02,
            fixed_stop_loss_pct=0.01,
            take_profit_pct=0.015,
            max_position_fraction=0.5,
        ),
    )

    assert result.preview is not None
    assert result.preview.side == "SELL"
    assert not result.preview.reduce_only
    assert result.submission is not None and result.submission.validated_only
    assert client.position == 0


def test_live_validation_records_dynamic_leverage_decision(tmp_path) -> None:
    client = FakeFuturesClient()
    gateway = FuturesTradingGateway(
        client,
        LiveTradingConfig(
            market_type="usd_m_futures",
            max_order_quote=1000,
            max_balance_fraction=0.10,
            exchange_protection_enabled=False,
            leverage=3,
        ),
    )
    result = run_live_rl_cycle(
        make_calibrated_frame("2026-01-01T00:00:00Z"),
        make_policy(0.50),
        gateway=gateway,
        storage_root=tmp_path,
        risk_config=RiskConfig(
            max_risk_per_trade=0.05,
            fixed_stop_loss_pct=0.01,
            take_profit_pct=0.03,
            max_position_fraction=0.8,
            dynamic_leverage=DynamicLeverageConfig(enabled=True),
        ),
    )

    assert result.signal.leverage in {2, 3}
    assert result.signal.leverage_net_expectancy is not None
    assert result.signal.leverage_net_expectancy > 0
    cycles = pd.read_csv(result.paths.cycles_csv)
    assert cycles.iloc[-1]["leverage"] == result.signal.leverage
    assert "符合" in result.signal.leverage_reason


def test_reversal_closes_short_with_reduce_only_before_opening_long(tmp_path) -> None:
    client = FakeFuturesClient()
    gateway = make_gateway(client, enabled=True)
    risk = RiskConfig(
        max_risk_per_trade=0.02,
        fixed_stop_loss_pct=0.01,
        take_profit_pct=0.015,
        max_position_fraction=0.5,
    )
    opened = run_live_rl_cycle(
        make_frame("2026-01-01T00:00:00Z"),
        make_policy(-0.25),
        gateway=gateway,
        storage_root=tmp_path,
        risk_config=risk,
        execute=True,
        confirmation="EXECUTE TESTNET",
    )
    assert opened.preview is not None and opened.preview.side == "SELL"
    assert client.position < 0
    positions = load_positions(live_trading_paths(tmp_path, "testnet"))
    assert positions["usd_m_futures:BTC/USDT"].side == "SHORT"

    closed = run_live_rl_cycle(
        make_frame("2026-01-01T00:30:00Z"),
        make_policy(0.25),
        gateway=gateway,
        storage_root=tmp_path,
        risk_config=risk,
        execute=True,
        confirmation="EXECUTE TESTNET",
    )

    assert closed.reason == "reverse_flatten_first"
    assert closed.preview is not None and closed.preview.side == "BUY"
    assert closed.preview.reduce_only
    assert client.position == 0
    assert "usd_m_futures:BTC/USDT" not in load_positions(
        live_trading_paths(tmp_path, "testnet")
    )


def test_exchange_flat_reconciliation_keeps_manual_emergency_halt(tmp_path) -> None:
    client = FakeFuturesClient()
    gateway = make_gateway(client, enabled=True)
    risk = RiskConfig(
        max_risk_per_trade=0.02,
        fixed_stop_loss_pct=0.01,
        take_profit_pct=0.015,
        max_position_fraction=0.5,
    )
    run_live_rl_cycle(
        make_frame("2026-01-01T00:00:00Z"),
        make_policy(-0.25),
        gateway=gateway,
        storage_root=tmp_path,
        risk_config=risk,
        execute=True,
        confirmation="EXECUTE TESTNET",
    )
    paths = live_trading_paths(tmp_path, "testnet")
    activate_emergency_halt(paths, "manual_review_required")
    client.position = 0

    result = run_live_rl_cycle(
        make_frame("2026-01-01T00:30:00Z"),
        make_policy(0.0),
        gateway=gateway,
        storage_root=tmp_path,
        risk_config=risk,
        execute=True,
        confirmation="EXECUTE TESTNET",
    )

    assert result.reason == "exchange_position_closed"
    assert emergency_halt_reason(paths) == "manual_review_required"


def test_cancelled_protection_does_not_block_reduce_only_exit(tmp_path) -> None:
    client = FakeFuturesClient()
    gateway = FuturesTradingGateway(
        client,
        LiveTradingConfig(
            market_type="usd_m_futures",
            trading_enabled=True,
            max_order_quote=1000,
            max_balance_fraction=1.0,
            exchange_protection_enabled=True,
            leverage=2,
        ),
    )
    risk = RiskConfig(
        max_risk_per_trade=0.02,
        fixed_stop_loss_pct=0.01,
        take_profit_pct=0.015,
        max_position_fraction=0.5,
    )
    run_live_rl_cycle(
        make_frame("2026-01-01T00:00:00Z"),
        make_policy(-0.25),
        gateway=gateway,
        storage_root=tmp_path,
        risk_config=risk,
        execute=True,
        confirmation="EXECUTE TESTNET",
    )
    for algo in client.algos.values():
        algo["algoStatus"] = "CANCELED"

    result = run_live_rl_cycle(
        make_frame("2026-01-01T00:30:00Z"),
        make_policy(0.0),
        gateway=gateway,
        storage_root=tmp_path,
        risk_config=risk,
        execute=True,
        confirmation="EXECUTE TESTNET",
    )

    assert result.preview is not None and result.preview.reduce_only
    assert client.position == 0
    assert emergency_halt_reason(result.paths) is not None


def test_live_policy_receives_complete_risk_and_trade_plan_context(tmp_path) -> None:
    client = FakeFuturesClient()
    gateway = make_gateway(client, enabled=True)
    risk = RiskConfig(
        max_risk_per_trade=0.02,
        fixed_stop_loss_pct=0.01,
        take_profit_pct=0.015,
        max_position_fraction=0.5,
    )

    def complete_policy(target: float) -> LoadedRLPolicy:
        policy = make_policy(target)
        policy.model.observation_space = spaces.Box(-10, 10, shape=(15,))
        return replace(
            policy,
            env_config=replace(
                policy.env_config,
                include_risk_context=True,
                include_trade_plan_context=True,
            ),
        )

    run_live_rl_cycle(
        make_frame("2026-01-01T00:00:00Z"),
        complete_policy(-0.25),
        gateway=gateway,
        storage_root=tmp_path,
        risk_config=risk,
        execute=True,
        confirmation="EXECUTE TESTNET",
    )
    policy = complete_policy(-0.25)
    run_live_rl_cycle(
        make_frame("2026-01-01T00:30:00Z"),
        policy,
        gateway=gateway,
        storage_root=tmp_path,
        risk_config=risk,
    )

    account_context = policy.model.last_observation[-9:]
    assert account_context[5] > 0  # 停損距離
    assert account_context[6] > 0  # 實際清算距離
    assert account_context[8] > 0  # 停利距離


def test_live_reduce_only_exit_does_not_require_model_quality_gate(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("AI_QUANT_LIVE_TRADING_ENABLED", "YES")
    client = FakeFuturesClient("-0.004")
    gateway = FuturesTradingGateway(
        client,
        LiveTradingConfig(
            environment="live",
            market_type="usd_m_futures",
            trading_enabled=True,
            max_order_quote=1000,
            max_balance_fraction=1.0,
            exchange_protection_enabled=False,
            leverage=2,
        ),
    )
    paths = live_trading_paths(tmp_path, "live")
    save_positions(
        paths,
        {
            "usd_m_futures:BTC/USDT": LivePosition(
                symbol="BTC/USDT",
                quantity=0.004,
                entry_price=50000,
                stop_loss=50500,
                take_profit=49250,
                entry_order_id="entry-1",
                opened_at="2026-01-01T00:00:00+00:00",
                updated_at="2026-01-01T00:00:00+00:00",
                protection_status="disabled",
                market_type="usd_m_futures",
                side="SHORT",
                leverage=2,
                margin_type="ISOLATED",
            )
        },
    )

    def forbidden_quality_check() -> None:
        raise AssertionError("降低風險不應呼叫模型品質閘門")

    result = run_futures_live_target_cycle(
        LiveSignal(
            timestamp="2026-01-01T00:15:00+00:00",
            symbol="BTC/USDT",
            target_fraction=0.0,
            action_signal=0,
            close=50000,
            atr=200,
        ),
        model_dir=tmp_path / "missing-model-is-allowed-for-exit",
        gateway=gateway,
        storage_root=tmp_path,
        risk_config=RiskConfig(
            max_risk_per_trade=0.02,
            fixed_stop_loss_pct=0.01,
            take_profit_pct=0.015,
            max_position_fraction=0.5,
        ),
        execute=True,
        confirmation="EXECUTE LIVE",
        execution_eligibility=forbidden_quality_check,
    )

    assert result.preview is not None and result.preview.reduce_only
    assert client.position == 0
