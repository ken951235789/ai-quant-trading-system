"""RL 目標持倉接入 Binance 安全交易流程的測試。"""

import json
from pathlib import Path

from gymnasium import spaces
import numpy as np
import pandas as pd

from ai_quant_trading.live_trading.config import LiveTradingConfig
from ai_quant_trading.live_trading.gateway import LiveTradingGateway
from ai_quant_trading.live_trading.rl_service import (
    ensure_rl_market_execution_eligible,
    run_live_rl_cycle,
)
from ai_quant_trading.live_trading.storage import live_trading_paths
from ai_quant_trading.reinforcement_learning import LoadedRLPolicy, PortfolioEnvConfig
from ai_quant_trading.risk import RiskConfig
from tests.live_trading.test_service import FakeClient


class FixedRLModel:
    """測試用模型，固定輸出 25% 目標持倉。"""

    observation_space = spaces.Box(-10, 10, shape=(6,))

    def predict(self, observation, deterministic=True):
        return np.array([0.25], dtype=np.float32), None


def make_policy() -> LoadedRLPolicy:
    return LoadedRLPolicy(
        model=FixedRLModel(),
        training_dir=Path("models/rl/test"),
        environment_dir=Path("models/rl/environment"),
        training_metadata={},
        environment_metadata={
            "source": {"exchange": "binance", "symbol": "BTC/USDT"},
            "feature_columns": ["return_1"],
            "normalization": {
                "mean": {"return_1": 0.01},
                "std": {"return_1": 0.02},
            },
        },
        feature_columns=["return_1"],
        env_config=PortfolioEnvConfig(max_position_fraction=0.5),
    )


def make_frame() -> pd.DataFrame:
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


def test_rl_cycle_uses_target_as_position_cap_and_only_validates(tmp_path) -> None:
    client = FakeClient()
    gateway = LiveTradingGateway(
        client,
        LiveTradingConfig(max_order_quote=1_000, max_balance_fraction=1.0),
    )

    result = run_live_rl_cycle(
        make_frame(),
        make_policy(),
        gateway=gateway,
        storage_root=tmp_path,
        risk_config=RiskConfig(max_position_fraction=1.0),
    )

    assert result.signal.action_signal == 1
    # 一般專家的單一標的上限是 20%，SAC 的 25% 目標必須由投資組合主管裁切。
    assert result.signal.target_fraction == 0.20
    assert result.preview is not None
    assert float(result.preview.estimated_notional) <= 275.0
    assert result.submission is not None and result.submission.validated_only
    assert client.placed == []
    audit_path = live_trading_paths(tmp_path, "testnet").audit_jsonl
    events = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    role_event = next(item for item in events if item["event_type"] == "rl_role_decision")
    assert role_event["event_type"] == "rl_role_decision"
    assert {
        "data_engineer",
        "finbert_analyst",
        "model_monitor",
        "transformer_analyst",
        "sac_trader",
        "fixed_risk_manager",
        "portfolio_manager",
    }.issubset(role_event["details"]["roles"])


def test_single_market_policy_rejects_unseen_live_symbol() -> None:
    ensure_rl_market_execution_eligible(make_policy(), "BTC/USDT")

    import pytest

    with pytest.raises(ValueError, match="禁止直接送到 Live"):
        ensure_rl_market_execution_eligible(make_policy(), "ETH/USDT")
