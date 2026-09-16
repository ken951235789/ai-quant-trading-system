"""20 根 K 線完整決策、風控、成交與稽核紀錄測試。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ai_quant_trading.paper_trading import PaperTradingConfig, run_rl_paper_cycle
from ai_quant_trading.paper_trading.storage import read_account_csv
from ai_quant_trading.reinforcement_learning import LoadedRLPolicy, PortfolioEnvConfig
from ai_quant_trading.risk import RiskConfig


class _FeatureTargetModel:
    """將標準化測試特徵當成 SAC 連續目標曝險。"""

    class ObservationSpace:
        shape = (13,)

    observation_space = ObservationSpace()

    def predict(self, observation, deterministic=True):
        del deterministic
        return np.asarray([observation[0]], dtype=np.float32), None


def _policy() -> LoadedRLPolicy:
    return LoadedRLPolicy(
        model=_FeatureTargetModel(),
        training_dir=Path("training/sac_twenty_bar"),
        environment_dir=Path("environment"),
        training_metadata={"training_config": {"algorithm": "sac"}},
        environment_metadata={
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
        },
        feature_columns=["model_feature"],
        env_config=PortfolioEnvConfig(
            max_position_fraction=0.5,
            allow_short=True,
            max_short_fraction=0.5,
            execution_mode="perpetual",
            expert_kind="short_term",
            include_risk_context=True,
            leverage=2.0,
            max_leverage=3.0,
            max_margin_fraction=0.5,
        ),
    )


def _twenty_bars() -> pd.DataFrame:
    timestamps = pd.date_range("2026-08-20T00:00:00Z", periods=20, freq="15min")
    model_targets = [0.30] * 7 + [-0.30] * 7 + [0.20] * 6
    close = pd.Series(
        [100, 101, 102, 103, 104, 105, 106, 105, 104, 103, 102, 101, 100, 99,
         100, 101, 102, 103, 104, 105],
        dtype=float,
    )
    bull = pd.Series([0.75] * 7 + [0.10] * 7 + [0.70] * 6)
    bear = pd.Series([0.10] * 7 + [0.75] * 7 + [0.15] * 6)
    return_5 = pd.Series([0.005] * 7 + [-0.005] * 7 + [0.004] * 6)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "exchange": "binance_futures",
            "symbol": "BTC/USDT",
            "interval": "15m",
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_000.0,
            "atr_14": 2.0,
            "funding_rate": 0.0,
            "model_feature": model_targets,
            "transformer_available": 1.0,
            "transformer_return_1": return_5 / 5,
            "transformer_return_5": return_5,
            "transformer_return_20": return_5 * 2,
            "transformer_volatility": 0.01,
            "transformer_bull_probability": bull,
            "transformer_bear_probability": bear,
            "transformer_uncertainty": 0.20,
        }
    )


def test_twenty_bar_end_to_end_cycle_is_idempotent(tmp_path) -> None:
    market = _twenty_bars()
    arguments = {
        "policy": _policy(),
        "model_dir": "training/sac_twenty_bar",
        "root_dir": tmp_path,
        "account_id": "twenty_bar_cycle",
        "config": PaperTradingConfig(
            initial_capital=1_000,
            fee_rate=0.0004,
            slippage_rate=0.0002,
            position_fraction=0.5,
            allow_short=True,
            max_short_fraction=0.5,
            execution_mode="perpetual",
            leverage=2.0,
            max_margin_fraction=0.5,
        ),
        "risk_config": RiskConfig(
            max_risk_per_trade=0.01,
            fixed_stop_loss_pct=0.03,
            take_profit_pct=0.02,
            max_drawdown_limit=0.10,
            max_position_fraction=0.5,
        ),
        "entry_threshold": 0.05,
        "exit_threshold": 0.01,
    }
    run_rl_paper_cycle(market.iloc[:1], **arguments)
    result = run_rl_paper_cycle(market, **arguments)
    duplicate = run_rl_paper_cycle(market, **arguments)

    predictions = read_account_csv(result.paths.predictions_csv)
    performance = read_account_csv(result.paths.performance_csv)
    orders = read_account_csv(result.paths.orders_csv)

    assert len(predictions) == 20
    assert len(performance) == 20
    assert len(orders) >= 2
    assert {"bull_trend", "bear_trend"}.issubset(set(predictions["market_regime"]))
    assert predictions["capital_multiplier"].notna().all()
    assert predictions["target_risk_multiplier"].notna().all()
    assert predictions["portfolio_gross_exposure"].notna().all()
    assert predictions["decision_guard_reason"].astype(str).str.len().gt(0).all()
    assert performance["equity"].gt(0).all()
    assert not duplicate.processed
    assert duplicate.message == "最新 K 線已處理，沒有重複下單。"
