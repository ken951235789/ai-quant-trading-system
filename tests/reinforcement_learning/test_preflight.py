"""正式訓練前資料與風控契約測試。"""

from __future__ import annotations

import pandas as pd

from ai_quant_trading.reinforcement_learning import (
    PortfolioEnvConfig,
    assess_pretraining_readiness,
)


def _split(start: str, periods: int = 8) -> pd.DataFrame:
    timestamp = pd.date_range(start, periods=periods, freq="15min", tz="UTC")
    close = pd.Series(range(100, 100 + periods), dtype=float)
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "open": close,
            "high": close + 2,
            "low": close - 2,
            "close": close + 1,
            "volume": 10.0,
            "return_1": 0.001,
        }
    )


def _config() -> PortfolioEnvConfig:
    return PortfolioEnvConfig(
        execution_mode="perpetual",
        allow_short=True,
        max_short_fraction=0.5,
        leverage=2.0,
        max_leverage=3.0,
        risk_per_trade=0.0025,
        max_drawdown_limit=0.10,
        spread_rate=0.0002,
    )


def test_preflight_accepts_valid_small_smoke_data_with_coverage_warning() -> None:
    report = assess_pretraining_readiness(
        metadata={
            "source": {
                "exchange": "binance_futures",
                "symbol": "BTC/USDT",
                "interval": "15m",
            }
        },
        feature_columns=["return_1"],
        env_config=_config(),
        train=_split("2026-01-01"),
        validation=_split("2026-01-02"),
        test=_split("2026-01-03"),
    )

    assert report.eligible
    assert not report.formal_research_ready
    assert any(item.code == "coverage.rows" for item in report.findings)


def test_preflight_blocks_future_target_and_split_overlap() -> None:
    train = _split("2026-01-01")
    validation = _split("2026-01-01T01:00:00Z")
    test = _split("2026-01-03")
    for frame in (train, validation, test):
        frame["future_return"] = 0.01

    report = assess_pretraining_readiness(
        metadata={},
        feature_columns=["return_1", "future_return"],
        env_config=_config(),
        train=train,
        validation=validation,
        test=test,
    )

    assert not report.eligible
    codes = {item.code for item in report.findings}
    assert "features.lookahead" in codes
    assert "split.overlap" in codes


def test_formal_preflight_blocks_sparse_finbert_context() -> None:
    report = assess_pretraining_readiness(
        metadata={
            "ai_context": {
                "finbert_enabled": True,
                "finbert_config": {"minimum_training_coverage": 0.30},
                "coverage": {"aggregate": {"finbert": 0.05}},
            }
        },
        feature_columns=["return_1", "finbert_available"],
        env_config=_config(),
        train=_split("2026-01-01"),
        validation=_split("2026-01-02"),
        test=_split("2026-01-03"),
        minimum_formal_rows=10,
        minimum_formal_days=1,
    )

    assert not report.eligible
    assert any(
        item.code == "ai.finbert" and item.severity == "error"
        for item in report.findings
    )
