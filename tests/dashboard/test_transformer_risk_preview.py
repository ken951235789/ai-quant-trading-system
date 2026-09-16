"""Transformer 單次風控預覽測試。"""

from __future__ import annotations

import pandas as pd

from ai_quant_trading.dashboard.transformer_risk_preview import (
    summarize_transformer_risk,
)


def test_bullish_prediction_allows_only_same_direction_confirmation() -> None:
    preview = summarize_transformer_risk(
        pd.Series(
            {
                "timestamp": "2026-08-21T00:00:00Z",
                "close": 70_000.0,
                "transformer_available": 1.0,
                "transformer_bull_probability": 0.70,
                "transformer_bear_probability": 0.10,
                "transformer_uncertainty": 0.30,
                "transformer_return_1": 0.001,
                "transformer_return_5": 0.004,
                "transformer_return_20": 0.009,
                "transformer_volatility": 0.002,
            }
        ),
        checkpoint_name="btc_transformer",
    )

    assert preview.trend == "偏多"
    assert "確認多頭" in preview.risk_conclusion
    assert preview.checkpoint_name == "btc_transformer"


def test_uncertain_prediction_does_not_increase_risk() -> None:
    preview = summarize_transformer_risk(
        pd.Series(
            {
                "timestamp": "2026-08-21T00:00:00Z",
                "close": 70_000.0,
                "transformer_available": 1.0,
                "transformer_bull_probability": 0.60,
                "transformer_bear_probability": 0.20,
                "transformer_uncertainty": 0.80,
                "transformer_return_1": 0.001,
                "transformer_return_5": 0.004,
                "transformer_return_20": 0.009,
                "transformer_volatility": 0.002,
            }
        ),
        checkpoint_name="btc_transformer",
    )

    assert preview.trend == "中性"
    assert "不建立或增加部位" in preview.risk_conclusion
