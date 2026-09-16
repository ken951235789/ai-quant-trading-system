"""Transformer 輸出欄位與未訓練時的安全預設值。"""

from __future__ import annotations

import pandas as pd


TRANSFORMER_CONTEXT_COLUMNS = [
    "transformer_return_1",
    "transformer_return_5",
    "transformer_return_20",
    "transformer_volatility",
    "transformer_bull_probability",
    "transformer_bear_probability",
    "transformer_uncertainty",
    "transformer_available",
    "transformer_probability_calibrated",
    "transformer_input_coverage",
    "transformer_input_valid",
    *[
        f"transformer_{name}_{horizon}"
        for horizon in (1, 5, 20)
        for name in (
            "down_probability",
            "neutral_probability",
            "up_probability",
            "movement_down_probability",
            "movement_neutral_probability",
            "movement_up_probability",
            "side_down_probability",
            "side_up_probability",
            "return_q10",
            "return_q50",
            "return_q90",
            "long_edge",
            "short_edge",
            "downside_excursion",
            "upside_excursion",
            "tradeability",
        )
    ],
    "transformer_volatility_regime_low_probability",
    "transformer_volatility_regime_normal_probability",
    "transformer_volatility_regime_high_probability",
    "transformer_timeframe_fast_attention",
    "transformer_timeframe_medium_attention",
    "transformer_timeframe_slow_attention",
]


def ensure_transformer_context(frame: pd.DataFrame) -> pd.DataFrame:
    """補齊 Transformer 特徵；沒有正式模型輸出時標記為不可用。"""
    result = frame.copy()
    for column in TRANSFORMER_CONTEXT_COLUMNS:
        if column not in result:
            result[column] = 0.0
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)
    result["transformer_available"] = result["transformer_available"].clip(0.0, 1.0)
    return result
