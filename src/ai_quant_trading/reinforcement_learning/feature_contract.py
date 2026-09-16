"""新版短線 RL 的固定特徵清單與預期報酬語意契約。"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from ai_quant_trading.features.market_context import MARKET_CONTEXT_COLUMNS, market_timeframes


SHORT_TERM_FEATURE_CONTRACT_VERSION = "btc_short_compact_v2"
SHORT_TERM_FEATURE_BUDGET = 112
COMPACT_TIMEFRAME_FEATURES = (
    "ema_20_50_atr",
    "macd_histogram_atr",
    "atr_pct",
    "relative_volume_20",
    "smc_structure_direction",
    "funding_rate",
    "rsi_14",
    "bb_width_20",
    "volume_delta_ratio",
    "range_position_20",
    "adx_14",
    "open_interest_change",
    "ema_50_200_atr",
    "candle_body_pct",
)
COMPACT_TRANSFORMER_FEATURES = (
    "u_transformer_return_5",
    "u_transformer_return_q10_5",
    "u_transformer_return_q90_5",
    "u_transformer_net_edge_advantage_5",
    "u_transformer_expected_best_edge_5",
    "u_transformer_tradeability_5",
    "u_transformer_uncertainty",
    "u_transformer_available",
    "u_transformer_execution_signal",
    "u_transformer_setup_signal",
    "u_transformer_trend_signal",
    "u_transformer_signal_disagreement",
    "u_transformer_signal_entropy",
    "u_transformer_decision_confidence",
    "u_transformer_no_trade_pressure",
    "u_transformer_volatility",
    "u_transformer_volatility_regime_high_probability",
    "u_transformer_side_direction_edge_5",
)


def build_expected_return_contract(horizon: int = 5) -> dict[str, object]:
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("expected_return 的 horizon 必須是正整數 K 線根數")
    return {
        "schema_version": 1,
        "source_column": f"transformer_return_{horizon}",
        "horizon_bars": horizon,
        "units": "fractional_gross_return",
        "direction": "positive_long_negative_short",
    }


def resolve_expected_return_contract(ai_context: Mapping[str, object]) -> dict[str, object]:
    raw = ai_context.get("expected_return_contract")
    if raw is None:
        # 舊模型的執行流程曾固定使用 20 根，不可在搬家後悄悄改變語意。
        return build_expected_return_contract(20)
    if not isinstance(raw, Mapping):
        raise ValueError("模型 expected_return_contract 格式不正確")
    horizon = raw.get("horizon_bars")
    expected = build_expected_return_contract(horizon)
    if any(raw.get(key) != value for key, value in expected.items()):
        raise ValueError("模型 expected_return_contract 的來源、單位或方向不一致")
    return expected


def attach_expected_return(frame: pd.DataFrame, contract: Mapping[str, object]) -> pd.DataFrame:
    validated = resolve_expected_return_contract({"expected_return_contract": contract})
    source = str(validated["source_column"])
    if source not in frame:
        raise ValueError(f"模型指定的預期報酬來源不存在：{source}")
    result = frame.copy()
    result["expected_return"] = pd.to_numeric(result[source], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    result.attrs["expected_return_contract"] = validated
    return result


def select_compact_short_term_features(
    frame: pd.DataFrame,
    *,
    use_transformer: bool = True,
    use_finbert: bool = False,
    maximum_features: int = SHORT_TERM_FEATURE_BUDGET,
) -> list[str]:
    """依固定金融用途挑選，不利用驗證或 final holdout 的統計排序。"""
    if not 80 <= maximum_features <= 120:
        raise ValueError("短線精簡特徵預算必須介於 80 與 120")
    intervals = market_timeframes(list(frame.columns), prefix="u_mtf_")
    selected = [f"u_{column}" for column in MARKET_CONTEXT_COLUMNS if f"u_{column}" in frame]
    if use_transformer:
        selected.extend(column for column in COMPACT_TRANSFORMER_FEATURES if column in frame)
    if use_finbert:
        selected.extend(
            column
            for column in (
                "u_finbert_sentiment",
                "u_finbert_confidence",
                "u_finbert_recency",
                "u_finbert_available",
            )
            if column in frame
        )
    for interval in intervals:
        selected.extend(
            column
            for column in (
                f"u_mtf_{interval}_available",
                f"u_mtf_{interval}_age_ratio",
            )
            if column in frame
        )
    if intervals:
        # 同時平衡週期與金融用途；不重複加入決策週期的 u_rsi/u_return 等欄位。
        for feature in COMPACT_TIMEFRAME_FEATURES:
            for interval in intervals:
                column = f"u_mtf_{interval}_{feature}"
                if column in frame:
                    selected.append(column)
    else:
        selected.extend(
            column
            for column in (
                "u_return_1",
                "u_return_5",
                "u_return_20",
                "u_ema_20_gap",
                "u_ema_60_gap",
                "u_rsi_14",
                "u_macd_histogram_pct",
                "u_atr_pct",
                "u_bb_zscore_20",
                "u_bb_width_20",
                "u_volume_zscore_20",
                "u_vwap_gap",
                "u_range_pct",
                "u_efficiency_ratio_20",
                "u_funding_rate",
                "u_open_interest_change",
                "u_spread_rate",
            )
            if column in frame
        )
    selected = list(dict.fromkeys(selected))[:maximum_features]
    frame.attrs["rl_feature_contract"] = compact_feature_metadata(selected)
    return selected


def compact_feature_metadata(columns: list[str]) -> dict[str, object]:
    return {
        "version": SHORT_TERM_FEATURE_CONTRACT_VERSION,
        "selection": "semantic_timeframe_indicator_round_robin",
        "maximum_features": SHORT_TERM_FEATURE_BUDGET,
        "feature_columns": list(columns),
        "timeframes": list(market_timeframes(columns, prefix="u_mtf_")),
        "raw_ohlcv_as_features": False,
    }
