"""訓練與推論共用的比例化價格及多週期市場情境。"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ai_quant_trading.market_clock import interval_duration


MARKET_CONTEXT_COLUMNS = (
    "market_trend_consensus",
    "market_fast_trend",
    "market_slow_trend",
    "market_cross_timeframe_disagreement",
    "market_direction_entropy",
    "market_reversal_pressure",
    "market_trend_reversal_conflict",
    "market_fast_slow_conflict",
    "market_timeframe_coverage",
)
NORMALIZED_PRICE_COLUMNS = (
    "macd_pct",
    "macd_signal_pct",
    "macd_histogram_pct",
    "atr_pct",
    "price_body_ratio",
    "price_range_ratio",
    "price_close_location",
)
PRICE_LEVEL_COLUMNS = frozenset(
    {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap",
        "obv",
        "atr_14",
        "macd",
        "macd_signal",
        "macd_histogram",
        "mark_price",
        "index_price",
        "bb_middle_20",
        "bb_upper_20",
        "bb_lower_20",
    }
)


def is_price_level_feature(column: str) -> bool:
    """原始價格保留給標籤與撮合，不能成為新版模型的輸入捷徑。"""
    name = column.lower()
    return name in PRICE_LEVEL_COLUMNS or re.fullmatch(r"(?:sma|ema)_\d+", name) is not None


def market_timeframes(
    columns: list[str] | tuple[str, ...], *, prefix: str = "mtf_"
) -> tuple[str, ...]:
    intervals = {
        match.group(1)
        for column in columns
        if (match := re.match(rf"^{re.escape(prefix)}(\d+[mhd])_", column))
    }
    return tuple(sorted(intervals, key=lambda value: interval_duration(value).total_seconds()))


def add_model_market_features(
    frame: pd.DataFrame, *, rebuild_context: bool = False
) -> pd.DataFrame:
    """只使用本列已對齊的資料，不讀取未來值或 Transformer 預測。"""
    result = frame.copy()

    def numeric(column: str) -> pd.Series:
        if column not in result:
            return pd.Series(np.nan, index=result.index, dtype=float)
        return pd.to_numeric(result[column], errors="coerce").replace([np.inf, -np.inf], np.nan)

    close = numeric("close").where(lambda values: values > 0)
    normalized: dict[str, pd.Series] = {}
    for source, target in (
        ("macd", "macd_pct"),
        ("macd_signal", "macd_signal_pct"),
        ("macd_histogram", "macd_histogram_pct"),
        ("atr_14", "atr_pct"),
    ):
        if source in result and "close" in result:
            normalized[target] = numeric(source) / close
    if {"open", "high", "low", "close"}.issubset(result.columns):
        open_price = numeric("open").where(lambda values: values > 0)
        high, low = numeric("high"), numeric("low")
        normalized.update(
            {
                "price_body_ratio": (close - open_price) / open_price,
                "price_range_ratio": (high - low) / open_price,
                "price_close_location": ((close - low) - (high - close))
                / (high - low).replace(0, np.nan),
            }
        )
    if normalized:
        result = result.drop(columns=list(normalized), errors="ignore")
        result = pd.concat([result, pd.DataFrame(normalized, index=result.index)], axis=1)

    # 精簡 CSV 可能只保留部分原始指標，不能用較少欄位改寫已由完整資料算出的共識。
    if not rebuild_context and set(MARKET_CONTEXT_COLUMNS).issubset(result.columns):
        result.attrs = dict(frame.attrs)
        return result
    intervals = market_timeframes(list(result.columns))
    if not intervals:
        # 沒有多週期來源時不捏造共識；coverage=0 告知使用者與下游模型。
        context = pd.DataFrame(0.0, index=result.index, columns=MARKET_CONTEXT_COLUMNS)
    else:
        trends, reversals, validities = {}, {}, {}
        for interval in intervals:
            prefix = f"mtf_{interval}_"
            components = pd.concat(
                [
                    np.tanh(numeric(prefix + "ema_20_50_atr")),
                    numeric(prefix + "supertrend_direction_10_3").clip(-1, 1),
                    numeric(prefix + "smc_structure_direction").clip(-1, 1),
                ],
                axis=1,
            )
            available = numeric(prefix + "available").fillna(0).ge(0.5)
            age = numeric(prefix + "age_ratio")
            valid = available & age.ge(0) & age.le(1.0) & components.notna().any(axis=1)
            trends[interval] = components.mean(axis=1).where(valid)
            reversals[interval] = (
                pd.concat(
                    [
                        -np.tanh(numeric(prefix + "bb_zscore_20") / 2),
                        -(2 * numeric(prefix + "rsi_14").clip(0, 1) - 1),
                    ],
                    axis=1,
                )
                .mean(axis=1)
                .where(valid)
            )
            validities[interval] = valid.astype(float)
        trend_frame = pd.DataFrame(trends, index=result.index)
        consensus = trend_frame.mean(axis=1).fillna(0).clip(-1, 1)

        def group_mean(group: list[str]) -> pd.Series:
            if not group:
                return pd.Series(0.0, index=result.index)
            return trend_frame[group].mean(axis=1).fillna(0).clip(-1, 1)

        fast = group_mean(
            [value for value in intervals if interval_duration(value).total_seconds() <= 300]
        )
        slow = group_mean(
            [value for value in intervals if interval_duration(value).total_seconds() >= 14_400]
        )
        counts = trend_frame.notna().sum(axis=1).replace(0, np.nan)
        probabilities = pd.concat(
            [
                trend_frame.lt(-0.15).sum(axis=1) / counts,
                (trend_frame.abs().le(0.15)).sum(axis=1) / counts,
                trend_frame.gt(0.15).sum(axis=1) / counts,
            ],
            axis=1,
        ).fillna(0)
        entropy = -(probabilities * np.log(probabilities.clip(lower=1e-12))).sum(axis=1) / np.log(3)
        reversal = pd.DataFrame(reversals, index=result.index).mean(axis=1).fillna(0).clip(-1, 1)
        context = pd.DataFrame(
            {
                "market_trend_consensus": consensus,
                "market_fast_trend": fast,
                "market_slow_trend": slow,
                "market_cross_timeframe_disagreement": (
                    trend_frame.max(axis=1) - trend_frame.min(axis=1)
                ).fillna(0)
                / 2,
                "market_direction_entropy": entropy.clip(0, 1),
                "market_reversal_pressure": reversal,
                "market_trend_reversal_conflict": (-consensus * reversal).clip(0, 1),
                "market_fast_slow_conflict": (-fast * slow).clip(0, 1),
                "market_timeframe_coverage": pd.DataFrame(validities, index=result.index).mean(
                    axis=1
                ),
            },
            index=result.index,
        )
    result = result.drop(columns=list(MARKET_CONTEXT_COLUMNS), errors="ignore")
    result = pd.concat([result, context], axis=1)
    result.attrs = dict(frame.attrs)
    return result
