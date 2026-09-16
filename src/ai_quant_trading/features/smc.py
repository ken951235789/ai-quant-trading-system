"""可回測、因果且不重繪的 SMC 市場結構特徵。"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd


CAUSAL_SMC_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "smc_swing_high_distance_atr",
    "smc_swing_low_distance_atr",
    "smc_structure_direction",
    "smc_bos_up",
    "smc_bos_down",
    "smc_choch_up",
    "smc_choch_down",
    "smc_liquidity_sweep_high",
    "smc_liquidity_sweep_low",
    "smc_range_position_20",
    "smc_bullish_fvg_atr",
    "smc_bearish_fvg_atr",
)


def _causal_atr(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    close = pd.to_numeric(frame["close"], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def build_causal_smc_features(
    frame: pd.DataFrame,
    *,
    atr: pd.Series | None = None,
    swing_left: int = 3,
    swing_right: int = 3,
    range_window: int = 20,
) -> pd.DataFrame:
    """建立確認式 Swing、BOS/CHoCH、掃流動性與三根 K 線 FVG。

    Pivot 只有在右側 `swing_right` 根全部收盤後才確認，並從下一根 K 線開始使用。
    因此新增未來資料不會改寫已輸出的歷史特徵。
    """
    required = {"high", "low", "close"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"SMC 特徵缺少必要欄位：{missing}")
    if swing_left < 1 or swing_right < 1:
        raise ValueError("Swing 左右確認根數必須大於 0")
    if range_window < 2:
        raise ValueError("SMC 區間視窗至少需要 2 根 K 線")

    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    volatility = (
        pd.to_numeric(atr, errors="coerce")
        if atr is not None
        else _causal_atr(frame)
    ).replace(0, np.nan)

    confirmation_window = swing_left + swing_right + 1
    candidate_high = high.shift(swing_right)
    candidate_low = low.shift(swing_right)
    confirmed_high = candidate_high.where(
        candidate_high.eq(
            high.rolling(confirmation_window, min_periods=confirmation_window).max()
        )
    )
    confirmed_low = candidate_low.where(
        candidate_low.eq(
            low.rolling(confirmation_window, min_periods=confirmation_window).min()
        )
    )

    # 同一根收盤才確認的 Swing 從下一根開始可用，刻意多延遲一根以避免重繪。
    swing_high = confirmed_high.ffill().shift(1)
    swing_low = confirmed_low.ffill().shift(1)
    break_up = (close > swing_high) & (close.shift(1) <= swing_high)
    break_down = (close < swing_low) & (close.shift(1) >= swing_low)

    direction_event = pd.Series(np.nan, index=frame.index, dtype="float64")
    direction_event = direction_event.mask(break_up, 1.0).mask(break_down, -1.0)
    previous_direction = direction_event.ffill().shift(1).fillna(0.0)
    structure_direction = direction_event.ffill().fillna(0.0)

    previous_high = high.rolling(range_window, min_periods=range_window).max().shift(1)
    previous_low = low.rolling(range_window, min_periods=range_window).min().shift(1)
    previous_range = (previous_high - previous_low).replace(0, np.nan)

    bullish_fvg = (low - high.shift(2)).clip(lower=0.0) / volatility
    bearish_fvg = (low.shift(2) - high).clip(lower=0.0) / volatility
    result = pd.DataFrame(
        {
            "smc_swing_high_distance_atr": (swing_high - close) / volatility,
            "smc_swing_low_distance_atr": (close - swing_low) / volatility,
            "smc_structure_direction": structure_direction,
            "smc_bos_up": (break_up & previous_direction.ge(0)).astype("float64"),
            "smc_bos_down": (break_down & previous_direction.le(0)).astype("float64"),
            "smc_choch_up": (break_up & previous_direction.lt(0)).astype("float64"),
            "smc_choch_down": (break_down & previous_direction.gt(0)).astype("float64"),
            "smc_liquidity_sweep_high": (
                (high > swing_high) & (close <= swing_high)
            ).astype("float64"),
            "smc_liquidity_sweep_low": (
                (low < swing_low) & (close >= swing_low)
            ).astype("float64"),
            "smc_range_position_20": (
                (close - previous_low) / previous_range
            ).clip(-1.0, 2.0),
            "smc_bullish_fvg_atr": bullish_fvg.clip(0.0, 10.0),
            "smc_bearish_fvg_atr": bearish_fvg.clip(0.0, 10.0),
        },
        index=frame.index,
    )
    return result.replace([np.inf, -np.inf], np.nan)
