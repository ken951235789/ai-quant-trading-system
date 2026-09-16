"""技術指標與模型標籤計算。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ai_quant_trading.data_collection.schemas import REQUIRED_OHLCV_COLUMNS
from ai_quant_trading.features.intraday import (
    INTRADAY_FEATURE_COLUMNS,
    build_intraday_features,
)
from ai_quant_trading.features.smc import (
    CAUSAL_SMC_FEATURE_COLUMNS,
    build_causal_smc_features,
)


MA_WINDOWS = (5, 20, 60, 200)
SHORT_TERM_FEATURE_COLUMNS = [
    "short_rsi_2",
    "short_ema_5_gap",
    "short_ema_200_gap",
    "short_bb_zscore_20",
    "short_reversal_entry",
    "short_reversal_exit",
    "short_reversal_position",
]
FEATURE_COLUMNS = [
    "return_1",
    "sma_5",
    "sma_20",
    "sma_60",
    "sma_200",
    "ema_5",
    "ema_20",
    "ema_60",
    "ema_200",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_histogram",
    "roc_12_pct",
    "atr_14",
    "bb_middle_20",
    "bb_upper_20",
    "bb_lower_20",
    "bb_width_20",
    "historical_volatility_20",
    "volume_change",
    "obv",
    "vwap",
    "higher_high",
    "higher_low",
    "breakout_20",
    *SHORT_TERM_FEATURE_COLUMNS,
    *INTRADAY_FEATURE_COLUMNS,
    *CAUSAL_SMC_FEATURE_COLUMNS,
]


def _calculate_rsi(close: pd.Series, window: int) -> pd.Series:
    """以 Wilder 平滑計算 RSI，並處理單邊上漲或價格不變。"""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(
        alpha=1 / window,
        adjust=False,
        min_periods=window,
    ).mean()
    loss = (-delta.clip(upper=0)).ewm(
        alpha=1 / window,
        adjust=False,
        min_periods=window,
    ).mean()
    relative_strength = gain / loss.where(loss != 0)
    rsi = 100 - (100 / (1 + relative_strength))
    rsi = rsi.mask((loss == 0) & (gain > 0), 100.0)
    return rsi.mask((loss == 0) & (gain == 0), 50.0)


def prepare_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """複製並整理 OHLCV，確保後續計算使用正確型別與時間順序。"""
    missing = [column for column in REQUIRED_OHLCV_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"OHLCV 缺少必要欄位：{missing}")
    if frame.empty:
        raise ValueError("OHLCV 資料是空的")

    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
    if result["timestamp"].isna().any():
        raise ValueError("OHLCV timestamp 含有無法解析的值")

    numeric_columns = ["open", "high", "low", "close", "volume"]
    result[numeric_columns] = result[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if result[numeric_columns].isna().any().any():
        raise ValueError("OHLCV 價格或成交量欄位含有非數字")

    result = result.sort_values("timestamp").reset_index(drop=True)
    if result["timestamp"].duplicated().any():
        raise ValueError("OHLCV timestamp 不可重複")
    return result


def add_trend_features(frame: pd.DataFrame) -> pd.DataFrame:
    """加入報酬率、簡單移動平均與指數移動平均。"""
    result = frame.copy()
    close = result["close"]
    result["return_1"] = close.pct_change(fill_method=None)

    for window in MA_WINDOWS:
        result[f"sma_{window}"] = close.rolling(window=window, min_periods=window).mean()
        result[f"ema_{window}"] = close.ewm(
            span=window,
            adjust=False,
            min_periods=window,
        ).mean()
    return result


def add_momentum_features(frame: pd.DataFrame) -> pd.DataFrame:
    """加入 RSI、MACD 與 ROC 動能指標。"""
    result = frame.copy()
    close = result["close"]

    result["rsi_14"] = _calculate_rsi(close, 14)

    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    result["macd"] = ema_12 - ema_26
    result["macd_signal"] = (
        result["macd"]
        .ewm(
            span=9,
            adjust=False,
            min_periods=9,
        )
        .mean()
    )
    result["macd_histogram"] = result["macd"] - result["macd_signal"]
    result["roc_12_pct"] = close.pct_change(periods=12, fill_method=None) * 100
    return result


def add_volatility_features(
    frame: pd.DataFrame,
    annualization_periods: int,
) -> pd.DataFrame:
    """加入 ATR、布林通道與年化歷史波動率。"""
    if annualization_periods <= 0:
        raise ValueError("annualization_periods 必須大於 0")

    result = frame.copy()
    previous_close = result["close"].shift(1)
    true_range = pd.concat(
        [
            result["high"] - result["low"],
            (result["high"] - previous_close).abs(),
            (result["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    result["atr_14"] = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    middle = result["close"].rolling(window=20, min_periods=20).mean()
    standard_deviation = result["close"].rolling(window=20, min_periods=20).std(ddof=0)
    result["bb_middle_20"] = middle
    result["bb_upper_20"] = middle + 2 * standard_deviation
    result["bb_lower_20"] = middle - 2 * standard_deviation
    result["bb_width_20"] = (result["bb_upper_20"] - result["bb_lower_20"]) / middle.where(
        middle != 0
    )

    log_return = np.log(result["close"] / result["close"].shift(1))
    result["historical_volatility_20"] = log_return.rolling(window=20, min_periods=20).std(
        ddof=1
    ) * np.sqrt(annualization_periods)
    return result


def add_volume_features(frame: pd.DataFrame) -> pd.DataFrame:
    """加入成交量變化、OBV 與從資料起點累積的 VWAP。"""
    result = frame.copy()
    result["volume_change"] = result["volume"].pct_change(fill_method=None)

    direction = np.sign(result["close"].diff()).fillna(0)
    result["obv"] = (direction * result["volume"]).cumsum()

    typical_price = (result["high"] + result["low"] + result["close"]) / 3
    cumulative_volume = result["volume"].cumsum()
    result["vwap"] = (typical_price * result["volume"]).cumsum() / cumulative_volume.where(
        cumulative_volume != 0
    )
    return result


def add_market_structure_features(frame: pd.DataFrame) -> pd.DataFrame:
    """加入高點、低點抬高與突破前 20 期高點的市場結構特徵。"""
    result = frame.copy()
    result["higher_high"] = (result["high"] > result["high"].shift(1)).astype("int8")
    result["higher_low"] = (result["low"] > result["low"].shift(1)).astype("int8")

    # 先 shift 再比較，確保突破特徵不會偷看本期之後的價格。
    previous_20_high = result["high"].rolling(window=20, min_periods=20).max().shift(1)
    result["breakout_20"] = (result["close"] > previous_20_high).astype("int8")
    return result


def add_short_term_reversal_features(
    frame: pd.DataFrame,
    *,
    entry_rsi: float = 5.0,
    exit_rsi: float = 65.0,
    trend_window: int = 200,
    rsi_window: int = 2,
    band_window: int = 20,
    band_std: float = 1.5,
    exit_ema_window: int = 5,
    max_holding_bars: int = 8,
) -> pd.DataFrame:
    """加入短線回檔的連續特徵、規則事件與僅向後看的參考持倉。"""
    if "close" not in frame.columns:
        raise ValueError("短線回檔特徵需要 close 欄位")
    if not 0 < entry_rsi < exit_rsi < 100:
        raise ValueError("RSI 門檻必須符合 0 < entry < exit < 100")
    if min(trend_window, rsi_window, band_window, exit_ema_window) <= 1:
        raise ValueError("所有短線指標週期都必須大於 1")
    if band_std <= 0 or max_holding_bars < 1:
        raise ValueError("band_std 必須大於 0，最大持有期至少為 1")

    result = frame.copy()
    close = pd.to_numeric(result["close"], errors="coerce")
    if close.isna().any():
        raise ValueError("close 含有非數字")
    rsi = _calculate_rsi(close, rsi_window)
    trend = close.ewm(
        span=trend_window,
        adjust=False,
        min_periods=trend_window,
    ).mean()
    exit_average = close.ewm(
        span=exit_ema_window,
        adjust=False,
        min_periods=exit_ema_window,
    ).mean()
    middle = close.rolling(band_window, min_periods=band_window).mean()
    deviation = close.rolling(band_window, min_periods=band_window).std(ddof=0)
    zscore = (close - middle) / deviation.where(deviation != 0)

    result["short_rsi_2"] = rsi
    result["short_ema_5_gap"] = close / exit_average.where(exit_average != 0) - 1
    result["short_ema_200_gap"] = close / trend.where(trend != 0) - 1
    result["short_bb_zscore_20"] = zscore
    ready = rsi.notna() & trend.notna() & exit_average.notna() & zscore.notna()
    entry = ready & (close > trend) & (zscore < -band_std) & (rsi <= entry_rsi)
    exit_rule = ready & (
        (close >= exit_average) | (rsi >= exit_rsi) | (close < trend)
    )
    result["short_reversal_entry"] = entry.astype("int8")
    result["short_reversal_exit"] = exit_rule.astype("int8")

    position = pd.Series(0, index=result.index, dtype="int8")
    holding = False
    holding_bars = 0
    for index in result.index:
        if not ready.loc[index]:
            continue
        if not holding:
            if entry.loc[index]:
                holding = True
                holding_bars = 0
        else:
            holding_bars += 1
            if exit_rule.loc[index] or holding_bars >= max_holding_bars:
                holding = False
                holding_bars = 0
        position.loc[index] = int(holding)
    result["short_reversal_position"] = position
    return result


def add_target(
    frame: pd.DataFrame,
    horizon: int = 1,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """加入未來 N 期報酬與二元方向標籤，兩者都不是模型輸入特徵。"""
    if horizon <= 0:
        raise ValueError("target horizon 必須大於 0")

    result = frame.copy()
    future_close = result["close"].shift(-horizon)
    if "timestamp" in result.columns:
        result["target_timestamp"] = result["timestamp"].shift(-horizon)
    result["target_horizon"] = horizon
    result["future_return"] = future_close / result["close"] - 1

    target = pd.Series(pd.NA, index=result.index, dtype="Int64")
    valid = result["future_return"].notna()
    target.loc[valid] = (result.loc[valid, "future_return"] > threshold).astype(int)
    result["target"] = target
    return result


def build_feature_dataset(
    frame: pd.DataFrame,
    target_horizon: int = 1,
    target_threshold: float = 0.0,
    annualization_periods: int = 252,
    drop_na: bool = True,
) -> pd.DataFrame:
    """依序計算完整特徵與 target，回傳可供模型使用的資料集。"""
    result = prepare_ohlcv(frame)
    result = add_trend_features(result)
    result = add_momentum_features(result)
    result = add_volatility_features(result, annualization_periods)
    result = add_volume_features(result)
    result = add_market_structure_features(result)
    result = add_short_term_reversal_features(result)
    result = build_intraday_features(result)
    smc = build_causal_smc_features(result, atr=result["atr_14"])
    result = pd.concat([result, smc], axis=1)
    result = add_target(result, target_horizon, target_threshold)

    if drop_na:
        required = [*FEATURE_COLUMNS, "future_return", "target"]
        result = result.dropna(subset=required).reset_index(drop=True)
    return result
