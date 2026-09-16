"""BTC 日內交易使用的因果式、比例化市場特徵。"""

from __future__ import annotations

import numpy as np
import pandas as pd


INTRADAY_FEATURE_COLUMNS = (
    "log_return_1",
    "return_3",
    "return_5",
    "return_20",
    "candle_body_pct",
    "range_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "close_location_value",
    "ema_20_gap",
    "ema_50_gap",
    "ema_200_gap",
    "ema_20_50_atr",
    "ema_50_200_atr",
    "ema_20_slope_5",
    "adx_14",
    "plus_di_14",
    "minus_di_14",
    "supertrend_direction_10_3",
    "supertrend_distance_atr",
    "donchian_position_20",
    "donchian_breakout_up_20",
    "donchian_breakout_down_20",
    "linear_regression_slope_20",
    "stoch_rsi_k_14",
    "stoch_rsi_d_3",
    "mfi_14",
    "realized_volatility_20",
    "parkinson_volatility_20",
    "keltner_position_20",
    "squeeze_on",
    "choppiness_14",
    "relative_volume_20",
    "volume_zscore_20",
    "cmf_20",
    "ad_line_slope_20",
    "volume_oscillator_5_20",
    "rolling_vwap_gap_atr",
    "taker_buy_ratio",
    "volume_delta_ratio",
    "cvd_pressure_20",
    "trade_intensity_20",
    "average_trade_size_ratio_20",
    "amihud_20",
    "microstructure_available",
    "funding_percentile_200",
    "funding_zscore_200",
    "open_interest_change_1",
    "open_interest_zscore_50",
    "spread_fraction",
    "derivatives_available",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "is_weekend",
    "asia_session",
    "europe_session",
    "us_session",
    "minutes_to_funding_ratio",
)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0.0, np.nan)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(
        alpha=1 / window, adjust=False, min_periods=window
    ).mean()
    loss = (-delta.clip(upper=0)).ewm(
        alpha=1 / window, adjust=False, min_periods=window
    ).mean()
    relative_strength = gain / loss.replace(0.0, np.nan)
    result = 100 - 100 / (1 + relative_strength)
    result = result.mask((loss == 0) & (gain > 0), 100.0)
    return result.mask((loss == 0) & (gain == 0), 50.0)


def _directional_movement(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = true_range.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    upward = high.diff()
    downward = -low.diff()
    plus_dm = upward.where((upward > downward) & (upward > 0), 0.0)
    minus_dm = downward.where((downward > upward) & (downward > 0), 0.0)
    plus_di = 100 * plus_dm.ewm(
        alpha=1 / window, adjust=False, min_periods=window
    ).mean() / atr.replace(0.0, np.nan)
    minus_di = 100 * minus_dm.ewm(
        alpha=1 / window, adjust=False, min_periods=window
    ).mean() / atr.replace(0.0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx = dx.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    return atr, adx / 100, plus_di / 100, minus_di / 100


def _supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr: pd.Series,
    multiplier: float = 3.0,
) -> tuple[pd.Series, pd.Series]:
    """使用已知 ATR 逐根更新 Supertrend，不回寫過去訊號。"""
    midpoint = (high + low) / 2
    # Pandas 3 的 Copy-on-Write 會回傳唯讀陣列；Supertrend 必須逐根更新軌道。
    upper = (midpoint + multiplier * atr).to_numpy(dtype=float, copy=True)
    lower = (midpoint - multiplier * atr).to_numpy(dtype=float, copy=True)
    prices = close.to_numpy(dtype=float)
    direction = np.full(len(close), np.nan, dtype=float)
    line = np.full(len(close), np.nan, dtype=float)
    for index in range(1, len(close)):
        if not np.isfinite(upper[index]) or not np.isfinite(lower[index]):
            continue
        previous_upper = upper[index - 1]
        previous_lower = lower[index - 1]
        previous_close = prices[index - 1]
        if np.isfinite(previous_upper) and previous_close <= previous_upper:
            upper[index] = min(upper[index], previous_upper)
        if np.isfinite(previous_lower) and previous_close >= previous_lower:
            lower[index] = max(lower[index], previous_lower)
        previous_direction = direction[index - 1]
        if not np.isfinite(previous_direction):
            previous_direction = 1.0 if prices[index] >= midpoint.iloc[index] else -1.0
        if previous_direction > 0 and prices[index] < lower[index]:
            direction[index] = -1.0
        elif previous_direction < 0 and prices[index] > upper[index]:
            direction[index] = 1.0
        else:
            direction[index] = previous_direction
        line[index] = lower[index] if direction[index] > 0 else upper[index]
    return (
        pd.Series(direction, index=close.index),
        pd.Series(line, index=close.index),
    )


def build_intraday_features(frame: pd.DataFrame) -> pd.DataFrame:
    """建立可供 Transformer 與 SAC 使用的 BTC 日內特徵。

    所有 rolling、ewm 與事件判斷都只使用目前 K 線及過去資料。缺少交易次數、
    主動成交或合約欄位時輸出中性值，並以 available 欄位告知模型。
    """
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"日內特徵缺少必要欄位：{missing}")

    result = frame.copy()
    timestamps = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
    open_price = _numeric(result, "open")
    high = _numeric(result, "high")
    low = _numeric(result, "low")
    close = _numeric(result, "close")
    volume = _numeric(result, "volume")
    previous_close = close.shift(1)
    atr, adx, plus_di, minus_di = _directional_movement(high, low, close)
    atr_safe = atr.replace(0.0, np.nan)

    result["log_return_1"] = np.log(close / previous_close)
    for window in (3, 5, 20):
        result[f"return_{window}"] = close.pct_change(window, fill_method=None)
    body_high = pd.concat([open_price, close], axis=1).max(axis=1)
    body_low = pd.concat([open_price, close], axis=1).min(axis=1)
    candle_range = (high - low).replace(0.0, np.nan)
    result["candle_body_pct"] = _safe_divide(close - open_price, previous_close)
    result["range_pct"] = _safe_divide(high - low, previous_close)
    result["upper_wick_pct"] = _safe_divide(high - body_high, previous_close)
    result["lower_wick_pct"] = _safe_divide(body_low - low, previous_close)
    result["close_location_value"] = ((close - low) - (high - close)) / candle_range

    ema_20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
    ema_50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    ema_200 = close.ewm(span=200, adjust=False, min_periods=200).mean()
    result["ema_20_gap"] = close / ema_20.replace(0.0, np.nan) - 1
    result["ema_50_gap"] = close / ema_50.replace(0.0, np.nan) - 1
    result["ema_200_gap"] = close / ema_200.replace(0.0, np.nan) - 1
    result["ema_20_50_atr"] = (ema_20 - ema_50) / atr_safe
    result["ema_50_200_atr"] = (ema_50 - ema_200) / atr_safe
    result["ema_20_slope_5"] = ema_20.pct_change(5, fill_method=None)
    result["adx_14"] = adx
    result["plus_di_14"] = plus_di
    result["minus_di_14"] = minus_di

    supertrend_direction, supertrend_line = _supertrend(high, low, close, atr)
    result["supertrend_direction_10_3"] = supertrend_direction
    result["supertrend_distance_atr"] = (close - supertrend_line) / atr_safe
    previous_high = high.rolling(20, min_periods=20).max().shift(1)
    previous_low = low.rolling(20, min_periods=20).min().shift(1)
    donchian_range = (previous_high - previous_low).replace(0.0, np.nan)
    result["donchian_position_20"] = (close - previous_low) / donchian_range
    result["donchian_breakout_up_20"] = (close > previous_high).astype("float64")
    result["donchian_breakout_down_20"] = (close < previous_low).astype("float64")
    x = np.arange(20, dtype=float)
    x_centered = x - x.mean()
    denominator = float(np.square(x_centered).sum())
    result["linear_regression_slope_20"] = close.rolling(20, min_periods=20).apply(
        lambda values: float(np.dot(values - values.mean(), x_centered) / denominator)
        / max(abs(float(values.mean())), 1e-12),
        raw=True,
    )

    rsi = _rsi(close)
    rsi_low = rsi.rolling(14, min_periods=14).min()
    rsi_high = rsi.rolling(14, min_periods=14).max()
    rsi_range = rsi_high - rsi_low
    stoch_rsi = (rsi - rsi_low) / rsi_range.replace(0.0, np.nan)
    stoch_rsi = stoch_rsi.mask(
        rsi_low.notna() & rsi_high.notna() & rsi_range.eq(0.0),
        0.5,
    )
    result["stoch_rsi_k_14"] = stoch_rsi.rolling(3, min_periods=3).mean()
    result["stoch_rsi_d_3"] = result["stoch_rsi_k_14"].rolling(3, min_periods=3).mean()
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume
    positive_flow = raw_money_flow.where(typical_price.diff() > 0, 0.0)
    negative_flow = raw_money_flow.where(typical_price.diff() < 0, 0.0)
    positive_sum = positive_flow.rolling(14, min_periods=14).sum()
    negative_sum = negative_flow.rolling(14, min_periods=14).sum()
    money_ratio = positive_sum / negative_sum.replace(0.0, np.nan)
    mfi = (100 - 100 / (1 + money_ratio)) / 100
    mfi = mfi.mask((negative_sum == 0) & (positive_sum > 0), 1.0)
    mfi = mfi.mask((positive_sum == 0) & (negative_sum > 0), 0.0)
    result["mfi_14"] = mfi.mask(
        (positive_sum == 0) & (negative_sum == 0),
        0.5,
    )

    log_return = result["log_return_1"]
    result["realized_volatility_20"] = np.sqrt(
        log_return.pow(2).rolling(20, min_periods=20).sum()
    )
    parkinson_variance = np.log(high / low.replace(0.0, np.nan)).pow(2)
    result["parkinson_volatility_20"] = np.sqrt(
        parkinson_variance.rolling(20, min_periods=20).mean() / (4 * np.log(2))
    )
    result["keltner_position_20"] = (close - ema_20) / (2 * atr_safe)
    close_std = close.rolling(20, min_periods=20).std(ddof=0)
    result["squeeze_on"] = (
        (ema_20 + 2 * close_std < ema_20 + 1.5 * atr)
        & (ema_20 - 2 * close_std > ema_20 - 1.5 * atr)
    ).astype("float64")
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    total_range = high.rolling(14, min_periods=14).max() - low.rolling(
        14, min_periods=14
    ).min()
    result["choppiness_14"] = np.log10(
        true_range.rolling(14, min_periods=14).sum() / total_range.replace(0.0, np.nan)
    ) / np.log10(14)

    volume_mean = volume.rolling(20, min_periods=20).mean()
    volume_std = volume.rolling(20, min_periods=20).std(ddof=0)
    rolling_volume = volume.rolling(20, min_periods=20).sum()
    result["relative_volume_20"] = volume / volume_mean.replace(0.0, np.nan)
    result["volume_zscore_20"] = (volume - volume_mean) / volume_std.replace(0.0, np.nan)
    money_flow_multiplier = ((close - low) - (high - close)) / candle_range
    money_flow_volume = money_flow_multiplier * volume
    result["cmf_20"] = money_flow_volume.rolling(20, min_periods=20).sum() / rolling_volume.replace(
        0.0, np.nan
    )
    ad_line = money_flow_volume.cumsum()
    result["ad_line_slope_20"] = ad_line.diff(20) / rolling_volume.replace(0.0, np.nan)
    volume_ema_5 = volume.ewm(span=5, adjust=False, min_periods=5).mean()
    volume_ema_20 = volume.ewm(span=20, adjust=False, min_periods=20).mean()
    result["volume_oscillator_5_20"] = volume_ema_5 / volume_ema_20.replace(0.0, np.nan) - 1
    rolling_vwap = (typical_price * volume).rolling(20, min_periods=20).sum() / rolling_volume.replace(
        0.0, np.nan
    )
    result["rolling_vwap_gap_atr"] = (close - rolling_vwap) / atr_safe

    taker_volume = _numeric(result, "taker_buy_base_volume")
    trades = _numeric(result, "number_of_trades")
    micro_available = taker_volume.notna() | trades.notna()
    taker_ratio = (taker_volume / volume.replace(0.0, np.nan)).clip(0.0, 1.0)
    delta_ratio = (2 * taker_ratio - 1).clip(-1.0, 1.0)
    cumulative_delta = (delta_ratio.fillna(0.0) * volume).cumsum()
    result["taker_buy_ratio"] = taker_ratio.fillna(0.5)
    result["volume_delta_ratio"] = delta_ratio.fillna(0.0)
    result["cvd_pressure_20"] = cumulative_delta.diff(20) / rolling_volume.replace(0.0, np.nan)
    trade_mean = trades.rolling(20, min_periods=20).mean()
    result["trade_intensity_20"] = (trades / trade_mean.replace(0.0, np.nan)).fillna(0.0)
    average_size = volume / trades.replace(0.0, np.nan)
    result["average_trade_size_ratio_20"] = (
        average_size / average_size.rolling(20, min_periods=20).mean().replace(0.0, np.nan)
    ).fillna(0.0)
    quote_volume = _numeric(result, "quote_asset_volume").combine_first(close * volume)
    result["amihud_20"] = (
        log_return.abs() / quote_volume.replace(0.0, np.nan)
    ).rolling(20, min_periods=20).mean()
    result["microstructure_available"] = micro_available.astype("float64")

    funding = _numeric(result, "funding_rate")
    funding_mean = funding.rolling(200, min_periods=20).mean()
    funding_std = funding.rolling(200, min_periods=20).std(ddof=0)
    result["funding_percentile_200"] = funding.rolling(200, min_periods=20).rank(pct=True).fillna(0.5)
    result["funding_zscore_200"] = ((funding - funding_mean) / funding_std.replace(0.0, np.nan)).fillna(0.0)
    open_interest = _numeric(result, "open_interest")
    if open_interest.notna().any():
        oi_change = open_interest.pct_change(fill_method=None)
    else:
        oi_change = _numeric(result, "open_interest_change")
    oi_mean = oi_change.rolling(50, min_periods=20).mean()
    oi_std = oi_change.rolling(50, min_periods=20).std(ddof=0)
    result["open_interest_change_1"] = oi_change.fillna(0.0)
    result["open_interest_zscore_50"] = ((oi_change - oi_mean) / oi_std.replace(0.0, np.nan)).fillna(0.0)
    spread_fraction = _numeric(result, "spread_bps") / 10_000
    result["spread_fraction"] = spread_fraction.fillna(0.0)
    result["derivatives_available"] = pd.concat(
        [funding, open_interest, _numeric(result, "open_interest_change"), spread_fraction],
        axis=1,
    ).notna().any(axis=1).astype("float64")

    hour = timestamps.dt.hour + timestamps.dt.minute / 60
    hour_angle = 2 * np.pi * hour / 24
    weekday_angle = 2 * np.pi * timestamps.dt.weekday / 7
    result["hour_sin"] = np.sin(hour_angle)
    result["hour_cos"] = np.cos(hour_angle)
    result["weekday_sin"] = np.sin(weekday_angle)
    result["weekday_cos"] = np.cos(weekday_angle)
    result["is_weekend"] = timestamps.dt.weekday.ge(5).astype("float64")
    result["asia_session"] = ((hour >= 0) & (hour < 9)).astype("float64")
    result["europe_session"] = ((hour >= 7) & (hour < 17)).astype("float64")
    result["us_session"] = ((hour >= 13) & (hour < 22)).astype("float64")
    hours_since_funding = np.mod(hour, 8.0)
    result["minutes_to_funding_ratio"] = (8.0 - hours_since_funding) / 8.0

    result.loc[:, INTRADAY_FEATURE_COLUMNS] = result.loc[
        :, INTRADAY_FEATURE_COLUMNS
    ].replace([np.inf, -np.inf], np.nan)
    return result
