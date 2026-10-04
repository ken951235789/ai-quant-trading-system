"""候選事件的固定特徵消融；不改訊號、標籤、成本或成交規則。"""

import numpy as np
import pandas as pd

from ai_quant_trading.transformer.strategy_events import validate_event_bars


FEATURE_SETS = ("existing", "compact", "compact_flow", "compact_setup", "compact_combined")
FLOW_COLUMNS = ("quote_asset_volume", "number_of_trades", "taker_buy_base_volume",
                "taker_buy_quote_volume")
FLOW_SETS = {"compact_flow", "compact_combined"}
SETUP_SETS = {"compact_setup", "compact_combined"}


def validate_flow(raw):
    """缺漏不可冒充零成交；只有交易所明確記錄的零量棒才使用中性值。"""
    missing = set(FLOW_COLUMNS).difference(raw)
    if missing:
        raise ValueError(f"成交特徵缺少原始欄位：{sorted(missing)}")
    values = raw[["volume", *FLOW_COLUMNS]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("成交來源含缺漏、負數或非有限值，不可補零訓練")
    volume, quote, count, buy, buy_quote = values.T
    if not np.equal(count, np.floor(count)).all():
        raise ValueError("成交筆數必須為整數")
    if (buy > volume + np.maximum(volume, 1) * 1e-9).any() or (
            buy_quote > quote + np.maximum(quote, 1) * 1e-9).any():
        raise ValueError("主動買入量大於總成交量")
    positive = volume > 0
    if ((count > 0) != positive).any() or ((quote > 0) != positive).any():
        raise ValueError("成交筆數、基礎量與報價量的零值狀態不一致")


def enrich_flow_source(raw, supplemental):
    """以相同時間與市場逐根核對 OHLCV，僅補入指定成交欄位，不帶入外部衍生特徵。"""
    validate_event_bars(raw)
    validate_event_bars(supplemental)
    def identity(x):
        return x.assign(timestamp=pd.to_datetime(x.timestamp, utc=True)).set_index("timestamp")
    left, right = identity(raw), identity(supplemental)
    aligned = right.reindex(left.index)
    if aligned.close.isna().any():
        raise ValueError("補充成交來源缺少研究時間戳")
    for col in ("open", "high", "low", "close", "volume"):
        if not np.array_equal(left[col].to_numpy(dtype=float), aligned[col].to_numpy(dtype=float)):
            raise ValueError(f"補充成交來源與研究 OHLCV 不一致：{col}")
    validate_flow(aligned)
    result = raw.copy().reset_index(drop=True)
    for col in FLOW_COLUMNS:
        if col in raw and not np.array_equal(raw[col].to_numpy(dtype=float), aligned[col].to_numpy(dtype=float)):
            raise ValueError(f"已有成交欄位與補充來源衝突：{col}")
        result[col] = aligned[col].to_numpy(dtype=float)
    return result


def compact_columns():
    """事先依機制固定，不用測試期相關性或重要性挑欄位。"""
    common = ("return_1", "return_4", "ema20_gap", "ema200_gap", "ema50_slope",
              "atr_pct", "adx_14", "di_balance", "rsi_14", "relative_volume_prev20")
    context = ("body_atr", "upper_wick_atr", "lower_wick_atr", "close_location",
               "previous_day_high_distance_atr", "previous_day_low_distance_atr",
               "day_vwap_distance_atr", "weekday_sin", "weekday_cos")
    return {f"mtf_{tf}_{name}" for tf in ("15m", "1h", "4h") for name in common} | {
        f"mtf_15m_context_{name}" for name in context} | {
        "mtf_15m_bollinger_z", "mtf_15m_bollinger_width",
        "mtf_15m_cost_roundtrip_over_atr", "mtf_15m_cost_minutes_to_settlement"}


def _flow_features(raw):
    validate_flow(raw)
    bars = raw.assign(timestamp=pd.to_datetime(raw.timestamp, utc=True)).set_index("timestamp")
    decisions = pd.DatetimeIndex(bars.index + pd.Timedelta(minutes=15))
    result = pd.DataFrame(index=raw.index)
    for tf, count in (("15min", 1), ("1h", 4), ("4h", 16)):
        grouped = bars.resample(tf, closed="left", label="left")
        summed = grouped[["volume", "taker_buy_base_volume"]].sum(min_count=count)
        summed = summed.loc[grouped.volume.count().eq(count)]
        ratio = 2 * summed.taker_buy_base_volume / summed.volume.replace(0, np.nan) - 1
        ratio = ratio.where(summed.volume.gt(0), 0.)
        ratio.index += pd.Timedelta(tf)
        token = "15m" if tf == "15min" else tf
        result[f"mtf_{token}_flow_imbalance"] = ratio.reindex(decisions, method="ffill").to_numpy()
    trades = raw.number_of_trades.astype(float)
    average = raw.quote_asset_volume / trades.replace(0, np.nan)
    # 分母只含過去棒；零成交大小不存在，保留 NaN 與模型缺值遮罩。
    result["mtf_15m_flow_trade_activity"] = trades / trades.shift(1).rolling(20).mean().replace(0, np.nan)
    result["mtf_15m_flow_average_trade_relative"] = average / average.shift(1).rolling(20).mean().replace(0, np.nan)
    result["mtf_15m_flow_has_trades"] = trades.gt(0).astype(float)
    return result


def _capped_age(event, cap=96):
    positions = pd.Series(np.arange(len(event)), index=event.index)
    last = positions.where(event).ffill()
    # 未觀察到事件表示最近窗口無事件，不宣稱知道更早的真正起點。
    return (positions - last).fillna(cap).clip(upper=cap) / cap


def _setup_features(frame):
    close, atr = frame.close, frame.event_atr.replace(0, np.nan)
    high20, low20 = frame.high.shift(1).rolling(20).max(), frame.low.shift(1).rolling(20).min()
    high96, low96 = frame.high.shift(1).rolling(96).max(), frame.low.shift(1).rolling(96).min()
    previous_high = close - frame.mtf_15m_context_previous_day_high_distance_atr * atr
    previous_low = close - frame.mtf_15m_context_previous_day_low_distance_atr * atr
    levels = pd.concat([high20, low20, high96, low96, previous_high, previous_low], axis=1)
    delta = levels.sub(close, axis=0).div(atr, axis=0)
    long_room = delta.where(delta >= 0).min(axis=1)
    short_room = -delta.where(delta <= 0).max(axis=1)
    valid = levels.notna().all(axis=1) & atr.notna()
    regime = frame.event_regime
    regime_age = regime.groupby(regime.ne(regime.shift()).cumsum()).cumcount().clip(upper=96) / 96
    values = {
        "long_room_atr": long_room.where(valid), "short_room_atr": short_room.where(valid),
        "long_level_known": long_room.notna().astype(float).where(valid),
        "short_level_known": short_room.notna().astype(float).where(valid),
        "pullback_from_high_atr": (high20 - close) / atr,
        "rebound_from_low_atr": (close - low20) / atr,
        "since_up_break_capped96": _capped_age(close > high20),
        "since_down_break_capped96": _capped_age(close < low20),
        "regime_age_capped96": regime_age,
    }
    return pd.DataFrame({f"mtf_15m_setup_{k}": v for k, v in values.items()}, index=frame.index)


def apply_feature_set(frame, raw, feature_set):
    if feature_set not in FEATURE_SETS:
        raise ValueError("未登記的候選特徵集合")
    if feature_set == "existing":
        return frame
    result = frame.copy()
    extra = []
    if feature_set in FLOW_SETS:
        extra.append(_flow_features(raw.reset_index(drop=True)))
    if feature_set in SETUP_SETS:
        extra.append(_setup_features(frame))
    allowed = compact_columns()
    missing = allowed.difference(result)
    if missing:
        raise ValueError(f"精簡特徵缺少必要的 context_v2/cost 欄位：{sorted(missing)}")
    result = result.drop(columns=[c for c in result if c.startswith("mtf_") and c not in allowed])
    for block in extra:
        for col in block:
            result[col] = block[col].to_numpy()
    return result
