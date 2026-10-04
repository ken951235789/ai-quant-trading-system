"""六個事前固定的複合策略假說，只使用決策前已收盤的同源行情。"""

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from ai_quant_trading.backtesting.grid_execution import GridParams, prepare_cache, signal_frame
from ai_quant_trading.transformer.strategy_events import _indicator_frame


FAMILIES = {
    "trend_pullback": "跨週期趨勢回調",
    "squeeze_breakout": "波動壓縮突破",
    "vwap_reversion": "區間 VWAP 均值回歸",
    "liquidity_reclaim": "假突破收回",
    "donchian_retest": "突破後回測確認",
    "regime_switch": "趨勢／震盪分流",
}
RISK_LEVELS = {"stop_atr": (1.5, 2.), "reward_r": (1., 2., 3.),
              "holding_bars": (32, 64), "cooldown_bars": (0, 4), "regime_exit": ("opposite", "disabled")}


@dataclass(frozen=True)
class ComplexParams:
    family: str
    timeframe: str
    strict: bool
    stop_atr: float
    reward_r: float
    holding_bars: int
    cooldown_bars: int
    regime_exit: str

    @property
    def signal_key(self):
        return self.family, self.timeframe, self.strict

    def execution(self):
        # 只借用已驗證的成交參數容器，不以其中 breakout 等欄位產生新策略訊號。
        return GridParams(self.timeframe, "breakout", 25, 2, 20, self.stop_atr, self.reward_r,
                          self.holding_bars, self.cooldown_bars, self.regime_exit)


def complex_grid():
    baseline = ComplexParams("original", "15m", False, 2., 2., 32, 4, "loss")
    return [baseline] + [ComplexParams(*v) for v in product(FAMILIES, ("15m", "1h"), (False, True), *RISK_LEVELS.values())]


def prepare_complex_cache(bars):
    cache = prepare_cache(bars)
    grouped = cache["raw"].set_index("timestamp").resample("4h", closed="left", label="left")
    sampled = grouped.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    sampled = sampled.loc[grouped.close.count().eq(16)]
    sampled.index += pd.Timedelta(hours=4)
    cache["4h"] = {"bars": sampled, "indicators": _indicator_frame(sampled)}
    cache["complex_features"] = {tf: feature_frame(cache, tf) for tf in ("15m", "1h")}
    return cache


def feature_frame(cache, timeframe):
    bars = cache[timeframe]["bars"].copy()
    indicators = cache[timeframe]["indicators"]
    close = bars.close
    bars["atr"] = indicators.atr_pct * close
    bars["ema20"] = close.ewm(span=20, adjust=False, min_periods=20).mean()
    bars["mid20"] = close.rolling(20).mean()
    bars["std20"] = close.rolling(20).std(ddof=0)
    bars["upper"] = bars.mid20 + 2 * bars.std20
    bars["lower"] = bars.mid20 - 2 * bars.std20
    bars["rsi"] = indicators.rsi_14 * 100
    bars["rvol"] = bars.volume / bars.volume.shift(1).rolling(20).mean().replace(0, np.nan)
    bars["prior_high"] = bars.high.shift(1).rolling(20).max()
    bars["prior_low"] = bars.low.shift(1).rolling(20).min()
    # 以開盤日歸屬 UTC 日 VWAP；午夜收盤的最後一棒仍屬於前一交易日。
    days = (bars.index - pd.Timedelta(timeframe)).floor("1D")
    typical = (bars.high + bars.low + close) / 3
    bars["vwap"] = (typical * bars.volume).groupby(days).cumsum() / bars.volume.groupby(days).cumsum().replace(0, np.nan)
    span = (bars.high - bars.low).replace(0, np.nan)
    bars["lower_wick"] = (bars[["open", "close"]].min(axis=1) - bars.low) / span
    bars["upper_wick"] = (bars.high - bars[["open", "close"]].max(axis=1)) / span
    for tf in ("1h", "4h"):
        higher = cache[tf]["indicators"].reindex(bars.index, method="ffill")
        bars[f"{tf}_direction"] = np.sign(higher.ema50_200_spread)
        bars[f"{tf}_adx"] = higher.adx_14 * 100
        bars[f"{tf}_slope"] = higher.ema50_slope
    return bars


def complex_signals(f: pd.DataFrame, family: str, strict: bool) -> pd.Series:
    """所有條件在本棒收盤可知；不使用未來 pivot 或真實流動性標籤。"""
    if family not in FAMILIES:
        raise ValueError("未登記的複合策略")
    a, c = f.atr, f.close
    threshold = 25 if strict else 20
    trend_long = (f["1h_direction"] == 1) & (f["4h_direction"] == 1) & (f["1h_adx"] >= threshold)
    trend_short = (f["1h_direction"] == -1) & (f["4h_direction"] == -1) & (f["1h_adx"] >= threshold)
    pull_long = trend_long & (f.low <= f.ema20) & (c > f.ema20) & (c > f.open) & f.rsi.between(50 if strict else 45, 65 if strict else 70)
    pull_short = trend_short & (f.high >= f.ema20) & (c < f.ema20) & (c < f.open) & f.rsi.between(35 if strict else 30, 50 if strict else 55)
    pull_long &= f.rvol >= (1. if strict else .8)
    pull_short &= f.rvol >= (1. if strict else .8)
    if strict:
        pull_long &= (c > f.high.shift(1)) & (f["4h_slope"] > 0)
        pull_short &= (c < f.low.shift(1)) & (f["4h_slope"] < 0)
    quiet = (f["1h_adx"] < (15 if strict else 20)) & (f["4h_slope"].abs() < (.002 if strict else .003))
    normal_volume = f.rvol.between(.5, 2.5 if strict else 3.)
    revert_long = quiet & normal_volume & (c.shift(1) < f.lower.shift(1)) & (c >= f.lower) & (c < f.vwap - .5 * a) & (f.rsi < (40 if strict else 45)) & (c > f.open)
    revert_short = quiet & normal_volume & (c.shift(1) > f.upper.shift(1)) & (c <= f.upper) & (c > f.vwap + .5 * a) & (f.rsi > (60 if strict else 55)) & (c < f.open)
    if family == "trend_pullback":
        long, short = pull_long, pull_short
    elif family == "squeeze_breakout":
        squeeze = 2 * f.std20 < (1.2 if strict else 1.5) * a
        recent = squeeze.shift(1).rolling(6, min_periods=6).max().eq(1)
        active = recent & (f.rvol >= (1.3 if strict else 1.))
        excess = .1 if strict else 0.
        long = active & trend_long & (c > f.prior_high + excess * a) & (c > f.open)
        short = active & trend_short & (c < f.prior_low - excess * a) & (c < f.open)
    elif family == "vwap_reversion":
        long, short = revert_long, revert_short
    elif family == "liquidity_reclaim":
        active = (f["1h_adx"] < (20 if strict else 25)) & (f.rvol >= (1.3 if strict else 1.))
        penetration, reclaim, wick = (.15, .05, .6) if strict else (.05, 0., .4)
        long = active & (f.low < f.prior_low - penetration * a) & (c > f.prior_low + reclaim * a) & (c > f.open) & (f.lower_wick >= wick)
        short = active & (f.high > f.prior_high + penetration * a) & (c < f.prior_high - reclaim * a) & (c < f.open) & (f.upper_wick >= wick)
        if strict:
            long &= f["4h_direction"] >= 0
            short &= f["4h_direction"] <= 0
    elif family == "donchian_retest":
        volume = f.rvol >= (1.3 if strict else 1.)
        up = trend_long & volume & (c > f.prior_high) & (c.shift(1) <= f.prior_high.shift(1))
        down = trend_short & volume & (c < f.prior_low) & (c.shift(1) >= f.prior_low.shift(1))
        # shift 後才延續已知突破價，禁止同一根棒同時宣告突破與回測。
        life = 4 if strict else 8
        up_level = f.prior_high.where(up).shift(1).ffill(limit=life - 1)
        down_level = f.prior_low.where(down).shift(1).ffill(limit=life - 1)
        tolerance = .1 if strict else .2
        failed_distance = .25 if strict else .5
        # 已被穿透作廢的突破價，不得在後面反彈時重新冒充有效回測。
        up_failed = (f.low < up_level - failed_distance * a).groupby(up.shift(1, fill_value=False).cumsum()).cummax()
        down_failed = (f.high > down_level + failed_distance * a).groupby(down.shift(1, fill_value=False).cumsum()).cummax()
        up_level = up_level.where(~up_failed)
        down_level = down_level.where(~down_failed)
        long = trend_long & f.rvol.ge(.9 if strict else .7) & (f.low <= up_level + tolerance * a) & (f.low >= up_level - failed_distance * a) & (c > up_level) & (c <= up_level + a) & (c > f.open)
        short = trend_short & f.rvol.ge(.9 if strict else .7) & (f.high >= down_level - tolerance * a) & (f.high <= down_level + failed_distance * a) & (c < down_level) & (c >= down_level - a) & (c < f.open)
    else:
        trending = f["1h_adx"] >= (30 if strict else 25)
        ranging = f["1h_adx"] < (15 if strict else 18)
        long = (trending & pull_long) | (ranging & revert_long)
        short = (trending & pull_short) | (ranging & revert_short)
    return pd.Series(np.select([long & ~short, short & ~long], [1, -1], 0), index=f.index, dtype=int)


def complex_frame(cache, params: ComplexParams):
    if params.family == "original":
        return signal_frame(cache, params.execution())
    features = cache["complex_features"][params.timeframe]
    signals = complex_signals(features, params.family, params.strict)
    decisions = cache["decisions"]
    hourly = cache["1h"]["indicators"]
    direction = np.sign(hourly.ema50_200_spread)
    active = hourly.adx_14.ge(.25)
    confirmed = active & active.shift(1, fill_value=False) & direction.eq(direction.shift(1))
    regime = direction.where(confirmed, 0).fillna(0)
    result = cache["raw"].copy()
    result["event_atr"] = features.atr.reindex(decisions, method="ffill").to_numpy()
    result["event_regime"] = regime.reindex(decisions, method="ffill").fillna(0).to_numpy()
    side = signals.reindex(decisions, fill_value=0).to_numpy()
    valid = (decisions >= decisions[0] + pd.Timedelta(hours=1000)) & np.isfinite(result.event_atr) & result.event_atr.gt(0)
    result["candidate_side"] = np.where(valid, side, 0)
    return result
