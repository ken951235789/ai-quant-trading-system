"""既有事件成交契約的 NumPy 批次加速；逐筆參考引擎仍是核對基準。"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from ai_quant_trading.transformer.strategy_events import (
    StrategyEventConfig, _indicator_frame, replay_event, validate_event_bars,
)


LEVELS = {
    "timeframe": ("15m", "1h"), "entry": ("breakout", "pullback"),
    "adx": (15, 20, 25), "confirmation": (1, 2), "lookback": (10, 20, 40),
    "stop_atr": (1.0, 1.5, 2.0), "reward_r": (1.0, 2.0, 3.0),
    "holding_bars": (16, 32, 64), "cooldown_bars": (0, 4),
    "regime_exit": ("loss", "opposite", "disabled"),
}
REASONS = {1: "stop", 2: "target", 3: "regime", 4: "time"}


@dataclass(frozen=True)
class GridParams:
    timeframe: str
    entry: str
    adx: int
    confirmation: int
    lookback: int
    stop_atr: float
    reward_r: float
    holding_bars: int
    cooldown_bars: int
    regime_exit: str

    def __post_init__(self):
        for key, choices in LEVELS.items():
            if getattr(self, key) not in choices:
                raise ValueError(f"參數不在本次預登記搜尋範圍：{key}")

    @property
    def signal_key(self) -> tuple:
        return self.timeframe, self.entry, self.adx, self.confirmation, self.lookback


def parameter_grid() -> list[GridParams]:
    return [GridParams(*values) for values in product(*LEVELS.values())]


def prepare_cache(bars: pd.DataFrame) -> dict:
    validate_event_bars(bars)
    raw = bars.copy().reset_index(drop=True)
    raw["timestamp"] = pd.to_datetime(raw.timestamp, utc=True)
    indexed = raw.set_index("timestamp")
    cache = {"raw": raw, "decisions": pd.DatetimeIndex(raw.timestamp + pd.Timedelta(minutes=15))}
    for token, period, size in (("15m", "15min", 1), ("1h", "1h", 4)):
        grouped = indexed.resample(period, closed="left", label="left")
        sampled = grouped.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        sampled = sampled.loc[grouped.close.count().eq(size)]
        sampled.index += pd.Timedelta(period)
        cache[token] = {"bars": sampled, "indicators": _indicator_frame(sampled)}
    return cache


def signal_frame(cache: dict, params: GridParams) -> pd.DataFrame:
    decisions = cache["decisions"]
    hourly = cache["1h"]["indicators"]
    direction = np.sign(hourly.ema50_200_spread)
    active = hourly.adx_14 * 100 >= params.adx
    if params.confirmation == 2:
        active &= active.shift(1, fill_value=False) & direction.eq(direction.shift(1))
    regime = direction.where(active, 0).fillna(0).reindex(decisions, method="ffill").fillna(0).to_numpy()
    bars = cache[params.timeframe]["bars"]
    close = bars.close
    if params.entry == "breakout":
        long = close > bars.high.shift(1).rolling(params.lookback).max()
        short = close < bars.low.shift(1).rolling(params.lookback).min()
    else:
        ema = close.ewm(span=params.lookback, adjust=False, min_periods=params.lookback).mean()
        long = (close.shift(1) <= ema.shift(1)) & (close > ema) & (close > bars.open)
        short = (close.shift(1) >= ema.shift(1)) & (close < ema) & (close < bars.open)
    side = pd.Series(np.select([long & ~short, short & ~long], [1, -1], 0), index=bars.index)
    # 不 ffill 進場訊號：小時訊號只有真正收盤的一個決策時點有效。
    signal = side.reindex(decisions, fill_value=0).to_numpy()
    atr = (cache[params.timeframe]["indicators"].atr_pct * close).reindex(decisions, method="ffill").to_numpy()
    valid = (decisions >= decisions[0] + pd.Timedelta(hours=1000)) & np.isfinite(atr) & (atr > 0)
    result = cache["raw"].copy()
    result["event_atr"] = atr
    result["event_regime"] = regime
    result["candidate_side"] = np.where(valid & (signal == regime), signal, 0)
    return result


def _settle(table: dict, frame: pd.DataFrame, cost: dict) -> dict:
    result = dict(table)
    side = table["side"]
    friction = (cost["slippage"] + cost["spread"] / 2) / 10000
    fill = table["exit_reference"] * (1 - side * friction)
    entry = table["entry_price"]
    fees = cost["fee"] / 10000 * (1 + fill / entry)
    stamps = frame.timestamp.dt.as_unit("ns").astype("int64").to_numpy()
    bucket = 8 * 3600 * 1_000_000_000
    count = stamps[table["exit_endpoint"]] // bucket - stamps[table["entry_endpoint"]] // bucket
    funding = count * cost["funding"] / 10000
    result.update(exit_price=fill, fee_return=fees, funding_return=funding,
                  net_return=side * (fill / entry - 1) - fees - funding)
    return result


def batch_events(frame: pd.DataFrame, params: GridParams, cost: dict) -> dict:
    """一次求出候選的最早退出，順序完全沿用 regime、stop、target 契約。"""
    if min(cost.values()) < 0 or not np.isfinite(list(cost.values())).all():
        raise ValueError("成本必須有限且非負")
    points = np.flatnonzero(frame.candidate_side.to_numpy())
    points = points[points + 65 < len(frame)]
    side = frame.candidate_side.to_numpy(dtype=int)[points]
    distance = frame.event_atr.to_numpy()[points]
    opening, high, low = (frame[column].to_numpy(dtype=float) for column in ("open", "high", "low"))
    friction = (cost["slippage"] + cost["spread"] / 2) / 10000
    entry = opening[points + 1] * (1 + side * friction)
    stop = entry - side * params.stop_atr * distance
    target = entry + side * params.stop_atr * params.reward_r * distance
    if not np.isfinite([entry, stop, target, distance]).all() or (np.minimum(stop, target) <= 0).any() or (distance <= 0).any():
        raise ValueError("候選保護價不合法")
    future = points[:, None] + np.arange(1, 65)[None, :]
    stops = np.where(side[:, None] == 1, low[future] <= stop[:, None], high[future] >= stop[:, None])
    targets = np.where(side[:, None] == 1, high[future] >= target[:, None], low[future] <= target[:, None])
    regime = frame.event_regime.to_numpy()[future - 1]
    changes = np.zeros_like(stops)
    if params.regime_exit == "loss":
        changes = regime != side[:, None]
    elif params.regime_exit == "opposite":
        changes = regime == -side[:, None]
    changes[:, 0] = False
    codes = np.where(changes, 3, np.where(stops, 1, np.where(targets, 2, 0)))
    has_exit = codes.any(axis=1)
    # 尋找第一個退出事件，不能用原因編碼的大小排序。
    offset = (codes != 0).argmax(axis=1)
    exits = np.where(has_exit, points + offset + 1, points + 65)
    reasons = np.where(has_exit, codes[np.arange(len(points)), offset], 4)
    references = opening[exits].copy()
    stop_fill = np.where(side == 1, np.minimum(references, stop), np.maximum(references, stop))
    references = np.where(reasons == 1, stop_fill, np.where(reasons == 2, target, references))
    return _settle({"endpoint": points, "entry_endpoint": points + 1, "exit_endpoint": exits,
        "side": side, "entry_price": entry, "stop_price": stop, "target_price": target,
        "exit_reference": references, "reason_code": reasons}, frame, cost)


def holding_limit(events: dict, frame: pd.DataFrame, holding: int, cost: dict) -> dict:
    if holding not in LEVELS["holding_bars"]:
        raise ValueError("持有上限不在範圍")
    deadline = events["endpoint"] + holding + 1
    # 截止棒的開盤已平倉，不能再吃到那一棒盤中的停利。
    forced = events["exit_endpoint"] >= deadline
    result = dict(events)
    result["exit_endpoint"] = np.where(forced, deadline, events["exit_endpoint"])
    result["reason_code"] = np.where(forced, 4, events["reason_code"])
    result["exit_reference"] = np.where(forced, frame.open.to_numpy()[deadline], events["exit_reference"])
    return _settle(result, frame, cost)


def scheduled_indices(events: dict, start: int, stop: int, cooldown: int) -> np.ndarray:
    points, exits = events["endpoint"], events["exit_endpoint"]
    index, end = np.searchsorted(points, [start, stop - 65])
    selected = []
    while index < end:
        selected.append(index)
        index = np.searchsorted(points, exits[index] + cooldown + 1)
    return np.asarray(selected, dtype=int)


def reference_check(events: dict, frame: pd.DataFrame, params: GridParams, cost: dict, index: int) -> None:
    config = StrategyEventConfig(max_holding_bars=params.holding_bars, cooldown_bars=params.cooldown_bars,
        stop_atr=params.stop_atr, target_atr=params.stop_atr * params.reward_r,
        spread_bps=cost["spread"], funding_reserve_bps_per_settlement=cost["funding"])
    reference = replay_event(frame, int(events["endpoint"][index]), config,
        fee_bps_per_side=cost["fee"], slippage_bps_per_side=cost["slippage"], regime_exit=params.regime_exit)
    for key, value in reference.items():
        if key == "exit_reason":
            if REASONS[int(events["reason_code"][index])] != value:
                raise ValueError("批次引擎退出原因與參考引擎不同")
        else:
            np.testing.assert_allclose(events[key][index], value, rtol=1e-12, atol=1e-14,
                                       err_msg=f"批次與參考引擎不同：{key}")
