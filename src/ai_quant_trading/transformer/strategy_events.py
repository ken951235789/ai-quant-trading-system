"""Transformer 固定策略的因果事件與共用成交回放；不依賴 RL 執行引擎。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ai_quant_trading.features.intraday import _directional_movement, _rsi


@dataclass(frozen=True, slots=True)
class StrategyEventConfig:
    """規則版本固定；資金費率是雙向成本準備金，不冒充歷史實際費率。"""

    max_holding_bars: int = 32
    cooldown_bars: int = 4
    warmup_hours: int = 1000
    stop_atr: float = 2.0
    target_atr: float = 4.0
    adx_threshold: float = 25.0
    spread_bps: float = 1.0
    funding_reserve_bps_per_settlement: float = 1.0
    minimum_probability: float = 0.55

    def __post_init__(self) -> None:
        if self.max_holding_bars < 1 or self.cooldown_bars < 0 or self.warmup_hours < 200:
            raise ValueError("事件持有／冷卻設定不合法，暖機至少 200 小時")
        values = (self.stop_atr, self.target_atr, self.adx_threshold, self.spread_bps,
                  self.funding_reserve_bps_per_settlement, self.minimum_probability)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("事件參數必須為有限數值")
        if self.stop_atr <= 0 or self.target_atr <= 0 or not 0 <= self.adx_threshold <= 100:
            raise ValueError("ATR 倍數或 ADX 門檻不合法")
        if min(self.spread_bps, self.funding_reserve_bps_per_settlement) < 0:
            raise ValueError("成本假設不可為負")
        if not 0 < self.minimum_probability < 1:
            raise ValueError("機率門檻必須介於 0 與 1")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_event_bars(frame: pd.DataFrame) -> None:
    """拒絕缺棒、重複、混市場、未收盤與不合法 OHLC，不默默修補。"""
    required = {"timestamp", "open", "high", "low", "close", "volume",
                "symbol", "exchange", "interval"}
    if required.difference(frame):
        raise ValueError(f"事件資料缺少欄位：{sorted(required.difference(frame))}")
    if len(frame) < 2:
        raise ValueError("事件資料至少需要兩根 K 棒")
    for column, allowed in (("symbol", {"BTCUSDT", "BTC/USDT", "BTC/USDT:USDT"}),
                            ("exchange", {"binance_futures", "binance_usdm"}),
                            ("interval", {"15m"})):
        if frame[column].isna().any() or not set(frame[column].astype(str)).issubset(allowed):
            raise ValueError(f"事件模式僅接受 Binance USD-M BTCUSDT 15m：{column}")
        if frame[column].nunique() != 1:
            raise ValueError(f"事件資料混有不同身分：{column}")
    times = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    if times.isna().any() or not times.diff().iloc[1:].eq(pd.Timedelta(minutes=15)).all():
        raise ValueError("事件資料必須依序且每 15 分鐘連續，不可重複或缺棒")
    if not times.eq(times.dt.floor("15min")).all():
        raise ValueError("timestamp 必須是 UTC 15 分鐘 K 棒開盤時間")
    if times.iloc[-1] + pd.Timedelta(minutes=15) > pd.Timestamp.now(tz="UTC"):
        raise ValueError("事件資料包含未收盤 K 棒")
    prices = frame[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices[:, :4] <= 0).any() or (prices[:, 4] < 0).any():
        raise ValueError("事件資料價格／成交量不合法")
    if ((prices[:, 1] < prices[:, [0, 2, 3]].max(axis=1)).any()
            or (prices[:, 2] > prices[:, [0, 1, 3]].min(axis=1)).any()):
        raise ValueError("事件資料 OHLC 高低價不一致")


def _indicator_frame(bars: pd.DataFrame) -> pd.DataFrame:
    close, high, low = bars["close"], bars["high"], bars["low"]
    atr, adx, plus, minus = _directional_movement(high, low, close)
    ema20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    ema200 = close.ewm(span=200, adjust=False, min_periods=200).mean()
    volume_mean = bars["volume"].rolling(20, min_periods=20).mean().replace(0, np.nan)
    mid = close.rolling(20, min_periods=20).mean()
    std = close.rolling(20, min_periods=20).std(ddof=0)
    return pd.DataFrame({
        "return_1": close.pct_change(fill_method=None),
        "return_4": close.pct_change(4, fill_method=None),
        "ema20_gap": close / ema20 - 1,
        "ema50_gap": close / ema50 - 1,
        "ema200_gap": close / ema200 - 1,
        "ema50_200_spread": ema50 / ema200 - 1,
        "ema50_slope": ema50.pct_change(4, fill_method=None),
        "atr_pct": atr / close,
        "adx_14": adx,
        "di_balance": plus - minus,
        "rsi_14": _rsi(close) / 100,
        "bollinger_z": (close - mid) / std.replace(0, np.nan),
        "bollinger_width": 4 * std / mid,
        "relative_volume": bars["volume"] / volume_mean,
        "realized_volatility": close.pct_change(fill_method=None).rolling(20).std(ddof=0),
        "range_fraction": (high - low) / close,
    }, index=bars.index)


def prepare_event_frame(frame: pd.DataFrame, config: StrategyEventConfig) -> pd.DataFrame:
    """從同一份 15m 重採樣，只有已收盤的 1h／4h 資訊能進入特徵。"""
    validate_event_bars(frame)
    result = frame[["timestamp", "open", "high", "low", "close", "volume",
                    "symbol", "exchange", "interval"]].copy().reset_index(drop=True)
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    bars = result.set_index("timestamp")
    decision_times = pd.DatetimeIndex(result["timestamp"] + pd.Timedelta(minutes=15))
    aligned_hour = None
    for token, rule, count in (("15m", "15min", 1), ("1h", "1h", 4), ("4h", "4h", 16)):
        sampled = bars.resample(rule, label="left", closed="left").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
        })
        counts = bars["close"].resample(rule).count()
        sampled = sampled.loc[counts.eq(count)]
        indicators = _indicator_frame(sampled)
        if token == "1h":
            direction = np.sign(indicators["ema50_200_spread"])
            active = indicators["adx_14"] * 100 >= config.adx_threshold
            # 前後兩根已收盤小時棒皆同向，才視為可交易趨勢。
            indicators["regime"] = direction.where(
                active & active.shift(1, fill_value=False) & direction.eq(direction.shift(1)), 0,
            )
        indicators.index = indicators.index + pd.Timedelta(rule)
        aligned = indicators.reindex(decision_times, method="ffill")
        for name in indicators:
            if name != "regime":
                result[f"mtf_{token}_{name}"] = aligned[name].to_numpy()
        if token == "1h":
            aligned_hour = aligned["regime"].to_numpy()
    atr, _, _, _ = _directional_movement(bars["high"], bars["low"], bars["close"])
    prior_high = result["high"].shift(1).rolling(20).max()
    prior_low = result["low"].shift(1).rolling(20).min()
    result["event_regime"] = aligned_hour
    result["event_atr"] = atr.to_numpy()
    side = np.where((result["close"] > prior_high) & (result["event_regime"] == 1), 1,
                    np.where((result["close"] < prior_low) & (result["event_regime"] == -1), -1, 0))
    warm = decision_times >= decision_times[0] + pd.Timedelta(hours=config.warmup_hours)
    result["candidate_side"] = np.where(warm & np.isfinite(atr.to_numpy()), side, 0)
    result["mtf_15m_breakout_high_distance"] = result["close"] / prior_high - 1
    result["mtf_15m_breakout_low_distance"] = result["close"] / prior_low - 1
    result["hour_sin"] = np.sin(decision_times.hour * 2 * np.pi / 24)
    result["hour_cos"] = np.cos(decision_times.hour * 2 * np.pi / 24)
    return result


def replay_event(
    frame: pd.DataFrame, endpoint: int, config: StrategyEventConfig, *,
    fee_bps_per_side: float, slippage_bps_per_side: float,
) -> dict[str, float | int | str]:
    """訊號收盤後下一開盤進場；跳空取較差價，同棒停利停損採停損優先。"""
    side = int(frame.iloc[endpoint]["candidate_side"])
    if side not in (-1, 1) or endpoint + config.max_holding_bars + 1 >= len(frame):
        raise ValueError("事件方向不合法，或沒有足夠的出場執行 K 棒")
    if not np.isfinite([fee_bps_per_side, slippage_bps_per_side]).all():
        raise ValueError("交易成本必須為有限數值")
    if min(fee_bps_per_side, slippage_bps_per_side) < 0:
        raise ValueError("交易成本不可為負")
    friction = (slippage_bps_per_side + config.spread_bps / 2) / 10_000
    entry_index = endpoint + 1
    entry = float(frame.iloc[entry_index]["open"]) * (1 + side * friction)
    distance = float(frame.iloc[endpoint]["event_atr"])
    stop = entry - side * config.stop_atr * distance
    target = entry + side * config.target_atr * distance
    if not np.isfinite([distance, entry, stop, target]).all() or min(distance, stop, target) <= 0:
        raise ValueError("事件 ATR 或保護價格不合法")
    exit_index = endpoint + config.max_holding_bars + 1
    exit_reference = float(frame.iloc[exit_index]["open"])
    reason = "time"
    for index in range(entry_index, exit_index):
        row = frame.iloc[index]
        opening = float(row["open"])
        # 前一收盤已失去趨勢，於此開盤出場，不能偷看本根收盤制度。
        if index > entry_index and int(frame.iloc[index - 1]["event_regime"]) != side:
            exit_index, exit_reference, reason = index, opening, "regime"
            break
        stop_hit = row["low"] <= stop if side == 1 else row["high"] >= stop
        target_hit = row["high"] >= target if side == 1 else row["low"] <= target
        if stop_hit:
            exit_index = index
            exit_reference = min(opening, stop) if side == 1 else max(opening, stop)
            reason = "stop"
            break
        if target_hit:
            # 停利不計正向跳空改善，但仍扣市場單成本。
            exit_index, exit_reference, reason = index, target, "target"
            break
    exit_fill = exit_reference * (1 - side * friction)
    fees = fee_bps_per_side / 10_000 * (1 + exit_fill / entry)
    start = pd.Timestamp(frame.iloc[entry_index]["timestamp"])
    end = pd.Timestamp(frame.iloc[exit_index]["timestamp"])
    # 結算發生在 K 棒開盤邊界；棒內已退出不應多收下一棒的費用。
    # 邊界進場視為結算後、邊界出場視為結算後，避免低估已持有部位的成本。
    settlements = pd.date_range(start.floor("8h"), end.ceil("8h"), freq="8h")
    count = int(((settlements > start) & (settlements <= end)).sum())
    funding = count * config.funding_reserve_bps_per_settlement / 10_000
    net = side * (exit_fill / entry - 1) - fees - funding
    return {"endpoint": endpoint, "entry_endpoint": entry_index, "exit_endpoint": exit_index,
            "side": side, "entry_price": entry, "exit_price": exit_fill,
            "stop_price": stop, "target_price": target, "net_return": net,
            "fee_return": fees, "funding_return": funding, "exit_reason": reason}


def build_event_outcomes(frame: pd.DataFrame, config: StrategyEventConfig, *,
                         fee_bps_per_side: float, slippage_bps_per_side: float) -> pd.DataFrame:
    """標記所有候選，不因模型接受／拒絕而提前丟掉後續機會。"""
    endpoints = np.flatnonzero(frame["candidate_side"].to_numpy())
    rows = [replay_event(frame, int(point), config, fee_bps_per_side=fee_bps_per_side,
                         slippage_bps_per_side=slippage_bps_per_side)
            for point in endpoints if point + config.max_holding_bars + 1 < len(frame)]
    return pd.DataFrame(rows)


def evaluate_event_predictions(predictions: pd.DataFrame, horizon: int,
                               config: StrategyEventConfig, minimum_edge_bps: float
                               ) -> dict[str, float]:
    """相同事件成交結果比較固定策略與 AI 過濾；不把重疊交易全部相加。"""
    required = ["series_index", "endpoint", "event_exit_endpoint",
                f"tradeability_probability_{horizon}", f"predicted_return_{horizon}",
                f"actual_return_{horizon}"]
    if set(required).difference(predictions):
        raise ValueError("事件預測缺少成交或模型欄位")
    if (not np.isfinite(predictions[required].to_numpy(dtype=float)).all()
            or predictions.duplicated(["series_index", "endpoint"]).any()
            or (predictions["event_exit_endpoint"] <= predictions["endpoint"]).any()
            or not predictions[f"tradeability_probability_{horizon}"].between(0, 1).all()):
        raise ValueError("事件預測含非有限值、重複列、錯誤出場時間或非法機率")
    result: dict[str, float] = {}
    for name in ("baseline", "filtered"):
        returns: list[float] = []
        for _, group in predictions.groupby("series_index", sort=False):
            next_signal = -1
            for row in group.sort_values("endpoint").to_dict("records"):
                if row["endpoint"] < next_signal:
                    continue
                if name == "filtered" and (
                    row[f"tradeability_probability_{horizon}"] < config.minimum_probability
                    or row[f"predicted_return_{horizon}"] <= minimum_edge_bps / 10_000
                ):
                    continue
                returns.append(float(row[f"actual_return_{horizon}"]))
                # 出場當根不反手，之後完整等待 cooldown 根。
                next_signal = int(row["event_exit_endpoint"]) + config.cooldown_bars + 1
        values = np.asarray(returns)
        result[f"event_{name}_trades"] = float(len(values))
        result[f"event_{name}_net_expectancy"] = float(values.mean()) if len(values) else 0.0
        result[f"event_{name}_win_rate"] = float((values > 0).mean()) if len(values) else 0.0
        result[f"event_{name}_notional_return_sum"] = float(values.sum())
        result[f"event_{name}_has_trades"] = float(bool(len(values)))
    result["event_executed_trade_count_ratio"] = (
        result["event_filtered_trades"] / max(result["event_baseline_trades"], 1)
    )
    result["event_filter_coverage"] = float((
        (predictions[f"tradeability_probability_{horizon}"] >= config.minimum_probability)
        & (predictions[f"predicted_return_{horizon}"] > minimum_edge_bps / 10_000)
    ).mean()) if len(predictions) else 0.0
    return result
