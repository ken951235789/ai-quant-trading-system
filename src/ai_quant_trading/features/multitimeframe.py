"""將同一市場的多週期 K 線以已收盤時間做因果對齊。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd

from ai_quant_trading.data_collection.assets import (
    infer_asset_class,
    market_file_symbol_slug,
)
from ai_quant_trading.features.builder import infer_annualization_periods
from ai_quant_trading.features.indicators import FEATURE_COLUMNS, build_feature_dataset
from ai_quant_trading.features.smc import (
    CAUSAL_SMC_FEATURE_COLUMNS,
    build_causal_smc_features,
)
from ai_quant_trading.features.market_context import add_model_market_features
from ai_quant_trading.market_clock import (
    BTC_MULTITIMEFRAME_INTERVALS as _BTC_MULTITIMEFRAME_INTERVALS,
    completed_bars_only,
    interval_duration,
)


MULTITIMEFRAME_SCHEMA_VERSION = 5
BTC_MULTITIMEFRAME_INTERVALS = _BTC_MULTITIMEFRAME_INTERVALS

# 每個週期都輸出相同的比例化特徵，避免價格尺度不同時無法共同學習。
MULTITIMEFRAME_FEATURE_GROUPS = {
    "趨勢": (
        "ema_20_gap",
        "ema_20_atr",
        "ema_50_atr",
        "ema_200_atr",
        "ema_20_50_atr",
        "ema_50_200_atr",
        "ema_20_slope_5",
        "ema_60_gap",
        "ema_200_gap",
        "sma_200_gap",
        "adx_14",
        "plus_di_14",
        "minus_di_14",
        "supertrend_direction_10_3",
        "supertrend_distance_atr",
        "donchian_position_20",
        "donchian_breakout_up_20",
        "donchian_breakout_down_20",
        "linear_regression_slope_20",
    ),
    "動能": (
        "return_1",
        "return_5",
        "rsi_14",
        "macd_histogram_pct",
        "macd_histogram_atr",
        "roc_12_pct",
        "stoch_rsi_k_14",
        "stoch_rsi_d_3",
        "mfi_14",
    ),
    "波動": (
        "atr_pct",
        "bb_zscore_20",
        "bb_width_20",
        "historical_volatility_20",
        "choppiness_14",
        "realized_volatility_20",
        "parkinson_volatility_20",
        "keltner_position_20",
        "squeeze_on",
    ),
    "成交量": (
        "volume_change",
        "volume_zscore_20",
        "relative_volume_20",
        "trade_intensity_20",
        "taker_buy_ratio",
        "vwap_gap_20",
        "obv_pressure_5",
        "cmf_20",
        "volume_delta_ratio",
        "cvd_pressure_20",
        "average_trade_size_ratio_20",
        "amihud_20",
        "microstructure_available",
    ),
    "市場結構": (
        "candle_body_pct",
        "range_pct",
        "upper_wick_pct",
        "lower_wick_pct",
        "wick_imbalance",
        "higher_high",
        "higher_low",
        "breakout_20",
        "previous_high_20_atr",
        "previous_low_20_atr",
        "range_position_20",
    ),
    "因果 SMC": CAUSAL_SMC_FEATURE_COLUMNS,
    "合約與流動性": (
        "funding_rate",
        "funding_percentile_200",
        "open_interest_change",
        "funding_zscore_200",
        "open_interest_zscore_50",
        "spread_fraction",
        "derivatives_available",
    ),
    "市場狀態": (
        "trend_regime",
        "high_volatility_regime",
        "low_volatility_regime",
    ),
}
MULTITIMEFRAME_FEATURE_NAMES = tuple(
    feature
    for features in MULTITIMEFRAME_FEATURE_GROUPS.values()
    for feature in features
)
MULTITIMEFRAME_METADATA_COLUMNS = frozenset(
    {
        "mtf_decision_interval",
        "mtf_source_intervals",
        "mtf_schema_version",
    }
)


def multitimeframe_numeric_columns(frame: pd.DataFrame) -> list[str]:
    """依固定 CSV 順序列出可供 Transformer／PPO 使用的多週期數值欄位。"""
    return [
        column
        for column in frame.columns
        if column.startswith("mtf_")
        and column not in MULTITIMEFRAME_METADATA_COLUMNS
        and pd.to_numeric(frame[column], errors="coerce").notna().any()
    ]


def multitimeframe_source_intervals(frame: pd.DataFrame) -> tuple[str, ...]:
    """讀取融合資料宣告的來源週期，保留原本由快到慢的順序。"""
    if frame.empty or "mtf_source_intervals" not in frame:
        return ()
    return tuple(
        value
        for value in str(frame.iloc[0]["mtf_source_intervals"]).split("|")
        if value
    )


@dataclass(frozen=True, slots=True)
class MultiTimeframeArtifact:
    """一個市場完成多週期融合後的檔案與覆蓋摘要。"""

    output_path: Path
    manifest_path: Path
    exchange: str
    symbol: str
    decision_interval: str
    source_intervals: tuple[str, ...]
    rows: int
    feature_columns: int
    coverage: dict[str, float]


def _first_text(frame: pd.DataFrame, column: str, default: str = "") -> str:
    if frame.empty or column not in frame or pd.isna(frame.iloc[0][column]):
        return default
    return str(frame.iloc[0][column]).strip()


def _interval_token(interval: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", interval.strip().lower()).strip("_")
    if not token:
        raise ValueError("K 線週期不可為空")
    return token


def _interval_sort_key(interval: str) -> tuple[float, str]:
    duration = interval_duration(interval)
    seconds = duration.total_seconds() if duration is not None else float("inf")
    return seconds, interval


def _bar_close_time(frame: pd.DataFrame, interval: str) -> pd.Series:
    """優先採交易所 close_time；缺少時才用 timestamp 加週期。"""
    if "close_time_ms" in frame:
        close_time = pd.to_datetime(
            frame["close_time_ms"], unit="ms", utc=True, errors="coerce"
        )
    else:
        close_time = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    duration = interval_duration(interval)
    if duration is None:
        raise ValueError(f"無法判斷 {interval} 的 K 線長度")
    fallback = frame["timestamp"] + duration
    return close_time.where(close_time.notna(), fallback)


def _prepare_feature_frame(frame: pd.DataFrame, interval: str) -> pd.DataFrame:
    """接受 Raw 或 Step 3 CSV，缺少技術指標時自動補齊。"""
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{interval} K 線缺少必要欄位：{missing}")
    result = completed_bars_only(frame).reset_index(drop=True)
    if result.empty:
        raise ValueError(f"{interval} 沒有已收盤 K 線")
    result["timestamp"] = pd.to_datetime(
        result["timestamp"], utc=True, errors="coerce", format="mixed"
    )
    result = (
        result.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )
    if not set(FEATURE_COLUMNS).issubset(result.columns):
        result = build_feature_dataset(
            result,
            target_horizon=1,
            annualization_periods=infer_annualization_periods(result),
            drop_na=False,
        )
    return result


def _directional_movement_14(
    frame: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """計算比例化 ADX、+DI、-DI，全部只使用當下以前資料。"""
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    upward = high.diff()
    downward = -low.diff()
    plus_dm = upward.where((upward > downward) & (upward > 0), 0.0)
    minus_dm = downward.where((downward > upward) & (downward > 0), 0.0)
    atr = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / atr
    denominator = (plus_di + minus_di).replace(0, np.nan)
    directional_index = 100 * (plus_di - minus_di).abs() / denominator
    adx = directional_index.ewm(alpha=1 / 14, adjust=False, min_periods=1).mean()
    return adx / 100, plus_di / 100, minus_di / 100


def _adx_14(frame: pd.DataFrame) -> pd.Series:
    """保留舊欄位介面，回傳比例化 ADX。"""
    return _directional_movement_14(frame)[0]


def _choppiness_14(frame: pd.DataFrame) -> pd.Series:
    """計算 Choppiness Index，數值越高代表市場越偏震盪。"""
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    total_range = high.rolling(14, min_periods=14).max() - low.rolling(
        14, min_periods=14
    ).min()
    ratio = true_range.rolling(14, min_periods=14).sum() / total_range.replace(0, np.nan)
    return np.log10(ratio.clip(lower=1e-12)) / np.log10(14)


def _snapshot_features(frame: pd.DataFrame, interval: str) -> pd.DataFrame:
    """把單一週期轉成可跨價格尺度比較的市場快照。"""
    token = _interval_token(interval)
    close = pd.to_numeric(frame["close"], errors="coerce")
    open_price = pd.to_numeric(frame["open"], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    volume = pd.to_numeric(frame["volume"], errors="coerce")

    def numeric(column: str) -> pd.Series:
        if column not in frame:
            return pd.Series(np.nan, index=frame.index, dtype="float64")
        return pd.to_numeric(frame[column], errors="coerce")

    volume_mean = volume.rolling(20, min_periods=20).mean()
    volume_std = volume.rolling(20, min_periods=20).std(ddof=0)
    typical_price = (high + low + close) / 3
    rolling_volume = volume.rolling(20, min_periods=20).sum()
    rolling_vwap = (typical_price * volume).rolling(20, min_periods=20).sum()
    rolling_vwap = rolling_vwap / rolling_volume.replace(0, np.nan)
    close_mean = close.rolling(20, min_periods=20).mean()
    close_std = close.rolling(20, min_periods=20).std(ddof=0)
    direction = np.sign(close.diff()).fillna(0.0)
    obv = (direction * volume).cumsum()
    obv_pressure = obv.diff(5) / volume.rolling(5, min_periods=5).sum().replace(0, np.nan)
    money_flow_multiplier = (
        ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    )
    chaikin_money_flow = (money_flow_multiplier * volume).rolling(
        20, min_periods=20
    ).sum() / rolling_volume.replace(0, np.nan)
    upper_wick = high - pd.concat([open_price, close], axis=1).max(axis=1)
    lower_wick = pd.concat([open_price, close], axis=1).min(axis=1) - low
    atr = numeric("atr_14").replace(0, np.nan)
    ema_20 = numeric("ema_20")
    ema_50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    ema_200 = numeric("ema_200")
    adx, plus_di, minus_di = _directional_movement_14(frame)
    previous_high_20 = high.rolling(20, min_periods=20).max().shift(1)
    previous_low_20 = low.rolling(20, min_periods=20).min().shift(1)
    range_20 = (previous_high_20 - previous_low_20).replace(0, np.nan)
    number_of_trades = numeric("number_of_trades")
    trade_mean = number_of_trades.rolling(20, min_periods=20).mean()
    taker_buy_volume = numeric("taker_buy_base_volume")
    taker_buy_ratio = (taker_buy_volume / volume.replace(0, np.nan)).clip(0.0, 1.0)
    funding_rate = numeric("funding_rate")
    funding_percentile = funding_rate.rolling(200, min_periods=20).rank(pct=True)
    open_interest_change = numeric("open_interest_change")
    spread_fraction = numeric("spread_bps") / 10_000
    derivatives_available = pd.concat(
        [funding_rate, open_interest_change, spread_fraction], axis=1
    ).notna().any(axis=1).astype("float64")
    atr_fraction = atr / close.replace(0, np.nan)
    volatility_percentile = atr_fraction.rolling(200, min_periods=50).rank(pct=True)
    trend_regime = np.sign(ema_20 - ema_50).where(adx >= 0.20, 0.0)
    smc = build_causal_smc_features(frame, atr=atr)

    snapshot = pd.DataFrame(
        {
            "_available_at": _bar_close_time(frame, interval),
            f"mtf_{token}_return_1": numeric("return_1"),
            f"mtf_{token}_return_5": close.pct_change(5, fill_method=None),
            f"mtf_{token}_ema_20_gap": close / ema_20.replace(0, np.nan) - 1,
            f"mtf_{token}_ema_20_atr": (close - ema_20) / atr,
            f"mtf_{token}_ema_50_atr": (close - ema_50) / atr,
            f"mtf_{token}_ema_200_atr": (close - ema_200) / atr,
            f"mtf_{token}_ema_20_50_atr": (ema_20 - ema_50) / atr,
            f"mtf_{token}_ema_50_200_atr": (ema_50 - ema_200) / atr,
            f"mtf_{token}_ema_20_slope_5": ema_20.pct_change(5, fill_method=None),
            f"mtf_{token}_ema_60_gap": close / numeric("ema_60").replace(0, np.nan) - 1,
            f"mtf_{token}_ema_200_gap": close / ema_200.replace(0, np.nan) - 1,
            f"mtf_{token}_sma_200_gap": close / numeric("sma_200").replace(0, np.nan) - 1,
            f"mtf_{token}_adx_14": adx,
            f"mtf_{token}_plus_di_14": plus_di,
            f"mtf_{token}_minus_di_14": minus_di,
            f"mtf_{token}_supertrend_direction_10_3": numeric(
                "supertrend_direction_10_3"
            ),
            f"mtf_{token}_supertrend_distance_atr": numeric(
                "supertrend_distance_atr"
            ),
            f"mtf_{token}_donchian_position_20": numeric("donchian_position_20"),
            f"mtf_{token}_donchian_breakout_up_20": numeric(
                "donchian_breakout_up_20"
            ),
            f"mtf_{token}_donchian_breakout_down_20": numeric(
                "donchian_breakout_down_20"
            ),
            f"mtf_{token}_linear_regression_slope_20": numeric(
                "linear_regression_slope_20"
            ),
            f"mtf_{token}_rsi_14": numeric("rsi_14") / 100,
            f"mtf_{token}_macd_histogram_pct": (
                numeric("macd_histogram") / close.replace(0, np.nan)
            ),
            f"mtf_{token}_macd_histogram_atr": numeric("macd_histogram") / atr,
            f"mtf_{token}_roc_12_pct": numeric("roc_12_pct") / 100,
            f"mtf_{token}_stoch_rsi_k_14": numeric("stoch_rsi_k_14"),
            f"mtf_{token}_stoch_rsi_d_3": numeric("stoch_rsi_d_3"),
            f"mtf_{token}_mfi_14": numeric("mfi_14"),
            f"mtf_{token}_atr_pct": atr_fraction,
            f"mtf_{token}_bb_zscore_20": (
                (close - close_mean) / close_std.replace(0, np.nan)
            ),
            f"mtf_{token}_bb_width_20": numeric("bb_width_20"),
            f"mtf_{token}_historical_volatility_20": numeric(
                "historical_volatility_20"
            ),
            f"mtf_{token}_choppiness_14": _choppiness_14(frame),
            f"mtf_{token}_realized_volatility_20": numeric(
                "realized_volatility_20"
            ),
            f"mtf_{token}_parkinson_volatility_20": numeric(
                "parkinson_volatility_20"
            ),
            f"mtf_{token}_keltner_position_20": numeric("keltner_position_20"),
            f"mtf_{token}_squeeze_on": numeric("squeeze_on"),
            f"mtf_{token}_volume_change": numeric("volume_change").clip(-1.0, 10.0),
            f"mtf_{token}_volume_zscore_20": (
                (volume - volume_mean) / volume_std.replace(0, np.nan)
            ),
            f"mtf_{token}_relative_volume_20": volume / volume_mean.replace(0, np.nan),
            f"mtf_{token}_trade_intensity_20": (
                number_of_trades / trade_mean.replace(0, np.nan)
            ).fillna(0.0),
            f"mtf_{token}_taker_buy_ratio": taker_buy_ratio.fillna(0.5),
            f"mtf_{token}_vwap_gap_20": (close - rolling_vwap) / atr,
            f"mtf_{token}_obv_pressure_5": obv_pressure,
            f"mtf_{token}_cmf_20": chaikin_money_flow,
            f"mtf_{token}_volume_delta_ratio": numeric("volume_delta_ratio"),
            f"mtf_{token}_cvd_pressure_20": numeric("cvd_pressure_20"),
            f"mtf_{token}_average_trade_size_ratio_20": numeric(
                "average_trade_size_ratio_20"
            ),
            f"mtf_{token}_amihud_20": numeric("amihud_20"),
            f"mtf_{token}_microstructure_available": numeric(
                "microstructure_available"
            ),
            f"mtf_{token}_candle_body_pct": (
                (close - open_price) / open_price.replace(0, np.nan)
            ),
            f"mtf_{token}_range_pct": (high - low) / close.replace(0, np.nan),
            f"mtf_{token}_upper_wick_pct": upper_wick / close.replace(0, np.nan),
            f"mtf_{token}_lower_wick_pct": lower_wick / close.replace(0, np.nan),
            f"mtf_{token}_wick_imbalance": (
                (lower_wick - upper_wick) / close.replace(0, np.nan)
            ),
            f"mtf_{token}_higher_high": numeric("higher_high"),
            f"mtf_{token}_higher_low": numeric("higher_low"),
            f"mtf_{token}_breakout_20": numeric("breakout_20"),
            f"mtf_{token}_previous_high_20_atr": (previous_high_20 - close) / atr,
            f"mtf_{token}_previous_low_20_atr": (close - previous_low_20) / atr,
            f"mtf_{token}_range_position_20": (
                (close - previous_low_20) / range_20
            ).clip(0.0, 1.0),
            **{
                f"mtf_{token}_{column}": smc[column]
                for column in CAUSAL_SMC_FEATURE_COLUMNS
            },
            f"mtf_{token}_funding_rate": funding_rate.fillna(0.0),
            f"mtf_{token}_funding_percentile_200": funding_percentile.fillna(0.5),
            f"mtf_{token}_open_interest_change": open_interest_change.fillna(0.0),
            f"mtf_{token}_funding_zscore_200": numeric(
                "funding_zscore_200"
            ).fillna(0.0),
            f"mtf_{token}_open_interest_zscore_50": numeric(
                "open_interest_zscore_50"
            ).fillna(0.0),
            f"mtf_{token}_spread_fraction": spread_fraction.fillna(0.0),
            f"mtf_{token}_derivatives_available": derivatives_available,
            f"mtf_{token}_trend_regime": trend_regime,
            f"mtf_{token}_high_volatility_regime": (
                volatility_percentile >= 0.80
            ).astype("float64"),
            f"mtf_{token}_low_volatility_regime": (
                volatility_percentile <= 0.20
            ).astype("float64"),
        }
    )
    value_columns = [column for column in snapshot if column.startswith("mtf_")]
    snapshot[value_columns] = snapshot[value_columns].replace([np.inf, -np.inf], np.nan)
    # K 線是否可用只由收盤時間決定；個別長週期指標仍在暖機時，
    # 會在因果對齊後填成中性值，不能因此抹掉整根已收盤 K 線。
    snapshot = snapshot.dropna(subset=["_available_at"])
    return (
        snapshot.sort_values("_available_at")
        .drop_duplicates("_available_at", keep="last")
        .reset_index(drop=True)
    )


def build_multitimeframe_frame(
    frames: Mapping[str, pd.DataFrame],
    decision_interval: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """建立單一決策時鐘的多週期特徵，不使用當時尚未收盤的 K 線。"""
    normalized = {str(key).strip().lower(): value for key, value in frames.items()}
    decision = decision_interval.strip().lower()
    if decision not in normalized:
        raise ValueError(f"多週期資料缺少決策週期 {decision}")
    if len(normalized) < 2:
        raise ValueError("多週期資料至少需要兩種 K 線週期")

    prepared = {
        interval: _prepare_feature_frame(frame, interval)
        for interval, frame in normalized.items()
    }
    identities = {
        (
            _first_text(frame, "exchange").lower(),
            _first_text(frame, "symbol").upper(),
        )
        for frame in prepared.values()
    }
    if len(identities) != 1:
        raise ValueError("多週期融合只能包含同一個交易所與標的")

    result = prepared[decision].copy()
    result["_decision_time"] = _bar_close_time(result, decision)
    result = result.sort_values("_decision_time").reset_index(drop=True)
    hour = result["_decision_time"].dt.hour + result["_decision_time"].dt.minute / 60
    weekday = result["_decision_time"].dt.weekday
    result["mtf_context_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    result["mtf_context_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    result["mtf_context_weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    result["mtf_context_weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    result["mtf_context_asia_session"] = ((hour >= 0) & (hour < 8)).astype("float64")
    result["mtf_context_europe_session"] = ((hour >= 7) & (hour < 16)).astype("float64")
    result["mtf_context_us_session"] = ((hour >= 13) & (hour < 22)).astype("float64")
    result["mtf_context_weekend"] = (weekday >= 5).astype("float64")
    coverage: dict[str, float] = {}
    ordered_intervals = tuple(sorted(prepared, key=_interval_sort_key))

    for interval in ordered_intervals:
        token = _interval_token(interval)
        snapshot = _snapshot_features(prepared[interval], interval)
        result = pd.merge_asof(
            result,
            snapshot,
            left_on="_decision_time",
            right_on="_available_at",
            direction="backward",
            allow_exact_matches=True,
        )
        value_columns = [
            column
            for column in result
            if column.startswith(f"mtf_{token}_")
            and not column.endswith(("_available", "_age_ratio"))
        ]
        available = result["_available_at"].notna()
        duration = interval_duration(interval)
        if duration is None:
            raise ValueError(f"無法解析多週期 K 線：{interval}")
        age = (result["_decision_time"] - result["_available_at"]) / duration
        freshness = pd.DataFrame(
            {
                f"mtf_{token}_available": available.astype("float64"),
                f"mtf_{token}_age_ratio": pd.to_numeric(
                    age, errors="coerce"
                ).fillna(100.0).clip(0.0, 100.0),
            },
            index=result.index,
        )
        # 多週期會增加數百欄；整批串接可避免 DataFrame 記憶體碎片化。
        result = pd.concat([result, freshness], axis=1).copy()
        result[value_columns] = result[value_columns].fillna(0.0)
        coverage[interval] = float(available.mean())
        result = result.drop(columns=["_available_at"])

    result = result.drop(columns=["_decision_time"]).copy()
    metadata = pd.DataFrame(
        {
            "mtf_decision_interval": decision,
            "mtf_source_intervals": "|".join(ordered_intervals),
            "mtf_schema_version": MULTITIMEFRAME_SCHEMA_VERSION,
        },
        index=result.index,
    )
    result = pd.concat([result, metadata], axis=1)
    return add_model_market_features(result, rebuild_context=True), coverage


def _group_sources(
    source_paths: Sequence[str | Path],
) -> dict[tuple[str, str], dict[str, Path]]:
    groups: dict[tuple[str, str], dict[str, Path]] = {}
    for value in source_paths:
        path = Path(value).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"找不到多週期來源：{path}")
        first = pd.read_csv(path, nrows=1)
        if first.empty:
            raise ValueError(f"{path.name} 是空檔案")
        exchange = _first_text(first, "exchange", "unknown").lower()
        symbol = _first_text(first, "symbol", path.stem)
        interval = _first_text(first, "interval").lower()
        if not interval:
            raise ValueError(f"{path.name} 缺少 interval")
        key = exchange, symbol.upper()
        if interval in groups.setdefault(key, {}):
            raise ValueError(f"{exchange}:{symbol} 重複選到 {interval} 資料")
        groups[key][interval] = path
    return groups


def build_multitimeframe_feature_datasets(
    source_paths: Sequence[str | Path],
    decision_interval: str,
    output_dir: str | Path,
) -> tuple[MultiTimeframeArtifact, ...]:
    """依交易所與標的分組，批次建立可供 Transformer／PPO 共用的 CSV。"""
    if not source_paths:
        raise ValueError("請至少選擇兩份不同週期的特徵資料")
    decision = decision_interval.strip().lower()
    target_dir = Path(output_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[MultiTimeframeArtifact] = []

    for (exchange, _), interval_paths in _group_sources(source_paths).items():
        if decision not in interval_paths:
            continue
        if len(interval_paths) < 2:
            continue
        frames = {
            interval: pd.read_csv(path)
            for interval, path in interval_paths.items()
        }
        fused, coverage = build_multitimeframe_frame(frames, decision)
        first = fused.iloc[0]
        symbol = str(first.get("symbol", "unknown"))
        asset_class = infer_asset_class(exchange, symbol)
        symbol_slug = market_file_symbol_slug(symbol, asset_class)
        intervals = tuple(sorted(interval_paths, key=_interval_sort_key))
        interval_slug = "-".join(_interval_token(value) for value in intervals)
        stem = (
            f"features_mtf_{asset_class}_{exchange}_{symbol_slug}_"
            f"{_interval_token(decision)}_{interval_slug}_h1"
        )
        output_path = target_dir / f"{stem}.csv"
        manifest_path = target_dir / f"{stem}.json"
        temporary = output_path.with_suffix(f".csv.{uuid4().hex}.tmp")
        try:
            fused.to_csv(temporary, index=False, encoding="utf-8")
            temporary.replace(output_path)
        finally:
            temporary.unlink(missing_ok=True)
        manifest = {
            "schema_version": MULTITIMEFRAME_SCHEMA_VERSION,
            "exchange": exchange,
            "symbol": symbol,
            "decision_interval": decision,
            "source_intervals": list(intervals),
            "source_paths": {
                interval: str(path) for interval, path in interval_paths.items()
            },
            "rows": len(fused),
            "coverage": coverage,
            "causal_alignment": "source_close_time <= decision_close_time",
            "feature_groups": {
                group: list(features)
                for group, features in MULTITIMEFRAME_FEATURE_GROUPS.items()
            },
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        mtf_columns = [column for column in fused if column.startswith("mtf_")]
        artifacts.append(
            MultiTimeframeArtifact(
                output_path=output_path,
                manifest_path=manifest_path,
                exchange=exchange,
                symbol=symbol,
                decision_interval=decision,
                source_intervals=intervals,
                rows=len(fused),
                feature_columns=len(mtf_columns),
                coverage=coverage,
            )
        )

    if not artifacts:
        raise ValueError(
            f"沒有標的同時具備 {decision} 與至少一個其他週期；請重新選擇來源"
        )
    return tuple(artifacts)


def refresh_multitimeframe_feature_datasets(
    output_dir: str | Path,
) -> tuple[tuple[MultiTimeframeArtifact, ...], tuple[str, ...]]:
    """依既有 manifest 重建固定多週期檔，供啟動增量更新後同步使用。"""
    root = Path(output_dir).resolve()
    artifacts: list[MultiTimeframeArtifact] = []
    errors: list[str] = []
    for manifest_path in sorted(root.glob("features_mtf_*.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            source_values = list(dict(manifest.get("source_paths", {})).values())
            sources: list[Path] = []
            for value in source_values:
                configured = Path(str(value))
                candidate = configured if configured.is_file() else root / configured.name
                if not candidate.is_file():
                    raise FileNotFoundError(f"找不到來源 {configured.name}")
                sources.append(candidate)
            artifacts.extend(
                build_multitimeframe_feature_datasets(
                    sources,
                    str(manifest.get("decision_interval", "")),
                    root,
                )
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            errors.append(f"{manifest_path.name}：{exc}")
    return tuple(artifacts), tuple(errors)
