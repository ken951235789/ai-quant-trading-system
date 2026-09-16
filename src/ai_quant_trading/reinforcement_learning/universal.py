"""跨市場通用 RL 資料集與環境成品。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from ai_quant_trading.reinforcement_learning.config import (
    ExpertKind,
    PortfolioEnvConfig,
    RLSplitConfig,
)
from ai_quant_trading.reinforcement_learning.dataset import MARKET_COLUMNS
from ai_quant_trading.features.multitimeframe import multitimeframe_numeric_columns
from ai_quant_trading.features.market_context import MARKET_CONTEXT_COLUMNS, add_model_market_features
from ai_quant_trading.reinforcement_learning.feature_contract import (
    attach_expected_return,
    build_expected_return_contract,
    compact_feature_metadata,
    select_compact_short_term_features,
)


MULTIMODAL_RL_FEATURE_COLUMNS = [
    "u_finbert_sentiment",
    "u_finbert_positive",
    "u_finbert_negative",
    "u_finbert_confidence",
    "u_finbert_news_count_log",
    "u_finbert_sentiment_change",
    "u_finbert_recency",
    "u_finbert_available",
    "u_transformer_return_1",
    "u_transformer_return_5",
    "u_transformer_return_20",
    "u_transformer_volatility",
    "u_transformer_bull_probability",
    "u_transformer_bear_probability",
    "u_transformer_uncertainty",
    "u_transformer_available",
    *[
        f"u_transformer_{name}_{horizon}"
        for horizon in (1, 5, 20)
        for name in (
            "direction_edge",
            "movement_direction_edge",
            "movement_neutral_probability",
            "side_direction_edge",
            "neutral_probability",
            "return_q10",
            "return_q50",
            "return_q90",
            "interval_width",
            "long_edge",
            "short_edge",
            "downside_excursion",
            "upside_excursion",
            "tradeability",
        )
    ],
    "u_transformer_volatility_regime_low_probability",
    "u_transformer_volatility_regime_normal_probability",
    "u_transformer_volatility_regime_high_probability",
    "u_transformer_timeframe_fast_attention",
    "u_transformer_timeframe_medium_attention",
    "u_transformer_timeframe_slow_attention",
    "u_transformer_trend_signal",
    "u_transformer_setup_signal",
    "u_transformer_execution_signal",
    "u_transformer_consensus_signal",
    "u_transformer_signal_disagreement",
    "u_transformer_signal_entropy",
    "u_transformer_decision_confidence",
    "u_transformer_no_trade_pressure",
    "u_transformer_net_edge_advantage_5",
    "u_transformer_expected_best_edge_5",
]

ADVANCED_RL_FEATURE_COLUMNS = [
    "u_macro_yield_curve",
    "u_macro_vix_level",
    "u_macro_vix_zscore",
    "u_macro_dollar_change",
    "u_macro_high_yield_spread",
    "u_macro_available",
    "u_breadth_advance_ratio",
    "u_breadth_above_sma_20",
    "u_breadth_above_sma_50",
    "u_breadth_above_sma_200",
    "u_breadth_new_high_ratio",
    "u_breadth_new_low_ratio",
    "u_breadth_median_return",
    "u_breadth_dispersion",
    "u_breadth_available",
    "u_fundamental_revenue_growth",
    "u_fundamental_profit_margin",
    "u_fundamental_operating_margin",
    "u_fundamental_debt_to_equity",
    "u_fundamental_ocf_margin",
    "u_fundamentals_available",
    "u_global_long_short_pressure",
    "u_taker_buy_sell_pressure",
    "u_basis_rate",
    "u_orderbook_imbalance_5",
    "u_orderbook_imbalance_10",
    "u_orderbook_imbalance_20",
    "u_orderbook_microprice_deviation",
    "u_orderbook_spread_rate",
    "u_orderbook_available",
]


UNIVERSAL_RL_FEATURE_COLUMNS = [
    "u_return_1",
    "u_sma_5_gap",
    "u_sma_20_gap",
    "u_sma_60_gap",
    "u_sma_200_gap",
    "u_ema_5_gap",
    "u_ema_20_gap",
    "u_ema_60_gap",
    "u_ema_200_gap",
    "u_rsi_14",
    "u_macd_pct",
    "u_macd_signal_pct",
    "u_macd_histogram_pct",
    "u_roc_12_pct",
    "u_atr_pct",
    "u_bb_zscore_20",
    "u_bb_width_20",
    "u_historical_volatility_20",
    "u_volume_change",
    "u_vwap_gap",
    "u_higher_high",
    "u_higher_low",
    "u_breakout_20",
    "u_short_rsi_2",
    "u_short_ema_5_gap",
    "u_short_ema_200_gap",
    "u_short_bb_zscore_20",
    "u_short_reversal_entry",
    "u_short_reversal_exit",
    "u_short_reversal_position",
    "u_return_5",
    "u_return_20",
    "u_return_60",
    "u_return_120",
    "u_realized_volatility_5",
    "u_realized_volatility_20",
    "u_realized_volatility_60",
    "u_volatility_ratio_5_20",
    "u_downside_volatility_20",
    "u_drawdown_20",
    "u_drawdown_60",
    "u_drawdown_252",
    "u_volume_zscore_20",
    "u_dollar_volume_zscore_20",
    "u_range_pct",
    "u_gap_return",
    "u_efficiency_ratio_20",
    "u_distance_high_20",
    "u_distance_low_20",
    "u_trend_spread_20_60",
    "u_trend_spread_60_200",
    "u_hour_sin",
    "u_hour_cos",
    "u_weekday_sin",
    "u_weekday_cos",
    "u_is_crypto",
    "u_funding_rate",
    "u_open_interest_change",
    "u_spread_rate",
    "u_derivatives_context_available",
    *ADVANCED_RL_FEATURE_COLUMNS,
    *MULTIMODAL_RL_FEATURE_COLUMNS,
]


LONG_TERM_RL_FEATURE_COLUMNS = [
    "u_return_1",
    "u_return_5",
    "u_return_20",
    "u_return_60",
    "u_return_120",
    "u_sma_20_gap",
    "u_sma_60_gap",
    "u_sma_200_gap",
    "u_ema_20_gap",
    "u_ema_60_gap",
    "u_ema_200_gap",
    "u_rsi_14",
    "u_macd_histogram_pct",
    "u_roc_12_pct",
    "u_atr_pct",
    "u_bb_width_20",
    "u_historical_volatility_20",
    "u_realized_volatility_20",
    "u_realized_volatility_60",
    "u_volatility_ratio_5_20",
    "u_downside_volatility_20",
    "u_volume_zscore_20",
    "u_dollar_volume_zscore_20",
    "u_vwap_gap",
    "u_higher_high",
    "u_higher_low",
    "u_breakout_20",
    "u_drawdown_20",
    "u_drawdown_60",
    "u_drawdown_252",
    "u_efficiency_ratio_20",
    "u_distance_high_20",
    "u_trend_spread_20_60",
    "u_trend_spread_60_200",
    "u_is_crypto",
    "u_funding_rate",
    "u_open_interest_change",
    "u_spread_rate",
    "u_derivatives_context_available",
    *ADVANCED_RL_FEATURE_COLUMNS,
    *MULTIMODAL_RL_FEATURE_COLUMNS,
]


SHORT_TERM_RL_FEATURE_COLUMNS = [
    "u_return_1",
    "u_return_5",
    "u_return_20",
    "u_sma_5_gap",
    "u_sma_20_gap",
    "u_ema_5_gap",
    "u_ema_20_gap",
    "u_ema_60_gap",
    "u_rsi_14",
    "u_macd_pct",
    "u_macd_histogram_pct",
    "u_atr_pct",
    "u_bb_zscore_20",
    "u_bb_width_20",
    "u_realized_volatility_5",
    "u_realized_volatility_20",
    "u_volatility_ratio_5_20",
    "u_downside_volatility_20",
    "u_volume_change",
    "u_volume_zscore_20",
    "u_dollar_volume_zscore_20",
    "u_vwap_gap",
    "u_higher_high",
    "u_higher_low",
    "u_breakout_20",
    "u_short_rsi_2",
    "u_short_ema_5_gap",
    "u_short_ema_200_gap",
    "u_short_bb_zscore_20",
    "u_short_reversal_entry",
    "u_short_reversal_exit",
    "u_range_pct",
    "u_gap_return",
    "u_efficiency_ratio_20",
    "u_distance_high_20",
    "u_distance_low_20",
    "u_hour_sin",
    "u_hour_cos",
    "u_weekday_sin",
    "u_weekday_cos",
    "u_is_crypto",
    "u_funding_rate",
    "u_open_interest_change",
    "u_spread_rate",
    "u_derivatives_context_available",
    *ADVANCED_RL_FEATURE_COLUMNS,
    *MULTIMODAL_RL_FEATURE_COLUMNS,
]


UNIVERSAL_SOURCE_FEATURE_COLUMNS = [
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
    "vwap",
    "higher_high",
    "higher_low",
    "breakout_20",
    "short_rsi_2",
    "short_ema_5_gap",
    "short_ema_200_gap",
    "short_bb_zscore_20",
    "short_reversal_entry",
    "short_reversal_exit",
    "short_reversal_position",
]

def _multitimeframe_source_columns(frame: pd.DataFrame) -> list[str]:
    """依 CSV 欄位順序列出多週期數值特徵，排除描述欄位。"""
    return multitimeframe_numeric_columns(frame)


@dataclass(slots=True)
class UniversalMarketDataset:
    """單一市場在通用資料集中的時序切分。"""

    frame: pd.DataFrame
    train_end: int
    validation_end: int

    @property
    def train(self) -> pd.DataFrame:
        return self.frame.iloc[: self.train_end].reset_index(drop=True)

    @property
    def validation(self) -> pd.DataFrame:
        return self.frame.iloc[self.train_end : self.validation_end].reset_index(drop=True)

    @property
    def test(self) -> pd.DataFrame:
        return self.frame.iloc[self.validation_end :].reset_index(drop=True)


@dataclass(slots=True)
class PreparedUniversalRLDataset:
    """使用共同標準化參數的多市場 RL 資料。"""

    markets: dict[str, UniversalMarketDataset]
    feature_columns: list[str]
    feature_mean: pd.Series
    feature_std: pd.Series
    split_config: RLSplitConfig
    ai_coverage: dict[str, dict[str, float]]
    use_finbert: bool = True
    use_transformer: bool = True
    expected_return_contract: dict[str, object] = field(default_factory=dict)
    feature_contract: dict[str, object] = field(default_factory=dict)

    @property
    def observation_size(self) -> int:
        return len(self.feature_columns) + 5


@dataclass(frozen=True, slots=True)
class UniversalRLArtifactPaths:
    """通用 RL 環境的中繼資料與市場檔案路徑。"""

    run_dir: Path
    metadata_json: Path
    diagnostic_csv: Path


def add_universal_rl_features(
    frame: pd.DataFrame, *, feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """把價格型指標轉成跨資產可比較的比例與 Z-score。"""
    required = [*MARKET_COLUMNS, *UNIVERSAL_SOURCE_FEATURE_COLUMNS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"通用 RL 資料缺少欄位：{missing}")
    # 模擬交易可能重複準備同一份資料；先移除舊衍生欄，避免同名欄位重複。
    result = add_model_market_features(frame.drop(
        columns=[column for column in frame.columns if column.startswith("u_")],
        errors="ignore",
    ))
    numeric = ["close", *UNIVERSAL_SOURCE_FEATURE_COLUMNS]
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="coerce")
    close = result["close"]

    result["u_return_1"] = result["return_1"]
    for kind in ["sma", "ema"]:
        for window in [5, 20, 60, 200]:
            average = result[f"{kind}_{window}"]
            result[f"u_{kind}_{window}_gap"] = close / average.where(average != 0) - 1
    result["u_rsi_14"] = result["rsi_14"] / 100
    result["u_macd_pct"] = result["macd"] / close
    result["u_macd_signal_pct"] = result["macd_signal"] / close
    result["u_macd_histogram_pct"] = result["macd_histogram"] / close
    result["u_roc_12_pct"] = result["roc_12_pct"] / 100
    result["u_atr_pct"] = result["atr_14"] / close
    band_deviation = (result["bb_upper_20"] - result["bb_lower_20"]) / 4
    result["u_bb_zscore_20"] = (
        (close - result["bb_middle_20"]) / band_deviation.where(band_deviation != 0)
    )
    result["u_bb_width_20"] = result["bb_width_20"]
    result["u_historical_volatility_20"] = result["historical_volatility_20"]
    result["u_volume_change"] = result["volume_change"].clip(-1, 5)
    rolling_volume = result["volume"].rolling(20, min_periods=20).sum()
    rolling_vwap = (close * result["volume"]).rolling(20, min_periods=20).sum()
    rolling_vwap = rolling_vwap / rolling_volume.where(rolling_volume != 0)
    result["u_vwap_gap"] = close / rolling_vwap.where(rolling_vwap != 0) - 1
    result["u_higher_high"] = result["higher_high"]
    result["u_higher_low"] = result["higher_low"]
    result["u_breakout_20"] = result["breakout_20"]
    result["u_short_rsi_2"] = result["short_rsi_2"] / 100
    for column in [
        "short_ema_5_gap",
        "short_ema_200_gap",
        "short_bb_zscore_20",
        "short_reversal_entry",
        "short_reversal_exit",
        "short_reversal_position",
    ]:
        result[f"u_{column}"] = result[column]

    returns = close.pct_change(fill_method=None)
    log_returns = np.log(close / close.shift(1))
    for window in [5, 20, 60, 120]:
        result[f"u_return_{window}"] = close.pct_change(window, fill_method=None)
    for window in [5, 20, 60]:
        result[f"u_realized_volatility_{window}"] = log_returns.rolling(
            window, min_periods=window
        ).std(ddof=0)
    result["u_volatility_ratio_5_20"] = (
        result["u_realized_volatility_5"]
        / result["u_realized_volatility_20"].where(
            result["u_realized_volatility_20"] != 0
        )
    )
    downside = returns.clip(upper=0)
    result["u_downside_volatility_20"] = downside.rolling(20, min_periods=20).std(ddof=0)
    for window in [20, 60, 252]:
        rolling_high = close.rolling(window, min_periods=window).max()
        result[f"u_drawdown_{window}"] = close / rolling_high.where(rolling_high != 0) - 1

    volume_mean = result["volume"].rolling(20, min_periods=20).mean()
    volume_std = result["volume"].rolling(20, min_periods=20).std(ddof=0)
    result["u_volume_zscore_20"] = (
        (result["volume"] - volume_mean) / volume_std.where(volume_std != 0)
    )
    dollar_volume = close * result["volume"]
    dollar_mean = dollar_volume.rolling(20, min_periods=20).mean()
    dollar_std = dollar_volume.rolling(20, min_periods=20).std(ddof=0)
    result["u_dollar_volume_zscore_20"] = (
        (dollar_volume - dollar_mean) / dollar_std.where(dollar_std != 0)
    )
    result["u_range_pct"] = (result["high"] - result["low"]) / close
    result["u_gap_return"] = result["open"] / close.shift(1) - 1
    path_length = close.diff().abs().rolling(20, min_periods=20).sum()
    result["u_efficiency_ratio_20"] = (
        close.diff(20).abs() / path_length.where(path_length != 0)
    )
    high_20 = result["high"].rolling(20, min_periods=20).max()
    low_20 = result["low"].rolling(20, min_periods=20).min()
    result["u_distance_high_20"] = close / high_20.where(high_20 != 0) - 1
    result["u_distance_low_20"] = close / low_20.where(low_20 != 0) - 1
    result["u_trend_spread_20_60"] = (
        result["sma_20"] / result["sma_60"].where(result["sma_60"] != 0) - 1
    )
    result["u_trend_spread_60_200"] = (
        result["sma_60"] / result["sma_200"].where(result["sma_200"] != 0) - 1
    )
    timestamps = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
    hour_angle = 2 * np.pi * timestamps.dt.hour / 24
    weekday_angle = 2 * np.pi * timestamps.dt.weekday / 7
    result["u_hour_sin"] = np.sin(hour_angle)
    result["u_hour_cos"] = np.cos(hour_angle)
    result["u_weekday_sin"] = np.sin(weekday_angle)
    result["u_weekday_cos"] = np.cos(weekday_angle)
    exchange = result["exchange"].astype(str).str.lower()
    result["u_is_crypto"] = exchange.isin(
        {"binance", "binance_futures", "bybit", "bybit_futures"}
    ).astype("int8")

    # 多週期建構器已先做因果對齊；此處只轉成通用環境的 u_ 特徵契約。
    mtf_features: dict[str, pd.Series] = {}
    mtf_sources = (
        [column for column in result if column.startswith("mtf_") and f"u_{column}" in feature_columns]
        if feature_columns is not None else _multitimeframe_source_columns(result)
    )
    for column in mtf_sources:
        target = f"u_{column}"
        if feature_columns is not None and target not in feature_columns:
            continue
        values = pd.to_numeric(result[column], errors="coerce")
        if column.endswith("_available"):
            mtf_features[target] = values.fillna(0.0).clip(0.0, 1.0)
        elif column.endswith("_age_ratio"):
            mtf_features[target] = values.fillna(100.0).clip(0.0, 100.0)
        else:
            mtf_features[target] = values.fillna(0.0).clip(-20.0, 20.0)
    if mtf_features:
        result = pd.concat([result, pd.DataFrame(mtf_features, index=result.index)], axis=1)

    def optional_numeric(column: str) -> pd.Series:
        if column not in result:
            return pd.Series(np.nan, index=result.index, dtype="float64")
        return pd.to_numeric(result[column], errors="coerce")

    funding = optional_numeric("funding_rate")
    open_interest_change = optional_numeric("open_interest_change")
    spread_bps = optional_numeric("spread_bps")
    available = result.get("derivatives_context_available")
    if available is None:
        available = funding.notna() | open_interest_change.notna() | spread_bps.notna()
    result["u_funding_rate"] = funding.fillna(0.0).clip(-0.05, 0.05)
    result["u_open_interest_change"] = open_interest_change.fillna(0.0).clip(-1.0, 5.0)
    result["u_spread_rate"] = spread_bps.fillna(0.0).clip(0.0, 1_000.0) / 10_000
    result["u_derivatives_context_available"] = (
        pd.to_numeric(available, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    )
    # 前面已建立大量技術欄位，先整理記憶體區塊再一次加入多模態特徵。
    result = result.copy()

    macro_available = optional_numeric("macro_available")
    breadth_available = optional_numeric("breadth_available")
    fundamentals_available = optional_numeric("fundamentals_available")
    orderbook_available = optional_numeric("orderbook_available")
    advanced = pd.DataFrame(
        {
            "u_macro_yield_curve": (
                optional_numeric("macro_yield_curve_10y_2y")
                .fillna(0.0)
                .clip(-10.0, 10.0)
                / 100
            ),
            "u_macro_vix_level": (
                optional_numeric("macro_vixcls_value").fillna(0.0).clip(0.0, 100.0)
                / 100
            ),
            "u_macro_vix_zscore": (
                optional_numeric("macro_vixcls_zscore_60")
                .fillna(0.0)
                .clip(-5.0, 5.0)
                / 5
            ),
            "u_macro_dollar_change": (
                optional_numeric("macro_dtwexbgs_pct_change")
                .fillna(0.0)
                .clip(-0.2, 0.2)
            ),
            "u_macro_high_yield_spread": (
                optional_numeric("macro_bamlh0a0hym2_value")
                .fillna(0.0)
                .clip(0.0, 25.0)
                / 100
            ),
            "u_macro_available": macro_available.fillna(0.0).clip(0.0, 1.0),
            "u_breadth_advance_ratio": (
                optional_numeric("breadth_advance_ratio").fillna(0.5).clip(0.0, 1.0)
            ),
            "u_breadth_above_sma_20": (
                optional_numeric("breadth_pct_above_sma_20")
                .fillna(0.5)
                .clip(0.0, 1.0)
            ),
            "u_breadth_above_sma_50": (
                optional_numeric("breadth_pct_above_sma_50")
                .fillna(0.5)
                .clip(0.0, 1.0)
            ),
            "u_breadth_above_sma_200": (
                optional_numeric("breadth_pct_above_sma_200")
                .fillna(0.5)
                .clip(0.0, 1.0)
            ),
            "u_breadth_new_high_ratio": (
                optional_numeric("breadth_new_high_ratio_20")
                .fillna(0.0)
                .clip(0.0, 1.0)
            ),
            "u_breadth_new_low_ratio": (
                optional_numeric("breadth_new_low_ratio_20")
                .fillna(0.0)
                .clip(0.0, 1.0)
            ),
            "u_breadth_median_return": (
                optional_numeric("breadth_median_return")
                .fillna(0.0)
                .clip(-0.5, 0.5)
            ),
            "u_breadth_dispersion": (
                optional_numeric("breadth_return_dispersion")
                .fillna(0.0)
                .clip(0.0, 0.5)
            ),
            "u_breadth_available": breadth_available.fillna(0.0).clip(0.0, 1.0),
            "u_fundamental_revenue_growth": (
                optional_numeric("fundamental_revenue_growth")
                .fillna(0.0)
                .clip(-2.0, 5.0)
            ),
            "u_fundamental_profit_margin": (
                optional_numeric("fundamental_profit_margin")
                .fillna(0.0)
                .clip(-2.0, 2.0)
            ),
            "u_fundamental_operating_margin": (
                optional_numeric("fundamental_operating_margin")
                .fillna(0.0)
                .clip(-2.0, 2.0)
            ),
            "u_fundamental_debt_to_equity": (
                optional_numeric("fundamental_debt_to_equity")
                .fillna(0.0)
                .clip(-10.0, 10.0)
                / 10
            ),
            "u_fundamental_ocf_margin": (
                optional_numeric("fundamental_ocf_margin")
                .fillna(0.0)
                .clip(-2.0, 2.0)
            ),
            "u_fundamentals_available": (
                fundamentals_available.fillna(0.0).clip(0.0, 1.0)
            ),
            "u_global_long_short_pressure": (
                np.log(
                    optional_numeric("global_long_short_ratio")
                    .fillna(1.0)
                    .clip(0.05, 20.0)
                )
                / np.log(20.0)
            ),
            "u_taker_buy_sell_pressure": (
                np.log(
                    optional_numeric("taker_buy_sell_ratio")
                    .fillna(1.0)
                    .clip(0.05, 20.0)
                )
                / np.log(20.0)
            ),
            "u_basis_rate": (
                optional_numeric("basis_rate").fillna(0.0).clip(-0.2, 0.2)
            ),
            "u_orderbook_imbalance_5": (
                optional_numeric("orderbook_imbalance_5")
                .fillna(0.0)
                .clip(-1.0, 1.0)
            ),
            "u_orderbook_imbalance_10": (
                optional_numeric("orderbook_imbalance_10")
                .fillna(0.0)
                .clip(-1.0, 1.0)
            ),
            "u_orderbook_imbalance_20": (
                optional_numeric("orderbook_imbalance_20")
                .fillna(0.0)
                .clip(-1.0, 1.0)
            ),
            "u_orderbook_microprice_deviation": (
                optional_numeric("orderbook_microprice_deviation_bps")
                .fillna(0.0)
                .clip(-1_000.0, 1_000.0)
                / 10_000
            ),
            "u_orderbook_spread_rate": (
                optional_numeric("orderbook_spread_bps")
                .fillna(0.0)
                .clip(0.0, 1_000.0)
                / 10_000
            ),
            "u_orderbook_available": (
                orderbook_available.fillna(0.0).clip(0.0, 1.0)
            ),
        },
        index=result.index,
    )

    finbert_sentiment = optional_numeric("finbert_sentiment")
    finbert_positive = optional_numeric("finbert_positive")
    finbert_negative = optional_numeric("finbert_negative")
    finbert_confidence = optional_numeric("finbert_confidence")
    finbert_news_count = optional_numeric("finbert_news_count")
    finbert_change = optional_numeric("finbert_sentiment_change")
    finbert_age = optional_numeric("finbert_hours_since_news")
    finbert_available = optional_numeric("finbert_available")
    if "finbert_available" not in result:
        finbert_available = (
            finbert_sentiment.notna()
            | finbert_positive.notna()
            | finbert_negative.notna()
        ).astype(float)
    transformer_available = optional_numeric("transformer_available")
    if "transformer_available" not in result:
        transformer_available = pd.Series(0.0, index=result.index, dtype="float64")
    multimodal = pd.DataFrame(
        {
            "u_finbert_sentiment": finbert_sentiment.fillna(0.0).clip(-1.0, 1.0),
            "u_finbert_positive": finbert_positive.fillna(0.0).clip(0.0, 1.0),
            "u_finbert_negative": finbert_negative.fillna(0.0).clip(0.0, 1.0),
            "u_finbert_confidence": finbert_confidence.fillna(0.0).clip(0.0, 1.0),
            "u_finbert_news_count_log": (
                np.log1p(finbert_news_count.fillna(0.0).clip(0.0, 100.0))
                / np.log1p(100.0)
            ),
            "u_finbert_sentiment_change": finbert_change.fillna(0.0).clip(-2.0, 2.0),
            "u_finbert_recency": (
                np.exp(-finbert_age.fillna(1_000.0).clip(lower=0.0) / 24.0)
                * finbert_available.fillna(0.0).clip(0.0, 1.0)
            ),
            "u_finbert_available": finbert_available.fillna(0.0).clip(0.0, 1.0),
            "u_transformer_return_1": (
                optional_numeric("transformer_return_1").fillna(0.0).clip(-1.0, 1.0)
            ),
            "u_transformer_return_5": (
                optional_numeric("transformer_return_5").fillna(0.0).clip(-1.0, 1.0)
            ),
            "u_transformer_return_20": (
                optional_numeric("transformer_return_20").fillna(0.0).clip(-1.0, 1.0)
            ),
            "u_transformer_volatility": (
                optional_numeric("transformer_volatility").fillna(0.0).clip(0.0, 1.0)
            ),
            "u_transformer_bull_probability": (
                optional_numeric("transformer_bull_probability")
                .fillna(0.0)
                .clip(0.0, 1.0)
            ),
            "u_transformer_bear_probability": (
                optional_numeric("transformer_bear_probability")
                .fillna(0.0)
                .clip(0.0, 1.0)
            ),
            "u_transformer_uncertainty": (
                optional_numeric("transformer_uncertainty").fillna(0.0).clip(0.0, 1.0)
            ),
            "u_transformer_available": (
                transformer_available.fillna(0.0).clip(0.0, 1.0)
            ),
        },
        index=result.index,
    )
    for horizon in (1, 5, 20):
        down = optional_numeric(
            f"transformer_down_probability_{horizon}"
        ).fillna(0.0).clip(0.0, 1.0)
        up = optional_numeric(
            f"transformer_up_probability_{horizon}"
        ).fillna(0.0).clip(0.0, 1.0)
        neutral = optional_numeric(
            f"transformer_neutral_probability_{horizon}"
        ).fillna(0.0).clip(0.0, 1.0)
        movement_down = optional_numeric(
            f"transformer_movement_down_probability_{horizon}"
        ).fillna(0.0).clip(0.0, 1.0)
        movement_neutral = optional_numeric(
            f"transformer_movement_neutral_probability_{horizon}"
        ).fillna(0.0).clip(0.0, 1.0)
        movement_up = optional_numeric(
            f"transformer_movement_up_probability_{horizon}"
        ).fillna(0.0).clip(0.0, 1.0)
        side_down = optional_numeric(
            f"transformer_side_down_probability_{horizon}"
        ).fillna(0.0).clip(0.0, 1.0)
        side_up = optional_numeric(
            f"transformer_side_up_probability_{horizon}"
        ).fillna(0.0).clip(0.0, 1.0)
        side_total = side_down + side_up
        legacy_side_total = down + up
        side_down = side_down.where(
            side_total > 0,
            down / legacy_side_total.where(legacy_side_total > 0),
        ).fillna(0.0)
        side_up = side_up.where(
            side_total > 0,
            up / legacy_side_total.where(legacy_side_total > 0),
        ).fillna(0.0)
        lower = optional_numeric(
            f"transformer_return_q10_{horizon}"
        ).fillna(0.0).clip(-1.0, 1.0)
        median = optional_numeric(
            f"transformer_return_q50_{horizon}"
        ).fillna(0.0).clip(-1.0, 1.0)
        upper = optional_numeric(
            f"transformer_return_q90_{horizon}"
        ).fillna(0.0).clip(-1.0, 1.0)
        multimodal[f"u_transformer_direction_edge_{horizon}"] = up - down
        multimodal[f"u_transformer_movement_direction_edge_{horizon}"] = (
            movement_up - movement_down
        )
        multimodal[f"u_transformer_movement_neutral_probability_{horizon}"] = (
            movement_neutral
        )
        multimodal[f"u_transformer_side_direction_edge_{horizon}"] = (
            side_up - side_down
        )
        multimodal[f"u_transformer_neutral_probability_{horizon}"] = neutral
        multimodal[f"u_transformer_return_q10_{horizon}"] = lower
        multimodal[f"u_transformer_return_q50_{horizon}"] = median
        multimodal[f"u_transformer_return_q90_{horizon}"] = upper
        multimodal[f"u_transformer_interval_width_{horizon}"] = (
            upper - lower
        ).clip(0.0, 2.0)
        for name in (
            "long_edge",
            "short_edge",
            "downside_excursion",
            "upside_excursion",
        ):
            multimodal[f"u_transformer_{name}_{horizon}"] = optional_numeric(
                f"transformer_{name}_{horizon}"
            ).fillna(0.0).clip(-1.0 if "edge" in name else 0.0, 1.0)
        multimodal[f"u_transformer_tradeability_{horizon}"] = optional_numeric(
            f"transformer_tradeability_{horizon}"
        ).fillna(0.0).clip(0.0, 1.0)
    for name in ("low", "normal", "high"):
        multimodal[f"u_transformer_volatility_regime_{name}_probability"] = (
            optional_numeric(
                f"transformer_volatility_regime_{name}_probability"
            ).fillna(0.0).clip(0.0, 1.0)
        )
    for name in ("fast", "medium", "slow"):
        multimodal[f"u_transformer_timeframe_{name}_attention"] = optional_numeric(
            f"transformer_timeframe_{name}_attention"
        ).fillna(0.0).clip(0.0, 1.0)
    available = multimodal["u_transformer_available"]
    confidence = (1.0 - multimodal["u_transformer_uncertainty"]).clip(0.0, 1.0)
    role_signals: dict[int, pd.Series] = {}
    for horizon in (1, 5, 20):
        role_signals[horizon] = (
            multimodal[f"u_transformer_side_direction_edge_{horizon}"]
            * multimodal[f"u_transformer_tradeability_{horizon}"]
            * confidence
            * available
        )
    multimodal["u_transformer_execution_signal"] = role_signals[1]
    multimodal["u_transformer_setup_signal"] = role_signals[5]
    multimodal["u_transformer_trend_signal"] = role_signals[20]
    multimodal["u_transformer_consensus_signal"] = (
        0.15 * role_signals[1]
        + 0.35 * role_signals[5]
        + 0.50 * role_signals[20]
    )
    signal_frame = pd.DataFrame(role_signals, index=result.index)
    multimodal["u_transformer_signal_disagreement"] = (
        signal_frame.max(axis=1) - signal_frame.min(axis=1)
    ) * available
    # 分歧保留下來作為風險情境，而不是強迫三個預測 horizon 同方向。
    horizon_probabilities = []
    for horizon in (1, 5, 20):
        values = pd.concat([
            optional_numeric(f"transformer_{name}_probability_{horizon}")
            for name in ("down", "neutral", "up")
        ], axis=1).fillna(0).clip(0, 1)
        values.columns = ["down", "neutral", "up"]
        values = values.div(values.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        horizon_probabilities.append(values)
    average_probability = sum(horizon_probabilities) / len(horizon_probabilities)
    multimodal["u_transformer_signal_entropy"] = (
        -(average_probability * np.log(average_probability.clip(lower=1e-12))).sum(axis=1)
        / np.log(3) * available
    ).clip(0, 1)
    weighted_tradeability = (
        0.15 * multimodal["u_transformer_tradeability_1"]
        + 0.35 * multimodal["u_transformer_tradeability_5"]
        + 0.50 * multimodal["u_transformer_tradeability_20"]
    )
    multimodal["u_transformer_decision_confidence"] = (
        confidence * weighted_tradeability * available
    )
    multimodal["u_transformer_no_trade_pressure"] = (
        (1.0 - weighted_tradeability) * available
    )
    multimodal["u_transformer_net_edge_advantage_5"] = (
        multimodal["u_transformer_long_edge_5"]
        - multimodal["u_transformer_short_edge_5"]
    ) * available
    multimodal["u_transformer_expected_best_edge_5"] = (
        multimodal[["u_transformer_long_edge_5", "u_transformer_short_edge_5"]]
        .max(axis=1)
        .clip(-1.0, 1.0)
        * multimodal["u_transformer_tradeability_5"]
        * available
    )
    market_context = pd.DataFrame({
        f"u_{column}": pd.to_numeric(result[column], errors="coerce").fillna(0)
        for column in MARKET_CONTEXT_COLUMNS
    }, index=result.index)
    if feature_columns is not None:
        result = result.drop(columns=[
            column for column in result if column.startswith("u_") and column not in feature_columns
        ])
        advanced = advanced[[column for column in advanced if column in feature_columns]]
        multimodal = multimodal[[column for column in multimodal if column in feature_columns]]
        market_context = market_context[[column for column in market_context if column in feature_columns]]
    result = pd.concat([result, advanced, multimodal, market_context], axis=1)
    result.attrs = dict(frame.attrs)
    return result


def compact_short_term_feature_columns(
    frame: pd.DataFrame, *, use_transformer: bool = True, use_finbert: bool = False,
    maximum_features: int = 112,
) -> list[str]:
    """依欄位 schema 選擇，第一根暖機缺值不應改變整套模型的輸入。"""
    sample = frame.head(1).copy()
    for column in sample:
        if column.startswith("mtf_") and column not in {"mtf_source_intervals", "mtf_decision_interval"}:
            sample[column] = pd.to_numeric(sample[column], errors="coerce").fillna(0)
    schema = add_universal_rl_features(sample)
    return select_compact_short_term_features(
        schema, use_transformer=use_transformer, use_finbert=use_finbert,
        maximum_features=maximum_features,
    )


def build_compact_short_term_frame(
    frame: pd.DataFrame, *, use_transformer: bool = True, use_finbert: bool = False,
    maximum_features: int = 112,
) -> tuple[pd.DataFrame, list[str]]:
    """只建立固定契約需要的 u_ 欄，降低五年資料記憶體。"""
    columns = compact_short_term_feature_columns(
        frame, use_transformer=use_transformer, use_finbert=use_finbert,
        maximum_features=maximum_features,
    )
    result = add_universal_rl_features(frame, feature_columns=columns)
    result.attrs["rl_feature_contract"] = compact_feature_metadata(columns)
    return result, columns


def prepare_universal_rl_dataset(
    frames: dict[str, pd.DataFrame],
    split_config: RLSplitConfig | None = None,
    expert_kind: ExpertKind = "general",
    *,
    use_finbert: bool = True,
    use_transformer: bool = True,
) -> PreparedUniversalRLDataset:
    """逐市場切分，再只用所有市場的訓練段估計共同標準化參數。"""
    if len(frames) < 2:
        raise ValueError("通用 RL 至少需要兩個市場")
    if expert_kind == "long_term":
        selected_features = list(LONG_TERM_RL_FEATURE_COLUMNS)
    elif expert_kind == "short_term":
        selected_features = list(SHORT_TERM_RL_FEATURE_COLUMNS)
    else:
        selected_features = list(UNIVERSAL_RL_FEATURE_COLUMNS)
    mtf_schemas = [
        tuple(f"u_{column}" for column in _multitimeframe_source_columns(frame))
        for frame in frames.values()
    ]
    if any(mtf_schemas):
        expected = mtf_schemas[0]
        if not expected or any(schema != expected for schema in mtf_schemas[1:]):
            raise ValueError("通用 RL 的所有市場必須使用相同多週期特徵架構")
        selected_features.extend(
            column for column in expected if column not in selected_features
        )
    if expert_kind == "short_term":
        selected_features = compact_short_term_feature_columns(
            next(iter(frames.values())),
            use_finbert=use_finbert, use_transformer=use_transformer,
        )
    split = split_config or RLSplitConfig()
    prepared: dict[str, UniversalMarketDataset] = {}
    ai_coverage: dict[str, dict[str, float]] = {}
    expected_contracts: list[dict[str, object]] = []
    for market, source in frames.items():
        horizon = 5 if expert_kind == "short_term" else 20
        contract = dict(source.attrs.get("expected_return_contract", {}))
        if use_transformer and f"transformer_return_{horizon}" in source:
            source = attach_expected_return(source, contract or build_expected_return_contract(horizon))
            contract = dict(source.attrs["expected_return_contract"])
        expected_contracts.append(contract)
        result = add_universal_rl_features(source)
        ai_coverage[market] = {
            "finbert": float(result["u_finbert_available"].mean()),
            "transformer": float(result["u_transformer_available"].mean()),
            "multitimeframe": float(
                result[
                    [
                        column
                        for column in selected_features
                        if column.startswith("u_mtf_")
                        and column.endswith("_available")
                    ]
                ].mean(axis=1).mean()
            )
            if any(
                column.startswith("u_mtf_") and column.endswith("_available")
                for column in selected_features
            )
            else 0.0,
        }
        if not use_finbert:
            result[[column for column in selected_features if "finbert" in column]] = 0.0
        if not use_transformer:
            result[
                [column for column in selected_features if "transformer" in column]
            ] = 0.0
        if expert_kind == "short_term":
            market_schema = select_compact_short_term_features(
                result, use_finbert=use_finbert, use_transformer=use_transformer,
            )
            if market_schema != selected_features:
                raise ValueError("短線通用 RL 的所有市場必須使用相同精簡特徵契約")
        required = [*MARKET_COLUMNS, *selected_features]
        required.extend(column for column in (
            "expected_return", "funding_rate", "spread_bps", "event_blackout",
            "transformer_available", "finbert_available",
        ) if column in result and column not in required)
        if contract and str(contract["source_column"]) in result:
            required.append(str(contract["source_column"]))
        result = result[required].copy()
        result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
        numeric = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            *selected_features,
        ]
        result[numeric] = result[numeric].apply(pd.to_numeric, errors="coerce")
        result[numeric] = result[numeric].replace([np.inf, -np.inf], np.nan)
        result = result.dropna(subset=["timestamp", *numeric]).sort_values("timestamp")
        result = result.drop_duplicates("timestamp", keep="last").reset_index(drop=True)
        rows = len(result)
        train_end = int(rows * split.train_fraction)
        validation_end = train_end + int(rows * split.validation_fraction)
        sizes = [train_end, validation_end - train_end, rows - validation_end]
        if min(sizes) < split.min_rows_per_split:
            raise ValueError(f"{market} 的訓練／驗證／測試資料不足：{sizes}")
        prepared[market] = UniversalMarketDataset(result, train_end, validation_end)

    train_features = pd.concat(
        [market.train[selected_features] for market in prepared.values()],
        ignore_index=True,
    )
    feature_mean = train_features.mean()
    feature_std = train_features.std(ddof=0)
    # 多市場中仍可能出現近乎常數的特徵，避免浮點誤差被標準化放大。
    feature_std = feature_std.mask(feature_std.abs() < 1e-12, 1.0).fillna(1.0)
    for market in prepared.values():
        market.frame[selected_features] = (
            (market.frame[selected_features] - feature_mean) / feature_std
        ).clip(-10.0, 10.0)
    if any(contract != expected_contracts[0] for contract in expected_contracts[1:]):
        raise ValueError("通用 RL 各市場的 expected_return 契約必須相同")
    return PreparedUniversalRLDataset(
        prepared,
        selected_features,
        feature_mean,
        feature_std,
        split,
        ai_coverage,
        use_finbert,
        use_transformer,
        expected_contracts[0],
        compact_feature_metadata(selected_features) if expert_kind == "short_term" else {},
    )


def _market_slug(name: str, index: int) -> str:
    safe = "".join(character if character.isalnum() else "-" for character in name)
    return f"{index:02d}_{safe.strip('-') or 'market'}"


def save_universal_rl_environment(
    dataset: PreparedUniversalRLDataset,
    env_config: PortfolioEnvConfig,
    output_dir: str | Path,
    *,
    source_paths: dict[str, str] | None = None,
    diagnostic: pd.DataFrame | None = None,
    transformer_checkpoint: str | Path | None = None,
    finbert_scored_news_path: str | Path | None = None,
    finbert_config: dict[str, object] | None = None,
    transformer_provenance: dict[str, object] | None = None,
) -> UniversalRLArtifactPaths:
    """保存通用環境，每個市場保留獨立 train／validation／test。"""
    if transformer_checkpoint is not None and not transformer_provenance:
        raise ValueError("帶 Transformer 的通用 RL 環境缺少時間隔離來源證明")
    if transformer_checkpoint is not None and not dataset.expected_return_contract:
        raise ValueError("新 Transformer RL 環境必須明確指定 expected_return_contract，不可猜測預測週期")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = Path(output_dir).resolve() / f"{stamp}_universal_{len(dataset.markets)}markets"
    markets_dir = run_dir / "markets"
    markets_dir.mkdir(parents=True, exist_ok=False)
    transformer_relative_path: str | None = None
    if transformer_checkpoint is not None:
        source_checkpoint = Path(transformer_checkpoint).resolve()
        if not source_checkpoint.is_file():
            raise FileNotFoundError(f"找不到 Transformer 模型：{source_checkpoint}")
        ai_dir = run_dir / "ai"
        ai_dir.mkdir()
        bundled_checkpoint = ai_dir / "transformer_model.pt"
        shutil.copy2(source_checkpoint, bundled_checkpoint)
        transformer_relative_path = "ai/transformer_model.pt"
    market_payload: dict[str, object] = {}
    totals = {"train": 0, "validation": 0, "test": 0}
    for index, (name, market) in enumerate(dataset.markets.items()):
        slug = _market_slug(name, index)
        target = markets_dir / slug
        target.mkdir()
        files = {}
        sizes = {}
        for split_name in ["train", "validation", "test"]:
            frame = getattr(market, split_name)
            relative = Path("markets") / slug / f"{split_name}.csv"
            frame.to_csv(run_dir / relative, index=False, encoding="utf-8")
            files[split_name] = str(relative).replace("\\", "/")
            sizes[split_name] = len(frame)
            totals[split_name] += len(frame)
        first = market.frame.iloc[0]
        market_payload[name] = {
            "source_path": (source_paths or {}).get(name),
            "exchange": str(first.get("exchange", "unknown")),
            "symbol": str(first.get("symbol", name)),
            "interval": str(first.get("interval", "unknown")),
            "start": str(market.frame.iloc[0]["timestamp"]),
            "end": str(market.frame.iloc[-1]["timestamp"]),
            "files": files,
            "split_rows": sizes,
        }

    diagnostic_csv = run_dir / "diagnostic.csv"
    (diagnostic if diagnostic is not None else pd.DataFrame()).to_csv(
        diagnostic_csv,
        index=False,
        encoding="utf-8",
    )
    intervals = {
        str(market.frame.iloc[0].get("interval", "unknown"))
        for market in dataset.markets.values()
    }
    coverage_values = list(dataset.ai_coverage.values())
    coverage_names = sorted(
        {name for item in coverage_values for name in item}
    )
    aggregate_coverage = {
        name: float(np.mean([item.get(name, 0.0) for item in coverage_values]))
        if coverage_values
        else 0.0
        for name in coverage_names
    }
    finbert_source = (
        str(finbert_scored_news_path).replace("\\", "/")
        if finbert_scored_news_path is not None
        else None
    )
    runtime_ready = bool(
        (not dataset.use_transformer or transformer_relative_path)
        and (not dataset.use_finbert or finbert_source)
    )
    payload = {
        "status": "environment_ready",
        "training_started": False,
        "environment_kind": "universal",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "exchange": "multi_market",
            "symbol": "UNIVERSAL",
            "interval": next(iter(intervals)) if len(intervals) == 1 else "mixed",
            "market_count": len(dataset.markets),
        },
        "markets": market_payload,
        "feature_columns": dataset.feature_columns,
        "feature_contract": dataset.feature_contract,
        "feature_transform": "universal_ratios",
        "observation_size": len(dataset.feature_columns)
        + (5 if env_config.include_position_context else 3)
        + (7 if env_config.include_risk_context else 0)
        + (2 if env_config.include_trade_plan_context else 0),
        "action": {
            "type": "continuous_target_position_fraction",
            "minimum": -env_config.max_short_fraction,
            "maximum": env_config.max_position_fraction,
        },
        "reward": {
            "formula": (
                "log_return - drawdown_penalty - turnover_penalty - "
                "downside_penalty - concentration_penalty - risk_termination_penalty"
            ),
            "next_open_execution": True,
        },
        "expert": {
            "kind": env_config.expert_kind,
            "rebalance_deadband": env_config.rebalance_deadband,
            "minimum_holding_bars": env_config.minimum_holding_bars,
            "soft_drawdown_limit": env_config.soft_drawdown_limit,
            "allow_short": env_config.allow_short,
            "max_short_fraction": env_config.max_short_fraction,
        },
        "ai_context": {
            "finbert_enabled": dataset.use_finbert,
            "transformer_enabled": dataset.use_transformer,
            "multitimeframe_enabled": any(
                column.startswith("u_mtf_") for column in dataset.feature_columns
            ),
            "multitimeframe_features": [
                column
                for column in dataset.feature_columns
                if column.startswith("u_mtf_")
            ],
            "availability_flags_required": True,
            "runtime_ready": runtime_ready,
            "finbert_scored_news_path": finbert_source,
            "finbert_config": finbert_config or {},
            "transformer_checkpoint": transformer_relative_path,
            "transformer_provenance": transformer_provenance or {},
            "expected_return_contract": (
                dataset.expected_return_contract
                if transformer_relative_path is not None else None
            ),
            "coverage": {
                "aggregate": aggregate_coverage,
                "markets": dataset.ai_coverage,
            },
        },
        "environment_config": env_config.to_dict(),
        "split_config": dataset.split_config.to_dict(),
        "split_rows": totals,
        "normalization": {
            "fit_on": "all_markets_train_only",
            "clip": [-10.0, 10.0],
            "mean": {key: float(value) for key, value in dataset.feature_mean.items()},
            "std": {key: float(value) for key, value in dataset.feature_std.items()},
        },
    }
    metadata_json = run_dir / "environment.json"
    metadata_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return UniversalRLArtifactPaths(run_dir, metadata_json, diagnostic_csv)
