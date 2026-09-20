"""建立不跨市場、依時間切分的 Transformer 訓練資料。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ai_quant_trading.features.contracts import forbidden_model_feature_reason
from ai_quant_trading.features.market_context import (
    MARKET_CONTEXT_COLUMNS,
    NORMALIZED_PRICE_COLUMNS,
    add_model_market_features,
    is_price_level_feature,
    market_timeframes,
)
from ai_quant_trading.transformer.config import (
    TemporalTransformerConfig,
    TransformerTrainingConfig,
)
from ai_quant_trading.features.multitimeframe import (
    MULTITIMEFRAME_FEATURE_GROUPS,
)


IDENTITY_COLUMNS = {
    "timestamp",
    "symbol",
    "exchange",
    "interval",
    "collected_at",
    "open_time_ms",
    "close_time_ms",
    "target_timestamp",
    "target_horizon",
    "future_return",
    "target",
    "mtf_schema_version",
}

# 特徵數不足時優先保留這些已知且可跨市場解讀的欄位。
FEATURE_PRIORITY = (
    *MARKET_CONTEXT_COLUMNS,
    *NORMALIZED_PRICE_COLUMNS,
    "return_1",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_histogram",
    "roc_12_pct",
    "atr_14",
    "bb_width_20",
    "historical_volatility_20",
    "volume_change",
    "obv",
    "vwap",
    "higher_high",
    "higher_low",
    "breakout_20",
    "short_rsi_2",
    "short_ema_5_gap",
    "short_ema_200_gap",
    "short_bb_zscore_20",
    "ema_20_50_atr",
    "ema_50_200_atr",
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
    "volume_delta_ratio",
    "cvd_pressure_20",
    "trade_intensity_20",
    "microstructure_available",
    "funding_percentile_200",
    "funding_zscore_200",
    "open_interest_change_1",
    "open_interest_zscore_50",
    "derivatives_available",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "is_weekend",
    "minutes_to_funding_ratio",
    "finbert_sentiment",
    "finbert_positive",
    "finbert_negative",
    "finbert_confidence",
    "finbert_news_count",
    "finbert_sentiment_change",
    "finbert_age_hours",
    "macro_yield_curve_10y_2y",
    "macro_vixcls_value",
    "macro_vixcls_zscore_60",
    "macro_dtwexbgs_pct_change",
    "macro_bamlh0a0hym2_value",
    "breadth_advance_ratio",
    "breadth_pct_above_sma_20",
    "breadth_pct_above_sma_50",
    "breadth_pct_above_sma_200",
    "breadth_new_high_ratio_20",
    "breadth_new_low_ratio_20",
    "fundamental_revenue_growth",
    "fundamental_profit_margin",
    "fundamental_operating_margin",
    "fundamental_debt_to_equity",
    "fundamental_ocf_margin",
    "global_long_short_ratio",
    "taker_buy_sell_ratio",
    "basis_rate",
    "orderbook_imbalance_5",
    "orderbook_imbalance_10",
    "orderbook_imbalance_20",
    "orderbook_microprice_deviation_bps",
    "orderbook_spread_bps",
    "spread_bps",
    "microprice_deviation_bps",
    "bid_depth_5",
    "ask_depth_5",
    "imbalance_5",
    "bid_depth_10",
    "ask_depth_10",
    "imbalance_10",
    "bid_depth_20",
    "ask_depth_20",
    "imbalance_20",
    "mark_price",
    "index_price",
    "premium_index",
    "liquidation_long_ratio",
    "liquidation_short_ratio",
)

FEATURE_GROUP_NAMES = ("fast", "medium", "slow")
FAST_INTERVALS = {"1m", "3m", "5m"}
MEDIUM_INTERVALS = {"15m", "30m", "1h"}
SLOW_INTERVALS = {"4h", "12h", "1d"}
FAST_FEATURE_MARKERS = (
    "orderbook",
    "microprice",
    "spread",
    "depth",
    "imbalance",
    "taker",
    "cvd",
    "volume_delta",
    "trade_intensity",
    "average_trade",
    "large_trade",
    "amihud",
    "candle_",
    "wick_",
)
SLOW_FEATURE_MARKERS = (
    "funding",
    "open_interest",
    "basis",
    "premium",
    "mark_price",
    "index_price",
    "liquidation",
    "long_short",
    "finbert",
    "sentiment",
    "macro_",
    "vix",
    "dxy",
    "yield",
    "breadth",
    "hour_",
    "weekday_",
    "weekend",
    "session",
)


@dataclass(frozen=True, slots=True)
class TransformerScaler:
    """只由訓練區段估計的缺值與標準化參數。"""

    feature_columns: tuple[str, ...]
    medians: tuple[float, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    return_means: tuple[float, ...]
    return_scales: tuple[float, ...]
    volatility_mean: float
    volatility_scale: float
    regime_thresholds: tuple[float, ...]
    edge_means: tuple[float, ...] = ()
    edge_scales: tuple[float, ...] = ()
    excursion_scales: tuple[float, ...] = ()
    volatility_regime_thresholds: tuple[float, ...] = ()
    feature_group_ids: tuple[int, ...] = ()
    feature_group_names: tuple[str, ...] = FEATURE_GROUP_NAMES
    label_mode: str = "close_to_close"

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_columns": list(self.feature_columns),
            "medians": list(self.medians),
            "means": list(self.means),
            "scales": list(self.scales),
            "return_means": list(self.return_means),
            "return_scales": list(self.return_scales),
            "volatility_mean": self.volatility_mean,
            "volatility_scale": self.volatility_scale,
            "regime_thresholds": list(self.regime_thresholds),
            "edge_means": list(self.edge_means),
            "edge_scales": list(self.edge_scales),
            "excursion_scales": list(self.excursion_scales),
            "volatility_regime_thresholds": list(self.volatility_regime_thresholds),
            "feature_group_ids": list(self.feature_group_ids),
            "feature_group_names": list(self.feature_group_names),
            "label_mode": self.label_mode,
        }


@dataclass(frozen=True, slots=True)
class TransformerSourceSummary:
    """單一市場資料在訓練工作中的摘要。"""

    path: Path
    symbol: str
    exchange: str
    interval: str
    rows: int
    start_at: str
    end_at: str

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path.resolve()),
            "symbol": self.symbol,
            "exchange": self.exchange,
            "interval": self.interval,
            "rows": self.rows,
            "start_at": self.start_at,
            "end_at": self.end_at,
        }


@dataclass(slots=True)
class _PreparedSeries:
    features: np.ndarray
    feature_mask: np.ndarray
    future_returns: np.ndarray
    future_volatility: np.ndarray
    future_directions: np.ndarray
    movement_directions: np.ndarray
    future_sides: np.ndarray
    regime_source: np.ndarray
    edge_returns: np.ndarray
    future_excursions: np.ndarray
    tradeability: np.ndarray
    volatility_regime_source: np.ndarray
    train_endpoints: np.ndarray
    validation_endpoints: np.ndarray
    test_endpoints: np.ndarray


class MarketSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    """以終點索引即時切出序列，避免預先展開造成大量記憶體占用。"""

    def __init__(
        self,
        series: Sequence[_PreparedSeries],
        split: str,
        sequence_length: int,
        scaler: TransformerScaler,
    ) -> None:
        endpoint_name = f"{split}_endpoints"
        self.series = tuple(series)
        self.sequence_length = sequence_length
        self.return_means = np.asarray(scaler.return_means, dtype=np.float32)
        self.return_scales = np.asarray(scaler.return_scales, dtype=np.float32)
        self.volatility_mean = np.float32(scaler.volatility_mean)
        self.volatility_scale = np.float32(scaler.volatility_scale)
        self.regime_thresholds = np.asarray(
            scaler.regime_thresholds,
            dtype=np.float32,
        )
        horizon_count = len(scaler.return_means)
        self.edge_means = (
            np.asarray(scaler.edge_means, dtype=np.float32).reshape(
                horizon_count,
                2,
            )
            if scaler.edge_means
            else np.empty((0, 2), dtype=np.float32)
        )
        self.edge_scales = (
            np.asarray(scaler.edge_scales, dtype=np.float32).reshape(
                horizon_count,
                2,
            )
            if scaler.edge_scales
            else np.empty((0, 2), dtype=np.float32)
        )
        self.excursion_scales = (
            np.asarray(
                scaler.excursion_scales,
                dtype=np.float32,
            ).reshape(horizon_count, 2)
            if scaler.excursion_scales
            else np.empty(
                (0, 2),
                dtype=np.float32,
            )
        )
        self.volatility_regime_thresholds = np.asarray(
            scaler.volatility_regime_thresholds,
            dtype=np.float32,
        )
        references: list[tuple[int, int]] = []
        for series_index, item in enumerate(self.series):
            endpoints = getattr(item, endpoint_name)
            references.extend((series_index, int(endpoint)) for endpoint in endpoints)
        self.references = tuple(references)

    def __len__(self) -> int:
        return len(self.references)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        series_index, endpoint = self.references[index]
        item = self.series[series_index]
        start = endpoint - self.sequence_length + 1
        features = item.features[start : endpoint + 1]
        feature_mask = item.feature_mask[start : endpoint + 1]
        returns = (item.future_returns[endpoint] - self.return_means) / self.return_scales
        volatility = (
            item.future_volatility[endpoint] - self.volatility_mean
        ) / self.volatility_scale
        regime = int(
            np.searchsorted(
                self.regime_thresholds,
                item.regime_source[endpoint],
                side="right",
            )
        )
        sample = {
            "features": torch.from_numpy(features),
            "feature_mask": torch.from_numpy(feature_mask),
            "future_returns": torch.from_numpy(returns.astype(np.float32)),
            "volatility": torch.tensor([volatility], dtype=torch.float32),
            "future_directions": torch.from_numpy(
                item.future_directions[endpoint].astype(np.int64)
            ),
            "movement_directions": torch.from_numpy(
                item.movement_directions[endpoint].astype(np.int64)
            ),
            "future_sides": torch.from_numpy(item.future_sides[endpoint].astype(np.int64)),
            "regime": torch.tensor(regime, dtype=torch.long),
        }
        if self.edge_means.size:
            edges = (item.edge_returns[endpoint] - self.edge_means) / self.edge_scales
            excursions = item.future_excursions[endpoint] / self.excursion_scales
            volatility_regime = int(
                np.searchsorted(
                    self.volatility_regime_thresholds,
                    item.volatility_regime_source[endpoint],
                    side="right",
                )
            )
            sample.update(
                {
                    "edge_returns": torch.from_numpy(edges.astype(np.float32)),
                    "excursions": torch.from_numpy(excursions.astype(np.float32)),
                    "tradeability": torch.from_numpy(
                        item.tradeability[endpoint].astype(np.float32)
                    ),
                    "volatility_regime": torch.tensor(
                        volatility_regime,
                        dtype=torch.long,
                    ),
                }
            )
        return sample


@dataclass(frozen=True, slots=True)
class PreparedTransformerData:
    """訓練器使用的三份 Dataset 與可重現中繼資料。"""

    train: MarketSequenceDataset
    validation: MarketSequenceDataset
    test: MarketSequenceDataset
    scaler: TransformerScaler
    sources: tuple[TransformerSourceSummary, ...]
    resolved_model_config: TemporalTransformerConfig

    @property
    def sample_counts(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
        }


def _is_eligible_feature(column: str) -> bool:
    lowered = column.lower()
    return (
        lowered not in IDENTITY_COLUMNS
        and not is_price_level_feature(lowered)
        and forbidden_model_feature_reason(
            lowered,
            forbid_transformer_outputs=True,
        )
        is None
    )


def _load_source(
    path: Path,
    max_rows: int | None,
) -> tuple[pd.DataFrame, TransformerSourceSummary]:
    if not path.exists():
        raise FileNotFoundError(f"找不到特徵資料：{path}")
    frame = pd.read_csv(path)
    required = {"timestamp", "close"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path.name} 缺少必要欄位：{missing}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = (
        frame.dropna(subset=["timestamp", "close"])
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )
    if max_rows is not None and len(frame) > max_rows:
        frame = frame.tail(max_rows).reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"{path.name} 沒有可用資料")
    first = frame.iloc[0]
    summary = TransformerSourceSummary(
        path=path.resolve(),
        symbol=str(first.get("symbol", path.stem)),
        exchange=str(first.get("exchange", "")),
        interval=str(first.get("interval", "")),
        rows=len(frame),
        start_at=pd.Timestamp(frame.iloc[0]["timestamp"]).isoformat(),
        end_at=pd.Timestamp(frame.iloc[-1]["timestamp"]).isoformat(),
    )
    return add_model_market_features(frame), summary


def transformer_feature_group_ids(columns: Sequence[str]) -> tuple[int, ...]:
    """依金融用途將欄位固定分派到快速、中速與慢速分支。"""
    groups: list[int] = []
    for column in columns:
        lowered = column.lower()
        match = re.match(r"^mtf_([^_]+)_", lowered)
        interval = match.group(1) if match else ""
        if interval in FAST_INTERVALS:
            groups.append(0)
        elif interval in MEDIUM_INTERVALS:
            groups.append(1)
        elif interval in SLOW_INTERVALS:
            groups.append(2)
        elif any(marker in lowered for marker in FAST_FEATURE_MARKERS):
            groups.append(0)
        elif any(marker in lowered for marker in SLOW_FEATURE_MARKERS):
            groups.append(2)
        else:
            groups.append(1)
    return tuple(groups)


def _interval_minutes(frame: pd.DataFrame) -> float:
    raw = str(frame.iloc[0].get("interval", "5m")).strip().lower()
    match = re.fullmatch(r"(\d+)([mhd])", raw)
    if not match:
        return 5.0
    value = float(match.group(1))
    return value * {"m": 1.0, "h": 60.0, "d": 1_440.0}[match.group(2)]


def _numeric_column(frame: pd.DataFrame, candidates: Sequence[str]) -> pd.Series:
    for column in candidates:
        if column in frame:
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.notna().any():
                return values
    return pd.Series(np.nan, index=frame.index, dtype="float64")


def _future_window_extreme(values: pd.Series, horizon: int, mode: str) -> pd.Series:
    shifted = values.shift(-1)
    rolling = shifted.rolling(horizon, min_periods=horizon)
    extreme = rolling.max() if mode == "max" else rolling.min()
    return extreme.shift(-(horizon - 1))


def _v3_targets(
    frame: pd.DataFrame,
    model_config: TemporalTransformerConfig,
    training_config: TransformerTrainingConfig,
) -> dict[str, np.ndarray]:
    """建立下一根可成交、扣除成本的多空與路徑風險標籤。"""
    close = pd.to_numeric(frame["close"], errors="coerce")
    open_price = _numeric_column(frame, ("open",)).fillna(close)
    high = _numeric_column(frame, ("high",)).fillna(close)
    low = _numeric_column(frame, ("low",)).fillna(close)
    entry = open_price.shift(-1).where(open_price.shift(-1) > 0)
    spread_bps = _numeric_column(
        frame,
        (
            "orderbook_spread_bps",
            "spread_bps",
            "mtf_5m_spread_fraction",
            "mtf_15m_spread_fraction",
        ),
    )
    if spread_bps.name and str(spread_bps.name).endswith("spread_fraction"):
        spread_bps = spread_bps * 10_000
    spread_bps = spread_bps.fillna(0.0).clip(
        lower=0.0,
        upper=training_config.max_observed_spread_bps,
    )
    fixed_cost = 2 * (training_config.fee_bps_per_side + training_config.slippage_bps_per_side)
    round_trip_cost = (fixed_cost + spread_bps) / 10_000
    funding_rate = _numeric_column(
        frame,
        ("funding_rate", "mtf_5m_funding_rate", "mtf_15m_funding_rate"),
    ).fillna(0.0)
    interval_minutes = _interval_minutes(frame)
    edge_buffer = training_config.direction_threshold_bps / 10_000

    returns: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    edges: list[np.ndarray] = []
    excursions: list[np.ndarray] = []
    tradeability: list[np.ndarray] = []
    movement_directions: list[np.ndarray] = []
    future_sides: list[np.ndarray] = []
    future_return_paths: list[np.ndarray] = []
    atr_fraction = _numeric_column(
        frame,
        ("mtf_15m_atr_pct", "atr_pct"),
    )
    if atr_fraction.isna().all():
        atr_fraction = _numeric_column(frame, ("atr_14",)) / close.where(close > 0)
    atr_fraction = atr_fraction.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    minimum_movement = training_config.movement_threshold_bps / 10_000
    for horizon in model_config.return_horizons:
        exit_price = close.shift(-horizon)
        gross_return = exit_price / entry - 1.0
        log_return = np.log(exit_price / entry)
        funding_cost = funding_rate * (horizon * interval_minutes / 480.0)
        long_edge = log_return - round_trip_cost - funding_cost
        short_edge = -log_return - round_trip_cost + funding_cost
        edge_pair = np.column_stack(
            [
                long_edge.to_numpy(dtype=np.float64),
                short_edge.to_numpy(dtype=np.float64),
            ]
        )
        best_direction = np.where(
            (long_edge > edge_buffer) & (long_edge >= short_edge),
            2,
            np.where(short_edge > edge_buffer, 0, 1),
        )
        movement_buffer = np.maximum(
            minimum_movement,
            atr_fraction.to_numpy(dtype=np.float64)
            * training_config.movement_atr_multiplier
            * np.sqrt(horizon),
        )
        log_return_values = log_return.to_numpy(dtype=np.float64)
        movement_direction = np.where(
            log_return_values > movement_buffer,
            2,
            np.where(log_return_values < -movement_buffer, 0, 1),
        )
        future_side = np.where(log_return_values >= 0.0, 1, 0)
        future_high = _future_window_extreme(high, horizon, "max")
        future_low = _future_window_extreme(low, horizon, "min")
        downside = (1.0 - future_low / entry).clip(lower=0.0)
        upside = (future_high / entry - 1.0).clip(lower=0.0)

        returns.append(gross_return.to_numpy(dtype=np.float64))
        directions.append(np.asarray(best_direction, dtype=np.int64))
        edges.append(edge_pair)
        excursions.append(
            np.column_stack(
                [
                    downside.to_numpy(dtype=np.float64),
                    upside.to_numpy(dtype=np.float64),
                ]
            )
        )
        tradeability.append(
            np.asarray(np.maximum(long_edge, short_edge) > edge_buffer, dtype=np.float64)
        )
        movement_directions.append(np.asarray(movement_direction, dtype=np.int64))
        future_sides.append(np.asarray(future_side, dtype=np.int64))

    max_horizon = max(model_config.return_horizons)
    one_bar_returns = close.pct_change(fill_method=None)
    for step in range(1, max_horizon + 1):
        future_return_paths.append(one_bar_returns.shift(-step).to_numpy(dtype=np.float64))
    future_volatility = (
        pd.DataFrame(np.column_stack(future_return_paths)).std(axis=1, ddof=0).to_numpy()
    )
    future_returns = np.column_stack(returns)
    edge_returns = np.stack(edges, axis=1)
    future_excursions = np.stack(excursions, axis=1)
    future_directions = np.column_stack(directions)
    tradeability_array = np.column_stack(tradeability)
    movement_direction_array = np.column_stack(movement_directions)
    future_side_array = np.column_stack(future_sides)
    risk_adjusted_trend = future_returns[:, -1] / np.maximum(
        future_volatility * np.sqrt(max_horizon),
        1e-8,
    )
    return {
        "future_returns": future_returns,
        "future_directions": future_directions,
        "movement_directions": movement_direction_array,
        "future_sides": future_side_array,
        "future_volatility": future_volatility,
        "regime_source": risk_adjusted_trend,
        "edge_returns": edge_returns,
        "future_excursions": future_excursions,
        "tradeability": tradeability_array,
        "volatility_regime_source": future_volatility,
    }


def _select_feature_columns(
    frames: Sequence[pd.DataFrame],
    maximum_features: int,
) -> tuple[str, ...]:
    if maximum_features < 1:
        raise ValueError("Transformer 特徵預算必須大於 0")
    frames = [add_model_market_features(frame) for frame in frames]
    common = set(frames[0].columns)
    for frame in frames[1:]:
        common.intersection_update(frame.columns)
    numeric_candidates: list[str] = []
    for column in frames[0].columns:
        if column not in common or not _is_eligible_feature(column):
            continue
        converted = pd.to_numeric(frames[0][column], errors="coerce")
        if converted.notna().any():
            numeric_candidates.append(column)
    mtf_candidates = [column for column in numeric_candidates if column.startswith("mtf_")]
    intervals = market_timeframes(mtf_candidates)
    metadata_priority = [
        f"mtf_{interval}_{name}"
        for interval in intervals
        for name in ("available", "age_ratio")
        if f"mtf_{interval}_{name}" in mtf_candidates
    ]
    grouped_queues: list[list[str]] = []
    grouped_columns: set[str] = set(metadata_priority)
    for feature_names in MULTITIMEFRAME_FEATURE_GROUPS.values():
        for interval in intervals:
            queue = [
                f"mtf_{interval}_{name}"
                for name in feature_names
                if f"mtf_{interval}_{name}" in mtf_candidates
            ]
            grouped_queues.append(queue)
            grouped_columns.update(queue)
    grouped_queues.append(
        sorted(column for column in mtf_candidates if column not in grouped_columns)
    )

    # 每輪從「金融用途 × 時間週期」各取一欄，避免短週期吃掉全部容量。
    balanced_priority: list[str] = []
    offsets = [0] * len(grouped_queues)
    while True:
        added = False
        for queue_index, queue in enumerate(grouped_queues):
            offset = offsets[queue_index]
            if offset < len(queue):
                balanced_priority.append(queue[offset])
                offsets[queue_index] += 1
                added = True
        if not added:
            break
    mtf_priority = metadata_priority + [
        column for column in balanced_priority if column not in metadata_priority
    ]
    base_candidates = [column for column in numeric_candidates if column not in mtf_candidates]
    priority = [column for column in FEATURE_PRIORITY if column in base_candidates]
    remaining = sorted(column for column in base_candidates if column not in priority)
    base_priority = priority + remaining
    if mtf_priority:
        # 多週期資料預設保留四分之三容量給各時間尺度，仍留下基礎週期欄位。
        base_budget = min(len(base_priority), maximum_features, max(4, maximum_features // 4))
        mtf_budget = max(0, maximum_features - base_budget)
        selected = mtf_priority[:mtf_budget] + base_priority[:base_budget]
    else:
        selected = base_priority[:maximum_features]
    if not selected:
        raise ValueError("找不到可供 Transformer 使用的共同數值特徵")
    return tuple(selected)


def _split_endpoints(
    rows: int,
    sequence_length: int,
    max_horizon: int,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    train_cut = int(rows * train_fraction)
    validation_cut = int(rows * (train_fraction + validation_fraction))
    first_endpoint = sequence_length - 1

    def endpoints(start: int, stop_exclusive: int) -> np.ndarray:
        first = max(first_endpoint, start)
        last_exclusive = stop_exclusive - max_horizon
        if last_exclusive <= first:
            return np.empty(0, dtype=np.int64)
        return np.arange(first, last_exclusive, dtype=np.int64)

    return (
        endpoints(0, train_cut),
        endpoints(train_cut, validation_cut),
        endpoints(validation_cut, rows),
        train_cut,
    )


def _safe_scale(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    scale = np.nanstd(values, axis=axis)
    return np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)


def _prune_training_features(
    frames: Sequence[pd.DataFrame],
    feature_columns: Sequence[str],
    training_config: TransformerTrainingConfig,
) -> tuple[str, ...]:
    """只以訓練切片移除常數與高度重複欄位，避免看見驗證／測試資料。"""
    if not feature_columns:
        return ()
    sample_blocks: list[np.ndarray] = []
    rows_per_source = max(
        100,
        training_config.correlation_sample_rows // max(len(frames), 1),
    )
    for frame in frames:
        train_cut = int(len(frame) * training_config.train_fraction)
        numeric = (
            frame.iloc[:train_cut]
            .loc[:, feature_columns]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
        )
        values = numeric.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64)
        if len(values) > rows_per_source:
            indices = np.linspace(
                0,
                len(values) - 1,
                rows_per_source,
                dtype=np.int64,
            )
            values = values[indices]
        sample_blocks.append(values)
    sample = np.concatenate(sample_blocks, axis=0)
    finite_counts = np.isfinite(sample).sum(axis=0)
    medians = np.nanmedian(sample, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    filled = np.where(np.isfinite(sample), sample, medians)
    scales = np.nanstd(filled, axis=0)

    kept_indices = [
        index
        for index in range(len(feature_columns))
        if finite_counts[index] > 0
        and (
            not training_config.drop_constant_features
            or (np.isfinite(scales[index]) and scales[index] > 1e-10)
        )
    ]
    threshold = training_config.max_feature_correlation
    if threshold is not None and len(kept_indices) > 1:
        normalized = filled[:, kept_indices]
        normalized = (normalized - np.mean(normalized, axis=0)) / np.maximum(
            np.std(normalized, axis=0), 1e-12
        )
        correlation = np.abs(
            np.nan_to_num(
                np.corrcoef(normalized, rowvar=False),
                nan=0.0,
                posinf=1.0,
                neginf=1.0,
            )
        )
        independent: list[int] = []
        for local_index, original_index in enumerate(kept_indices):
            if any(
                correlation[local_index, previous_local] >= threshold
                for previous_local in independent
            ):
                continue
            independent.append(local_index)
        kept_indices = [kept_indices[index] for index in independent]
    return tuple(feature_columns[index] for index in kept_indices)


def prepare_transformer_datasets(
    source_paths: Sequence[str | Path],
    model_config: TemporalTransformerConfig,
    training_config: TransformerTrainingConfig,
) -> PreparedTransformerData:
    """讀取多市場特徵資料，使用純時間切分建立訓練、驗證與測試集。"""
    paths = tuple(Path(path).resolve() for path in source_paths)
    if not paths:
        raise ValueError("至少選擇一份特徵 CSV")
    loaded = [_load_source(path, training_config.max_rows_per_source) for path in paths]
    frames = [item[0] for item in loaded]
    sources = tuple(item[1] for item in loaded)
    intervals = {source.interval for source in sources if source.interval}
    if len(intervals) > 1:
        raise ValueError("同一次訓練只能使用相同 K 線週期的資料")

    feature_columns = _select_feature_columns(frames, model_config.input_features)
    feature_columns = _prune_training_features(
        frames,
        feature_columns,
        training_config,
    )
    if not feature_columns:
        raise ValueError("Transformer 訓練區段沒有任何可用數值特徵")
    max_horizon = max(model_config.return_horizons)
    raw_series: list[dict[str, object]] = []
    training_feature_blocks: list[np.ndarray] = []
    training_return_blocks: list[np.ndarray] = []
    training_volatility_blocks: list[np.ndarray] = []
    training_regime_blocks: list[np.ndarray] = []
    training_edge_blocks: list[np.ndarray] = []
    training_excursion_blocks: list[np.ndarray] = []
    training_volatility_regime_blocks: list[np.ndarray] = []

    for frame in frames:
        numeric = frame.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
        feature_values = numeric.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64)
        feature_mask = np.isfinite(feature_values).astype(np.float32)
        targets = _v3_targets(frame, model_config, training_config)
        future_returns = targets["future_returns"]
        future_directions = targets["future_directions"]
        movement_directions = targets["movement_directions"]
        future_sides = targets["future_sides"]
        future_volatility = targets["future_volatility"]
        regime_source = targets["regime_source"]
        edge_returns = targets["edge_returns"]
        future_excursions = targets["future_excursions"]
        tradeability = targets["tradeability"]
        volatility_regime_source = targets["volatility_regime_source"]
        train_ep, validation_ep, test_ep, train_cut = _split_endpoints(
            len(frame),
            model_config.sequence_length,
            max_horizon,
            training_config.train_fraction,
            training_config.validation_fraction,
        )
        if not len(train_ep) or not len(validation_ep) or not len(test_ep):
            raise ValueError("資料不足以完成時間切分；請增加 K 線筆數、縮短序列或縮短預測週期")
        training_feature_blocks.append(feature_values[:train_cut])
        training_return_blocks.append(future_returns[train_ep])
        training_volatility_blocks.append(future_volatility[train_ep])
        training_regime_blocks.append(regime_source[train_ep])
        training_edge_blocks.append(edge_returns[train_ep])
        training_excursion_blocks.append(future_excursions[train_ep])
        training_volatility_regime_blocks.append(volatility_regime_source[train_ep])
        raw_series.append(
            {
                "features": feature_values,
                "feature_mask": feature_mask,
                "future_returns": future_returns,
                "future_volatility": future_volatility,
                "future_directions": future_directions,
                "movement_directions": movement_directions,
                "future_sides": future_sides,
                "regime_source": regime_source,
                "edge_returns": edge_returns,
                "future_excursions": future_excursions,
                "tradeability": tradeability,
                "volatility_regime_source": volatility_regime_source,
                "train_endpoints": train_ep,
                "validation_endpoints": validation_ep,
                "test_endpoints": test_ep,
            }
        )

    train_features = np.concatenate(training_feature_blocks)
    medians = np.nanmedian(train_features, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    filled_train = np.where(np.isfinite(train_features), train_features, medians)
    means = np.mean(filled_train, axis=0)
    scales = _safe_scale(filled_train, axis=0)

    train_returns = np.concatenate(training_return_blocks)
    return_means = np.nanmean(train_returns, axis=0)
    return_means = np.where(np.isfinite(return_means), return_means, 0.0)
    return_scales = _safe_scale(train_returns, axis=0)
    train_volatility = np.concatenate(training_volatility_blocks)
    volatility_mean = 0.0
    if not np.isfinite(volatility_mean):
        volatility_mean = 0.0
    volatility_scale = float(_safe_scale(train_volatility))
    regime_values = np.concatenate(training_regime_blocks)
    regime_values = regime_values[np.isfinite(regime_values)]
    if not len(regime_values):
        raise ValueError("訓練區段無法建立行情狀態標籤")
    quantiles = np.linspace(0, 1, model_config.regime_classes + 1)[1:-1]
    regime_thresholds = np.quantile(regime_values, quantiles)
    train_edges = np.concatenate(training_edge_blocks)
    edge_mean_values = np.nanmean(train_edges, axis=0)
    edge_mean_values = np.where(np.isfinite(edge_mean_values), edge_mean_values, 0.0)
    edge_scale_values = _safe_scale(train_edges, axis=0)
    train_excursions = np.concatenate(training_excursion_blocks)
    excursion_scale_values = np.nanquantile(train_excursions, 0.95, axis=0)
    excursion_scale_values = np.where(
        np.isfinite(excursion_scale_values) & (excursion_scale_values > 1e-8),
        excursion_scale_values,
        1.0,
    )
    volatility_regime_values = np.concatenate(training_volatility_regime_blocks)
    volatility_regime_values = volatility_regime_values[np.isfinite(volatility_regime_values)]
    if not len(volatility_regime_values):
        raise ValueError("訓練區段無法建立波動制度標籤")
    volatility_quantiles = np.linspace(
        0,
        1,
        model_config.volatility_regime_classes + 1,
    )[1:-1]
    volatility_regime_thresholds = tuple(
        float(value)
        for value in np.quantile(
            volatility_regime_values,
            volatility_quantiles,
        )
    )
    edge_means = tuple(float(value) for value in edge_mean_values.reshape(-1))
    edge_scales = tuple(float(value) for value in edge_scale_values.reshape(-1))
    excursion_scales = tuple(float(value) for value in excursion_scale_values.reshape(-1))

    feature_group_ids = transformer_feature_group_ids(feature_columns)

    scaler = TransformerScaler(
        feature_columns=feature_columns,
        medians=tuple(float(value) for value in medians),
        means=tuple(float(value) for value in means),
        scales=tuple(float(value) for value in scales),
        return_means=tuple(float(value) for value in return_means),
        return_scales=tuple(float(value) for value in return_scales),
        volatility_mean=volatility_mean,
        volatility_scale=volatility_scale,
        regime_thresholds=tuple(float(value) for value in regime_thresholds),
        edge_means=edge_means,
        edge_scales=edge_scales,
        excursion_scales=excursion_scales,
        volatility_regime_thresholds=volatility_regime_thresholds,
        feature_group_ids=feature_group_ids,
        feature_group_names=FEATURE_GROUP_NAMES,
        label_mode=(
            "hierarchical_movement_side_cost_aware_tradeability"
            if model_config.hierarchical_direction
            else "next_open_cost_aware_long_short"
        ),
    )

    prepared_series: list[_PreparedSeries] = []
    for item in raw_series:
        raw_features = np.asarray(item["features"], dtype=np.float64)
        filled = np.where(np.isfinite(raw_features), raw_features, medians)
        standardized = ((filled - means) / scales).astype(np.float32)
        prepared_series.append(
            _PreparedSeries(
                features=standardized,
                feature_mask=np.asarray(item["feature_mask"], dtype=np.float32),
                future_returns=np.asarray(item["future_returns"], dtype=np.float32),
                future_volatility=np.asarray(
                    item["future_volatility"],
                    dtype=np.float32,
                ),
                future_directions=np.asarray(
                    item["future_directions"],
                    dtype=np.int64,
                ),
                movement_directions=np.asarray(
                    item["movement_directions"],
                    dtype=np.int64,
                ),
                future_sides=np.asarray(
                    item["future_sides"],
                    dtype=np.int64,
                ),
                regime_source=np.asarray(item["regime_source"], dtype=np.float32),
                edge_returns=np.asarray(item["edge_returns"], dtype=np.float32),
                future_excursions=np.asarray(
                    item["future_excursions"],
                    dtype=np.float32,
                ),
                tradeability=np.asarray(item["tradeability"], dtype=np.float32),
                volatility_regime_source=np.asarray(
                    item["volatility_regime_source"],
                    dtype=np.float32,
                ),
                train_endpoints=np.asarray(item["train_endpoints"], dtype=np.int64),
                validation_endpoints=np.asarray(
                    item["validation_endpoints"],
                    dtype=np.int64,
                ),
                test_endpoints=np.asarray(item["test_endpoints"], dtype=np.int64),
            )
        )

    resolved_config = TemporalTransformerConfig(
        input_features=len(feature_columns),
        sequence_length=model_config.sequence_length,
        d_model=model_config.d_model,
        n_heads=model_config.n_heads,
        n_layers=model_config.n_layers,
        feedforward_dim=model_config.feedforward_dim,
        dropout=model_config.dropout,
        latent_dim=model_config.latent_dim,
        return_horizons=model_config.return_horizons,
        regime_classes=model_config.regime_classes,
        architecture_version=model_config.architecture_version,
        local_kernel_size=model_config.local_kernel_size,
        quantile_levels=model_config.quantile_levels,
        patch_size=model_config.patch_size,
        patch_stride=model_config.patch_stride,
        feature_group_ids=feature_group_ids,
        feature_group_names=FEATURE_GROUP_NAMES,
        volatility_regime_classes=model_config.volatility_regime_classes,
        hierarchical_direction=model_config.hierarchical_direction,
        horizon_adapter_dim=model_config.horizon_adapter_dim,
    )
    return PreparedTransformerData(
        train=MarketSequenceDataset(
            prepared_series,
            "train",
            model_config.sequence_length,
            scaler,
        ),
        validation=MarketSequenceDataset(
            prepared_series,
            "validation",
            model_config.sequence_length,
            scaler,
        ),
        test=MarketSequenceDataset(
            prepared_series,
            "test",
            model_config.sequence_length,
            scaler,
        ),
        scaler=scaler,
        sources=sources,
        resolved_model_config=resolved_config,
    )
