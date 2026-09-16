"""進階市場資料的因果式合併與市場廣度特徵。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from ai_quant_trading.data_collection.assets import market_file_symbol_slug
from ai_quant_trading.data_collection.csv_storage import save_canonical_context_csv


@dataclass(frozen=True, slots=True)
class AdvancedFeatureArtifact:
    """單一特徵檔完成進階資料融合後的摘要。"""

    path: Path
    rows: int
    added_columns: tuple[str, ...]
    available_rows: int


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def build_market_breadth(
    markets: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """由多標的已知價格建立不使用未來資料的橫斷面市場廣度。"""
    if len(markets) < 2:
        raise ValueError("市場廣度至少需要兩個標的")
    blocks: list[pd.DataFrame] = []
    for name, source in markets.items():
        if not {"timestamp", "close"}.issubset(source.columns):
            raise ValueError(f"{name} 缺少 timestamp 或 close")
        block = source[["timestamp", "close"]].copy()
        block["timestamp"] = pd.to_datetime(
            block["timestamp"], utc=True, errors="coerce", format="mixed"
        )
        block["close"] = pd.to_numeric(block["close"], errors="coerce")
        block = (
            block.dropna()
            .sort_values("timestamp")
            .drop_duplicates("timestamp", keep="last")
        )
        block["symbol"] = name
        block["return_1"] = block["close"].pct_change(fill_method=None)
        for window in (20, 50, 200):
            moving_average = block["close"].rolling(window, min_periods=window).mean()
            block[f"above_sma_{window}"] = (
                block["close"] > moving_average
            ).where(moving_average.notna())
        high_20 = block["close"].rolling(20, min_periods=20).max()
        low_20 = block["close"].rolling(20, min_periods=20).min()
        block["new_high_20"] = (block["close"] >= high_20).where(high_20.notna())
        block["new_low_20"] = (block["close"] <= low_20).where(low_20.notna())
        blocks.append(block)
    combined = pd.concat(blocks, ignore_index=True)
    rows: list[dict[str, float | int | pd.Timestamp]] = []
    advance_decline_line = 0.0
    for timestamp, group in combined.groupby("timestamp", sort=True):
        returns = _numeric(group, "return_1")
        valid_returns = returns.dropna()
        advances = int((returns > 0).sum())
        declines = int((returns < 0).sum())
        unchanged = int((returns == 0).sum())
        advance_decline_line += advances - declines
        denominator = advances + declines
        row: dict[str, float | int | pd.Timestamp] = {
            "timestamp": timestamp,
            "breadth_market_count": int(group["symbol"].nunique()),
            "breadth_advance_ratio": advances / denominator if denominator else 0.5,
            "breadth_decline_ratio": declines / denominator if denominator else 0.5,
            "breadth_unchanged_count": unchanged,
            "breadth_advance_decline_line": advance_decline_line,
            "breadth_median_return": (
                float(valid_returns.median()) if len(valid_returns) else np.nan
            ),
            "breadth_return_dispersion": (
                float(valid_returns.std(ddof=0)) if len(valid_returns) else np.nan
            ),
        }
        for window in (20, 50, 200):
            values = group[f"above_sma_{window}"].dropna()
            row[f"breadth_pct_above_sma_{window}"] = (
                float(values.astype(float).mean()) if len(values) else np.nan
            )
        high_values = group["new_high_20"].dropna()
        low_values = group["new_low_20"].dropna()
        row["breadth_new_high_ratio_20"] = (
            float(high_values.astype(float).mean()) if len(high_values) else np.nan
        )
        row["breadth_new_low_ratio_20"] = (
            float(low_values.astype(float).mean()) if len(low_values) else np.nan
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    result["breadth_available"] = (
        result["breadth_market_count"].ge(2).astype("int8")
    )
    result["timestamp"] = result["timestamp"].map(lambda value: value.isoformat())
    return result.reset_index(drop=True)


def market_breadth_path(processed_dir: str | Path) -> Path:
    """市場廣度使用固定檔名，更新時直接覆寫。"""
    return Path(processed_dir) / "advanced" / "market_breadth_latest.csv"


def save_market_breadth(
    frame: pd.DataFrame,
    processed_dir: str | Path,
) -> Path:
    return save_canonical_context_csv(
        frame,
        market_breadth_path(processed_dir),
        key_columns=("timestamp",),
        merge_existing=False,
    )


def _merge_asof_context(
    source: pd.DataFrame,
    context: pd.DataFrame,
    *,
    columns: Iterable[str],
    age_column: str,
    max_age: pd.Timedelta | None = None,
) -> pd.DataFrame:
    """只向後尋找當時最近已公開資料，並保存資料年齡。"""
    result = source.copy()
    if context.empty or "timestamp" not in context:
        for column in columns:
            if column not in result:
                result[column] = np.nan
        result[age_column] = np.nan
        return result
    result["timestamp"] = pd.to_datetime(
        result["timestamp"], utc=True, errors="coerce", format="mixed"
    )
    right = context.copy()
    right["context_available_at"] = pd.to_datetime(
        right["timestamp"], utc=True, errors="coerce", format="mixed"
    )
    selected = [column for column in columns if column in right]
    right = (
        right[["context_available_at", *selected]]
        .dropna(subset=["context_available_at"])
        .sort_values("context_available_at")
        .drop_duplicates("context_available_at", keep="last")
    )
    rename_overlaps = {
        column: f"{column}__advanced"
        for column in selected
        if column in result.columns
    }
    right = right.rename(columns=rename_overlaps)
    merged = pd.merge_asof(
        result.sort_values("timestamp"),
        right,
        left_on="timestamp",
        right_on="context_available_at",
        direction="backward",
        tolerance=max_age,
    )
    for column, temporary in rename_overlaps.items():
        merged[column] = _numeric(merged, temporary).combine_first(
            _numeric(merged, column)
        )
        merged = merged.drop(columns=temporary)
    merged[age_column] = (
        merged["timestamp"] - merged["context_available_at"]
    ).dt.total_seconds() / 86_400
    return merged.drop(columns="context_available_at").reset_index(drop=True)


def attach_macro_context(
    frame: pd.DataFrame,
    macro_frames: Iterable[pd.DataFrame],
) -> pd.DataFrame:
    """把每個 FRED 序列依其保守公開時間向後合併。"""
    result = frame.copy()
    available_columns: list[str] = []
    for context in macro_frames:
        if context.empty or "series_id" not in context:
            continue
        series_values = context["series_id"].dropna().astype(str)
        if series_values.empty:
            continue
        series_id = series_values.iloc[0].lower()
        prefix = f"macro_{series_id}"
        renamed = context.rename(
            columns={
                "value": f"{prefix}_value",
                "change": f"{prefix}_change",
                "pct_change": f"{prefix}_pct_change",
                "zscore_60": f"{prefix}_zscore_60",
            }
        )
        columns = [
            f"{prefix}_value",
            f"{prefix}_change",
            f"{prefix}_pct_change",
            f"{prefix}_zscore_60",
        ]
        result = _merge_asof_context(
            result,
            renamed,
            columns=columns,
            age_column=f"{prefix}_age_days",
        )
        flag = f"{prefix}_available"
        result[flag] = _numeric(result, f"{prefix}_value").notna().astype("int8")
        available_columns.append(flag)
    if {"macro_dgs10_value", "macro_dgs2_value"}.issubset(result.columns):
        result["macro_yield_curve_10y_2y"] = (
            _numeric(result, "macro_dgs10_value")
            - _numeric(result, "macro_dgs2_value")
        )
    result["macro_available"] = (
        result[available_columns].max(axis=1).astype("int8")
        if available_columns
        else 0
    )
    return result


def attach_advanced_context(
    frame: pd.DataFrame,
    *,
    macro_frames: Iterable[pd.DataFrame] = (),
    fundamentals: pd.DataFrame | None = None,
    breadth: pd.DataFrame | None = None,
    derivatives: pd.DataFrame | None = None,
    order_book: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """將進階資料依可用時間融合到特徵列，絕不使用未來一筆。"""
    result = attach_macro_context(frame, macro_frames)
    if fundamentals is not None:
        fundamental_columns = [
            *[
                column
                for column in (
                    "revenue",
                    "net_income",
                    "operating_income",
                    "assets",
                    "liabilities",
                    "equity",
                    "diluted_eps",
                    "operating_cash_flow",
                    "capital_expenditure",
                    "fundamental_revenue_growth",
                    "fundamental_profit_margin",
                    "fundamental_operating_margin",
                    "fundamental_debt_to_equity",
                    "fundamental_ocf_margin",
                )
                if column in fundamentals
            ],
            "fundamentals_available",
        ]
        result = _merge_asof_context(
            result,
            fundamentals,
            columns=fundamental_columns,
            age_column="fundamentals_age_days",
        )
        result["fundamentals_available"] = _numeric(
            result, "fundamentals_available"
        ).fillna(0).clip(0, 1)
    if breadth is not None:
        breadth_columns = [
            column
            for column in breadth.columns
            if column.startswith("breadth_")
        ]
        result = _merge_asof_context(
            result,
            breadth,
            columns=breadth_columns,
            age_column="breadth_age_days",
            max_age=pd.Timedelta(days=7),
        )
    if derivatives is not None:
        derivative_columns = [
            column
            for column in (
                "funding_rate",
                "mark_price",
                "open_interest",
                "open_interest_value",
                "open_interest_change",
                "spread_bps",
                "global_long_short_ratio",
                "global_long_account",
                "global_short_account",
                "taker_buy_sell_ratio",
                "taker_buy_volume",
                "taker_sell_volume",
                "basis_rate",
                "basis_price",
                "index_price",
                "derivatives_context_available",
            )
            if column in derivatives
        ]
        result = _merge_asof_context(
            result,
            derivatives,
            columns=derivative_columns,
            age_column="derivatives_age_days",
            max_age=pd.Timedelta(days=3),
        )
        result["derivatives_context_available"] = result[
            derivative_columns
        ].notna().any(axis=1).astype("int8")
    if order_book is not None:
        renamed = order_book.rename(
            columns={
                column: f"orderbook_{column}"
                for column in order_book.columns
                if column not in {"timestamp", "symbol", "exchange", "collected_at"}
            }
        )
        orderbook_columns = [
            column for column in renamed if column.startswith("orderbook_")
        ]
        result = _merge_asof_context(
            result,
            renamed,
            columns=orderbook_columns,
            age_column="orderbook_age_days",
            max_age=pd.Timedelta(days=1),
        )
        result["orderbook_available"] = result[orderbook_columns].notna().any(
            axis=1
        ).astype("int8")
    result["timestamp"] = pd.to_datetime(
        result["timestamp"], utc=True, errors="coerce", format="mixed"
    ).map(lambda value: value.isoformat())
    return result


def _read_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
        return None


def enrich_feature_csv(
    feature_path: str | Path,
    *,
    raw_dir: str | Path,
    processed_dir: str | Path,
) -> AdvancedFeatureArtifact:
    """自動尋找相符的進階資料，直接更新原本的 Step 3 特徵檔。"""
    path = Path(feature_path)
    source = pd.read_csv(path)
    if source.empty or "timestamp" not in source:
        raise ValueError(f"{path.name} 沒有可融合的特徵資料")
    first = source.iloc[0]
    symbol = str(first.get("symbol", "")).upper()
    exchange = str(first.get("exchange", "")).lower()
    interval = str(first.get("interval", ""))
    macro_frames = [
        frame
        for candidate in sorted((Path(raw_dir) / "macro" / "fred").glob("*.csv"))
        if (frame := _read_if_exists(candidate)) is not None
    ]
    fundamental_path = (
        Path(raw_dir)
        / "us_equity"
        / "sec"
        / "fundamentals"
        / f"fundamentals_SEC_{symbol.replace('.', '-')}_latest.csv"
    )
    breadth_path = market_breadth_path(processed_dir)
    derivatives_path: Path | None = None
    orderbook_path: Path | None = None
    if exchange in {"binance", "binance_futures", "bybit"}:
        slug = market_file_symbol_slug(symbol, "crypto")
        derivatives_path = (
            Path(raw_dir)
            / "crypto"
            / "binance"
            / "derivatives_context"
            / f"context_crypto_binance_{slug}_{interval}_latest.csv"
        )
        orderbook_path = (
            Path(raw_dir)
            / "crypto"
            / "binance"
            / "order_book"
            / f"orderbook_crypto_binance_{slug}_latest.csv"
        )
    result = attach_advanced_context(
        source,
        macro_frames=macro_frames,
        fundamentals=_read_if_exists(fundamental_path),
        breadth=_read_if_exists(breadth_path),
        derivatives=_read_if_exists(derivatives_path) if derivatives_path else None,
        order_book=_read_if_exists(orderbook_path) if orderbook_path else None,
    )
    added = tuple(column for column in result.columns if column not in source.columns)
    availability = [
        column
        for column in result.columns
        if column.endswith("_available") or column.endswith("_context_available")
    ]
    available_rows = (
        int(result[availability].fillna(0).max(axis=1).gt(0).sum())
        if availability
        else 0
    )
    save_canonical_context_csv(
        result,
        path,
        key_columns=("timestamp",),
        merge_existing=False,
    )
    return AdvancedFeatureArtifact(path.resolve(), len(result), added, available_rows)
