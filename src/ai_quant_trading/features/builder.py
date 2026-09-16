"""從原始 CSV 建立並儲存特徵資料集。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pandas as pd

from ai_quant_trading.data_collection.assets import (
    infer_asset_class,
    market_file_symbol_slug,
)
from ai_quant_trading.data_collection.validators import validate_ohlcv_dataframe
from ai_quant_trading.features.indicators import FEATURE_COLUMNS, build_feature_dataset
from ai_quant_trading.market_clock import completed_bars_only


@dataclass(frozen=True, slots=True)
class FeatureBuildArtifact:
    """單一 Raw CSV 完成特徵工程後的批次摘要。"""

    input_path: Path
    output_path: Path
    rows: int
    columns: int


def infer_annualization_periods(frame: pd.DataFrame) -> int:
    """依資產類別與 K 線週期推定一年包含的交易期數。"""
    if frame.empty:
        return 252
    exchange = _first_value(frame, "exchange", "").lower()
    interval = _first_value(frame, "interval", "1d").lower()
    is_crypto = exchange in {"binance", "binance_futures", "bybit"}

    crypto_periods = {
        "1s": 365 * 24 * 60 * 60,
        "1m": 365 * 24 * 60,
        "3m": 365 * 24 * 20,
        "5m": 365 * 24 * 12,
        "15m": 365 * 24 * 4,
        "30m": 365 * 24 * 2,
        "1h": 365 * 24,
        "2h": 365 * 12,
        "4h": 365 * 6,
        "6h": 365 * 4,
        "8h": 365 * 3,
        "12h": 365 * 2,
        "1d": 365,
        "3d": 122,
        "1w": 52,
    }
    equity_periods = {
        "1m": 252 * 390,
        "2m": 252 * 195,
        "5m": 252 * 78,
        "15m": 252 * 26,
        "30m": 252 * 13,
        "60m": 252 * 7,
        "1h": 252 * 7,
        "90m": 252 * 4,
        "1d": 252,
        "5d": 50,
        "1wk": 52,
        "1mo": 12,
        "3mo": 4,
    }
    periods = crypto_periods if is_crypto else equity_periods
    return periods.get(interval, 365 if is_crypto else 252)


def _first_value(frame: pd.DataFrame, column: str, default: str) -> str:
    if column not in frame.columns or frame.empty or pd.isna(frame.iloc[0][column]):
        return default
    return str(frame.iloc[0][column])


def save_feature_csv(
    frame: pd.DataFrame,
    source_frame: pd.DataFrame,
    output_dir: str | Path,
    target_horizon: int,
) -> Path:
    """同一市場與預測期只保留一份最新特徵資料集。"""
    if frame.empty:
        raise ValueError(
            "資料不足，無法建立模型資料集；MA200 暖機與 target 至少需要 "
            f"{200 + target_horizon} 根已收盤 K 線。"
        )
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    exchange = _first_value(source_frame, "exchange", "unknown").lower()
    raw_symbol = _first_value(source_frame, "symbol", "unknown")
    asset_class = infer_asset_class(exchange, raw_symbol)
    symbol = market_file_symbol_slug(raw_symbol, asset_class)
    interval = _first_value(source_frame, "interval", "unknown")
    filename = f"features_{asset_class}_{exchange}_{symbol}_{interval}_h{target_horizon}.csv"
    path = target_dir / filename
    legacy_pattern = (
        f"features_{asset_class}_{exchange}_{symbol}_{interval}_*_h{target_horizon}.csv"
    )
    old_files = list(target_dir.glob(legacy_pattern))
    temporary = path.with_suffix(f"{path.suffix}.{uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    for old_path in old_files:
        if old_path.resolve() != path.resolve():
            old_path.unlink(missing_ok=True)
    return path


def build_features_from_csv(
    input_path: str | Path,
    output_dir: str | Path = "data/processed",
    target_horizon: int = 1,
    target_threshold: float = 0.0,
    annualization_periods: int | None = None,
    drop_na: bool = True,
) -> tuple[pd.DataFrame, Path]:
    """讀取原始 OHLCV CSV、驗證、計算特徵並儲存結果。"""
    source_path = Path(input_path)
    if not source_path.exists():
        raise FileNotFoundError(f"找不到輸入 CSV：{source_path}")

    source_frame = pd.read_csv(source_path)
    validate_ohlcv_dataframe(source_frame)
    source_frame = completed_bars_only(source_frame).reset_index(drop=True)
    if source_frame.empty:
        raise ValueError("輸入資料沒有任何已收盤 K 線")
    periods = (
        annualization_periods
        if annualization_periods is not None
        else infer_annualization_periods(source_frame)
    )
    feature_frame = build_feature_dataset(
        source_frame,
        target_horizon=target_horizon,
        target_threshold=target_threshold,
        annualization_periods=periods,
        drop_na=False,
    )
    if drop_na:
        # 最新幾根 K 線還沒有未來 target，但必須保留給 Transformer 正式推論。
        # 訓練器會自行依時間切分並忽略尾端沒有標籤的樣本。
        feature_frame = feature_frame.dropna(subset=FEATURE_COLUMNS).reset_index(
            drop=True
        )
    if drop_na and feature_frame.empty:
        raise ValueError(
            "資料不足，無法建立模型資料集；請讓系統自動下載至少 "
            f"{200 + target_horizon} 根已收盤 K 線。"
        )
    output_path = save_feature_csv(
        feature_frame,
        source_frame,
        output_dir,
        target_horizon,
    )
    return feature_frame, output_path


def build_features_from_csv_batch(
    input_paths: list[str | Path] | tuple[str | Path, ...],
    output_dir: str | Path = "data/processed",
    target_horizon: int = 1,
    target_threshold: float = 0.0,
    annualization_periods: int | None = None,
    drop_na: bool = True,
) -> tuple[FeatureBuildArtifact, ...]:
    """依序處理多份 OHLCV；每個標的仍保存為獨立且可覆蓋的 CSV。"""
    if not input_paths:
        raise ValueError("批次特徵工程至少需要一份輸入 CSV")
    artifacts: list[FeatureBuildArtifact] = []
    for input_path in input_paths:
        frame, output_path = build_features_from_csv(
            input_path,
            output_dir=output_dir,
            target_horizon=target_horizon,
            target_threshold=target_threshold,
            annualization_periods=annualization_periods,
            drop_na=drop_na,
        )
        artifacts.append(
            FeatureBuildArtifact(
                input_path=Path(input_path),
                output_path=output_path,
                rows=len(frame),
                columns=len(frame.columns),
            )
        )
    return tuple(artifacts)
