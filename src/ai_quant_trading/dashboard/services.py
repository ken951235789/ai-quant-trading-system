"""Dashboard 使用的資料載入與整理服務。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ai_quant_trading.features.builder import infer_annualization_periods
from ai_quant_trading.features.indicators import FEATURE_COLUMNS, build_feature_dataset


@dataclass(frozen=True, slots=True)
class MarketSummary:
    """圖表上方使用的市場資料摘要。"""

    bars: int
    latest_close: float
    period_return: float
    latest_volume: float
    start_time: pd.Timestamp
    end_time: pd.Timestamp


def parse_symbols(value: str) -> list[str]:
    """把逗號或換行分隔的標的文字轉成去重後清單。"""
    normalized = value.replace("\n", ",")
    symbols: list[str] = []
    seen: set[str] = set()
    for item in normalized.split(","):
        symbol = item.strip().upper()
        if symbol and symbol not in seen:
            symbols.append(symbol)
            seen.add(symbol)
    return symbols


def list_ohlcv_files(raw_dir: str | Path) -> list[Path]:
    """列出所有資料源的 OHLCV CSV，最近修改的排在前面。"""
    paths = Path(raw_dir).rglob("ohlcv/*.csv")
    return sorted(paths, key=lambda path: path.stat().st_mtime, reverse=True)


def list_processed_files(processed_dir: str | Path) -> list[Path]:
    """列出 processed 內可供圖表或模型使用的 CSV。"""
    paths = Path(processed_dir).glob("*.csv")
    return sorted(paths, key=lambda path: path.stat().st_mtime, reverse=True)


def load_market_frame(path: str | Path) -> pd.DataFrame:
    """載入市場 CSV，並將時間與 OHLCV 轉為可計算型別。"""
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"找不到 CSV：{source_path}")

    frame = pd.read_csv(source_path)
    if "timestamp" not in frame.columns:
        raise ValueError("CSV 缺少 timestamp 欄位")

    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
    if result["timestamp"].isna().any():
        raise ValueError("CSV timestamp 含有無法解析的值")

    for column in ["open", "high", "low", "close", "volume"]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.sort_values("timestamp").reset_index(drop=True)


def ensure_chart_features(frame: pd.DataFrame) -> pd.DataFrame:
    """原始 OHLCV 尚無技術指標時，在記憶體中補齊圖表需要的特徵。"""
    if set(FEATURE_COLUMNS).issubset(frame.columns):
        return frame.copy()
    annualization_periods = infer_annualization_periods(frame)
    return build_feature_dataset(
        frame,
        target_horizon=1,
        annualization_periods=annualization_periods,
        drop_na=False,
    )


def summarize_market(frame: pd.DataFrame) -> MarketSummary:
    """計算目前畫面資料範圍的收盤價、報酬與成交量摘要。"""
    required = {"timestamp", "close", "volume"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"市場摘要缺少欄位：{missing}")
    if frame.empty:
        raise ValueError("無法摘要空資料")

    first_close = float(frame.iloc[0]["close"])
    latest_close = float(frame.iloc[-1]["close"])
    period_return = latest_close / first_close - 1 if first_close != 0 else 0.0
    return MarketSummary(
        bars=len(frame),
        latest_close=latest_close,
        period_return=period_return,
        latest_volume=float(frame.iloc[-1]["volume"]),
        start_time=pd.Timestamp(frame.iloc[0]["timestamp"]),
        end_time=pd.Timestamp(frame.iloc[-1]["timestamp"]),
    )


def relative_file_label(path: Path, project_root: str | Path) -> str:
    """產生介面選單使用的專案相對路徑。"""
    try:
        return str(path.resolve().relative_to(Path(project_root).resolve()))
    except ValueError:
        return str(path)
