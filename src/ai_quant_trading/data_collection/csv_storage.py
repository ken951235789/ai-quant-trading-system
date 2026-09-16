"""市場資料 CSV 儲存工具。"""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import time
from uuid import uuid4

import pandas as pd

from ai_quant_trading.data_collection.assets import (
    asset_class_folder,
    infer_asset_class,
    market_file_symbol_slug,
)


def symbol_to_file_slug(symbol: str) -> str:
    """把 BTC/USDT 轉成適合檔名使用的 BTCUSDT。"""
    return symbol.replace("/", "").replace("-", "").upper()


def _canonical_ohlcv_path(
    raw_dir: str | Path,
    exchange: str,
    symbol: str,
    interval: str,
) -> Path:
    """同一市場只保留一份可持續更新的 OHLCV 檔案。"""
    asset_class = infer_asset_class(exchange, symbol)
    target_dir = Path(raw_dir) / asset_class_folder(asset_class) / exchange / "ohlcv"
    symbol_slug = market_file_symbol_slug(symbol, asset_class)
    return target_dir / f"ohlcv_{asset_class}_{exchange}_{symbol_slug}_{interval}_latest.csv"


def canonical_ohlcv_path(
    raw_dir: str | Path,
    exchange: str,
    symbol: str,
    interval: str,
) -> Path:
    """取得固定 OHLCV 路徑，供即時串流與 REST 補漏共用。"""
    return _canonical_ohlcv_path(raw_dir, exchange, symbol, interval)


def _canonical_market_data_path(
    raw_dir: str | Path,
    exchange: str,
    symbol: str,
) -> Path:
    """24 小時市場快照使用固定檔名，每次以新快照取代。"""
    asset_class = infer_asset_class(exchange, symbol)
    target_dir = Path(raw_dir) / asset_class_folder(asset_class) / exchange / "market_data"
    symbol_slug = market_file_symbol_slug(symbol, asset_class)
    return target_dir / f"market_{asset_class}_{exchange}_{symbol_slug}_latest.csv"


def _canonical_derivatives_context_path(
    raw_dir: str | Path,
    symbol: str,
    interval: str,
    exchange: str = "binance_futures",
) -> Path:
    """同一永續合約週期只保留一份可更新的市場脈絡檔。"""
    symbol_slug = market_file_symbol_slug(symbol, "crypto")
    target_dir = Path(raw_dir) / "crypto" / exchange / "derivatives_context"
    return target_dir / f"context_crypto_{exchange}_{symbol_slug}_{interval}_latest.csv"


def canonical_order_book_path(
    raw_dir: str | Path,
    symbol: str,
    exchange: str = "binance_futures",
) -> Path:
    """回傳同一加密貨幣唯一的深度快照歷史檔。"""
    symbol_slug = market_file_symbol_slug(symbol, "crypto")
    target_dir = Path(raw_dir) / "crypto" / exchange / "order_book"
    return target_dir / f"orderbook_crypto_{exchange}_{symbol_slug}_latest.csv"


@contextmanager
def _exclusive_file_lock(target: Path, timeout_seconds: float = 30.0):
    """以小型 lock 檔保護背景模擬與手動下載的同時更新。"""
    lock_path = target.with_suffix(target.suffix + ".lock")
    deadline = time.monotonic() + timeout_seconds
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(descriptor, str(os.getpid()).encode("ascii"))
        except FileExistsError:
            try:
                stale = time.time() - lock_path.stat().st_mtime > 120
            except FileNotFoundError:
                continue
            if stale:
                lock_path.unlink(missing_ok=True)
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"等待市場資料檔案鎖逾時：{target.name}")
            time.sleep(0.05)
    try:
        yield
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    """先完整寫入同目錄暫存檔，再取代正式檔案。"""
    temporary = path.with_suffix(f"{path.suffix}.{uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _matching_ohlcv_files(path: Path) -> list[Path]:
    """找出固定檔與更新前留下的日期範圍檔。"""
    prefix = path.name.removesuffix("latest.csv")
    return sorted(candidate for candidate in path.parent.glob(f"{prefix}*.csv") if candidate.is_file())


def _merge_ohlcv_history(frame: pd.DataFrame, existing_files: list[Path]) -> pd.DataFrame:
    """合併歷史與新下載資料；相同 timestamp 以新資料為準。"""
    frames: list[pd.DataFrame] = []
    for path in existing_files:
        try:
            existing = pd.read_csv(path)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            continue
        if "timestamp" in existing.columns and not existing.empty:
            frames.append(existing)
    frames.append(frame.copy())
    merged = pd.concat(frames, ignore_index=True, sort=False)
    # 舊檔可能是 +00:00，新 API 資料可能是毫秒 Z；Pandas 2 需明確允許混合 ISO 格式。
    merged["timestamp"] = pd.to_datetime(
        merged["timestamp"],
        utc=True,
        errors="coerce",
        format="mixed",
    )
    merged = merged.dropna(subset=["timestamp"])
    if "close_time_ms" in merged.columns:
        close_times = pd.to_numeric(merged["close_time_ms"], errors="coerce")
        # 固定歷史檔只保存已收盤 K 線；即時未收盤狀態由 WebSocket 記憶體快照提供。
        merged = merged[close_times.isna() | (close_times <= int(time.time() * 1000))]
    merged = merged.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    merged["timestamp"] = merged["timestamp"].map(lambda value: value.isoformat())
    preferred = list(frame.columns)
    remaining = [column for column in merged.columns if column not in preferred]
    return merged[preferred + remaining].reset_index(drop=True)


def ohlcv_output_path(
    frame: pd.DataFrame,
    raw_dir: str | Path,
    exchange: str,
    symbol: str,
    interval: str,
) -> Path:
    """計算同一市場唯一的 OHLCV 固定路徑，不寫入檔案。"""
    if frame.empty:
        raise ValueError("無法替空的 OHLCV 資料建立路徑")
    return _canonical_ohlcv_path(raw_dir, exchange, symbol, interval)


def market_data_output_path(
    frame: pd.DataFrame,
    raw_dir: str | Path,
    exchange: str,
    symbol: str,
) -> Path:
    """計算同一市場唯一的快照固定路徑，不寫入檔案。"""
    if frame.empty:
        raise ValueError("無法替空的 Market Data 建立路徑")
    return _canonical_market_data_path(raw_dir, exchange, symbol)


def save_ohlcv_csv(
    frame: pd.DataFrame,
    raw_dir: str | Path,
    exchange: str,
    symbol: str,
    interval: str,
    *,
    max_rows: int | None = None,
) -> Path:
    """合併新舊 OHLCV，並覆寫同一市場唯一的固定 CSV。"""
    if frame.empty:
        raise ValueError("不儲存空的 OHLCV CSV")
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows 必須大於 0")

    path = _canonical_ohlcv_path(raw_dir, exchange, symbol, interval)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_file_lock(path):
        existing_files = _matching_ohlcv_files(path)
        merged = _merge_ohlcv_history(frame, existing_files)
        if max_rows is not None:
            merged = merged.tail(max_rows).reset_index(drop=True)
        _atomic_write_csv(merged, path)
        for old_path in existing_files:
            if old_path.resolve() != path.resolve():
                old_path.unlink(missing_ok=True)
    return path


def save_market_data_csv(
    frame: pd.DataFrame,
    raw_dir: str | Path,
    exchange: str,
    symbol: str,
) -> Path:
    """以固定檔名覆寫最新 24 小時市場統計快照。"""
    if frame.empty:
        raise ValueError("不儲存空的 Market Data CSV")

    path = _canonical_market_data_path(raw_dir, exchange, symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_file_lock(path):
        prefix = path.name.removesuffix("latest.csv")
        old_files = list(path.parent.glob(f"{prefix}*.csv"))
        _atomic_write_csv(frame, path)
        for old_path in old_files:
            if old_path.resolve() != path.resolve():
                old_path.unlink(missing_ok=True)
    return path


def save_derivatives_context_csv(
    frame: pd.DataFrame,
    raw_dir: str | Path,
    symbol: str,
    interval: str,
    exchange: str = "binance_futures",
) -> Path:
    """合併 funding／OI 歷史並覆寫固定檔，避免每次更新產生新檔。"""
    if frame.empty:
        raise ValueError("不儲存空的衍生品市場脈絡 CSV")
    path = _canonical_derivatives_context_path(raw_dir, symbol, interval, exchange)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_file_lock(path):
        prefix = path.name.removesuffix("latest.csv")
        existing_files = sorted(path.parent.glob(f"{prefix}*.csv"))
        merged = _merge_ohlcv_history(frame, existing_files)
        _atomic_write_csv(merged, path)
        for old_path in existing_files:
            if old_path.resolve() != path.resolve():
                old_path.unlink(missing_ok=True)
    return path


def save_canonical_context_csv(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    key_columns: tuple[str, ...] = ("timestamp",),
    merge_existing: bool = True,
) -> Path:
    """以原子覆寫保存進階資料，可選擇合併舊資料並依鍵值去重。"""
    if frame.empty:
        raise ValueError("不儲存空的進階資料 CSV")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    missing = [column for column in key_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"進階資料缺少去重鍵：{missing}")
    with _exclusive_file_lock(target):
        frames: list[pd.DataFrame] = []
        if merge_existing and target.exists():
            try:
                frames.append(pd.read_csv(target))
            except (OSError, pd.errors.ParserError, UnicodeDecodeError):
                pass
        frames.append(frame.copy())
        merged = pd.concat(frames, ignore_index=True, sort=False)
        if "timestamp" in merged:
            merged["timestamp"] = pd.to_datetime(
                merged["timestamp"], utc=True, errors="coerce", format="mixed"
            )
            merged = merged.dropna(subset=["timestamp"])
        merged = merged.drop_duplicates(list(key_columns), keep="last")
        if "timestamp" in merged:
            merged = merged.sort_values("timestamp")
            merged["timestamp"] = merged["timestamp"].map(lambda value: value.isoformat())
        _atomic_write_csv(merged.reset_index(drop=True), target)
    return target


def save_order_book_context_csv(
    frame: pd.DataFrame,
    raw_dir: str | Path,
    symbol: str,
    exchange: str = "binance_futures",
) -> Path:
    """累積深度快照並覆寫固定檔名，供短線模型向後合併。"""
    return save_canonical_context_csv(
        frame,
        canonical_order_book_path(raw_dir, symbol, exchange),
        key_columns=("timestamp",),
        merge_existing=True,
    )
