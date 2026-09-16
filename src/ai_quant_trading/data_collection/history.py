"""以官方月檔與 REST 補漏準備大型 BTC 永續合約歷史，不一次載入五年分鐘線。"""

from __future__ import annotations

import hashlib
from io import BytesIO
import os
from pathlib import Path
import re
import sqlite3
import time
from uuid import uuid4
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests

from ai_quant_trading.data_collection.binance import INTERVAL_TO_MS, BinanceApiError
from ai_quant_trading.data_collection.binance_futures import BinanceFuturesPublicClient
from ai_quant_trading.data_collection.csv_storage import (
    _exclusive_file_lock,
    canonical_ohlcv_path,
)
from ai_quant_trading.data_collection.http_client import create_secure_session
from ai_quant_trading.data_collection.schemas import OHLCV_COLUMNS
from ai_quant_trading.data_collection.validators import validate_ohlcv_dataframe
from ai_quant_trading.market_clock import BTC_SUPPORTED_MULTITIMEFRAME_INTERVALS
from ai_quant_trading.persistence import write_json_atomic


ARCHIVE_ROOT = "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT"
ARCHIVE_COLUMNS = (
    "open_time_ms", "open", "high", "low", "close", "volume", "close_time_ms",
    "quote_asset_volume", "number_of_trades", "taker_buy_base_volume",
    "taker_buy_quote_volume", "ignore",
)


def decode_monthly_klines(content: bytes, interval: str) -> pd.DataFrame:
    """只在記憶體解析 ZIP 內的 CSV，不解壓到磁碟，也不接受任意路徑。"""
    with ZipFile(BytesIO(content)) as archive:
        members = archive.infolist()
        if len(members) != 1 or not members[0].filename.endswith(".csv"):
            raise ValueError("官方 K 線 ZIP 必須只有一個 CSV")
        if members[0].file_size > 128 * 1024**2:
            raise ValueError("官方月檔解壓大小超出限制")
        with archive.open(members[0]) as stream:
            first_line = stream.readline().decode("utf-8-sig").strip()
        has_header = not first_line.split(",", 1)[0].isdigit()
        with archive.open(members[0]) as stream:
            frame = pd.read_csv(
                stream, names=list(ARCHIVE_COLUMNS), header=0 if has_header else None,
                encoding="utf-8-sig",
            )
    frame = frame.drop(columns="ignore").apply(pd.to_numeric, errors="raise")
    if frame.empty:
        raise ValueError("官方 K 線月檔為空")
    # USD-M 月檔的時間單位是毫秒；拒絕把現貨微秒檔混入合約資料。
    if frame["open_time_ms"].max() >= 100_000_000_000_000:
        raise ValueError("K 線月檔不是預期的毫秒時間戳")
    frame["timestamp"] = pd.to_datetime(frame["open_time_ms"], unit="ms", utc=True).map(
        lambda value: value.isoformat()
    )
    frame["symbol"] = "BTC/USDT"
    frame["exchange"] = "binance_futures"
    frame["interval"] = interval
    frame["collected_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    validate_history_chunk(frame, interval)
    return frame[list(OHLCV_COLUMNS)]


def validate_history_chunk(frame: pd.DataFrame, interval: str) -> None:
    """除了 OHLCV，也檢查有限值、交易量、週期邊界與開收盤時間一致性。"""
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce", format="mixed")
    validate_ohlcv_dataframe(frame.assign(timestamp=timestamps))
    numeric_columns = [
        "open", "high", "low", "close", "volume", "quote_asset_volume",
        "number_of_trades", "taker_buy_base_volume", "taker_buy_quote_volume",
        "open_time_ms", "close_time_ms",
    ]
    values = frame[numeric_columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("歷史 K 線含有缺值或無限大")
    if (values[numeric_columns[4:9]] < 0).any().any():
        raise ValueError("歷史 K 線含有負交易量或負交易次數")
    duration = INTERVAL_TO_MS[interval]
    opened = values["open_time_ms"]
    if (opened % duration != 0).any():
        raise ValueError("歷史 K 線開盤時間未對齊週期邊界")
    if (values["close_time_ms"] != opened + duration - 1).any():
        raise ValueError("歷史 K 線收盤時間不符合週期")
    if (values["close_time_ms"] > int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)).any():
        raise ValueError("歷史 K 線尚未收盤")
    if (timestamps.astype("int64") // 1_000_000 != opened).any():
        raise ValueError("timestamp 與原始開盤時間不一致")
    for column, expected in (
        ("symbol", "BTC/USDT"), ("exchange", "binance_futures"), ("interval", interval),
    ):
        if not frame[column].eq(expected).all():
            raise ValueError(f"歷史 K 線市場身分不一致：{column}")


class HistoryStaging:
    """下載期間使用可續傳的 SQLite 暫存，完成後仍發布原本的固定 CSV。"""

    def __init__(self, path: Path, interval: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.interval = interval
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA cache_size=-32768")
        self.connection.execute("CREATE TABLE IF NOT EXISTS bars (open_time_ms INTEGER PRIMARY KEY)")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS sources (url TEXT PRIMARY KEY, sha256 TEXT, rows INTEGER)"
        )

    def close(self) -> None:
        self.connection.close()

    def upsert(self, frame: pd.DataFrame, *, update_columns: list[str] | None = None) -> None:
        if frame.empty:
            return
        validate_history_chunk(frame, self.interval)
        frame = frame.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="mixed").map(
            lambda value: value.isoformat()
        )
        columns = list(frame.columns)
        if any(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None for name in columns):
            raise ValueError("歷史 CSV 含有不支援的欄位名稱")
        known = {row[1] for row in self.connection.execute("PRAGMA table_info(bars)")}
        for name in columns:
            if name not in known:
                # 欄位名稱已通過上方識別字白名單，這裡不能使用 SQL 值參數取代識別字。
                self.connection.execute(f'ALTER TABLE bars ADD COLUMN "{name}"')
        names = ",".join(f'"{name}"' for name in columns)
        values = ",".join("?" for _ in columns)
        updates = [name for name in (columns if update_columns is None else update_columns)
                   if name != "open_time_ms" and name in columns]
        conflict = "DO NOTHING" if not updates else "DO UPDATE SET " + ",".join(  # nosec B608
            f'"{name}"=excluded."{name}"' for name in updates
        )
        # names 與 conflict 只由已驗證欄位組成；資料值仍全部使用參數化查詢。
        sql = (
            f"INSERT INTO bars ({names}) VALUES ({values}) "  # nosec B608
            f"ON CONFLICT(open_time_ms) {conflict}"
        )
        clean = frame.astype(object).where(frame.notna(), None)
        with self.connection:
            self.connection.executemany(sql, clean.itertuples(index=False, name=None))

    def merge_existing(
        self, path: Path, *, refresh_context: bool = False, heartbeat: Path | None = None,
    ) -> None:
        if not path.exists():
            return
        for chunk in pd.read_csv(path, chunksize=25_000):
            updates = [column for column in chunk if column not in OHLCV_COLUMNS] if refresh_context else []
            self.upsert(chunk, update_columns=updates)
            if heartbeat is not None:
                os.utime(heartbeat, None)

    def missing_ranges(self, start_ms: int, end_ms: int) -> list[tuple[int, int]]:
        """回傳缺少的開盤時間區間，終點不包含；不補造缺失 K 線。"""
        duration = INTERVAL_TO_MS[self.interval]
        expected = start_ms
        gaps: list[tuple[int, int]] = []
        cursor = self.connection.execute(
            "SELECT open_time_ms FROM bars WHERE open_time_ms >= ? AND open_time_ms < ? ORDER BY open_time_ms",
            (start_ms, end_ms),
        )
        for (opened,) in cursor:
            if opened > expected:
                gaps.append((expected, opened))
            expected = opened + duration
        if expected < end_ms:
            gaps.append((expected, end_ms))
        return gaps

    def record_source(self, url: str, checksum: str, rows: int) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO sources VALUES (?, ?, ?)", (url, checksum, rows)
            )

    def publish(self, target: Path) -> None:
        """發布時使用既有檔案鎖，保留新增市場資料與衍生品欄位，再原子替換。"""
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(f".csv.{uuid4().hex}.tmp")
        try:
            with _exclusive_file_lock(target):
                lock_path = target.with_suffix(target.suffix + ".lock")
                self.merge_existing(target, refresh_context=True, heartbeat=lock_path)
                if self.connection.execute("SELECT COUNT(*) FROM bars").fetchone()[0] == 0:
                    raise ValueError("不發布空的歷史 CSV")
                columns = [row[1] for row in self.connection.execute("PRAGMA table_info(bars)")]
                ordered = list(OHLCV_COLUMNS) + [name for name in columns if name not in OHLCV_COLUMNS]
                # ordered 來自固定 OHLCV 欄位與 SQLite schema，不接受任意 SQL 片段。
                query = (
                    "SELECT "  # nosec B608
                    + ",".join(f'"{name}"' for name in ordered)
                    + " FROM bars ORDER BY open_time_ms"
                )
                with temporary.open("w", encoding="utf-8", newline="") as stream:
                    for index, chunk in enumerate(pd.read_sql_query(query, self.connection, chunksize=25_000)):
                        os.utime(lock_path, None)
                        validate_history_chunk(chunk, self.interval)
                        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], utc=True, format="mixed").map(
                            lambda value: value.isoformat()
                        )
                        chunk.to_csv(stream, index=False, header=index == 0)
                    stream.flush()
                    os.fsync(stream.fileno())
                for attempt in range(12):
                    try:
                        temporary.replace(target)
                        break
                    except PermissionError:
                        if attempt == 11:
                            raise
                        time.sleep(min(0.02 * 2**attempt, 0.25))
        finally:
            temporary.unlink(missing_ok=True)


def fetch_monthly_archive(
    session: requests.Session, interval: str, month: pd.Timestamp,
) -> tuple[pd.DataFrame, str, str] | None:
    name = f"BTCUSDT-{interval}-{month:%Y-%m}.zip"
    url = f"{ARCHIVE_ROOT}/{interval}/{name}"
    response = session.get(url, timeout=60)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    checksum_response = session.get(url + ".CHECKSUM", timeout=30)
    checksum_response.raise_for_status()
    parts = checksum_response.text.strip().split()
    if len(parts) != 2 or parts[1].lstrip("*") != name or not re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]):
        raise ValueError(f"官方 CHECKSUM 格式不正確：{name}")
    checksum = hashlib.sha256(response.content).hexdigest()
    if checksum != parts[0].lower():
        raise ValueError(f"官方月檔 SHA-256 不符：{name}")
    frame = decode_monthly_klines(response.content, interval)
    next_month = month + pd.offsets.MonthBegin(1)
    lower, upper = int(month.timestamp() * 1000), int(next_month.timestamp() * 1000)
    if not frame["open_time_ms"].between(lower, upper - 1).all():
        raise ValueError(f"官方月檔包含非指定月份 K 線：{name}")
    return frame, url, checksum


def fill_rest_range(
    staging: HistoryStaging, client: BinanceFuturesPublicClient, start_ms: int, end_ms: int,
) -> None:
    duration = INTERVAL_TO_MS[staging.interval]
    for first in range(start_ms, end_ms, duration * 499):
        stop = min(first + duration * 499, end_ms)
        frame = client.fetch_ohlcv(
            "BTC/USDT", staging.interval,
            start=pd.Timestamp(first, unit="ms", tz="UTC").isoformat(),
            end=pd.Timestamp(stop - 1, unit="ms", tz="UTC").isoformat(), limit=499,
        )
        if not frame.empty:
            if not frame["open_time_ms"].between(first, stop - 1).all():
                raise ValueError("REST 回傳了要求區間以外的 K 線")
            staging.upsert(frame)
        time.sleep(0.08)


def download_btc_history(
    project_root: Path, *, start: str, end: str | None = None, warmup_bars: int = 250,
    intervals: tuple[str, ...] = BTC_SUPPORTED_MULTITIMEFRAME_INTERVALS,
) -> dict[str, object]:
    """補齊九週期已收盤歷史；不建特徵、不訓練、不讀密鑰、不下單。"""
    if warmup_bars < 200 or not intervals or any(value not in BTC_SUPPORTED_MULTITIMEFRAME_INTERVALS for value in intervals):
        raise ValueError("週期必須為 BTC 支援週期，指標暖機至少 200 根")
    begin = pd.Timestamp(start)
    begin = begin.tz_localize("UTC") if begin.tzinfo is None else begin.tz_convert("UTC")
    session = create_secure_session()
    client = BinanceFuturesPublicClient(session=session)
    server = session.get("https://fapi.binance.com/fapi/v1/time", timeout=30)
    server.raise_for_status()
    now = pd.Timestamp(int(server.json()["serverTime"]), unit="ms", tz="UTC")
    cutoff = pd.Timestamp(end) if end else now
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    cutoff = min(cutoff, now, pd.Timestamp.now(tz="UTC"))
    if cutoff <= begin:
        raise ValueError("資料結束時間必須晚於開始時間")
    root = project_root.resolve()
    raw_dir = root / "data" / "raw"
    manifest_path = canonical_ohlcv_path(raw_dir, "binance_futures", "BTC/USDT", "15m").parent / "btc_futures_history_manifest.json"
    manifest: dict[str, object] = {
        "schema_version": 1, "market": "binance_futures", "symbol": "BTC/USDT",
        "training_start_utc": begin.isoformat(), "snapshot_cutoff_utc": cutoff.isoformat(),
        "timestamp_semantics": "UTC bar open time", "warmup_bars": warmup_bars,
        "status": "RUNNING", "datasets": [],
    }
    write_json_atomic(manifest_path, manifest)
    datasets: list[dict[str, object]] = []
    try:
        for index, interval in enumerate(dict.fromkeys(intervals), start=1):
            duration = INTERVAL_TO_MS[interval]
            start_ms = int(begin.timestamp() * 1000) // duration * duration - warmup_bars * duration
            end_ms = int(cutoff.timestamp() * 1000) // duration * duration
            target = canonical_ohlcv_path(raw_dir, "binance_futures", "BTC/USDT", interval)
            staging_path = root / "data" / "database" / "downloads" / f"btc_futures_{interval}.sqlite"
            staging = HistoryStaging(staging_path, interval)
            print(f"[{index}/{len(intervals)}] {interval}: 盤點及補漏", flush=True)
            completed = False
            try:
                staging.merge_existing(target)
                for gap_start, gap_end in staging.missing_ranges(start_ms, end_ms):
                    if (gap_end - gap_start) // duration < 5_000:
                        fill_rest_range(staging, client, gap_start, gap_end)
                        print(f"  {interval}: 小型缺口 REST 補漏完成", flush=True)
                        continue
                    first = pd.Timestamp(gap_start, unit="ms", tz="UTC")
                    month = first.normalize().replace(day=1)
                    while int(month.timestamp() * 1000) < gap_end:
                        following = month + pd.offsets.MonthBegin(1)
                        left = max(gap_start, int(month.timestamp() * 1000))
                        right = min(gap_end, int(following.timestamp() * 1000))
                        archived = None
                        if following <= cutoff:
                            archived = fetch_monthly_archive(session, interval, month)
                        if archived is not None:
                            frame, url, checksum = archived
                            frame = frame.loc[frame["open_time_ms"].between(left, right - 1)].copy()
                            staging.upsert(frame)
                            staging.record_source(url, checksum, len(frame))
                            print(f"  {interval} {month:%Y-%m}: 官方月檔 {len(frame):,} 根，SHA-256 通過", flush=True)
                        else:
                            fill_rest_range(staging, client, left, right)
                            print(f"  {interval} {month:%Y-%m}: REST 補漏完成", flush=True)
                        month = following
                gaps = staging.missing_ranges(start_ms, end_ms)
                for left, right in gaps:
                    fill_rest_range(staging, client, left, right)
                gaps = staging.missing_ranges(start_ms, end_ms)
                staging.publish(target)
                count, first, last = staging.connection.execute(
                    "SELECT COUNT(*), MIN(open_time_ms), MAX(open_time_ms) FROM bars"
                ).fetchone()
                digest = hashlib.sha256()
                with target.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                requested_count = staging.connection.execute(
                    "SELECT COUNT(*) FROM bars WHERE open_time_ms >= ? AND open_time_ms < ?", (start_ms, end_ms)
                ).fetchone()[0]
                missing = sum((right - left) // duration for left, right in gaps)
                item = {
                    "interval": interval, "path": str(target), "rows": count,
                    "requested_rows": requested_count, "expected_rows": (end_ms - start_ms) // duration,
                    "first_open_utc": pd.Timestamp(first, unit="ms", tz="UTC").isoformat(),
                    "last_open_utc": pd.Timestamp(last, unit="ms", tz="UTC").isoformat(),
                    "requested_start_utc": pd.Timestamp(start_ms, unit="ms", tz="UTC").isoformat(),
                    "requested_end_exclusive_utc": pd.Timestamp(end_ms, unit="ms", tz="UTC").isoformat(),
                    "missing_bars": missing, "missing_ranges_ms": gaps, "duplicates": 0,
                    "sha256": digest.hexdigest(), "bytes": target.stat().st_size,
                    "status": "PASS" if not missing else "WARN",
                    "archive_sources": [dict(url=row[0], sha256=row[1], rows=row[2]) for row in
                                        staging.connection.execute("SELECT url, sha256, rows FROM sources ORDER BY url")],
                }
                datasets.append(item)
                manifest["datasets"] = datasets
                write_json_atomic(manifest_path, manifest)
                print(f"  完成 {interval}: {count:,} 根，缺 {missing} 根，{target.stat().st_size / 1024**2:.1f} MiB", flush=True)
                completed = True
            finally:
                staging.close()
                if completed:
                    staging_path.unlink(missing_ok=True)
        manifest["status"] = "PASS" if all(item["status"] == "PASS" for item in datasets) else "WARN"
        manifest["completed_at_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
        write_json_atomic(manifest_path, manifest)
        print(f"資料清單與品質紀錄：{manifest_path}", flush=True)
        return manifest
    except (Exception, KeyboardInterrupt) as exc:
        manifest["status"] = "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAIL"
        manifest["error_type"] = type(exc).__name__
        write_json_atomic(manifest_path, manifest)
        raise
    finally:
        session.close()


def history_error_types() -> tuple[type[Exception], ...]:
    """供命令列統一處理已知下載錯誤。"""
    return (BinanceApiError, requests.RequestException, ValueError, OSError, sqlite3.Error)
