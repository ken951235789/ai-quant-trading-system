"""跨程序狀態檔使用的可靠寫入工具。"""

from __future__ import annotations

import json
from functools import lru_cache
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable

import pandas as pd


def write_json_atomic(
    path: str | Path,
    payload: object,
    *,
    default: Callable[[Any], object] | None = None,
    replace_attempts: int = 12,
) -> Path:
    """先完整落盤再原子取代，並處理 Windows 防毒或讀取程序的短暫鎖定。"""
    target = Path(path)
    if replace_attempts < 1:
        raise ValueError("replace_attempts 必須大於 0")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, mode="w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                indent=2,
                default=default,
            )
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(replace_attempts):
            try:
                os.replace(temporary, target)
                return target
            except PermissionError:
                if attempt + 1 >= replace_attempts:
                    raise
                time.sleep(min(0.02 * (2**attempt), 0.25))
    finally:
        temporary.unlink(missing_ok=True)


@lru_cache(maxsize=128)
def _read_csv_revision(
    path: str,
    modified_ns: int,
    changed_ns: int,
    size: int,
) -> pd.DataFrame:
    """依檔案版本解析 CSV；版本欄位只用於讓快取自動失效。"""
    del modified_ns, changed_ns, size
    return pd.read_csv(path)


def read_csv_snapshot(path: str | Path) -> pd.DataFrame:
    """讀取 CSV 快照；檔案未變時不重複解析，回傳值可由呼叫端安全修改。"""
    source = Path(path)
    stat = source.stat()
    frame = _read_csv_revision(
        str(source.resolve()),
        stat.st_mtime_ns,
        stat.st_ctime_ns,
        stat.st_size,
    )
    return frame.copy(deep=True)


def clear_csv_snapshot_cache() -> None:
    """測試或大量搬移資料後可主動清除 CSV 快取。"""
    _read_csv_revision.cache_clear()
