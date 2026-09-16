"""PostgreSQL Engine、連線池與健康檢查。"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from ai_quant_trading.database.config import DatabaseSettings


class DatabaseUnavailableError(RuntimeError):
    """資料庫無法使用；交易模式必須 fail closed。"""


def create_database_engine(settings: DatabaseSettings) -> Any:
    """建立 PostgreSQL 連線池，模組缺失時提供明確安裝訊息。"""
    if not settings.enabled:
        raise ValueError("只有 postgres 儲存模式需要建立資料庫 Engine")
    try:
        from sqlalchemy import create_engine
    except ImportError as exc:
        raise RuntimeError(
            '缺少 SQL 套件，請執行 python -m pip install -e ".[database]"'
        ) from exc
    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_size=settings.pool_size,
        max_overflow=settings.max_overflow,
        pool_recycle=1800,
        connect_args={"connect_timeout": settings.connect_timeout_seconds},
    )


@lru_cache(maxsize=4)
def engine_from_url(
    database_url: str,
    pool_size: int = 5,
    max_overflow: int = 5,
    connect_timeout_seconds: int = 5,
) -> Any:
    """依安全設定重用 Engine，避免 Dashboard 每次重跑都新增連線池。"""
    settings = DatabaseSettings(
        backend="postgres",
        database_url=database_url,
        pool_size=pool_size,
        max_overflow=max_overflow,
        connect_timeout_seconds=connect_timeout_seconds,
    )
    return create_database_engine(settings)


def check_database_health(engine: Any) -> tuple[bool, str]:
    """執行最小查詢；不把含密碼的連線字串寫進錯誤訊息。"""
    try:
        from sqlalchemy import text

        with engine.connect() as connection:
            value = connection.execute(text("SELECT 1")).scalar_one()
        return value == 1, "PostgreSQL 連線正常"
    except Exception as exc:
        return False, f"PostgreSQL 無法連線：{type(exc).__name__}: {exc}"
