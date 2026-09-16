"""資料庫環境變數與安全預設值。"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

from ai_quant_trading.runtime_secrets import runtime_env_path


StorageBackend = Literal["file", "postgres"]


@dataclass(frozen=True, slots=True)
class DatabaseSettings:
    """SQL 儲存設定；PostgreSQL 失效時不允許靜默改寫回檔案。"""

    backend: StorageBackend = "file"
    database_url: str = ""
    pool_size: int = 5
    max_overflow: int = 5
    connect_timeout_seconds: int = 5

    def __post_init__(self) -> None:
        if self.backend not in {"file", "postgres"}:
            raise ValueError("AI_QUANT_STORAGE_BACKEND 必須是 file 或 postgres")
        if self.backend == "postgres" and not self.database_url:
            raise ValueError("postgres 模式必須設定 AI_QUANT_DATABASE_URL")
        if self.database_url and not self.database_url.startswith(
            ("postgresql://", "postgresql+psycopg://")
        ):
            raise ValueError("交易狀態中心只接受 PostgreSQL 連線字串")
        if self.pool_size < 1 or self.max_overflow < 0:
            raise ValueError("資料庫連線池設定不合法")
        if self.connect_timeout_seconds < 1:
            raise ValueError("資料庫連線逾時必須大於 0")

    @classmethod
    def from_environment(cls, env_path: str | Path | None = None) -> "DatabaseSettings":
        """讀取 .env 與程序環境，程序環境的值優先。"""
        candidate = Path(env_path) if env_path is not None else runtime_env_path()
        if candidate.exists():
            load_dotenv(candidate, override=False)
        backend = os.getenv("AI_QUANT_STORAGE_BACKEND", "file").strip().lower()
        return cls(
            backend=backend,  # type: ignore[arg-type]
            database_url=os.getenv("AI_QUANT_DATABASE_URL", "").strip(),
            pool_size=int(os.getenv("AI_QUANT_DATABASE_POOL_SIZE", "5")),
            max_overflow=int(os.getenv("AI_QUANT_DATABASE_MAX_OVERFLOW", "5")),
            connect_timeout_seconds=int(
                os.getenv("AI_QUANT_DATABASE_CONNECT_TIMEOUT_SECONDS", "5")
            ),
        )

    @property
    def enabled(self) -> bool:
        return self.backend == "postgres"

    @property
    def redacted_url(self) -> str:
        """回傳不含密碼的連線描述，避免 log 洩漏憑證。"""
        if not self.database_url or "@" not in self.database_url:
            return self.database_url
        prefix, location = self.database_url.rsplit("@", 1)
        scheme, _, credentials = prefix.partition("://")
        username = credentials.split(":", 1)[0]
        return f"{scheme}://{username}:***@{location}"
