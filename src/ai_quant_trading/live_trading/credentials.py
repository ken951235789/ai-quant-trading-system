"""只從環境變數載入 Binance API 憑證。"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

from ai_quant_trading.live_trading.config import TradingEnvironment


@dataclass(frozen=True, slots=True)
class BinanceCredentials:
    """記憶體內使用的 API Key 與 HMAC Secret。"""

    api_key: str
    api_secret: str

    def __post_init__(self) -> None:
        if not self.api_key.strip() or not self.api_secret.strip():
            raise ValueError("Binance API Key 與 Secret 不可為空")


def credential_variable_names(environment: TradingEnvironment) -> tuple[str, str]:
    """回傳指定環境使用的變數名稱。"""
    prefix = f"BINANCE_{environment.upper()}"
    return f"{prefix}_API_KEY", f"{prefix}_API_SECRET"


def load_binance_credentials(
    environment: TradingEnvironment,
    env_path: str | Path | None = None,
) -> BinanceCredentials:
    """載入 `.env` 與系統環境變數；系統環境變數優先。"""
    if env_path is not None:
        load_dotenv(Path(env_path), override=False)
    key_name, secret_name = credential_variable_names(environment)
    api_key = os.getenv(key_name, "").strip()
    api_secret = os.getenv(secret_name, "").strip()
    if not api_key or not api_secret:
        raise ValueError(f"缺少 {key_name} 或 {secret_name}")
    return BinanceCredentials(api_key, api_secret)


def credentials_available(
    environment: TradingEnvironment,
    env_path: str | Path | None = None,
) -> bool:
    """不讀取憑證內容，只檢查指定環境是否已設定兩個變數。"""
    if env_path is not None:
        load_dotenv(Path(env_path), override=False)
    key_name, secret_name = credential_variable_names(environment)
    return bool(os.getenv(key_name, "").strip() and os.getenv(secret_name, "").strip())
