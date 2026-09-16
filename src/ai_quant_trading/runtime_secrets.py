"""集中管理原始碼版與封裝版的秘密設定檔位置。"""

from __future__ import annotations

import os
from pathlib import Path
import sys


APP_DIRECTORY_NAME = "AIQuantTradingSystem"


def user_config_directory() -> Path:
    """回傳目前使用者專用的設定目錄，不把密鑰放在可分享的 App 目錄。"""
    override = os.getenv("AI_QUANT_CONFIG_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    if os.name == "nt":
        base = os.getenv("LOCALAPPDATA", "").strip()
        if base:
            return Path(base).resolve() / APP_DIRECTORY_NAME
    base = os.getenv("XDG_CONFIG_HOME", "").strip()
    if base:
        return Path(base).expanduser().resolve() / APP_DIRECTORY_NAME
    return Path.home().resolve() / ".config" / APP_DIRECTORY_NAME


def runtime_env_path(
    runtime_root: str | Path | None = None,
    *,
    frozen: bool | None = None,
) -> Path:
    """原始碼版沿用專案 `.env`；封裝版改用使用者設定目錄。"""
    packaged = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
    if packaged:
        return user_config_directory() / ".env"
    root = Path(runtime_root).resolve() if runtime_root is not None else Path.cwd().resolve()
    return root / ".env"
