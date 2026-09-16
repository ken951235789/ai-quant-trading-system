"""原始碼模式與 PyInstaller 打包模式共用的路徑工具。"""

from __future__ import annotations

from pathlib import Path
import sys


def is_frozen_app() -> bool:
    """判斷目前是否由 PyInstaller EXE 執行。"""
    return bool(getattr(sys, "frozen", False))


def get_runtime_root() -> Path:
    """開發模式使用專案根目錄，打包模式使用 EXE 所在資料夾。"""
    if is_frozen_app():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[3]


def get_bundled_resource(relative_path: str | Path) -> Path:
    """取得 PyInstaller 內部資源；開發模式則從專案根目錄尋找。"""
    bundle_root = getattr(sys, "_MEIPASS", None)
    root = Path(bundle_root) if bundle_root else get_runtime_root()
    return root / relative_path
