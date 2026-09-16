"""建置可雙擊啟動的 Windows one-folder EXE。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import os
import shutil
import ssl
import subprocess
import sys

from ai_quant_trading.runtime_secrets import runtime_env_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = PROJECT_ROOT / "packaging" / "AIQuantTradingSystem.spec"
DIST_DIR = PROJECT_ROOT / "dist" / "AIQuantTradingSystem"
RUNTIME_BACKUP_DIR = PROJECT_ROOT / ".package-runtime-backup"
PRESERVED_RUNTIME_PATHS = ("data", "logs")
PROJECT_RUNTIME_DIRECTORIES = ("data", "configs", "docs", ".streamlit")
PACKAGING_DATA_EXCLUDES = {
    "models",
    "paper_trading_validation",
    "research",
}
PACKAGING_PROCESSED_EXCLUDES = {
    "backtests",
    "strategy_research",
    "short_term_research",
    "us_portfolio_research",
}
USER_ENV_PATH = runtime_env_path(PROJECT_ROOT, frozen=True)


def _validate_packaging_python() -> bool:
    """Windows 桌面版固定使用已驗證的 Python 3.12 工具鏈。"""
    if sys.version_info[:2] == (3, 12):
        return True
    print(
        "Windows App 請使用 Python 3.12 建置；目前版本是 "
        f"{sys.version_info.major}.{sys.version_info.minor}。\n"
        "這項限制可避免 Python 3.13／OpenSSL 與 PyInstaller 的啟動衝突。"
    )
    return False


def _validate_packaging_openssl() -> bool:
    """拒絕已知會讓 Windows WebSocket 崩潰的 OpenSSL 版本。"""
    incompatible = ("OpenSSL 3.5.7", "OpenSSL 3.6.3")
    if not ssl.OPENSSL_VERSION.startswith(incompatible):
        return True
    print(
        f"目前 {ssl.OPENSSL_VERSION} 有 Windows TLS 已知問題；"
        "請改用 OpenSSL 3.5.5/3.5.6 或已修復的新版本後再建置。"
    )
    return False


def _backup_runtime_data() -> None:
    """在 PyInstaller 清除 dist 前暫存桌面版產生的資料與帳本。"""
    if RUNTIME_BACKUP_DIR.exists():
        raise RuntimeError(
            f"發現尚未還原的桌面資料備份，請先檢查：{RUNTIME_BACKUP_DIR}"
        )
    sources = [DIST_DIR / relative for relative in PRESERVED_RUNTIME_PATHS]
    if not any(source.exists() for source in sources):
        return
    RUNTIME_BACKUP_DIR.mkdir(parents=True)
    for source in sources:
        if source.exists():
            shutil.move(str(source), RUNTIME_BACKUP_DIR / source.name)


def _copy_newer_file(source: str, destination: str) -> str:
    """合併資料時保留修改時間較新的桌面版檔案。"""
    source_path = Path(source)
    destination_path = Path(destination)
    if (
        destination_path.exists()
        and destination_path.stat().st_mtime_ns >= source_path.stat().st_mtime_ns
    ):
        return str(destination_path)
    return shutil.copy2(source_path, destination_path)


def _runtime_copy_ignore(directory: str, names: list[str]) -> set[str]:
    """桌面包略過可下載快取、測試資料與已淘汰的規則回測輸出。"""
    current = Path(directory).resolve()
    data_root = (PROJECT_ROOT / "data").resolve()
    processed_root = (data_root / "processed").resolve()
    if current == data_root:
        return set(names).intersection(PACKAGING_DATA_EXCLUDES)
    if current == processed_root:
        return set(names).intersection(PACKAGING_PROCESSED_EXCLUDES)
    return set()


def _remove_rebuildable_packaging_data() -> None:
    """舊桌面資料還原後，再移除不應隨 EXE 攜帶的大型快取。"""
    targets = [
        *(DIST_DIR / "data" / name for name in PACKAGING_DATA_EXCLUDES),
        *(
            DIST_DIR / "data" / "processed" / name
            for name in PACKAGING_PROCESSED_EXCLUDES
        ),
    ]
    dist_root = DIST_DIR.resolve()
    for target in targets:
        resolved = target.resolve()
        if not resolved.is_relative_to(dist_root):
            raise RuntimeError(f"拒絕清理桌面包以外的路徑：{resolved}")
        if resolved.is_dir():
            shutil.rmtree(resolved)
        elif resolved.exists():
            resolved.unlink()


def _remove_transient_runtime_files() -> None:
    """不要把上一次執行留下的鎖與停止請求帶進新版 App。"""
    data_dir = DIST_DIR / "data"
    if not data_dir.exists():
        return
    transient_names = {
        "startup_refresh.lock",
        "runner.lock",
        "stop.requested",
    }
    for path in data_dir.rglob("*"):
        if path.is_file() and path.name in transient_names:
            path.unlink(missing_ok=True)


def _restore_runtime_data() -> None:
    """把打包前暫存資料合併回新成品，避免更新 EXE 清空使用紀錄。"""
    if not RUNTIME_BACKUP_DIR.exists():
        return
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    for relative in PRESERVED_RUNTIME_PATHS:
        source = RUNTIME_BACKUP_DIR / relative
        if not source.exists():
            continue
        destination = DIST_DIR / relative
        if source.is_dir():
            shutil.copytree(
                source,
                destination,
                dirs_exist_ok=True,
                copy_function=_copy_newer_file,
            )
        elif not destination.exists() or source.stat().st_mtime_ns > destination.stat().st_mtime_ns:
            shutil.copy2(source, destination)
    shutil.rmtree(RUNTIME_BACKUP_DIR)


def _merge_project_runtime_directories() -> None:
    """只以較新的專案檔案補齊桌面目錄，不覆蓋較新的使用者資料。"""
    for directory in PROJECT_RUNTIME_DIRECTORIES:
        source = PROJECT_ROOT / directory
        if source.exists():
            shutil.copytree(
                source,
                DIST_DIR / directory,
                dirs_exist_ok=True,
                copy_function=_copy_newer_file,
                ignore=_runtime_copy_ignore if directory == "data" else None,
            )


def _copy_runtime_data() -> None:
    """把可寫入資料與設定放到 EXE 同層。"""
    _merge_project_runtime_directories()

    shutil.copy2(
        PROJECT_ROOT / "packaging" / "使用說明.txt",
        DIST_DIR / "使用說明.txt",
    )
    shutil.copy2(
        PROJECT_ROOT / "packaging" / "關閉 AI Quant App.cmd",
        DIST_DIR / "關閉 AI Quant App.cmd",
    )
    for filename in [
        "安裝無人值守排程.ps1",
        "安裝無人值守排程.cmd",
        "移除無人值守排程.cmd",
    ]:
        shutil.copy2(PROJECT_ROOT / "packaging" / filename, DIST_DIR / filename)
    shutil.copy2(PROJECT_ROOT / ".env.example", DIST_DIR / ".env.example")
    (DIST_DIR / "logs").mkdir(parents=True, exist_ok=True)

    # 背景工作 JSON 會包含建置電腦的絕對路徑；只攜帶模型成品，
    # 避免移到其他電腦後顯示一份已完成但路徑失效的進度工作。
    transient_jobs = (
        DIST_DIR
        / "data"
        / "processed"
        / "transformer"
        / "training_jobs"
    )
    if transient_jobs.exists():
        shutil.rmtree(transient_jobs)


def _harden_secret_acl(secret: Path) -> None:
    """限制 Windows 秘密檔僅目前帳號、SYSTEM 與系統管理員可讀。"""
    if os.name != "nt" or not secret.is_file():
        return
    user = os.getenv("USERNAME", "").strip()
    domain = os.getenv("USERDOMAIN", "").strip()
    if not user:
        raise RuntimeError("無法辨識目前 Windows 使用者，拒絕建立未受保護的秘密檔")
    identity = f"{domain}\\{user}" if domain else user
    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows")).resolve()
    icacls = system_root / "System32" / "icacls.exe"
    if not icacls.is_file():
        raise FileNotFoundError(f"找不到 Windows ACL 工具：{icacls}")
    subprocess.run(
        [
            str(icacls),
            str(secret),
            "/inheritance:r",
            "/grant:r",
            f"{identity}:(F)",
            "SYSTEM:(F)",
            "BUILTIN\\Administrators:(F)",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )


def _migrate_legacy_runtime_secret() -> Path | None:
    """把舊版 dist/.env 移到使用者設定目錄，避免分享 App 時外洩。"""
    legacy = DIST_DIR / ".env"
    if not legacy.is_file():
        return None
    USER_ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not USER_ENV_PATH.exists():
        destination = USER_ENV_PATH
    elif legacy.read_bytes() == USER_ENV_PATH.read_bytes():
        legacy.unlink()
        _harden_secret_acl(USER_ENV_PATH)
        return USER_ENV_PATH
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        destination = USER_ENV_PATH.with_name(f".env.migrated-{timestamp}.bak")
    shutil.move(str(legacy), destination)
    _harden_secret_acl(destination)
    return destination

def _validate_runtime_data() -> None:
    """確認桌面成品不是只有 EXE 與依賴，卻遺漏可操作資料。"""
    required = [
        DIST_DIR / "data" / "raw",
        DIST_DIR / "data" / "processed",
        DIST_DIR / "configs",
        DIST_DIR / "docs",
        DIST_DIR / "使用說明.txt",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"桌面成品缺少執行資料：{missing}")
    if not any((DIST_DIR / "data" / "processed").rglob("*.csv")):
        raise FileNotFoundError("桌面成品沒有任何 processed CSV，請檢查資料同步")
    if (DIST_DIR / ".env").exists():
        raise RuntimeError("桌面成品不可包含 .env 秘密檔")


def main() -> int:
    """呼叫 PyInstaller，完成後補上資料與中文說明。"""
    if not _validate_packaging_python() or not _validate_packaging_openssl():
        return 1

    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print('尚未安裝 PyInstaller，請執行：python -m pip install -e ".[package]"')
        return 1

    try:
        import gymnasium  # noqa: F401
        import stable_baselines3  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        print('尚未安裝 AI 或 RL 套件，請執行：python -m pip install -e ".[ai,rl]"')
        return 1

    print(
        f"PyTorch 建置：{torch.__version__}，CUDA runtime：{torch.version.cuda or '無（CPU-only）'}"
    )

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        str(SPEC_PATH),
    ]
    migrated_secret = _migrate_legacy_runtime_secret()
    if migrated_secret is not None:
        print(f"舊版秘密檔已移到使用者設定目錄：{migrated_secret}")
    _backup_runtime_data()
    try:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        _copy_runtime_data()
        _restore_runtime_data()
        # 舊桌面資料還原後再補一次，確保本次新增的模型與資料不會遺漏。
        _merge_project_runtime_directories()
        _remove_rebuildable_packaging_data()
        _remove_transient_runtime_files()
        _validate_runtime_data()
    finally:
        # 打包失敗時也把可恢復的帳戶資料放回 dist，不讓備份孤立。
        _restore_runtime_data()
    executable = DIST_DIR / "AIQuantTradingSystem.exe"
    if not executable.exists():
        raise FileNotFoundError(f"建置完成但找不到 EXE：{executable}")
    print(f"Windows App：{executable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
