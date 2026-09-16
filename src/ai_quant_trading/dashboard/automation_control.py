"""從 Dashboard 啟動可脫離視窗生命週期的自動交易背景程序。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from typing import Literal

from ai_quant_trading.automation import automation_paths, read_automation_status
from ai_quant_trading.dashboard.runtime import is_frozen_app


WorkerType = Literal["paper", "live"]
AUTOSTART_FILENAME = "autostart.json"
PORTABLE_PROJECT_PATH_FLAGS = frozenset(
    {
        "--input",
        "--model-dir",
        "--paper-dir",
        "--raw-dir",
        "--training-dir",
        "--transformer-checkpoint",
    }
)


def _process_is_alive(process_id: int | None) -> bool:
    """確認背景程序仍存在，避免舊狀態檔阻止自動恢復。"""
    if process_id is None or process_id <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(0x1000, False, process_id)
        if not handle:
            return False
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == 259
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(process_id, 0)
    except (OSError, ValueError):
        return False
    return True


def automation_autostart_enabled(automation_dir: str | Path) -> bool:
    """讀取帳戶是否應在 App 啟動時自動恢復背景程序。"""
    path = Path(automation_dir) / AUTOSTART_FILENAME
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return bool(payload.get("enabled", False))


def set_automation_autostart(automation_dir: str | Path, enabled: bool) -> Path:
    """原子化保存持續運行設定。"""
    target = Path(automation_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    path = target / AUTOSTART_FILENAME
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "enabled": bool(enabled),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def start_enabled_paper_workers(
    project_root: str | Path,
    paper_dir: str | Path,
) -> tuple[int, ...]:
    """恢復已啟用且目前沒有存活程序的 BTC WebSocket 模擬帳戶。"""
    root = Path(project_root).resolve()
    paper_root = Path(paper_dir).resolve()
    if not paper_root.is_dir():
        return ()
    started: list[int] = []
    for automation_dir in paper_root.glob("*/automation"):
        if not automation_autostart_enabled(automation_dir):
            continue
        status = read_automation_status(automation_paths(automation_dir))
        if status.state in {"running", "stopping"} and _process_is_alive(status.pid):
            continue
        config_path = automation_dir / "worker_config.json"
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            arguments = payload.get("arguments")
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if not isinstance(arguments, list) or not all(
            isinstance(item, str) for item in arguments
        ):
            continue
        if not arguments or arguments[0] != "rl-auto":
            continue
        if "--transport" not in arguments:
            continue
        transport_index = arguments.index("--transport") + 1
        if (
            transport_index >= len(arguments)
            or arguments[transport_index] not in {"websocket", "rest"}
        ):
            continue
        started.append(
            start_automation_worker("paper", root, automation_dir, arguments)
        )
    return tuple(started)


def _worker_command(worker_type: WorkerType, config_path: Path) -> list[str]:
    flag = "--paper-auto-worker" if worker_type == "paper" else "--live-auto-worker"
    arguments = [flag, "--worker-config", str(config_path)]
    if is_frozen_app():
        return [sys.executable, *arguments]
    return [sys.executable, "-m", "ai_quant_trading.dashboard.launcher", *arguments]


def _portable_worker_arguments(
    project_root: str | Path,
    arguments: list[str],
) -> list[str]:
    """將專案 data 內的絕對路徑改成可搬移的相對路徑。"""
    root = Path(project_root).resolve()
    portable = list(arguments)
    for index, flag in enumerate(portable[:-1]):
        if flag not in PORTABLE_PROJECT_PATH_FLAGS:
            continue
        value = Path(portable[index + 1])
        if not value.is_absolute():
            continue
        try:
            relative = value.resolve().relative_to(root)
        except ValueError:
            parts = value.parts
            data_index = next(
                (
                    part_index
                    for part_index, part in enumerate(parts)
                    if part.lower() == "data"
                ),
                None,
            )
            if data_index is None:
                continue
            relative = Path(*parts[data_index:])
        portable[index + 1] = str(relative)
    return portable


def start_automation_worker(
    worker_type: WorkerType,
    project_root: str | Path,
    automation_dir: str | Path,
    arguments: list[str],
) -> int:
    """保存非機密 CLI 參數並啟動隱藏程序，回傳 PID。"""
    if not arguments or arguments[0] not in {"auto", "rl-auto"}:
        raise ValueError("背景工作只接受 auto 或 rl-auto 子命令")
    root = Path(project_root).resolve()
    target = Path(automation_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    arguments = _portable_worker_arguments(root, arguments)
    config_path = target / "worker_config.json"
    temporary = config_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps({"arguments": arguments}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(config_path)
    log_path = target / "worker.log"
    environment = dict(os.environ)
    if is_frozen_app():
        environment["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    creation_flags = 0
    if sys.platform == "win32":
        creation_flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    with log_path.open("ab") as log_stream:
        process = subprocess.Popen(
            _worker_command(worker_type, config_path),
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            creationflags=creation_flags,
            env=environment,
            close_fds=True,
        )
    started_at = datetime.now(timezone.utc).isoformat()
    status_path = target / "status.json"
    status_temporary = status_path.with_suffix(".json.tmp")
    status_temporary.write_text(
        json.dumps(
            {
                "state": "running",
                "pid": process.pid,
                "started_at": started_at,
                "heartbeat_at": started_at,
                "stopped_at": None,
                "attempted_cycles": 0,
                "successful_cycles": 0,
                "consecutive_errors": 0,
                "last_message": "背景程序啟動中",
                "last_error": None,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    status_temporary.replace(status_path)
    return process.pid
