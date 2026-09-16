"""Windows 桌面視窗與 Streamlit 背景服務啟動器。"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import BinaryIO
from urllib.parse import urlsplit

from ai_quant_trading.dashboard.runtime import (
    get_bundled_resource,
    get_runtime_root,
    is_frozen_app,
)
from ai_quant_trading.performance import apply_thread_environment, clamp_cpu_threads


APP_TITLE = "AI Quant Trading System"
DEFAULT_DASHBOARD_CPU_THREADS = 2


def urlopen(url: str, timeout: float):
    """延後載入 SSL 網路層，讓封裝版背景服務能先初始化 Arrow。"""
    from urllib.request import urlopen as open_url

    parsed = urlsplit(url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("健康檢查只允許本機 HTTP 網址")
    # 前一行已把 scheme 與 host 限制為本機健康端點。
    return open_url(url, timeout=timeout)  # nosec B310


def _configure_logging(root: Path, filename: str = "launcher.log") -> None:
    """把啟動資訊寫入 EXE 同層的 logs 資料夾。"""
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / filename,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        encoding="utf-8",
        force=True,
    )


def _prepare_runtime_directories(root: Path) -> None:
    """建立桌面版執行期間需要的可寫入資料夾。"""
    for relative_dir in [
        "data/raw",
        "data/processed",
        "data/database",
        "data/paper_trading",
        "data/live_trading",
        "logs",
    ]:
        (root / relative_dir).mkdir(parents=True, exist_ok=True)


def _dashboard_cpu_threads() -> int:
    """讀取互動介面的 CPU 預算，無效設定會安全回到兩個執行緒。"""
    try:
        requested = int(
            os.environ.get(
                "AI_QUANT_DASHBOARD_CPU_THREADS",
                DEFAULT_DASHBOARD_CPU_THREADS,
            )
        )
        return clamp_cpu_threads(requested)
    except (TypeError, ValueError):
        return clamp_cpu_threads(DEFAULT_DASHBOARD_CPU_THREADS)


def _configure_current_process_threads() -> int:
    """在大型運算函式庫載入前限制目前 Dashboard／Worker 行程。"""
    threads = _dashboard_cpu_threads()
    apply_thread_environment(os.environ, threads, overwrite=True)
    return threads


def _acquire_single_instance(root: Path) -> BinaryIO | None:
    """鎖住桌面 App 執行個體，避免重複雙擊後多開服務。"""
    lock_path = root / "logs" / "dashboard-instance.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stream = lock_path.open("a+b")
    if lock_path.stat().st_size == 0:
        stream.write(b"0")
        stream.flush()
    stream.seek(0)
    try:
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:  # pragma: no cover - Windows 桌面版以外的開發環境
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        stream.close()
        return None
    return stream


def _release_single_instance(stream: BinaryIO) -> None:
    """釋放桌面 App 執行個體鎖。"""
    try:
        stream.seek(0)
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:  # pragma: no cover - Windows 桌面版以外的開發環境
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    finally:
        stream.close()


def _redirect_frozen_standard_streams(root: Path) -> None:
    """封裝版沒有主控台，將標準輸出導向 UTF-8 日誌供 Streamlit 使用。"""
    if not is_frozen_app():
        return
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = (log_dir / "streamlit-stdout.log").open(
        "a",
        encoding="utf-8",
        buffering=1,
    )
    sys.stderr = (log_dir / "streamlit-stderr.log").open(
        "a",
        encoding="utf-8",
        buffering=1,
    )


def _is_port_available(port: int) -> bool:
    # 先連線再嘗試綁定，才能同時辨識 IPv4 與 IPv6 wildcard 監聽。
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.2)
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            return False

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def find_available_port(start: int = 8501, attempts: int = 20) -> int:
    """從 8501 開始尋找可供本機背景服務使用的連接埠。"""
    for port in range(start, start + attempts):
        if _is_port_available(port):
            return port
    raise RuntimeError("找不到可供 Dashboard 使用的本機連接埠")


def _streamlit_entry_script(root: Path) -> Path:
    """依原始碼或 PyInstaller 模式取得 Streamlit 入口。"""
    return (
        get_bundled_resource(Path("app") / "streamlit_entry.py")
        if is_frozen_app()
        else root / "packaging" / "streamlit_entry.py"
    )


def _build_server_command(port: int) -> list[str]:
    """建立背景 Streamlit 子程序命令。"""
    arguments = ["--streamlit-server", "--port", str(port)]
    if is_frozen_app():
        return [sys.executable, *arguments]
    return [
        sys.executable,
        "-m",
        "ai_quant_trading.dashboard.launcher",
        *arguments,
    ]


def _build_server_environment() -> dict[str, str]:
    """讓封裝版背景服務成為獨立 PyInstaller 執行個體。"""
    environment = dict(os.environ)
    apply_thread_environment(
        environment,
        _dashboard_cpu_threads(),
        overwrite=True,
    )
    if is_frozen_app():
        environment["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    return environment


def _start_server_process(root: Path, port: int) -> subprocess.Popen[bytes]:
    """以隱藏子程序啟動 Streamlit，避免跳出命令提示字元。"""
    server_log = (root / "logs" / "streamlit-server.log").open("ab")
    creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    try:
        return subprocess.Popen(
            _build_server_command(port),
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            creationflags=creation_flags,
            env=_build_server_environment(),
        )
    finally:
        server_log.close()


def _startup_refresh_command() -> list[str]:
    """建立一次性啟動更新工作命令。"""
    arguments = ["--startup-refresh-worker"]
    if is_frozen_app():
        return [sys.executable, *arguments]
    return [sys.executable, "-m", "ai_quant_trading.dashboard.launcher", *arguments]


def start_startup_refresh_worker(root: str | Path) -> int:
    """啟動不阻塞介面的行情與新聞更新工作，回傳 PID。"""
    resolved_root = Path(root).resolve()
    _prepare_runtime_directories(resolved_root)
    log_path = resolved_root / "logs" / "startup-refresh.log"
    environment = _build_server_environment()
    creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    with log_path.open("ab") as log_stream:
        process = subprocess.Popen(
            _startup_refresh_command(),
            cwd=resolved_root,
            stdin=subprocess.DEVNULL,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            creationflags=creation_flags,
            env=environment,
            close_fds=True,
        )
    return process.pid


def _wait_until_ready(
    url: str,
    process: subprocess.Popen[bytes],
    timeout_seconds: float = 30.0,
) -> None:
    """等待 Streamlit 健康檢查成功，並提早回報子程序異常。"""
    deadline = time.monotonic() + timeout_seconds
    health_url = f"{url}/_stcore/health"
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"背景服務提前結束，錯誤碼：{return_code}")
        try:
            with urlopen(health_url, timeout=0.5) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.2)
    raise TimeoutError("背景服務啟動逾時")


def _stop_server_process(process: subprocess.Popen[bytes]) -> None:
    """視窗關閉後停止背景服務，不留下 localhost 程序。"""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _run_streamlit_server(root: Path, port: int) -> int:
    """在子程序主執行緒中運行阻塞式 Streamlit 事件迴圈。"""
    _configure_current_process_threads()
    _redirect_frozen_standard_streams(root)
    # 封裝版先載入 Arrow，避免 Streamlit 網路層先載入另一份 OpenSSL。
    import pyarrow
    from streamlit.web import bootstrap

    _ = pyarrow.__version__
    _configure_logging(root, "streamlit-server.log")
    _prepare_runtime_directories(root)
    os.chdir(root)
    entry_script = _streamlit_entry_script(root)
    if not entry_script.exists():
        raise FileNotFoundError(f"找不到 Dashboard 入口：{entry_script}")

    flag_options = {
        "global_developmentMode": False,
        "server_address": "127.0.0.1",
        "server_port": port,
        "server_headless": True,
        "server_fileWatcherType": "none",
        "browser_gatherUsageStats": False,
    }
    bootstrap.load_config_options(flag_options)
    bootstrap.run(str(entry_script), False, [], flag_options)
    return 0


def _run_desktop_app_locked(root: Path) -> int:
    """啟動背景服務並在原生 Windows WebView2 視窗中顯示 App。"""
    import webview

    _configure_logging(root)
    _prepare_runtime_directories(root)
    os.chdir(root)
    port = find_available_port()
    url = f"http://127.0.0.1:{port}"
    process = _start_server_process(root, port)
    try:
        _wait_until_ready(url, process)
        try:
            refresh_pid = start_startup_refresh_worker(root)
            logging.info("啟動資料更新工作，PID=%s", refresh_pid)
        except OSError:
            logging.exception("無法啟動資料更新工作")
        try:
            from ai_quant_trading.dashboard.automation_control import (
                start_enabled_paper_workers,
            )

            worker_pids = start_enabled_paper_workers(
                root,
                root / "data" / "paper_trading",
            )
            if worker_pids:
                logging.info("恢復持續運行模擬機器人，PID=%s", worker_pids)
        except (OSError, ValueError):
            logging.exception("無法恢復持續運行模擬機器人")
        try:
            from ai_quant_trading.operations.supervisor import supervise_testnet_once

            result = supervise_testnet_once(root)
            logging.info("Testnet 恢復檢查：%s", result.message)
        except (OSError, PermissionError, RuntimeError, ValueError):
            logging.exception("無法恢復 Testnet 無人交易")
        logging.info("開啟桌面 App：%s", url)
        webview.settings["ALLOW_DOWNLOADS"] = True
        webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = False
        webview.create_window(
            APP_TITLE,
            url,
            width=1440,
            height=900,
            min_size=(960, 640),
            maximized=True,
            background_color="#F4F7FA",
            text_select=True,
            zoomable=True,
        )
        webview.start(gui="edgechromium", private_mode=True)
        logging.info("桌面 App 已關閉")
        return 0
    finally:
        _stop_server_process(process)


def _run_desktop_app(root: Path) -> int:
    """只允許一份桌面視窗，避免多個 Dashboard 同時消耗 CPU。"""
    _prepare_runtime_directories(root)
    instance_lock = _acquire_single_instance(root)
    if instance_lock is None:
        _show_startup_error("AI Quant Trading System 已經開啟，請切換到既有視窗。")
        return 0
    try:
        return _run_desktop_app_locked(root)
    finally:
        _release_single_instance(instance_lock)


def _show_startup_error(message: str) -> None:
    """以 Windows 對話框顯示啟動錯誤，並保留日誌供排查。"""
    try:
        from tkinter import messagebox

        messagebox.showerror(APP_TITLE, message)
    except Exception:
        logging.exception("無法顯示啟動錯誤對話框")


def _parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--streamlit-server", action="store_true")
    parser.add_argument("--paper-auto-worker", action="store_true")
    parser.add_argument("--live-auto-worker", action="store_true")
    parser.add_argument("--startup-refresh-worker", action="store_true")
    parser.add_argument("--testnet-supervisor", action="store_true")
    parser.add_argument("--watchdog-worker", action="store_true")
    parser.add_argument("--worker-config")
    parser.add_argument("--port", type=int)
    parser.add_argument("--environment", choices=["testnet", "demo", "live"], default="testnet")
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--startup-grace-seconds", type=float, default=180.0)
    parsed, _ = parser.parse_known_args(arguments)
    return parsed


def _run_automation_worker(root: Path, config_path: str | None, worker_type: str) -> int:
    """讓原始碼與封裝版共用同一個獨立無人值守工作入口。"""
    _configure_current_process_threads()
    if not config_path:
        raise ValueError("背景工作缺少設定檔")
    path = Path(config_path).resolve()
    root_path = root.resolve()
    if not path.is_relative_to(root_path / "data"):
        raise ValueError("背景工作設定檔必須位於專案 data 資料夾")
    if not path.exists():
        raise FileNotFoundError(f"找不到背景工作設定：{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    arguments = payload.get("arguments")
    if not isinstance(arguments, list) or not all(isinstance(item, str) for item in arguments):
        raise ValueError("背景工作 arguments 必須是字串陣列")
    allowed_commands = {"auto", "rl-auto"}
    if not arguments or arguments[0] not in allowed_commands:
        raise ValueError("背景工作收到不允許的子命令")
    os.chdir(root)
    if worker_type == "paper":
        from ai_quant_trading.paper_trading.cli import main as worker_main
    else:
        from ai_quant_trading.live_trading.cli import main as worker_main
        arguments = [
            "--storage-root",
            str(root / "data" / "live_trading"),
            *arguments,
        ]
    return worker_main(arguments)


def _run_startup_refresh_worker(root: Path) -> int:
    """執行一次啟動更新，錯誤會寫入狀態檔而不阻擋 App。"""
    _configure_current_process_threads()
    from ai_quant_trading.startup_refresh import run_startup_refresh

    _configure_logging(root, "startup-refresh.log")
    _prepare_runtime_directories(root)
    os.chdir(root)
    result = run_startup_refresh(root)
    logging.info("啟動更新完成：%s", result.get("message", result.get("status")))
    return 0 if result.get("status") not in {"failed"} else 1


def _run_testnet_supervisor(root: Path, interval: float) -> int:
    """執行獨立 Supervisor，只恢復已授權的 Testnet Worker。"""
    _configure_current_process_threads()
    _configure_logging(root, "testnet-supervisor.log")
    _prepare_runtime_directories(root)
    os.chdir(root)
    from ai_quant_trading.operations.supervisor import run_testnet_supervisor

    return run_testnet_supervisor(root, interval_seconds=interval)


def _run_watchdog_worker(
    root: Path,
    *,
    environment: str,
    interval: float,
    port: int | None,
    startup_grace_seconds: float,
) -> int:
    """執行與 Dashboard、Supervisor 分離的 Watchdog 行程。"""
    _configure_current_process_threads()
    _configure_logging(root, "watchdog.log")
    _prepare_runtime_directories(root)
    os.chdir(root)
    from ai_quant_trading.operations.watchdog import main as watchdog_main

    arguments = [
        "--project-root",
        str(root),
        "--environment",
        environment,
        "--interval",
        str(interval),
        "--host",
        "127.0.0.1",
        "--port",
        str(port or 9108),
        "--startup-grace-seconds",
        str(startup_grace_seconds),
        "--only-when-armed",
    ]
    return watchdog_main(arguments)


def main(arguments: list[str] | None = None) -> int:
    """依命令列參數選擇桌面視窗或內部背景服務模式。"""
    root = get_runtime_root()
    options = _parse_arguments(arguments)
    try:
        if options.streamlit_server:
            if options.port is None:
                raise ValueError("背景服務缺少 --port")
            return _run_streamlit_server(root, options.port)
        if options.paper_auto_worker:
            _configure_logging(root, "paper-auto-worker.log")
            return _run_automation_worker(root, options.worker_config, "paper")
        if options.live_auto_worker:
            _configure_logging(root, "live-auto-worker.log")
            return _run_automation_worker(root, options.worker_config, "live")
        if options.startup_refresh_worker:
            return _run_startup_refresh_worker(root)
        if options.testnet_supervisor:
            return _run_testnet_supervisor(root, options.interval)
        if options.watchdog_worker:
            return _run_watchdog_worker(
                root,
                environment=options.environment,
                interval=options.interval,
                port=options.port,
                startup_grace_seconds=options.startup_grace_seconds,
            )
        return _run_desktop_app(root)
    except Exception as exc:
        _configure_logging(root)
        logging.exception("AI Quant Trading System 啟動失敗")
        if not (
            options.streamlit_server
            or options.paper_auto_worker
            or options.live_auto_worker
            or options.startup_refresh_worker
            or options.testnet_supervisor
            or options.watchdog_worker
        ):
            _show_startup_error(f"啟動失敗：{exc}\n\n請查看 logs\\launcher.log")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
