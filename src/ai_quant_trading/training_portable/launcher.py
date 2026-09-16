"""以原生 WebView 視窗啟動可攜式訓練中心。"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

from ai_quant_trading.performance import apply_thread_environment


APP_TITLE = "AI Quant 獨立訓練中心"


def _root() -> Path:
    return Path(os.environ.get("AI_QUANT_TRAINING_ROOT", Path.cwd())).resolve()


def _available_port(start: int = 8511) -> int:
    for port in range(start, start + 30):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
        return port
    raise RuntimeError("找不到可用的本機連接埠")


def _wait_ready(url: str, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"訓練中心背景服務提前結束：{process.returncode}")
        try:
            with urlopen(f"{url}/_stcore/health", timeout=0.5) as response:  # nosec B310
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.25)
    raise TimeoutError("訓練中心啟動逾時")


def main() -> int:
    """啟動 Streamlit 子程序，並在視窗關閉時一併停止服務。"""
    import webview

    root = _root()
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=logs / "training-launcher.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        encoding="utf-8",
        force=True,
    )
    entry = root / "training_app.py"
    if not entry.is_file():
        raise FileNotFoundError(f"找不到訓練入口：{entry}")
    port = _available_port()
    url = f"http://127.0.0.1:{port}"
    environment = dict(os.environ)
    apply_thread_environment(environment, 2, overwrite=True)
    environment["AI_QUANT_TRAINING_ROOT"] = str(root)
    environment["PYTHONPATH"] = str(root / "src")
    creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    with (logs / "training-server.log").open("ab") as output:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(entry),
                "--server.address=127.0.0.1",
                f"--server.port={port}",
                "--server.headless=true",
                "--server.fileWatcherType=none",
                "--browser.gatherUsageStats=false",
            ],
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            creationflags=creation_flags,
            env=environment,
        )
    try:
        _wait_ready(url, process)
        webview.settings["ALLOW_DOWNLOADS"] = True
        webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = False
        webview.create_window(
            APP_TITLE,
            url,
            width=1440,
            height=900,
            min_size=(960, 640),
            maximized=True,
            background_color="#f5f7f9",
            text_select=True,
            zoomable=True,
        )
        webview.start(gui="edgechromium", private_mode=True)
        return 0
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
