"""只負責恢復已由使用者授權的 Binance Testnet 無人交易 Worker。"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time

from ai_quant_trading.automation import automation_paths, read_automation_status
from ai_quant_trading.dashboard.automation_control import (
    automation_autostart_enabled,
    start_automation_worker,
)
from ai_quant_trading.live_trading.credentials import credentials_available
from ai_quant_trading.live_trading.notification import notify_event
from ai_quant_trading.live_trading.storage import (
    activate_emergency_halt,
    emergency_halt_reason,
    live_trading_paths,
)
from ai_quant_trading.runtime_secrets import runtime_env_path


@dataclass(frozen=True, slots=True)
class SupervisorResult:
    """單次 Supervisor 巡檢結果，不包含 API Key 或資產金額。"""

    action: str
    message: str
    pid: int | None = None


def _argument_value(arguments: list[str], flag: str) -> str | None:
    """取得單一命令列參數值；重複或缺值時視為不合法。"""
    indexes = [index for index, item in enumerate(arguments) if item == flag]
    if len(indexes) != 1:
        return None
    index = indexes[0] + 1
    if index >= len(arguments) or arguments[index].startswith("--"):
        return None
    return arguments[index]


def load_authorized_testnet_arguments(config_path: Path) -> list[str]:
    """讀取並驗證 Testnet 自動恢復設定，禁止藉設定檔切到正式環境。"""
    import json

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError("尚未保存 Testnet Worker 設定") from exc
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Testnet Worker 設定檔無法讀取") from exc
    arguments = payload.get("arguments")
    if not isinstance(arguments, list) or not all(
        isinstance(item, str) for item in arguments
    ):
        raise ValueError("Testnet Worker arguments 必須是字串陣列")
    if not arguments or arguments[0] != "rl-auto":
        raise ValueError("Testnet 自動恢復只允許 RL 無人交易")
    if _argument_value(arguments, "--environment") != "testnet":
        raise ValueError("自動恢復設定只能使用 Testnet")
    if "--execute" not in arguments:
        raise ValueError("自動恢復只接受已明確授權送到 Testnet 的設定")
    if _argument_value(arguments, "--confirm") != "EXECUTE TESTNET":
        raise ValueError("Testnet 確認文字不正確")
    if "--download-latest" not in arguments:
        raise ValueError("Testnet 無人交易必須使用最新市場資料")
    if "--disable-exchange-protection" in arguments:
        raise ValueError("Testnet 無人交易不可停用交易所保護單")
    if _argument_value(arguments, "--market-type") not in {None, "usd_m_futures"}:
        raise ValueError("Testnet 自動恢復只支援 USD-M Futures")
    return arguments


def supervise_testnet_once(
    project_root: str | Path,
    *,
    env_file: str | Path | None = None,
) -> SupervisorResult:
    """在安全條件成立時恢復 Testnet Worker；未授權時保持待命。"""
    root = Path(project_root).resolve()
    live_paths = live_trading_paths(root / "data" / "live_trading", "testnet")
    runner_paths = automation_paths(live_paths.environment_dir / "automation")
    if not automation_autostart_enabled(runner_paths.root_dir):
        return SupervisorResult("idle", "Testnet 開機自動恢復尚未啟用")
    halt = emergency_halt_reason(live_paths)
    if halt:
        return SupervisorResult("blocked", f"Testnet 緊急停機中：{halt}")

    status = read_automation_status(runner_paths)
    if status.state in {"running", "stopping"}:
        return SupervisorResult("healthy", "Testnet Worker 已在執行", status.pid)

    arguments = load_authorized_testnet_arguments(
        runner_paths.root_dir / "worker_config.json"
    )
    configured_env = _argument_value(arguments, "--env-file")
    secret_path = Path(configured_env) if configured_env else Path(
        env_file or runtime_env_path(root)
    )
    if not secret_path.is_absolute():
        secret_path = root / secret_path
    if not credentials_available("testnet", secret_path):
        raise ValueError("尚未設定 Binance Testnet API Key 與 Secret")

    pid = start_automation_worker(
        "live",
        root,
        runner_paths.root_dir,
        arguments,
    )
    return SupervisorResult("started", "Testnet Worker 已自動恢復", pid)


def run_testnet_supervisor(
    project_root: str | Path,
    *,
    interval_seconds: float = 30.0,
    restart_limit: int = 3,
    restart_window_minutes: float = 60.0,
    once: bool = False,
) -> int:
    """持續監督 Testnet Worker；重複崩潰時鎖住新增風險。"""
    if interval_seconds < 5:
        raise ValueError("Supervisor 巡檢間隔不可小於 5 秒")
    if restart_limit < 1:
        raise ValueError("Supervisor 重啟上限必須大於 0")
    if restart_window_minutes <= 0:
        raise ValueError("Supervisor 重啟觀察期間必須大於 0")

    root = Path(project_root).resolve()
    paths = live_trading_paths(root / "data" / "live_trading", "testnet")
    env_file = runtime_env_path(root)
    launches: deque[datetime] = deque()
    previous: tuple[str, str] | None = None
    while True:
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=restart_window_minutes)
        while launches and launches[0] < cutoff:
            launches.popleft()
        try:
            runner_paths = automation_paths(paths.environment_dir / "automation")
            armed = automation_autostart_enabled(runner_paths.root_dir)
            status = read_automation_status(runner_paths)
            needs_launch = armed and status.state not in {"running", "stopping"}
            halt = emergency_halt_reason(paths)
            if halt:
                result = SupervisorResult("blocked", f"Testnet 緊急停機中：{halt}")
            elif needs_launch and len(launches) >= restart_limit:
                reason = (
                    f"Supervisor 在 {restart_window_minutes:g} 分鐘內已啟動 Worker "
                    f"{len(launches)} 次，已停止自動恢復"
                )
                activate_emergency_halt(paths, reason)
                result = SupervisorResult("blocked", reason)
            else:
                result = supervise_testnet_once(root, env_file=env_file)
                if result.action == "started":
                    launches.append(now)
        except (OSError, PermissionError, RuntimeError, ValueError) as exc:
            result = SupervisorResult(
                "error",
                f"Testnet Supervisor：{type(exc).__name__}：{exc}",
            )

        signature = (result.action, result.message)
        if signature != previous:
            level = "critical" if result.action in {"blocked", "error"} else "info"
            notify_event(
                paths.notifications_log,
                level,
                result.message,
                env_path=env_file,
            )
            previous = signature
        if once:
            print(result.message)
            return 2 if result.action in {"blocked", "error"} else 0
        time.sleep(interval_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Binance Testnet 開機自動恢復 Supervisor")
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--restart-limit", type=int, default=3)
    parser.add_argument("--restart-window-minutes", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    return parser


def main(arguments: list[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    return run_testnet_supervisor(
        args.project_root,
        interval_seconds=args.interval,
        restart_limit=args.restart_limit,
        restart_window_minutes=args.restart_window_minutes,
        once=args.once,
    )


if __name__ == "__main__":
    raise SystemExit(main())
