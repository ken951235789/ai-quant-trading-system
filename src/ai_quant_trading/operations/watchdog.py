"""不依附 Dashboard 的交易系統 Watchdog 與 Prometheus 指標端點。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import shutil
from threading import Thread
import time
from typing import Any

from ai_quant_trading.automation import automation_paths, read_automation_status
from ai_quant_trading.dashboard.automation_control import automation_autostart_enabled
from ai_quant_trading.database.config import DatabaseSettings
from ai_quant_trading.database.engine import check_database_health
from ai_quant_trading.database.repository import repository_from_settings
from ai_quant_trading.live_trading.notification import notify_event
from ai_quant_trading.live_trading.storage import activate_emergency_halt, live_trading_paths
from ai_quant_trading.runtime_secrets import runtime_env_path


def _age_seconds(value: datetime | str | None, now: datetime) -> float:
    if value in {None, ""}:
        return float("inf")
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        parsed = value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max((now - parsed).total_seconds(), 0.0)


@dataclass(frozen=True, slots=True)
class WatchdogSnapshot:
    """單次巡檢結果；刻意不包含資產與持倉金額。"""

    database_up: bool
    user_stream_up: bool
    user_stream_heartbeat_age: float
    reconciliation_consistent: bool
    reconciliation_age: float
    automation_up: bool
    automation_heartbeat_age: float
    emergency_halt: bool
    disk_free_bytes: int
    healthy: bool
    problems: tuple[str, ...]

    def prometheus_text(self) -> str:
        """輸出 Prometheus exposition format。"""
        values = {
            "ai_quant_watchdog_up": 1,
            "ai_quant_database_up": int(self.database_up),
            "ai_quant_user_stream_up": int(self.user_stream_up),
            "ai_quant_user_stream_heartbeat_age_seconds": self.user_stream_heartbeat_age,
            "ai_quant_reconciliation_consistent": int(self.reconciliation_consistent),
            "ai_quant_reconciliation_age_seconds": self.reconciliation_age,
            "ai_quant_automation_up": int(self.automation_up),
            "ai_quant_automation_heartbeat_age_seconds": self.automation_heartbeat_age,
            "ai_quant_emergency_halt_active": int(self.emergency_halt),
            "ai_quant_disk_free_bytes": self.disk_free_bytes,
            "ai_quant_system_healthy": int(self.healthy),
        }
        lines = ["# AI Quant Trading System operational metrics"]
        for name, value in values.items():
            rendered = "+Inf" if value == float("inf") else str(value)
            lines.append(f"{name} {rendered}")
        return "\n".join(lines) + "\n"


class TradingWatchdog:
    """檢查資料庫、User Stream、對帳、Worker、停機旗標與磁碟。"""

    def __init__(
        self,
        project_root: Path,
        environment: str,
        *,
        stream_stale_seconds: float = 180.0,
        automation_stale_seconds: float = 180.0,
        reconciliation_stale_seconds: float = 20 * 60.0,
        minimum_disk_free_bytes: int = 5 * 1024**3,
        require_stream: bool = True,
        require_automation: bool = True,
        env_file: Path | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.environment = environment
        self.stream_stale_seconds = stream_stale_seconds
        self.automation_stale_seconds = automation_stale_seconds
        self.reconciliation_stale_seconds = reconciliation_stale_seconds
        self.minimum_disk_free_bytes = minimum_disk_free_bytes
        self.require_stream = require_stream
        self.require_automation = require_automation
        self.paths = live_trading_paths(
            self.project_root / "data" / "live_trading",
            environment,
        )
        self.env_file = env_file or runtime_env_path(self.project_root)
        settings = DatabaseSettings.from_environment(
            self.env_file
        )
        if not settings.enabled:
            raise ValueError("Watchdog 必須使用 PostgreSQL 交易狀態中心")
        self.repository = repository_from_settings(settings)
        self._latest_snapshot: WatchdogSnapshot | None = None

    def inspect(self) -> WatchdogSnapshot:
        """執行一次唯讀巡檢。"""
        now = datetime.now(timezone.utc)
        database_up, _ = check_database_health(self.repository.engine)
        checkpoint = None
        if database_up:
            checkpoint = self.repository.stream_checkpoint(
                self.environment,
                "usd_m_user_data",
            )
        stream_age = _age_seconds(
            checkpoint.get("heartbeat_at") if checkpoint else None,
            now,
        )
        stream_up = bool(
            checkpoint
            and checkpoint.get("state") == "connected"
            and stream_age <= self.stream_stale_seconds
        )
        reconciliation_consistent, reconciliation_age = (
            self._latest_reconciliation_status(now) if database_up else (False, float("inf"))
        )
        automation = read_automation_status(
            automation_paths(self.paths.environment_dir / "automation")
        )
        automation_age = _age_seconds(automation.heartbeat_at, now)
        automation_up = automation.state == "running" and automation_age <= self.automation_stale_seconds
        emergency_halt = self.paths.emergency_halt_json.exists()
        disk_free = shutil.disk_usage(self.project_root).free
        problems: list[str] = []
        if not database_up:
            problems.append("PostgreSQL 無法連線")
        if self.require_stream and not stream_up:
            problems.append("Binance 私有事件流中斷或心跳逾時")
        if not reconciliation_consistent:
            problems.append("最近一次交易所對帳不一致、尚未完成或已過期")
        if self.require_automation and not automation_up:
            problems.append("無人交易 Worker 未執行或心跳逾時")
        if emergency_halt:
            problems.append("緊急停機旗標已啟用")
        if disk_free < self.minimum_disk_free_bytes:
            problems.append("交易資料磁碟可用空間不足")
        snapshot = WatchdogSnapshot(
            database_up,
            stream_up,
            stream_age,
            reconciliation_consistent,
            reconciliation_age,
            automation_up,
            automation_age,
            emergency_halt,
            disk_free,
            not problems,
            tuple(problems),
        )
        self._latest_snapshot = snapshot
        return snapshot

    def _latest_reconciliation_status(self, now: datetime) -> tuple[bool, float]:
        from sqlalchemy import select

        from ai_quant_trading.database.models import ReconciliationRun

        with self.repository._session_factory() as session:
            row = session.scalar(
                select(ReconciliationRun)
                .where(ReconciliationRun.environment == self.environment)
                .order_by(ReconciliationRun.id.desc())
                .limit(1)
            )
            age = _age_seconds(row.occurred_at if row else None, now)
            return bool(
                row
                and row.consistent
                and age <= self.reconciliation_stale_seconds
            ), age

    def enforce_emergency_halt(self, snapshot: WatchdogSnapshot) -> bool:
        """重大基礎設施異常時鎖住新增風險；恢復後仍需人工解除。"""
        critical: list[str] = []
        if not snapshot.database_up:
            critical.append("PostgreSQL 無法連線")
        if self.require_stream and not snapshot.user_stream_up:
            critical.append("Binance 私有事件流中斷或心跳逾時")
        if not snapshot.reconciliation_consistent:
            critical.append("交易所對帳不一致或已過期")
        if snapshot.disk_free_bytes < self.minimum_disk_free_bytes:
            critical.append("交易資料磁碟可用空間不足")
        if not critical:
            return False
        activate_emergency_halt(self.paths, "Watchdog：" + "；".join(critical))
        return True

    @property
    def latest_snapshot(self) -> WatchdogSnapshot:
        return self._latest_snapshot or self.inspect()


def _serve_metrics(watchdog: TradingWatchdog, host: str, port: int) -> ThreadingHTTPServer:
    token = os.getenv("AI_QUANT_METRICS_TOKEN", "").strip()
    token_file = watchdog.project_root / "monitoring" / "secrets" / "metrics_token"
    if not token and token_file.is_file():
        token = token_file.read_text(encoding="utf-8").strip()
    if host not in {"127.0.0.1", "localhost", "::1"} and len(token) < 32:
        raise ValueError("非 loopback 指標端點必須設定至少 32 字元 AI_QUANT_METRICS_TOKEN")

    class MetricsHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/metrics":
                self.send_error(404)
                return
            if token and self.headers.get("Authorization") != f"Bearer {token}":
                self.send_error(401)
                return
            payload = watchdog.latest_snapshot.prometheus_text().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, _format: str, *_args: Any) -> None:
            return

    server = ThreadingHTTPServer((host, port), MetricsHandler)
    Thread(target=server.serve_forever, daemon=True, name="watchdog-metrics").start()
    return server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI 量化交易獨立 Watchdog")
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--environment", choices=["testnet", "demo", "live"], default="testnet")
    parser.add_argument("--env-file")
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9108)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--only-when-armed", action="store_true")
    parser.add_argument("--startup-grace-seconds", type=float, default=180.0)
    parser.add_argument("--no-require-stream", action="store_true")
    parser.add_argument("--no-require-automation", action="store_true")
    return parser


def main(arguments: list[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    root = Path(args.project_root)
    watchdog = TradingWatchdog(
        root,
        args.environment,
        require_stream=not args.no_require_stream,
        require_automation=not args.no_require_automation,
        env_file=Path(args.env_file) if args.env_file else None,
    )
    if args.once:
        snapshot = watchdog.inspect()
        print("正常" if snapshot.healthy else "異常：" + "；".join(snapshot.problems))
        return 0 if snapshot.healthy else 2
    # 即使 Testnet 尚未授權自動交易，也先提供健康端點，讓外部監控能辨識待命狀態。
    server = _serve_metrics(watchdog, args.host, args.port)
    previous: tuple[str, ...] | None = None
    armed_at: float | None = None
    try:
        while True:
            armed = not args.only_when_armed or automation_autostart_enabled(
                watchdog.paths.environment_dir / "automation"
            )
            if not armed:
                armed_at = None
                previous = None
                time.sleep(max(args.interval, 5.0))
                continue
            if armed_at is None:
                armed_at = time.monotonic()
            if time.monotonic() - armed_at < max(args.startup_grace_seconds, 0.0):
                time.sleep(max(args.interval, 5.0))
                continue
            snapshot = watchdog.inspect()
            watchdog.enforce_emergency_halt(snapshot)
            if snapshot.problems != previous:
                level = "info" if snapshot.healthy else "critical"
                message = "Watchdog 恢復正常" if snapshot.healthy else "；".join(snapshot.problems)
                notify_event(
                    watchdog.paths.notifications_log,
                    level,
                    message,
                    env_path=watchdog.env_file,
                )
                previous = snapshot.problems
            time.sleep(max(args.interval, 5.0))
    except KeyboardInterrupt:
        return 0
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
