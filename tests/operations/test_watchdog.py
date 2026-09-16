"""Watchdog 對過期狀態與硬性停機的測試。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from ai_quant_trading.operations import watchdog as watchdog_module
from ai_quant_trading.operations.watchdog import TradingWatchdog, WatchdogSnapshot


class _Session:
    def __init__(self, row) -> None:
        self.row = row

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def scalar(self, _query):
        return self.row


def _watchdog() -> TradingWatchdog:
    watchdog = object.__new__(TradingWatchdog)
    watchdog.environment = "testnet"
    watchdog.reconciliation_stale_seconds = 20 * 60
    watchdog.minimum_disk_free_bytes = 1024
    watchdog.require_stream = True
    watchdog.paths = SimpleNamespace()
    return watchdog


def test_watchdog_rejects_stale_successful_reconciliation() -> None:
    watchdog = _watchdog()
    row = SimpleNamespace(
        consistent=True,
        occurred_at=datetime.now(timezone.utc) - timedelta(minutes=30),
    )
    watchdog.repository = SimpleNamespace(_session_factory=lambda: _Session(row))

    consistent, age = watchdog._latest_reconciliation_status(datetime.now(timezone.utc))

    assert not consistent
    assert age >= 30 * 60


def test_watchdog_activates_halt_for_stream_failure(monkeypatch) -> None:
    watchdog = _watchdog()
    reasons: list[str] = []
    monkeypatch.setattr(
        "ai_quant_trading.operations.watchdog.activate_emergency_halt",
        lambda _paths, reason: reasons.append(reason),
    )
    snapshot = WatchdogSnapshot(
        database_up=True,
        user_stream_up=False,
        user_stream_heartbeat_age=999.0,
        reconciliation_consistent=True,
        reconciliation_age=10.0,
        automation_up=True,
        automation_heartbeat_age=10.0,
        emergency_halt=False,
        disk_free_bytes=4096,
        healthy=False,
        problems=("Binance 私有事件流中斷或心跳逾時",),
    )

    assert watchdog.enforce_emergency_halt(snapshot)
    assert reasons and "私有事件流" in reasons[0]


def test_watchdog_starts_metrics_while_automation_is_standby(tmp_path, monkeypatch) -> None:
    calls: list[str] = []

    class _FakeWatchdog:
        def __init__(self, project_root, environment, **_kwargs) -> None:
            self.paths = SimpleNamespace(
                environment_dir=project_root / "data" / "live_trading" / environment
            )

    class _FakeServer:
        def shutdown(self) -> None:
            calls.append("shutdown")

        def server_close(self) -> None:
            calls.append("close")

    monkeypatch.setattr(watchdog_module, "TradingWatchdog", _FakeWatchdog)
    monkeypatch.setattr(
        watchdog_module,
        "_serve_metrics",
        lambda *_args: calls.append("metrics") or _FakeServer(),
    )
    monkeypatch.setattr(watchdog_module, "automation_autostart_enabled", lambda _path: False)

    def stop_after_first_wait(_seconds: float) -> None:
        calls.append("wait")
        raise KeyboardInterrupt

    monkeypatch.setattr(watchdog_module.time, "sleep", stop_after_first_wait)

    result = watchdog_module.main(
        [
            "--project-root",
            str(tmp_path),
            "--only-when-armed",
            "--port",
            "0",
        ]
    )

    assert result == 0
    assert calls[:2] == ["metrics", "wait"]
    assert calls[-2:] == ["shutdown", "close"]
