"""Testnet 開機恢復 Supervisor 的安全邊界測試。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_quant_trading.dashboard.automation_control import set_automation_autostart
from ai_quant_trading.live_trading.storage import activate_emergency_halt, live_trading_paths
from ai_quant_trading.operations.supervisor import (
    load_authorized_testnet_arguments,
    supervise_testnet_once,
)


def _arguments(environment: str = "testnet") -> list[str]:
    return [
        "rl-auto",
        "--environment",
        environment,
        "--env-file",
        ".env",
        "--training-dir",
        "data/models/sac",
        "--download-latest",
        "--execute",
        "--confirm",
        f"EXECUTE {environment.upper()}",
    ]


def _configure(root: Path, arguments: list[str] | None = None) -> Path:
    paths = live_trading_paths(root / "data" / "live_trading", "testnet")
    automation_dir = paths.environment_dir / "automation"
    set_automation_autostart(automation_dir, True)
    (automation_dir / "worker_config.json").write_text(
        json.dumps({"arguments": arguments or _arguments()}),
        encoding="utf-8",
    )
    return automation_dir


def test_authorization_rejects_live_configuration(tmp_path: Path) -> None:
    automation_dir = _configure(tmp_path, _arguments("live"))

    with pytest.raises(ValueError, match="只能使用 Testnet"):
        load_authorized_testnet_arguments(automation_dir / "worker_config.json")


def test_authorization_requires_explicit_execution(tmp_path: Path) -> None:
    arguments = _arguments()
    arguments.remove("--execute")
    automation_dir = _configure(tmp_path, arguments)

    with pytest.raises(ValueError, match="明確授權"):
        load_authorized_testnet_arguments(automation_dir / "worker_config.json")


def test_supervisor_stays_idle_until_user_arms_it(tmp_path: Path) -> None:
    result = supervise_testnet_once(tmp_path)

    assert result.action == "idle"


def test_supervisor_does_not_start_during_emergency_halt(tmp_path: Path) -> None:
    _configure(tmp_path)
    paths = live_trading_paths(tmp_path / "data" / "live_trading", "testnet")
    activate_emergency_halt(paths, "演練停機")

    result = supervise_testnet_once(tmp_path)

    assert result.action == "blocked"


def test_supervisor_starts_authorized_testnet_worker(monkeypatch, tmp_path: Path) -> None:
    _configure(tmp_path)
    (tmp_path / ".env").write_text(
        "BINANCE_TESTNET_API_KEY=test-key\nBINANCE_TESTNET_API_SECRET=test-secret\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "ai_quant_trading.operations.supervisor.credentials_available",
        lambda _environment, _path: True,
    )
    launched: list[list[str]] = []
    monkeypatch.setattr(
        "ai_quant_trading.operations.supervisor.start_automation_worker",
        lambda _kind, _root, _automation, arguments: launched.append(arguments) or 4321,
    )

    result = supervise_testnet_once(tmp_path, env_file=tmp_path / ".env")

    assert result.action == "started"
    assert result.pid == 4321
    assert launched == [_arguments()]
