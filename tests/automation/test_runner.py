"""心跳、停止檔、鎖與錯誤熔斷測試。"""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from dataclasses import asdict

from ai_quant_trading.automation import (
    AutomationConfig,
    automation_paths,
    read_automation_status,
    request_automation_stop,
    run_automation,
)
from ai_quant_trading.automation.runner import AutomationStatus, _write_json_atomic


class AutomationRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = self.enterContext(tempfile.TemporaryDirectory())
        self.paths = automation_paths(self.temp_dir)

    def test_runs_requested_cycles_and_persists_heartbeat(self) -> None:
        calls = []

        status = run_automation(
            lambda: calls.append(len(calls)) or "完成",
            self.paths,
            AutomationConfig(poll_seconds=0, max_cycles=3),
        )

        self.assertEqual(status.state, "completed")
        self.assertEqual(status.successful_cycles, 3)
        self.assertEqual(len(calls), 3)
        self.assertEqual(read_automation_status(self.paths).state, "completed")
        self.assertFalse(self.paths.lock_file.exists())

    def test_cycle_can_request_safe_stop(self) -> None:
        def cycle():
            request_automation_stop(self.paths)
            return "已請求停止"

        status = run_automation(
            cycle,
            self.paths,
            AutomationConfig(poll_seconds=0),
        )

        self.assertEqual(status.state, "stopped")
        self.assertEqual(status.successful_cycles, 1)

    def test_consecutive_errors_trip_circuit_breaker(self) -> None:
        def broken_cycle():
            raise ValueError("資料源失敗")

        status = run_automation(
            broken_cycle,
            self.paths,
            AutomationConfig(
                poll_seconds=0,
                max_consecutive_errors=2,
            ),
        )

        self.assertEqual(status.state, "failed")
        self.assertEqual(status.attempted_cycles, 2)
        self.assertIn("ValueError", status.last_error or "")

    def test_stop_request_records_stopping_state(self) -> None:
        running = AutomationStatus(state="running", pid=os.getpid())
        _write_json_atomic(self.paths.status_json, asdict(running))

        request_automation_stop(self.paths)

        status = read_automation_status(self.paths)
        self.assertEqual(status.state, "stopping")
        self.assertTrue(self.paths.stop_file.exists())
        self.assertIn("等待", status.last_message)

    def test_active_process_lock_blocks_second_runner(self) -> None:
        self.paths.root_dir.mkdir(parents=True, exist_ok=True)
        self.paths.lock_file.write_text(str(os.getpid()), encoding="utf-8")

        with self.assertRaisesRegex(RuntimeError, "已在執行"):
            run_automation(
                lambda: None,
                self.paths,
                AutomationConfig(poll_seconds=0, max_cycles=1),
            )

    def test_status_write_retries_windows_access_denied(self) -> None:
        real_replace = os.replace
        attempts = 0

        def flaky_replace(source, destination):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise PermissionError(5, "存取被拒", str(destination))
            real_replace(source, destination)

        with patch(
            "ai_quant_trading.automation.runner.os.replace",
            side_effect=flaky_replace,
        ):
            _write_json_atomic(self.paths.status_json, {"state": "running"})

        self.assertEqual(attempts, 3)
        self.assertEqual(read_automation_status(self.paths).state, "running")
        self.assertFalse(list(self.paths.root_dir.glob(".status.json.*.tmp")))


if __name__ == "__main__":
    unittest.main()
