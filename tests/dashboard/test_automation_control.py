from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from ai_quant_trading.dashboard.automation_control import (
    _portable_worker_arguments,
    automation_autostart_enabled,
    set_automation_autostart,
    start_automation_worker,
    start_enabled_paper_workers,
)


class DashboardAutomationControlTest(unittest.TestCase):
    def test_worker_arguments_rebase_old_project_data_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "portable_app"
            arguments = _portable_worker_arguments(
                root,
                [
                    "rl-auto",
                    "--model-dir",
                    str(Path(temporary) / "old_app" / "data" / "models" / "sac"),
                    "--raw-dir",
                    str(root / "data" / "raw"),
                    "--symbol",
                    "BTC/USDT",
                ],
            )

        self.assertEqual(arguments[2], str(Path("data") / "models" / "sac"))
        self.assertEqual(arguments[4], str(Path("data") / "raw"))
        self.assertEqual(arguments[6], "BTC/USDT")

    def test_autostart_setting_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            automation_dir = Path(temporary) / "automation"

            set_automation_autostart(automation_dir, True)

            self.assertTrue(automation_autostart_enabled(automation_dir))
            set_automation_autostart(automation_dir, False)
            self.assertFalse(automation_autostart_enabled(automation_dir))

    @patch(
        "ai_quant_trading.dashboard.automation_control.start_automation_worker",
        return_value=8765,
    )
    def test_enabled_websocket_account_is_restored_once(
        self,
        mock_start: MagicMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            automation_dir = root / "data" / "paper_trading" / "btc" / "automation"
            set_automation_autostart(automation_dir, True)
            (automation_dir / "worker_config.json").write_text(
                json.dumps(
                    {
                        "arguments": [
                            "rl-auto",
                            "--account",
                            "btc",
                            "--transport",
                            "websocket",
                        ]
                    }
                ),
                encoding="utf-8",
            )

            pids = start_enabled_paper_workers(
                root,
                root / "data" / "paper_trading",
            )

        self.assertEqual(pids, (8765,))
        mock_start.assert_called_once()

    @patch("ai_quant_trading.dashboard.automation_control.is_frozen_app", return_value=False)
    @patch("ai_quant_trading.dashboard.automation_control.subprocess.Popen")
    def test_start_worker_immediately_records_running_status(
        self,
        mock_popen: MagicMock,
        _mock_frozen: MagicMock,
    ) -> None:
        process = MagicMock()
        process.pid = 4321
        mock_popen.return_value = process

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            automation_dir = root / "data" / "paper_trading" / "demo" / "automation"

            pid = start_automation_worker(
                worker_type="paper",
                project_root=root,
                automation_dir=automation_dir,
                arguments=["rl-auto", "--account", "demo"],
            )

            status = json.loads(
                (automation_dir / "status.json").read_text(encoding="utf-8")
            )

        self.assertEqual(pid, 4321)
        self.assertEqual(status["state"], "running")
        self.assertEqual(status["pid"], 4321)
        self.assertEqual(status["last_message"], "背景程序啟動中")
        self.assertIsNone(status["last_error"])


if __name__ == "__main__":
    unittest.main()
