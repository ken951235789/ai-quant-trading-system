from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from ai_quant_trading.dashboard.launcher import (
    _build_server_environment,
    _build_server_command,
    _run_desktop_app,
    _parse_arguments,
    _redirect_frozen_standard_streams,
    _run_automation_worker,
    _stop_server_process,
    _wait_until_ready,
    find_available_port,
)


class DashboardLauncherTest(unittest.TestCase):
    @patch("ai_quant_trading.dashboard.launcher._is_port_available")
    def test_find_available_port_skips_occupied_port(self, mock_available):
        mock_available.side_effect = [False, True]

        self.assertEqual(find_available_port(start=8501, attempts=2), 8502)

    @patch("ai_quant_trading.dashboard.launcher.is_frozen_app", return_value=True)
    @patch("ai_quant_trading.dashboard.launcher.sys.executable", "AIQuantTradingSystem.exe")
    def test_frozen_server_command_reuses_executable(self, _mock_frozen):
        self.assertEqual(
            _build_server_command(8502),
            [
                "AIQuantTradingSystem.exe",
                "--streamlit-server",
                "--port",
                "8502",
            ],
        )

    @patch("ai_quant_trading.dashboard.launcher.is_frozen_app", return_value=True)
    @patch.dict(
        "ai_quant_trading.dashboard.launcher.os.environ",
        {"EXISTING_VALUE": "kept"},
        clear=True,
    )
    def test_frozen_server_resets_pyinstaller_environment(self, _mock_frozen):
        environment = _build_server_environment()

        self.assertEqual(environment["PYINSTALLER_RESET_ENVIRONMENT"], "1")
        self.assertEqual(environment["EXISTING_VALUE"], "kept")
        self.assertEqual(environment["OMP_NUM_THREADS"], "2")
        self.assertEqual(environment["MKL_NUM_THREADS"], "2")

    @patch("ai_quant_trading.dashboard.launcher._show_startup_error")
    @patch("ai_quant_trading.dashboard.launcher._acquire_single_instance", return_value=None)
    def test_second_desktop_instance_is_rejected(self, _mock_lock, mock_error):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temporary:
            result = _run_desktop_app(Path(temporary))

        self.assertEqual(result, 0)
        mock_error.assert_called_once()

    def test_parse_server_arguments(self):
        options = _parse_arguments(["--streamlit-server", "--port", "8510"])

        self.assertTrue(options.streamlit_server)
        self.assertEqual(options.port, 8510)

    def test_parse_startup_refresh_arguments(self):
        options = _parse_arguments(["--startup-refresh-worker"])

        self.assertTrue(options.startup_refresh_worker)

    @patch("ai_quant_trading.dashboard.launcher.is_frozen_app", return_value=True)
    def test_frozen_standard_streams_are_redirected_to_logs(self, _mock_frozen):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch("ai_quant_trading.dashboard.launcher.sys.stdout", None), patch(
                "ai_quant_trading.dashboard.launcher.sys.stderr",
                None,
            ):
                _redirect_frozen_standard_streams(root)

                from ai_quant_trading.dashboard import launcher

                launcher.sys.stdout.write("stdout 測試\n")
                launcher.sys.stderr.write("stderr 測試\n")
                launcher.sys.stdout.close()
                launcher.sys.stderr.close()

            self.assertIn(
                "stdout 測試",
                (root / "logs" / "streamlit-stdout.log").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "stderr 測試",
                (root / "logs" / "streamlit-stderr.log").read_text(encoding="utf-8"),
            )

    @patch("ai_quant_trading.dashboard.launcher.urlopen")
    def test_wait_until_ready_accepts_health_response(self, mock_urlopen):
        response = MagicMock()
        response.status = 200
        mock_urlopen.return_value.__enter__.return_value = response
        process = MagicMock()
        process.poll.return_value = None

        _wait_until_ready("http://127.0.0.1:8502", process, timeout_seconds=1)

        mock_urlopen.assert_called_once_with(
            "http://127.0.0.1:8502/_stcore/health",
            timeout=0.5,
        )

    def test_stop_server_process_terminates_running_child(self):
        process = MagicMock()
        process.poll.return_value = None

        _stop_server_process(process)

        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=5)

    @patch("ai_quant_trading.live_trading.cli.main", return_value=0)
    @patch("ai_quant_trading.dashboard.launcher.os.chdir")
    def test_live_worker_accepts_rl_auto(self, _mock_chdir, mock_main):
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "data" / "worker.json"
            config.parent.mkdir()
            config.write_text(
                json.dumps({"arguments": ["rl-auto", "--download-latest"]}),
                encoding="utf-8",
            )

            result = _run_automation_worker(Path(temporary), str(config), "live")

        self.assertEqual(result, 0)
        mock_main.assert_called_once_with(
            [
                "--storage-root",
                str(Path(temporary) / "data" / "live_trading"),
                "rl-auto",
                "--download-latest",
            ]
        )

    @patch("ai_quant_trading.paper_trading.cli.main", return_value=0)
    @patch("ai_quant_trading.dashboard.launcher.os.chdir")
    def test_paper_worker_accepts_rl_auto(self, _mock_chdir, mock_main):
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "data" / "worker.json"
            config.parent.mkdir()
            config.write_text(
                json.dumps({"arguments": ["rl-auto", "--download-latest"]}),
                encoding="utf-8",
            )

            result = _run_automation_worker(Path(temporary), str(config), "paper")

        self.assertEqual(result, 0)
        mock_main.assert_called_once_with(["rl-auto", "--download-latest"])


if __name__ == "__main__":
    unittest.main()
