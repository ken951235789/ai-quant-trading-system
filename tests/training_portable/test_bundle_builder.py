"""獨立訓練包建置測試。"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_builder():
    path = PROJECT_ROOT / "scripts" / "build_training_bundle.py"
    spec = importlib.util.spec_from_file_location("build_training_bundle", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_builder_creates_training_only_runtime(tmp_path: Path) -> None:
    builder = _load_builder()
    output = tmp_path / "AIQuantTrainingPortable"

    assert builder.main(["--output", str(output), "--no-include-data", "--no-zip"]) == 0
    assert (output / "啟動訓練中心.cmd").is_file()
    assert (output / "安裝訓練環境.cmd").is_file()
    assert (output / "setup_training_env.ps1").read_bytes().startswith(b"\xef\xbb\xbf")
    assert (output / "啟動訓練中心.cmd").read_bytes().startswith(b"\xef\xbb\xbf")
    assert (output / "src" / "ai_quant_trading" / "training_portable" / "app.py").is_file()
    assert (output / "src" / "ai_quant_trading" / "performance.py").is_file()
    assert (output / "src" / "ai_quant_trading" / "persistence.py").is_file()
    assert (output / "src" / "ai_quant_trading" / "operations" / "integrity.py").is_file()
    assert (output / "src" / "ai_quant_trading" / "trading" / "model_guard.py").is_file()
    assert not any((output / "src").rglob("__pycache__"))
    assert not any((output / "src").rglob("*.pyc"))
    assert not (output / "src" / "ai_quant_trading" / "live_trading").exists()
    paper_support = output / "src" / "ai_quant_trading" / "paper_trading"
    assert (paper_support / "storage.py").is_file()
    assert not (paper_support / "engine.py").exists()
    assert not (paper_support / "realtime.py").exists()

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(output / "src")
    completed = subprocess.run(
        [sys.executable, "-c", "import ai_quant_trading.training_portable.app"],
        cwd=output,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_builder_refuses_to_delete_unmarked_custom_directory(tmp_path: Path) -> None:
    builder = _load_builder()
    output = tmp_path / "important-files"
    output.mkdir()
    (output / "keep.txt").write_text("do not delete", encoding="utf-8")

    with pytest.raises(ValueError, match="拒絕覆蓋"):
        builder.main(["--output", str(output), "--no-zip"])

    assert (output / "keep.txt").read_text(encoding="utf-8") == "do not delete"
