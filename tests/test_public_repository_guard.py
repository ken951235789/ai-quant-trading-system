"""公開邊界檢查只回報位置，避免掃描本身洩漏內容。"""

import importlib.util
from pathlib import Path
import subprocess

import pytest


spec = importlib.util.spec_from_file_location(
    "public_guard", Path(__file__).resolve().parents[1] / "scripts/check_public_repository.py",
)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


@pytest.mark.parametrize("name", [".env", "kaggle.json", "model.pt", "quotes.csv", "outputs/run.json"])
def test_private_artifacts_rejected(name):
    assert guard.inspect_file(name, b"fixture")


def test_public_examples_allowed():
    assert not guard.inspect_file(".env.example", b"API_KEY=\nEMAIL=user@example.com\n")
    assert not guard.inspect_file("data/raw/README.md", b"Public schema only")


def test_repository_exchange_credentials_are_empty():
    from dotenv import dotenv_values

    values = dotenv_values(Path(__file__).resolve().parents[1] / ".env.example")
    for environment in ("TESTNET", "DEMO", "LIVE"):
        for kind in ("KEY", "SECRET"):
            assert values[f"BINANCE_{environment}_API_{kind}"] == ""


def test_email_findings_do_not_echo_values():
    address = "fixture" + "@" + "private-mail.org"
    result = guard.inspect_file("note.md", address.encode())
    assert result[0]["rule"] == "non_example_email"
    assert address not in str(result)


def test_personal_paths_and_links_rejected():
    personal = "C:" + "\\Users\\" + "private-person\\file"
    assert guard.inspect_file("note.md", personal.encode())[0]["rule"] == "personal_windows_path"
    assert guard.inspect_file("linked.py", b"target", "120000")


def test_staged_blob_not_replaced_by_safe_worktree(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    file = tmp_path / "note.md"
    address = "fixture" + "@" + "private-mail.org"
    file.write_text(address, encoding="utf-8")
    subprocess.run(["git", "add", "note.md"], cwd=tmp_path, check=True)
    file.write_text("safe worktree", encoding="utf-8")
    records = guard.index_files(tmp_path)
    assert len(records) == 1
    assert guard.inspect_file(*records[0])
