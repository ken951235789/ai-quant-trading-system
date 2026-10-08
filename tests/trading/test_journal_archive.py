"""封存後可讀、損壞時拒絕刪檔、重跑不重複封存。"""

import json
import zipfile

import pytest

from ai_quant_trading.trading.journal import TradeJournal
from ai_quant_trading.trading.journal_archive import archive_journal, read_asset


def fixture(root):
    journal = TradeJournal(root)
    key = "a" * 64
    with journal.connect() as db:
        db.execute("INSERT INTO records(id,position_id,kind,timestamp,payload,status) VALUES (?,?,?,?,?,?)",
                   (key, key, "entry", "2026-01-01", "{}", "complete"))
    (root / f"{key}.json").write_text("{}")
    (root / f"{key}.png").write_bytes(b"png-test")
    return journal, key


def test_archive_roundtrip_and_idempotence(tmp_path):
    journal, key = fixture(tmp_path)
    assert archive_journal(journal, keep_recent=0, prune=True)["archived"] == 1
    assert not (tmp_path / f"{key}.png").exists()
    assert read_asset(tmp_path, key) == b"png-test"
    assert archive_journal(journal, keep_recent=0, prune=True)["archived"] == 0
    assert (tmp_path / "journal.sqlite3").exists()


def test_archive_mismatch_preserves_loose_files(tmp_path):
    journal, key = fixture(tmp_path)
    archive_journal(journal, keep_recent=0)
    (tmp_path / f"{key}.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="禁止刪除"):
        archive_journal(journal, keep_recent=0, prune=True)
    assert (tmp_path / f"{key}.png").read_bytes() == b"changed"


def test_corrupt_archive_is_not_displayed(tmp_path):
    journal, key = fixture(tmp_path)
    archive_journal(journal, keep_recent=0, prune=True)
    with zipfile.ZipFile(tmp_path / "archives" / f"{key}.zip", "w") as z:
        z.writestr(f"{key}.png", b"changed")
    with pytest.raises(ValueError):
        read_asset(tmp_path, key)
    with pytest.raises(ValueError):
        read_asset(tmp_path, "../outside")


def test_archive_recovers_receipt_interruption(tmp_path):
    journal, key = fixture(tmp_path)
    archive_journal(journal, keep_recent=0)
    (tmp_path / "archives" / f"{key}.json").unlink()
    archive_journal(journal, keep_recent=0, prune=True)
    assert json.loads((tmp_path / "archives" / f"{key}.json").read_text())["schema"] == "journal_archive_v1"
