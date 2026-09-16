"""交易稽核雜湊鏈測試。"""

from __future__ import annotations

import errno
import json

import ai_quant_trading.live_trading.audit as audit_module
from ai_quant_trading.live_trading.audit import append_audit_event, verify_audit_log


def test_audit_chain_accepts_ordered_events(tmp_path) -> None:
    path = tmp_path / "audit.jsonl"
    first = append_audit_event(path, "decision", {"target": 0.25})
    second = append_audit_event(path, "order", {"side": "BUY"}, severity="warning")

    result = verify_audit_log(path)

    assert result.valid
    assert result.event_count == 2
    assert second["previous_hash"] == first["event_hash"]
    assert result.last_hash == second["event_hash"]


def test_audit_chain_detects_modified_history(tmp_path) -> None:
    path = tmp_path / "audit.jsonl"
    append_audit_event(path, "decision", {"target": 0.25})
    append_audit_event(path, "order", {"side": "BUY"})
    lines = path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["details"]["target"] = 0.99
    lines[0] = json.dumps(first, ensure_ascii=False, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = verify_audit_log(path)

    assert not result.valid
    assert "可能已被修改" in result.reasons[0]


def test_audit_lock_retries_transient_windows_access_denied(tmp_path, monkeypatch) -> None:
    path = tmp_path / "audit.jsonl"
    real_open = audit_module.os.open
    attempts = 0

    def transient_open(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError(errno.EACCES, "temporary sharing conflict")
        return real_open(*args, **kwargs)

    monkeypatch.setattr(audit_module.os, "name", "nt")
    monkeypatch.setattr(audit_module.os, "open", transient_open)

    event = append_audit_event(path, "decision", {"target": 0.25})

    assert attempts >= 2
    assert event["event_type"] == "decision"
