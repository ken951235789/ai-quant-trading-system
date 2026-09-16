"""Live 安全演練測試。"""

from ai_quant_trading.live_trading.drills import (
    run_disconnect_reconnect_drill,
    run_kill_switch_drill,
    run_live_safety_drills,
    run_protection_order_drill,
)


def test_disconnect_reconnect_drill_passes() -> None:
    assert run_disconnect_reconnect_drill().passed


def test_protection_order_drill_passes() -> None:
    assert run_protection_order_drill().passed


def test_kill_switch_drill_passes() -> None:
    assert run_kill_switch_drill().passed


def test_live_gate_stays_locked_without_testnet_evidence(tmp_path) -> None:
    report = run_live_safety_drills(tmp_path)

    assert report["passed"]
    assert report["current_live_gate"]["locked"]
