"""每日營運報告與實盤前證據測試。"""

from __future__ import annotations

from datetime import date

import pytest

from ai_quant_trading.live_trading.audit import append_audit_event
from ai_quant_trading.live_trading.reporting import (
    build_daily_trading_report,
    save_daily_trading_report,
)
from ai_quant_trading.live_trading.storage import (
    CYCLE_COLUMNS,
    ORDER_COLUMNS,
    SNAPSHOT_COLUMNS,
    append_csv_row,
    live_trading_paths,
)


def test_daily_report_calculates_pnl_and_saves_both_formats(tmp_path) -> None:
    paths = live_trading_paths(tmp_path, "testnet")
    for timestamp, equity in (
        ("2026-08-23T00:00:00Z", 1_000),
        ("2026-08-23T01:00:00Z", 980),
        ("2026-08-23T02:00:00Z", 1_010),
    ):
        append_csv_row(
            paths.account_snapshots_csv,
            SNAPSHOT_COLUMNS,
            {"timestamp": timestamp, "estimated_equity": equity},
        )
    append_csv_row(
        paths.cycles_csv,
        CYCLE_COLUMNS,
        {"timestamp": "2026-08-23T01:00:00Z", "status": "hold"},
    )
    append_csv_row(
        paths.orders_csv,
        ORDER_COLUMNS,
        {
            "timestamp": "2026-08-23T01:00:00Z",
            "client_order_id": "aq1",
            "validated_only": False,
            "status": "FILLED",
        },
    )
    append_audit_event(paths.audit_jsonl, "order", {"client_order_id": "aq1"})

    report = build_daily_trading_report(paths, report_date=date(2026, 8, 23))
    json_path, markdown_path, saved = save_daily_trading_report(
        paths, report_date=date(2026, 8, 23)
    )

    assert report.pnl == 10
    assert report.return_rate == 0.01
    assert report.maximum_drawdown == pytest.approx(-0.02)
    assert report.submitted_orders == 1
    assert report.status == "normal"
    assert saved == report
    assert json_path.is_file() and markdown_path.is_file()
    assert "每日交易營運報告" in markdown_path.read_text(encoding="utf-8")


def test_daily_report_flags_unknown_order(tmp_path) -> None:
    paths = live_trading_paths(tmp_path, "testnet")
    append_csv_row(
        paths.orders_csv,
        ORDER_COLUMNS,
        {
            "timestamp": "2026-08-23T01:00:00Z",
            "client_order_id": "aq-unknown",
            "validated_only": False,
            "status": "UNKNOWN",
        },
    )

    report = build_daily_trading_report(paths, report_date=date(2026, 8, 23))

    assert report.status == "attention"
    assert report.unresolved_orders == 1
