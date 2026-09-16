"""執行不送單的事件、CSV、稽核鏈與資料庫壓力驗證。"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from time import perf_counter
import tempfile

import pandas as pd

from ai_quant_trading.database.config import DatabaseSettings
from ai_quant_trading.database.engine import check_database_health
from ai_quant_trading.database.repository import repository_from_settings
from ai_quant_trading.live_trading.audit import append_audit_event, verify_audit_log
from ai_quant_trading.live_trading.storage import append_csv_row
from ai_quant_trading.operations.integrity import build_artifact_manifest
from ai_quant_trading.persistence import write_json_atomic
from ai_quant_trading.runtime_secrets import runtime_env_path
from ai_quant_trading.trading import MarketEvent, MarketEventBus


def _events(count: int) -> list[MarketEvent]:
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    events: list[MarketEvent] = []
    for index in range(count):
        opened = started + timedelta(minutes=15 * index)
        open_ms = int(opened.timestamp() * 1000)
        price = 60_000.0 + index * 0.01
        events.append(
            MarketEvent.closed_kline(
                source="historical",
                exchange="binance_futures",
                symbol="BTC/USDT",
                interval="15m",
                open_time_ms=open_ms,
                close_time_ms=open_ms + 899_999,
                observed_at=opened + timedelta(minutes=15),
                payload={
                    "open": price,
                    "high": price + 1.0,
                    "low": price - 1.0,
                    "close": price + 0.2,
                    "volume": 10.0,
                },
            )
        )
    return events


def run_stress_test(
    project_root: str | Path,
    *,
    event_count: int = 20_000,
    csv_rows: int = 1_000,
    audit_events: int = 200,
    workers: int = 16,
    output_root: str | Path | None = None,
    check_postgres: bool = True,
) -> dict[str, object]:
    """驗證高量事件投遞及多執行緒檔案寫入不遺失、不破壞雜湊鏈。"""
    if min(event_count, csv_rows, audit_events, workers) <= 0:
        raise ValueError("壓力測試數量與 workers 必須大於 0")
    root = Path(project_root).resolve()
    with tempfile.TemporaryDirectory(prefix="ai-quant-stress-") as temporary:
        temp = Path(temporary)
        delivered: set[str] = set()
        bus = MarketEventBus([lambda event: delivered.add(event.event_id)], strict_continuity=True)
        events = _events(event_count)
        started = perf_counter()
        for event in events:
            bus.publish(event)
        event_seconds = perf_counter() - started

        csv_path = temp / "stress_rows.csv"
        columns = ["sequence", "status"]
        started = perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(
                executor.map(
                    lambda index: append_csv_row(
                        csv_path,
                        columns,
                        {"sequence": index, "status": "ok"},
                    ),
                    range(csv_rows),
                )
            )
        csv_seconds = perf_counter() - started
        csv_frame = pd.read_csv(csv_path)

        audit_path = temp / "stress_audit.jsonl"
        started = perf_counter()
        with ThreadPoolExecutor(max_workers=min(workers, 8)) as executor:
            list(
                executor.map(
                    lambda index: append_audit_event(
                        audit_path,
                        "stress_event",
                        {"sequence": index},
                    ),
                    range(audit_events),
                )
            )
        audit_seconds = perf_counter() - started
        audit = verify_audit_log(audit_path)

    postgres_enabled = False
    postgres_up: bool | None = None
    if check_postgres:
        settings = DatabaseSettings.from_environment(runtime_env_path(root))
        postgres_enabled = settings.enabled
        if postgres_enabled:
            repository = repository_from_settings(settings)
            postgres_up, _ = check_database_health(repository.engine)
            repository.engine.dispose()

    checks = {
        "event_delivery": len(delivered) == event_count and bus.metrics.delivered == event_count,
        "csv_integrity": len(csv_frame) == csv_rows
        and set(csv_frame["sequence"]) == set(range(csv_rows)),
        "audit_chain": audit.valid and audit.event_count == audit_events,
        "postgres": postgres_up is not False,
    }
    report: dict[str, object] = {
        "status": "passed" if all(checks.values()) else "failed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "events": {
            "count": event_count,
            "seconds": event_seconds,
            "per_second": event_count / max(event_seconds, 1e-12),
        },
        "csv": {
            "rows": csv_rows,
            "workers": workers,
            "seconds": csv_seconds,
            "per_second": csv_rows / max(csv_seconds, 1e-12),
        },
        "audit": {
            "events": audit_events,
            "seconds": audit_seconds,
            "per_second": audit_events / max(audit_seconds, 1e-12),
        },
        "postgres": {"enabled": postgres_enabled, "up": postgres_up},
    }
    destination = Path(output_root or root / "data" / "research" / "stress_tests").resolve()
    run_dir = destination / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    report_path = run_dir / "stress_summary.json"
    report["report_path"] = str(report_path)
    write_json_atomic(report_path, report)
    build_artifact_manifest(run_dir)
    return report


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AI Quant 無下單壓力測試")
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--event-count", type=int, default=20_000)
    parser.add_argument("--csv-rows", type=int, default=1_000)
    parser.add_argument("--audit-events", type=int, default=200)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output-root")
    parser.add_argument("--skip-postgres", action="store_true")
    args = parser.parse_args(arguments)
    report = run_stress_test(
        args.project_root,
        event_count=args.event_count,
        csv_rows=args.csv_rows,
        audit_events=args.audit_events,
        workers=args.workers,
        output_root=args.output_root,
        check_postgres=not args.skip_postgres,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
