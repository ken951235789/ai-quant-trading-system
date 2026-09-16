"""可重現壓力測試工具的低負載回歸測試。"""

from scripts.run_stress_test import run_stress_test


def test_stress_runner_preserves_all_records(tmp_path) -> None:
    report = run_stress_test(
        tmp_path,
        event_count=100,
        csv_rows=40,
        audit_events=20,
        workers=4,
        output_root=tmp_path / "reports",
        check_postgres=False,
    )

    assert report["status"] == "passed"
    assert all(report["checks"].values())
