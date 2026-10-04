"""驗證 AI Quant Engineering JSON 報告的核心契約與狀態一致性。"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPORT_TYPES = {
    "data_integrity",
    "model_research",
    "live_trading_safety",
    "release_quality",
    "security_review",
    "multi_module",
}
OVERALL_STATUSES = {"DRAFT", "PASS", "WARN", "FAIL", "BLOCKED"}
SEVERITIES = {"CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"}
FINDING_STATUSES = {"OPEN", "ACCEPTED", "RESOLVED"}
CHECK_STATUSES = {"PASS", "WARN", "FAIL", "BLOCKED", "NOT_APPLICABLE"}
FINDING_ID_PATTERN = re.compile(r"^[A-Z]+-[0-9]{3}$")


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _utc_datetime(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.utcoffset() == timezone.utc.utcoffset(parsed)


def validate_report(report: Any, *, allow_draft: bool = False) -> list[str]:
    """回傳所有契約錯誤；空陣列表示驗證通過。"""
    errors: list[str] = []
    if not isinstance(report, dict):
        return ["根節點必須是JSON object。"]

    required = {
        "schema_version", "report_id", "report_type", "generated_at_utc", "project",
        "scope", "overall_status", "summary", "changes", "findings", "checks",
        "metrics", "artifacts", "residual_risks", "next_actions",
    }
    missing = sorted(required.difference(report))
    if missing:
        errors.append(f"缺少必要欄位：{', '.join(missing)}")
        return errors

    if report["schema_version"] != "1.0.0":
        errors.append("schema_version必須是1.0.0。")
    if not _nonempty_string(report["report_id"]):
        errors.append("report_id不可為空。")
    if report["report_type"] not in REPORT_TYPES:
        errors.append("report_type不在允許清單。")
    if not _utc_datetime(report["generated_at_utc"]):
        errors.append("generated_at_utc必須是UTC ISO 8601時間。")
    if not _nonempty_string(report["scope"]):
        errors.append("scope不可為空。")

    overall_status = report["overall_status"]
    if overall_status not in OVERALL_STATUSES:
        errors.append("overall_status不在允許清單。")
    elif overall_status == "DRAFT" and not allow_draft:
        errors.append("最終報告不可維持DRAFT；需要時使用--allow-draft檢查草稿。")
    if overall_status != "DRAFT" and not _nonempty_string(report["summary"]):
        errors.append("最終報告的summary不可為空。")

    project = report["project"]
    if not isinstance(project, dict):
        errors.append("project必須是object。")
    else:
        if not _nonempty_string(project.get("name")):
            errors.append("project.name不可為空。")
        if not _nonempty_string(project.get("revision")):
            errors.append("project.revision不可為空。")

    list_fields = ("changes", "findings", "checks", "artifacts", "residual_risks", "next_actions")
    for field in list_fields:
        if not isinstance(report[field], list):
            errors.append(f"{field}必須是array。")
    if not isinstance(report["metrics"], dict):
        errors.append("metrics必須是object。")
    if not isinstance(report["findings"], list) or not isinstance(report["checks"], list):
        return errors

    finding_ids: set[str] = set()
    unresolved_high = False
    for index, finding in enumerate(report["findings"]):
        prefix = f"findings[{index}]"
        if not isinstance(finding, dict):
            errors.append(f"{prefix}必須是object。")
            continue
        finding_id = finding.get("id")
        if not isinstance(finding_id, str) or not FINDING_ID_PATTERN.fullmatch(finding_id):
            errors.append(f"{prefix}.id格式必須像DATA-001。")
        elif finding_id in finding_ids:
            errors.append(f"發現重複id：{finding_id}")
        else:
            finding_ids.add(finding_id)
        severity = finding.get("severity")
        status = finding.get("status")
        if severity not in SEVERITIES:
            errors.append(f"{prefix}.severity不在允許清單。")
        if status not in FINDING_STATUSES:
            errors.append(f"{prefix}.status不在允許清單。")
        if severity in {"CRITICAL", "HIGH"} and status != "RESOLVED":
            unresolved_high = True
        for field in ("category", "component", "title", "impact", "recommendation"):
            if not _nonempty_string(finding.get(field)):
                errors.append(f"{prefix}.{field}不可為空。")
        evidence = finding.get("evidence")
        if not isinstance(evidence, list) or not all(_nonempty_string(item) for item in evidence):
            errors.append(f"{prefix}.evidence必須是非空字串array。")
        elif severity in {"CRITICAL", "HIGH"} and not evidence:
            errors.append(f"{prefix}屬高風險，必須提供證據。")

    check_ids: set[str] = set()
    check_statuses: set[str] = set()
    for index, check in enumerate(report["checks"]):
        prefix = f"checks[{index}]"
        if not isinstance(check, dict):
            errors.append(f"{prefix}必須是object。")
            continue
        check_id = check.get("id")
        if not _nonempty_string(check_id):
            errors.append(f"{prefix}.id不可為空。")
        elif check_id in check_ids:
            errors.append(f"發現重複check id：{check_id}")
        else:
            check_ids.add(check_id)
        if not _nonempty_string(check.get("name")):
            errors.append(f"{prefix}.name不可為空。")
        status = check.get("status")
        if status not in CHECK_STATUSES:
            errors.append(f"{prefix}.status不在允許清單。")
        else:
            check_statuses.add(status)
        evidence = check.get("evidence")
        if not isinstance(evidence, list) or not all(_nonempty_string(item) for item in evidence):
            errors.append(f"{prefix}.evidence必須是非空字串array。")
        elif status in {"PASS", "WARN", "FAIL", "BLOCKED"} and not evidence:
            errors.append(f"{prefix}必須提供狀態證據。")

    if overall_status == "PASS" and unresolved_high:
        errors.append("存在未解決CRITICAL/HIGH發現時，整體狀態不得為PASS。")
    if "FAIL" in check_statuses and overall_status not in {"FAIL", "DRAFT"}:
        errors.append("存在失敗檢查時，整體狀態必須是FAIL。")
    if "BLOCKED" in check_statuses and overall_status == "PASS":
        errors.append("存在阻斷檢查時，整體狀態不得為PASS。")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--allow-draft", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"valid": False, "errors": [str(exc)]}, ensure_ascii=False, indent=2))
        return 1

    errors = validate_report(report, allow_draft=args.allow_draft)
    print(json.dumps({
        "valid": not errors,
        "report_id": report.get("report_id") if isinstance(report, dict) else None,
        "errors": errors,
    }, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
