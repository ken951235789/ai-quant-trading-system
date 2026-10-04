"""建立符合 AI Quant Engineering 契約的報告骨架。"""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


REPORT_TYPES = (
    "data_integrity",
    "model_research",
    "live_trading_safety",
    "release_quality",
    "security_review",
    "multi_module",
)


def build_report(*, report_type: str, project: str, revision: str, scope: str) -> dict[str, object]:
    """回傳尚待填寫的標準報告。"""
    timestamp = datetime.now(timezone.utc).replace(microsecond=0)
    report_id = f"{report_type.upper()}-{timestamp:%Y%m%d}-{uuid.uuid4().hex[:8].upper()}"
    return {
        "schema_version": "1.0.0",
        "report_id": report_id,
        "report_type": report_type,
        "generated_at_utc": timestamp.isoformat().replace("+00:00", "Z"),
        "project": {"name": project, "revision": revision, "environment": "unknown"},
        "scope": scope,
        "overall_status": "DRAFT",
        "summary": "",
        "changes": [],
        "findings": [],
        "checks": [],
        "metrics": {},
        "artifacts": [],
        "residual_risks": [],
        "next_actions": [],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--type", required=True, choices=REPORT_TYPES, dest="report_type")
    parser.add_argument("--project", required=True)
    parser.add_argument("--revision", default="unknown")
    parser.add_argument("--scope", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(
        report_type=args.report_type,
        project=args.project,
        revision=args.revision,
        scope=args.scope,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
