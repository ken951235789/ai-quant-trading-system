"""檢查既有 RL environment 是否已適合正式模型訓練。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ai_quant_trading.reinforcement_learning import assess_rl_environment_preflight


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RL 正式訓練前完整驗收")
    parser.add_argument("environment_dir", type=Path, help="包含 environment.json 的資料夾")
    parser.add_argument(
        "--allow-smoke",
        action="store_true",
        help="只有警告時仍回傳成功；正式訓練不要使用",
    )
    return parser


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = _parser().parse_args()
    report = assess_rl_environment_preflight(args.environment_dir)
    output = args.environment_dir / "pretraining_readiness.json"
    output.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"完整性：{'通過' if report.eligible else '阻擋'}")
    print(f"正式研究：{'可開始' if report.formal_research_ready else '尚未就緒'}")
    print(f"資料：{report.total_rows:,} 根，跨度 {report.observed_days:.1f} 天")
    for finding in report.findings:
        print(f"[{finding.severity.upper()}] {finding.code}: {finding.message}")
    print(f"報告：{output}")
    if not report.eligible:
        return 2
    if report.warnings and not args.allow_smoke:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
