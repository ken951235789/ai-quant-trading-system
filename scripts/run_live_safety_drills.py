"""執行不送出真實委託的 Live 安全演練。"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from ai_quant_trading.live_trading.drills import run_live_safety_drills


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    report = run_live_safety_drills(PROJECT_ROOT / "data" / "live_trading")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = PROJECT_ROOT / "data" / "research" / "live_safety_drills" / stamp
    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "report.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"演練報告：{output_path}")
    return 0 if bool(report["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
