"""重播全部低換手研究情境，核對逐筆成本與選擇結果。"""

import argparse
import json
from pathlib import Path

from ai_quant_trading.backtesting.candidate_optimization import audit_research
from ai_quant_trading.operations.integrity import build_artifact_manifest
from ai_quant_trading.persistence import write_json_atomic


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    args = parser.parse_args()
    result = audit_research(args.study)
    write_json_atomic(args.study / "audit.json", result)
    build_artifact_manifest(args.study, [p for p in args.study.rglob("*") if p.is_file()])
    print(json.dumps(result, ensure_ascii=False, indent=2))
