"""執行低換手規則研究；不訓練模型、不改變既有交易設定。"""

import argparse
from pathlib import Path

from ai_quant_trading.backtesting.candidate_optimization import run_research


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run_research(args.source, args.output)
    print({"selection": result["selection"]["selected"], "forward_gate": result["forward_gate"],
           "next_stage": result["next_stage"], "live_eligible": False})
