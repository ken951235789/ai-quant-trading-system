"""執行或核對續跑預登記的大型參數比較。"""

import argparse
from pathlib import Path

from ai_quant_trading.backtesting.parameter_sweep import run_sweep


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(run_sweep(args.source, args.output, resume=args.resume))
