"""執行六個家族的固定複合策略比較，不訓練或部署。"""

import argparse
from pathlib import Path

from ai_quant_trading.backtesting.complex_research import run_complex_research


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--previous", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(run_complex_research(args.source, args.previous, args.output))
