"""用保存的 SAC 與固定規則進行同環境樣本外比較。"""

import argparse
from pathlib import Path

from ai_quant_trading.backtesting.strategy_comparison import run_comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(run_comparison(args.training_dir, args.output))
