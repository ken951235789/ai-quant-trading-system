"""固定契約、Transformer 多種子驗證、品質通過後才訓練 SAC 部位管理。"""

import argparse
import json
from pathlib import Path

from ai_quant_trading.research.staged_training import run_staged_training


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--config", default="configs/btc15m_staged_research.json", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run_staged_training(args.source, args.config, args.output), ensure_ascii=False, indent=2))
