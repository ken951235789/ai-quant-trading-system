"""以固定規則、三組成本情境分析候選策略，並診斷已訓練模型的閘門。"""

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_quant_trading.backtesting.candidate_research import run_candidate_research  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model-study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_candidate_research(args.source.resolve(), args.model_study.resolve(), args.output.resolve())
    print(f"研究完成：{result}", flush=True)
