"""執行固定預算的四組事件消融與向前驗證；只在本機 CPU 訓練。"""

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_quant_trading.transformer.event_study import run_event_study  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/transformer_strategy_event_v2.example.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 137, 2026])
    args = parser.parse_args()
    result = run_event_study(args.source.resolve(), args.config.resolve(), args.output.resolve(), tuple(args.seeds))
    print(f"研究已完成：{result}", flush=True)


if __name__ == "__main__":
    main()
