"""匯出交易日誌的研究資料；不啟動模型訓練或下單。"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ai_quant_trading.trading.journal import TradeJournal  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--render-pending", action="store_true")
    args = parser.parse_args()
    if not args.output and not args.render_pending:
        parser.error("至少指定 output 或 render-pending")
    journal = TradeJournal(args.journal)
    if args.render_pending:
        print({"rendered": journal.render_pending(limit=100)})
    if args.output:
        print(journal.export_training_review(args.output))


if __name__ == "__main__":
    main()
