"""補齊 BTC 永續合約九週期歷史；重跑相同命令可接續未完成下載。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_quant_trading.data_collection.history import download_btc_history, history_error_types  # noqa: E402
from ai_quant_trading.market_clock import BTC_SUPPORTED_MULTITIMEFRAME_INTERVALS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="下載 BTC/USDT USD-M 永續合約九週期已收盤歷史")
    parser.add_argument("--start", required=True, help="研究開始時間，UTC，例如 2021-09-08")
    parser.add_argument("--end", help="資料截止時間，UTC；預設使用執行當下已收盤資料")
    parser.add_argument("--warmup-bars", type=int, default=250, help="各週期另保留的指標暖機根數，至少 200")
    parser.add_argument("--intervals", nargs="+", choices=BTC_SUPPORTED_MULTITIMEFRAME_INTERVALS,
                        default=list(BTC_SUPPORTED_MULTITIMEFRAME_INTERVALS))
    args = parser.parse_args()
    try:
        result = download_btc_history(
            PROJECT_ROOT, start=args.start, end=args.end, warmup_bars=args.warmup_bars,
            intervals=tuple(args.intervals),
        )
    except history_error_types() as exc:
        print(f"下載失敗，保留暫存供續傳：{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("下載已中斷；重跑相同命令可接續。", file=sys.stderr)
        return 130
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
