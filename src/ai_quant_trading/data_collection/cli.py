"""資料收集命令列介面。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ai_quant_trading.data_collection.binance import BinanceApiError
from ai_quant_trading.data_collection.collectors import (
    collect_binance_data,
    collect_yahoo_finance_data,
)
from ai_quant_trading.data_collection.yahoo_finance import YahooFinanceDataError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="下載加密貨幣或美股市場資料並儲存為 CSV")
    parser.add_argument(
        "--exchange",
        default="binance",
        choices=["binance", "yahoo"],
        help="資料來源。binance 用於加密貨幣，yahoo 用於美股研究資料。",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="標的清單。加密貨幣例如 BTC/USDT ETH/USDT；美股例如 AAPL MSFT SPY。",
    )
    parser.add_argument("--interval", default="1d", help="K 線時間週期，例如 1m、1h、1d。")
    parser.add_argument("--start", default=None, help="開始時間，例如 2024-01-01 或 ISO 時間。")
    parser.add_argument("--end", default=None, help="結束時間，例如 2024-02-01 或 ISO 時間。")
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="最多保留幾筆 K 線。Binance 單次 API 最大 1000；Yahoo 會在下載後取最後 N 筆。",
    )
    parser.add_argument("--raw-dir", default="data/raw", help="原始資料輸出資料夾。")
    parser.add_argument(
        "--skip-market-data",
        action="store_true",
        help="只下載 OHLCV，不下載 24 小時 market data 快照。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    symbols = args.symbols
    if symbols is None:
        symbols = (
            ["BTC/USDT", "ETH/USDT"]
            if args.exchange == "binance"
            else ["AAPL", "MSFT", "NVDA", "AMD", "SPY", "QQQ"]
        )

    try:
        if args.exchange == "binance":
            result = collect_binance_data(
                symbols=symbols,
                interval=args.interval,
                raw_dir=Path(args.raw_dir),
                start=args.start,
                end=args.end,
                limit=args.limit,
                include_market_data=not args.skip_market_data,
            )
        else:
            if args.skip_market_data:
                print("提示：Yahoo 美股第一版只下載 OHLCV，--skip-market-data 會被忽略。")
            result = collect_yahoo_finance_data(
                symbols=symbols,
                interval=args.interval,
                raw_dir=Path(args.raw_dir),
                start=args.start,
                end=args.end,
                limit=args.limit,
            )
    except (BinanceApiError, YahooFinanceDataError, ValueError, ImportError) as exc:
        print(f"資料下載失敗：{exc}", file=sys.stderr)
        return 1

    print("OHLCV CSV：")
    for path in result.ohlcv_files:
        print(f"  {path}")

    if result.market_data_files:
        print("Market Data CSV：")
        for path in result.market_data_files:
            print(f"  {path}")

    return 0
