"""執行歷史／即時統一事件重播與離線故障注入研究。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ai_quant_trading.research import run_replay_research


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="統一市場事件重播與故障注入研究")
    parser.add_argument("--input", required=True, help="單一標的、單一週期 OHLCV CSV")
    parser.add_argument("--output-root", default="data/research/event_replay")
    parser.add_argument("--limit", type=int, default=500, help="使用最後 N 根 K 線")
    parser.add_argument("--exchange", default="binance_futures")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--interval", default="15m")
    return parser


def main(arguments: list[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    artifacts = run_replay_research(
        args.input,
        output_root=args.output_root,
        limit=args.limit,
        project_root=Path.cwd(),
        default_exchange=args.exchange,
        default_symbol=args.symbol,
        default_interval=args.interval,
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": artifacts.passed,
                "experiment_dir": str(artifacts.experiment_dir),
                "report": str(artifacts.report_path),
                "manifest": str(artifacts.manifest_path),
                "results": str(artifacts.results_path),
                "artifact_manifest": str(artifacts.artifact_manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if artifacts.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
