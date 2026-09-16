"""特徵工程命令列介面。"""

from __future__ import annotations

import argparse

from ai_quant_trading.features.builder import build_features_from_csv_batch
from ai_quant_trading.features.indicators import FEATURE_COLUMNS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="把 OHLCV CSV 轉換成 AI 模型特徵資料集")
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="一或多份 Step 2 產生的 OHLCV CSV 路徑。",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="特徵資料集輸出資料夾。",
    )
    parser.add_argument(
        "--target-horizon",
        type=int,
        default=1,
        help="target 使用未來幾期報酬，預設 1 期。",
    )
    parser.add_argument(
        "--target-threshold",
        type=float,
        default=0.0,
        help="未來報酬高於此門檻時 target=1，例如 0.01 代表 1%%。",
    )
    parser.add_argument(
        "--annualization-periods",
        type=int,
        default=None,
        help="歷史波動率年化期數；日線加密貨幣自動用 365，美股用 252。",
    )
    parser.add_argument(
        "--keep-na",
        action="store_true",
        help="保留指標暖機期與最後 N 期的缺值，通常只用於檢查。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts = build_features_from_csv_batch(
        input_paths=args.input,
        output_dir=args.output_dir,
        target_horizon=args.target_horizon,
        target_threshold=args.target_threshold,
        annualization_periods=args.annualization_periods,
        drop_na=not args.keep_na,
    )

    for artifact in artifacts:
        print(f"特徵 CSV：{artifact.output_path}")
        print(f"輸出筆數：{artifact.rows}")
    print(f"模型特徵數：{len(FEATURE_COLUMNS)}")
    return 0
