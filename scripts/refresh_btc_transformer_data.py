"""重建 BTC Transformer 的衍生品脈絡、單週期特徵與多週期正式資料。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_quant_trading.data_collection.binance_futures import (  # noqa: E402
    BinanceFuturesPublicClient,
    attach_derivatives_context,
)
from ai_quant_trading.data_collection.csv_storage import (  # noqa: E402
    _atomic_write_csv,
    _exclusive_file_lock,
    canonical_ohlcv_path,
    save_derivatives_context_csv,
)
from ai_quant_trading.features.builder import build_features_from_csv_batch  # noqa: E402
from ai_quant_trading.features.multitimeframe import (  # noqa: E402
    build_multitimeframe_feature_datasets,
)


DEFAULT_INTERVALS = ("5m", "15m", "1h", "4h", "1d")
DERIVATIVE_VALUE_COLUMNS = (
    "funding_rate",
    "mark_price",
    "open_interest",
    "open_interest_value",
    "open_interest_change",
    "bid_price",
    "ask_price",
    "spread_bps",
    "global_long_short_ratio",
    "global_long_account",
    "global_short_account",
    "taker_buy_sell_ratio",
    "taker_buy_volume",
    "taker_sell_volume",
    "basis_rate",
    "basis_price",
    "index_price",
    "derivatives_context_available",
)


def _time_bounds(paths: list[Path]) -> tuple[pd.Timestamp, pd.Timestamp]:
    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    for path in paths:
        timestamps = pd.to_datetime(
            pd.read_csv(path, usecols=["timestamp"])["timestamp"],
            utc=True,
            errors="coerce",
            format="mixed",
        ).dropna()
        if timestamps.empty:
            raise ValueError(f"{path.name} 沒有有效時間")
        starts.append(timestamps.iloc[0])
        ends.append(timestamps.iloc[-1])
    return min(starts), max(ends)


def _replace_context(path: Path, context: pd.DataFrame) -> float:
    """移除舊脈絡後重新因果合併，避免舊 CSV 的缺口繼續殘留。"""
    frame = pd.read_csv(path, low_memory=False)
    base = frame.drop(columns=list(DERIVATIVE_VALUE_COLUMNS), errors="ignore")
    refreshed = attach_derivatives_context(base, context)
    availability = pd.to_numeric(
        refreshed["derivatives_context_available"], errors="coerce"
    ).fillna(0.0)
    coverage = float((availability >= 0.5).mean())
    with _exclusive_file_lock(path, timeout_seconds=120.0):
        _atomic_write_csv(refreshed, path)
    return coverage


def main() -> int:
    parser = argparse.ArgumentParser(
        description="重新整理 BTC 五週期衍生品脈絡並重建 Transformer 特徵"
    )
    parser.add_argument("--intervals", nargs="+", default=list(DEFAULT_INTERVALS))
    parser.add_argument("--start", help="衍生品歷史起點；預設由原始 K 線自動判斷")
    parser.add_argument("--end", help="衍生品歷史終點；預設由原始 K 線自動判斷")
    args = parser.parse_args()

    raw_dir = PROJECT_ROOT / "data" / "raw"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    paths = [
        canonical_ohlcv_path(raw_dir, "binance_futures", "BTC/USDT", interval)
        for interval in args.intervals
    ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"缺少原始 K 線：{missing}")

    detected_start, detected_end = _time_bounds(paths)
    start = pd.Timestamp(args.start) if args.start else detected_start
    end = pd.Timestamp(args.end) if args.end else detected_end
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")

    print(f"下載衍生品脈絡：{start.isoformat()} ～ {end.isoformat()}", flush=True)
    client = BinanceFuturesPublicClient()
    context = client.fetch_market_context(
        "BTC/USDT",
        "15m",
        start=start.isoformat(),
        end=end.isoformat(),
        include_advanced=True,
    )
    context_path = save_derivatives_context_csv(
        context,
        raw_dir,
        "BTC/USDT",
        "15m",
        exchange="binance_futures",
    )
    print(f"衍生品脈絡：{context_path}，{len(context):,} 筆", flush=True)

    for index, path in enumerate(paths, start=1):
        coverage = _replace_context(path, context)
        print(
            f"[{index}/{len(paths)}] {path.name} 衍生品覆蓋率 {coverage:.2%}",
            flush=True,
        )

    feature_paths: list[Path] = []
    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] 重建單週期特徵：{path.name}", flush=True)
        artifact = build_features_from_csv_batch(
            [path],
            output_dir=processed_dir,
            target_horizon=1,
            drop_na=True,
        )[0]
        feature_paths.append(artifact.output_path)
        print(
            f"  完成 {artifact.rows:,} 筆、{artifact.columns} 欄",
            flush=True,
        )

    artifacts = build_multitimeframe_feature_datasets(
        feature_paths,
        "15m",
        processed_dir,
    )
    if len(artifacts) != 1:
        raise RuntimeError(f"預期產生一份 BTC 多週期資料，實際為 {len(artifacts)}")
    artifact = artifacts[0]
    print(f"多週期正式資料：{artifact.output_path}", flush=True)
    print(f"資料列：{artifact.rows:,}，特徵欄：{artifact.feature_columns}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
