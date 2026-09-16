"""建立 Transformer V3 正式訓練所需的私人 Kaggle 資料包。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_quant_trading.features.contracts import ensure_safe_model_features  # noqa: E402
from ai_quant_trading.features.market_context import add_model_market_features, market_timeframes  # noqa: E402
from ai_quant_trading.reinforcement_learning.feature_contract import COMPACT_TIMEFRAME_FEATURES  # noqa: E402
from ai_quant_trading.transformer.dataset import _select_feature_columns  # noqa: E402
from ai_quant_trading.reinforcement_learning.universal import (  # noqa: E402
    UNIVERSAL_SOURCE_FEATURE_COLUMNS,
)


DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features_mtf_crypto_binance_futures_BTC-USDT_15m_5m-15m-1h-4h-1d_h1.csv"
)
KAGGLE_WHEELHOUSE = PROJECT_ROOT / "artifacts" / "kaggle_wheels"
SAFE_MARKET_COLUMNS = (
    "timestamp",
    "symbol",
    "exchange",
    "interval",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "funding_rate",
    "spread_bps",
    "open_interest_change",
    "global_long_short_ratio",
    "taker_buy_sell_ratio",
    "basis_rate",
    "derivatives_context_available",
    *UNIVERSAL_SOURCE_FEATURE_COLUMNS,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_source(destination: Path) -> None:
    """只複製訓練需要的 Python 套件，不包含 UI、快取與個人設定。"""
    shutil.copytree(
        SRC_DIR / "ai_quant_trading",
        destination / "src" / "ai_quant_trading",
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            "dashboard",
            ".env",
            ".env.*",
        ),
    )


def _write_reduced_csv(
    source: Path,
    destination: Path,
    *,
    maximum_features: int,
) -> dict[str, object]:
    """串流輸出必要欄位，同時檢查時間軸是否嚴格遞增。"""
    sample = add_model_market_features(pd.read_csv(source, nrows=5_000, low_memory=False))
    features = list(_select_feature_columns([sample], maximum_features))
    ensure_safe_model_features(
        features,
        model_name="Transformer V3",
        forbid_transformer_outputs=True,
    )
    if len(features) != maximum_features:
        raise ValueError(
            f"正式訓練需要 {maximum_features} 項特徵，目前只選到 {len(features)} 項"
        )
    sac_context = [
        f"mtf_{interval}_{name}" for interval in market_timeframes(list(sample.columns))
        for name in (*COMPACT_TIMEFRAME_FEATURES, "available", "age_ratio")
    ]
    selected = list(
        dict.fromkeys(
            column
            for column in (*SAFE_MARKET_COLUMNS, *features, *sac_context)
            if column in sample.columns
        )
    )
    missing_market = sorted({"timestamp", "open", "high", "low", "close"} - set(selected))
    if missing_market:
        raise ValueError(f"正式資料缺少必要市場欄位：{missing_market}")

    rows = 0
    first_timestamp: pd.Timestamp | None = None
    previous_timestamp: pd.Timestamp | None = None
    invalid_timestamps = 0
    duplicate_or_reversed = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()

    for chunk_index, chunk in enumerate(
        pd.read_csv(
            source,
            chunksize=10_000,
            low_memory=False,
        )
    ):
        chunk = add_model_market_features(chunk)[selected]
        timestamps = pd.to_datetime(chunk["timestamp"], utc=True, errors="coerce")
        invalid_timestamps += int(timestamps.isna().sum())
        valid = timestamps.dropna()
        if not valid.empty:
            if first_timestamp is None:
                first_timestamp = valid.iloc[0]
            if previous_timestamp is not None and valid.iloc[0] <= previous_timestamp:
                duplicate_or_reversed += 1
            duplicate_or_reversed += int((valid.diff().dropna() <= pd.Timedelta(0)).sum())
            previous_timestamp = valid.iloc[-1]
        chunk.to_csv(
            destination,
            mode="w" if chunk_index == 0 else "a",
            header=chunk_index == 0,
            index=False,
            encoding="utf-8",
        )
        rows += len(chunk)
        print(f"已整理 {rows:,} 根 K 線", flush=True)

    if invalid_timestamps:
        raise ValueError(f"資料包含 {invalid_timestamps} 筆無效時間")
    if duplicate_or_reversed:
        raise ValueError(f"時間軸包含 {duplicate_or_reversed} 筆重複或逆序資料")
    if first_timestamp is None or previous_timestamp is None:
        raise ValueError("正式資料沒有有效時間")
    return {
        "rows": rows,
        "columns": len(selected),
        "feature_count": len(features),
        "feature_columns": features,
        "feature_selection_version": "timeframe_indicator_balanced_v2",
        "ohlcv_storage_only": True,
        "start_at": first_timestamp.isoformat(),
        "end_at": previous_timestamp.isoformat(),
        "sha256": _sha256(destination),
        "size_bytes": destination.stat().st_size,
    }


def _archive_directory(source: Path, archive: Path) -> None:
    with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=6) as handle:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(source))


def _assert_no_sensitive_files(root: Path) -> None:
    forbidden_names = {".env", "kaggle.json", "id_rsa", "id_ed25519"}
    rejected = [path for path in root.rglob("*") if path.is_file() and path.name.lower() in forbidden_names]
    if rejected:
        raise ValueError(f"訓練包包含敏感檔案：{rejected}")


def build_bundle(source: Path, output_dir: Path) -> Path:
    if not source.is_file():
        raise FileNotFoundError(f"找不到 Transformer 正式資料：{source}")
    output_dir.mkdir(parents=True, exist_ok=True)
    staging = (output_dir / "bundle_staging").resolve()
    if output_dir.resolve() not in staging.parents:
        raise ValueError("暫存目錄必須位於正式訓練輸出目錄內")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    try:
        data_path = staging / "data" / "transformer_v3_formal.csv"
        data_summary = _write_reduced_csv(source, data_path, maximum_features=256)
        source_metadata: dict[str, object] = {}
        metadata_path = source.with_suffix(".json")
        if metadata_path.is_file():
            source_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            expected_rows = int(source_metadata.get("rows", data_summary["rows"]))
            if expected_rows != data_summary["rows"]:
                raise ValueError(
                    f"資料筆數與 metadata 不一致：CSV={data_summary['rows']}、metadata={expected_rows}"
                )

        _copy_source(staging)
        dependency_wheels: list[str] = []
        if KAGGLE_WHEELHOUSE.is_dir():
            wheelhouse_target = staging / "wheelhouse"
            shutil.copytree(KAGGLE_WHEELHOUSE, wheelhouse_target)
            dependency_wheels = sorted(path.name for path in wheelhouse_target.glob("*.whl"))
        manifest = {
            "schema_version": 1,
            "purpose": "Transformer V3 formal candidate training",
            "candidate_only": True,
            "source_snapshot": source.name,
            "source_metadata": source_metadata,
            "data": data_summary,
            "split": {"train": 0.60, "validation": 0.20, "test": 0.20},
            "causal_checks": {
                "timestamp_strictly_increasing": True,
                "forbidden_feature_check_passed": True,
                "scaler_fit_on_training_split_only": True,
                "next_open_cost_aware_targets": True,
            },
            "sensitive_files_included": False,
            "offline_dependency_wheels": dependency_wheels,
        }
        (staging / "formal_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _assert_no_sensitive_files(staging)
        archive = output_dir / "transformer_v3_formal_bundle.zip"
        _archive_directory(staging, archive)
        return archive
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def build_kaggle_files(output_dir: Path, username: str, runner: Path) -> None:
    dataset_dir = output_dir / "dataset"
    kernel_dir = output_dir / "kernel"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    kernel_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / "transformer_v3_formal_bundle.zip"
    shutil.copy2(archive, dataset_dir / archive.name)
    shutil.copy2(runner, kernel_dir / "train_transformer_v3.py")

    dataset_id = f"{username}/ai-quant-btc-transformer-v3-formal-input"
    (dataset_dir / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "AI Quant BTC Transformer V3 Formal Input",
                "id": dataset_id,
                "licenses": [{"name": "CC0-1.0"}],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": f"{username}/ai-quant-btc-transformer-v3-formal-training",
                "title": "AI Quant BTC Transformer V3 Formal Training",
                "code_file": "train_transformer_v3.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": True,
                "enable_internet": False,
                "machine_shape": "NvidiaTeslaT4",
                "dataset_sources": [dataset_id],
                "competition_sources": [],
                "kernel_sources": [],
                "model_sources": [],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="建立 Kaggle Transformer V3 正式訓練包")
    parser.add_argument("--username", required=True, help="Kaggle 使用者名稱")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "kaggle_transformer_v3_formal",
    )
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path(__file__).with_name("kaggle_transformer_v3_five_seed_runner.py"),
        help="要提交至 Kaggle 的正式訓練程式",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    archive = build_bundle(args.source.resolve(), output)
    build_kaggle_files(
        output,
        args.username,
        args.runner.resolve(),
    )
    print(f"正式訓練包：{archive}")
    print(f"壓縮後大小：{archive.stat().st_size / (1024**2):.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
