"""建立不含密鑰的 Kaggle Transformer V3／SAC 基準測試資料包。"""

from __future__ import annotations

import argparse
from collections import deque
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

from ai_quant_trading.transformer.dataset import _select_feature_columns  # noqa: E402
from ai_quant_trading.features.market_context import add_model_market_features  # noqa: E402


DEFAULT_FEATURE_SOURCE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features_mtf_crypto_binance_futures_BTC-USDT_15m_5m-15m-1h-4h-1d_h1.csv"
)
DEFAULT_FEATURE_METADATA = DEFAULT_FEATURE_SOURCE.with_suffix(".json")
DEFAULT_RL_ROOT = PROJECT_ROOT / "data" / "research" / "smoke_models" / "rl_environments"
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
)


def _latest_rl_environment(root: Path) -> Path:
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and (path / "environment.json").is_file()
        and (path / "train.csv").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"找不到 SAC 基準測試環境：{root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _read_transformer_tail(source: Path, rows: int) -> tuple[pd.DataFrame, int, list[str]]:
    """先以小樣本選欄，再串流保留最新資料，避免一次載入 1.3 GB CSV。"""
    sample = add_model_market_features(pd.read_csv(source, nrows=min(rows, 5_000), low_memory=False))
    feature_columns = list(_select_feature_columns([sample], 256))
    selected_columns = list(
        dict.fromkeys(
            column
            for column in (*SAFE_MARKET_COLUMNS, *feature_columns)
            if column in sample.columns
        )
    )
    chunks: deque[pd.DataFrame] = deque()
    kept_rows = 0
    total_rows = 0
    for chunk in pd.read_csv(
        source,
        chunksize=10_000,
        low_memory=False,
    ):
        chunk = add_model_market_features(chunk)[selected_columns]
        chunks.append(chunk)
        kept_rows += len(chunk)
        total_rows += len(chunk)
        while len(chunks) > 1 and kept_rows - len(chunks[0]) >= rows:
            kept_rows -= len(chunks.popleft())
    frame = pd.concat(list(chunks), ignore_index=True).tail(rows).reset_index(drop=True)
    return frame, total_rows, feature_columns


def _copy_source(destination: Path) -> None:
    package_source = SRC_DIR / "ai_quant_trading"
    shutil.copytree(
        package_source,
        destination / "src" / "ai_quant_trading",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "dashboard"),
    )


def _copy_rl_environment(source: Path, destination: Path) -> None:
    target = destination / "rl_environment"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("environment.json", "train.csv", "validation.csv", "test.csv"):
        shutil.copy2(source / name, target / name)

    metadata_path = target / "environment.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["status"] = "environment_ready"
    metadata["training_started"] = False
    metadata.pop("last_training", None)
    metadata["source_path"] = "Kaggle benchmark subset"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _archive_directory(source: Path, archive: Path) -> None:
    with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=6) as handle:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(source))


def build_bundle(output_dir: Path, rows: int) -> Path:
    if not DEFAULT_FEATURE_SOURCE.is_file():
        raise FileNotFoundError(f"找不到 Transformer 特徵資料：{DEFAULT_FEATURE_SOURCE}")
    output_dir.mkdir(parents=True, exist_ok=True)
    staging = (output_dir / "bundle_staging").resolve()
    if output_dir.resolve() not in staging.parents:
        raise ValueError("暫存目錄必須位於 Kaggle benchmark 輸出目錄內")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    frame, observed_rows, features = _read_transformer_tail(DEFAULT_FEATURE_SOURCE, rows)
    data_dir = staging / "data"
    data_dir.mkdir()
    transformer_csv = data_dir / "transformer_benchmark.csv"
    frame.to_csv(transformer_csv, index=False, encoding="utf-8")

    formal_rows = observed_rows
    if DEFAULT_FEATURE_METADATA.is_file():
        metadata = json.loads(DEFAULT_FEATURE_METADATA.read_text(encoding="utf-8"))
        formal_rows = int(metadata.get("rows", observed_rows))

    rl_environment = _latest_rl_environment(DEFAULT_RL_ROOT)
    _copy_source(staging)
    _copy_rl_environment(rl_environment, staging)
    manifest = {
        "schema_version": 1,
        "transformer_benchmark_rows": len(frame),
        "transformer_formal_rows": formal_rows,
        "transformer_feature_count": len(features),
        "transformer_interval": "15m",
        "rl_environment_source": rl_environment.name,
        "sensitive_files_included": False,
    }
    (staging / "benchmark_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    archive = output_dir / "benchmark_bundle.zip"
    _archive_directory(staging, archive)
    shutil.rmtree(staging)
    return archive


def build_kaggle_files(output_dir: Path, username: str, runner: Path) -> None:
    dataset_dir = output_dir / "dataset"
    kernel_dir = output_dir / "kernel"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    kernel_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_dir / "benchmark_bundle.zip", dataset_dir / "benchmark_bundle.zip")
    shutil.copy2(runner, kernel_dir / "benchmark.py")

    dataset_id = f"{username}/ai-quant-btc-benchmark-input"
    (dataset_dir / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "AI Quant BTC Benchmark Input",
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
                "id": f"{username}/ai-quant-transformer-v3-and-sac-benchmark",
                "title": "AI Quant Transformer V3 and SAC Benchmark",
                "code_file": "benchmark.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": True,
                "enable_internet": True,
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
    parser = argparse.ArgumentParser(description="建立 Kaggle 訓練速度基準測試")
    parser.add_argument("--username", required=True, help="Kaggle 使用者名稱")
    parser.add_argument("--rows", type=int, default=20_000, help="Transformer 測試資料筆數")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "kaggle_benchmark",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    archive = build_bundle(output, args.rows)
    build_kaggle_files(output, args.username, Path(__file__).with_name("kaggle_benchmark_runner.py"))
    print(f"Kaggle benchmark 套件：{archive}")
    print(f"大小：{archive.stat().st_size / (1024**2):.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
