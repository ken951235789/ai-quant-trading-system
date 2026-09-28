"""建立目前九週期 Transformer V3 + SAC 的私人 Kaggle 測試任務。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src" / "ai_quant_trading"
RAW_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "crypto"
    / "binance_futures"
    / "ohlcv"
)
WHEELHOUSE = PROJECT_ROOT / "artifacts" / "kaggle_wheels"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "kaggle_current_model_test"
DEFAULT_RUNNER = Path(__file__).with_name("kaggle_current_model_test_runner.py")
INTERVALS = ("1m", "3m", "5m", "15m", "30m", "1h", "4h", "12h", "1d")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _latest_timestamp(path: Path) -> pd.Timestamp:
    latest: pd.Timestamp | None = None
    for chunk in pd.read_csv(path, usecols=["timestamp"], chunksize=100_000):
        timestamps = pd.to_datetime(chunk["timestamp"], utc=True, errors="coerce")
        current = timestamps.max()
        if pd.notna(current) and (latest is None or current > latest):
            latest = current
    if latest is None:
        raise ValueError(f"{path.name} 沒有有效 timestamp")
    return latest


def _copy_window(
    source: Path,
    destination: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, object]:
    """分批複製公開行情，避免在 16 GB RAM 電腦一次載入五年 1m 資料。"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    rows = 0
    first: pd.Timestamp | None = None
    last: pd.Timestamp | None = None
    for chunk in pd.read_csv(source, chunksize=200_000, low_memory=False):
        timestamps = pd.to_datetime(chunk["timestamp"], utc=True, errors="coerce")
        selected = chunk.loc[timestamps.between(start, end, inclusive="both")].copy()
        if selected.empty:
            continue
        selected_timestamps = pd.to_datetime(
            selected["timestamp"], utc=True, errors="coerce"
        )
        selected = selected.loc[selected_timestamps.notna()]
        if selected.empty:
            continue
        selected_timestamps = selected_timestamps.loc[selected.index]
        if first is None:
            first = selected_timestamps.iloc[0]
        last = selected_timestamps.iloc[-1]
        selected.to_csv(
            destination,
            mode="w" if rows == 0 else "a",
            header=rows == 0,
            index=False,
            encoding="utf-8",
        )
        rows += len(selected)
    if rows == 0 or first is None or last is None:
        raise ValueError(f"{source.name} 在測試視窗內沒有資料")
    return {
        "rows": rows,
        "start_at": first.isoformat(),
        "end_at": last.isoformat(),
        "size_bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
    }


def _assert_safe_bundle(root: Path) -> None:
    forbidden = {".env", "kaggle.json", "id_rsa", "id_ed25519"}
    rejected = [
        path for path in root.rglob("*")
        if path.is_file() and path.name.lower() in forbidden
    ]
    if rejected:
        raise ValueError(f"Kaggle 測試包含有敏感檔案：{rejected}")


def build_bundle(
    output_dir: Path,
    *,
    research_days: int = 365,
    warmup_days: int = 250,
) -> Path:
    """封裝最近一年研究視窗及暖機資料，不包含帳號與交易狀態。"""
    decision_source = RAW_ROOT / (
        "ohlcv_crypto_binance_futures_BTC-USDT_15m_latest.csv"
    )
    latest = _latest_timestamp(decision_source)
    research_start = latest - pd.Timedelta(days=research_days)
    raw_start = research_start - pd.Timedelta(days=warmup_days)

    output_dir.mkdir(parents=True, exist_ok=True)
    staging = output_dir / "bundle_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        shutil.copytree(
            SOURCE_ROOT,
            staging / "src" / "ai_quant_trading",
            ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", ".env", ".env.*", "dashboard"
            ),
        )
        if not WHEELHOUSE.is_dir():
            raise FileNotFoundError("缺少 Kaggle 離線 wheelhouse")
        shutil.copytree(WHEELHOUSE, staging / "wheelhouse")

        files: dict[str, dict[str, object]] = {}
        for index, interval in enumerate(INTERVALS, start=1):
            name = (
                "ohlcv_crypto_binance_futures_BTC-USDT_"
                f"{interval}_latest.csv"
            )
            source = RAW_ROOT / name
            if not source.is_file():
                raise FileNotFoundError(f"缺少 {interval} 原始資料：{source}")
            target = staging / "data" / name
            files[interval] = _copy_window(
                source,
                target,
                start=raw_start,
                end=latest,
            )
            files[interval]["file"] = f"data/{name}"
            print(
                f"[{index}/{len(INTERVALS)}] {interval}: "
                f"{int(files[interval]['rows']):,} 根",
                flush=True,
            )

        manifest = {
            "schema_version": 1,
            "purpose": "current Transformer V3 + SAC end-to-end Kaggle test",
            "candidate_only": True,
            "exchange": "binance_futures",
            "symbol": "BTC/USDT",
            "decision_interval": "15m",
            "intervals": list(INTERVALS),
            "research_start_at": research_start.isoformat(),
            "raw_start_at": raw_start.isoformat(),
            "snapshot_end_at": latest.isoformat(),
            "files": files,
            "sensitive_files_included": False,
        }
        (staging / "current_model_test_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _assert_safe_bundle(staging)

        archive = output_dir / "current_model_test_bundle.zip"
        if archive.exists():
            archive.unlink()
        with ZipFile(archive, "w", ZIP_DEFLATED, compresslevel=6) as handle:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    handle.write(path, path.relative_to(staging))
        return archive
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def build_kaggle_files(
    output_dir: Path,
    username: str,
    runner: Path,
) -> tuple[str, str]:
    dataset_dir = output_dir / "dataset"
    kernel_dir = output_dir / "kernel"
    for directory in (dataset_dir, kernel_dir):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
    archive = output_dir / "current_model_test_bundle.zip"
    shutil.copy2(archive, dataset_dir / archive.name)
    shutil.copy2(runner, kernel_dir / "run_current_model_test.py")

    dataset_id = f"{username}/ai-quant-btc-current-model-test-input"
    kernel_id = f"{username}/ai-quant-btc-current-model-e2e-test"
    (dataset_dir / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "AI Quant BTC Current Model Test Input",
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
                "id": kernel_id,
                "title": "AI Quant BTC Current Model E2E Test",
                "code_file": "run_current_model_test.py",
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
    return dataset_id, kernel_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True, help="自己的 Kaggle 帳號")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--research-days", type=int, default=365)
    parser.add_argument("--warmup-days", type=int, default=250)
    args = parser.parse_args()
    if args.research_days < 180:
        raise ValueError("端到端測試至少需要 180 天研究資料")
    if args.warmup_days < 200:
        raise ValueError("暖機資料至少需要 200 天")

    output = args.output.resolve()
    archive = build_bundle(
        output,
        research_days=args.research_days,
        warmup_days=args.warmup_days,
    )
    dataset_id, kernel_id = build_kaggle_files(
        output,
        args.username,
        args.runner.resolve(),
    )
    print(f"Kaggle 測試包：{archive}")
    print(f"壓縮大小：{archive.stat().st_size / (1024**2):.1f} MB")
    print(f"Dataset：{dataset_id}")
    print(f"Kernel：{kernel_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
