"""建立 Transformer V3.1 與 SAC 三 seed 私人 Kaggle 任務。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from build_kaggle_transformer_v3_formal import DEFAULT_SOURCE, build_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "kaggle_transformer_sac_v31_three_seed"
DEFAULT_RUNNER = Path(__file__).with_name(
    "kaggle_transformer_sac_v31_three_seed_runner.py"
)


def build_kaggle_files(
    output_dir: Path,
    username: str,
    runner: Path,
) -> tuple[str, str]:
    """建立 Kaggle Dataset 與 Kernel 中繼資料。"""
    dataset_dir = output_dir / "dataset"
    kernel_dir = output_dir / "kernel"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    kernel_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / "transformer_v3_formal_bundle.zip"
    shutil.copy2(archive, dataset_dir / archive.name)
    shutil.copy2(runner, kernel_dir / "train_transformer_sac_v31.py")

    dataset_id = f"{username}/ai-quant-btc-transformer-sac-v31-input"
    kernel_id = f"{username}/ai-quant-btc-transformer-sac-v31-three-seed"
    (dataset_dir / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "AI Quant BTC Transformer SAC V31 Input",
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
                "title": "AI Quant BTC Transformer SAC V31 Three Seed",
                "code_file": "train_transformer_sac_v31.py",
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
    parser = argparse.ArgumentParser(description="建立 Transformer V3.1 + SAC Kaggle 任務")
    parser.add_argument("--username", required=True, help="Kaggle 使用者名稱")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    args = parser.parse_args()

    output = args.output.resolve()
    archive = build_bundle(args.source.resolve(), output)
    dataset_id, kernel_id = build_kaggle_files(
        output,
        args.username,
        args.runner.resolve(),
    )
    print(f"正式資料包：{archive}")
    print(f"壓縮後大小：{archive.stat().st_size / (1024**2):.1f} MB")
    print(f"Kaggle Dataset：{dataset_id}")
    print(f"Kaggle Kernel：{kernel_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
