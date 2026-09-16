"""建立重用正式 Transformer checkpoint 的 SAC 三 seed Kaggle 任務。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "kaggle_transformer_sac_v31_three_seed"
    / "result_v4"
    / "transformer_v31_three_seed"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "kaggle_sac_v31_corrected_three_seed"
DEFAULT_RUNNER = Path(__file__).with_name(
    "kaggle_sac_v31_corrected_three_seed_runner.py"
)
FORMAL_DATASET_ID = "ezkenn/ai-quant-btc-transformer-sac-v31-input"


def _find_selected_run(root: Path) -> Path:
    candidates = [path.parent for path in root.rglob("best_model.pt")]
    if len(candidates) != 1:
        raise RuntimeError(f"預期恰好一個已選 Transformer，實際找到 {len(candidates)} 個")
    return candidates[0]


def build_kaggle_files(
    output_dir: Path,
    username: str,
    selected_run: Path,
    runner: Path,
) -> tuple[str, str]:
    """建立小型 checkpoint Dataset 與 SAC Kernel 中繼資料。"""
    dataset_dir = output_dir / "transformer_dataset"
    kernel_dir = output_dir / "kernel"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    kernel_dir.mkdir(parents=True, exist_ok=True)

    for name in ("best_model.pt", "training.json", "artifact_manifest.json"):
        source = selected_run / name
        if not source.is_file():
            raise FileNotFoundError(f"已選 Transformer 缺少 {name}：{source}")
        shutil.copy2(source, dataset_dir / name)
    shutil.copy2(runner, kernel_dir / "train_sac_v31_corrected.py")

    dataset_id = f"{username}/ai-quant-btc-transformer-v31-selected"
    kernel_id = f"{username}/ai-quant-btc-sac-v31-corrected-three-seed"
    (dataset_dir / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "AI Quant BTC Transformer V31 Selected",
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
                "title": "AI Quant BTC SAC V31 Corrected Three Seed",
                "code_file": "train_sac_v31_corrected.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": True,
                "enable_internet": False,
                "machine_shape": "NvidiaTeslaT4",
                "dataset_sources": [FORMAL_DATASET_ID, dataset_id],
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
    parser = argparse.ArgumentParser(description="建立修正版 SAC 三 seed Kaggle 任務")
    parser.add_argument("--username", default="ezkenn")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    args = parser.parse_args()

    selected = _find_selected_run(args.result_root.resolve())
    dataset_id, kernel_id = build_kaggle_files(
        args.output.resolve(),
        args.username,
        selected,
        args.runner.resolve(),
    )
    print(f"已選 Transformer：{selected}")
    print(f"Kaggle Dataset：{dataset_id}")
    print(f"Kaggle Kernel：{kernel_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
