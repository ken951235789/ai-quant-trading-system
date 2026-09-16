"""建立配額節制版 SAC V3.1 正式候選 Kaggle 任務。"""

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
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "kaggle_sac_v31_formal_candidate"
DEFAULT_RUNNER = Path(__file__).with_name(
    "kaggle_sac_v31_corrected_three_seed_runner.py"
)
FORMAL_DATASET_ID = "ezkenn/ai-quant-btc-transformer-sac-v31-input"
SEEDS = (11, 42)
TOTAL_TIMESTEPS = 500_000
WALK_FORWARD_FOLDS = 2


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("seeds 必須是不可重複的非負整數")
    return seeds


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
    *,
    profile: str,
    seeds: tuple[int, ...],
    total_timesteps: int,
    walk_forward_folds: int,
    dataset_slug: str,
    kernel_slug: str,
) -> tuple[str, str]:
    """建立 checkpoint、工作設定與私人 Kaggle Kernel。"""
    dataset_dir = output_dir / "candidate_dataset"
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
    (dataset_dir / "sac_job_config.json").write_text(
        json.dumps(
            {
                "profile": profile,
                "seeds": list(seeds),
                "total_timesteps": total_timesteps,
                "walk_forward_folds": walk_forward_folds,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    shutil.copy2(runner, kernel_dir / "train_sac_v31_formal_candidate.py")

    dataset_id = f"{username}/{dataset_slug}"
    kernel_id = f"{username}/{kernel_slug}"
    kernel_title = " ".join(part.upper() for part in kernel_slug.split("-"))
    (dataset_dir / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "AI Quant BTC SAC V31 Formal Candidate Input",
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
                "title": kernel_title,
                "code_file": "train_sac_v31_formal_candidate.py",
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
    parser = argparse.ArgumentParser(description="建立配額節制版 SAC 正式候選")
    parser.add_argument("--username", default="ezkenn")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--profile", default="quota_aware_formal_candidate_500k")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--folds", type=int, default=WALK_FORWARD_FOLDS)
    parser.add_argument(
        "--dataset-slug",
        default="ai-quant-btc-sac-v31-formal-candidate-input",
    )
    parser.add_argument(
        "--kernel-slug",
        default="ai-quant-btc-sac-v31-formal-candidate",
    )
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    if args.timesteps < 100_000:
        raise ValueError("timesteps 不可低於 100,000")
    if not 0 <= args.folds <= 8:
        raise ValueError("folds 必須介於 0 與 8")
    selected = _find_selected_run(args.result_root.resolve())
    dataset_id, kernel_id = build_kaggle_files(
        args.output.resolve(),
        args.username,
        selected,
        args.runner.resolve(),
        profile=args.profile,
        seeds=seeds,
        total_timesteps=args.timesteps,
        walk_forward_folds=args.folds,
        dataset_slug=args.dataset_slug,
        kernel_slug=args.kernel_slug,
    )
    print(f"SAC 工作設定：seeds={seeds}、每個 run={args.timesteps:,} steps")
    print(f"Kaggle Dataset：{dataset_id}")
    print(f"Kaggle Kernel：{kernel_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
