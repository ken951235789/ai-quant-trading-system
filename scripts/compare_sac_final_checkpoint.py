"""補充比較預先定義的 final checkpoint，不依 Test 重選最佳模型。"""

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import argparse
import zipfile

import pandas as pd
import torch
from stable_baselines3 import SAC

from ai_quant_trading.backtesting.strategy_comparison import summarize
from ai_quant_trading.operations.integrity import sha256_file, verify_artifact_manifest, build_artifact_manifest
from ai_quant_trading.persistence import write_json_atomic
from ai_quant_trading.reinforcement_learning import PortfolioEnvConfig, evaluate_rl_model, load_rl_policy


def run(root: Path) -> None:
    if (root / "final_checkpoint_plan.json").exists():
        raise FileExistsError("補充研究已存在，不覆寫")
    verify_artifact_manifest(root)
    original = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    for path, digest in original["source_sha256"].items():
        if sha256_file(Path(path)) != digest:
            raise ValueError("原始比較輸入已改變")
    torch.set_num_threads(2)
    policy = load_rl_policy(original["training_dir"], device="cpu")
    final_path = policy.training_dir / "final_model.zip"
    with zipfile.ZipFile(policy.training_dir / "best_model" / "best_model.zip") as archive:
        selected_steps = json.loads(archive.read("data"))["num_timesteps"]
    with zipfile.ZipFile(final_path) as archive:
        final_steps = json.loads(archive.read("data"))["num_timesteps"]
    plan = {"created_at": datetime.now(timezone.utc).isoformat(), "scope": "checkpoint差異診斷，不重選模型",
        "selected_steps": selected_steps, "final_steps": final_steps, "final_model_sha256": sha256_file(final_path),
        "original_plan_sha256": sha256_file(root / "plan.json"), "code_sha256": sha256_file(Path(__file__)),
        "policy": "同一預定義final模型，在原比較的兩窗口與四成本重播，全部揭露；沒有新訓練。"}
    write_json_atomic(root / "final_checkpoint_plan.json", plan)
    model = SAC.load(final_path, device="cpu")
    rows = []
    for split in ("validation", "test"):
        raw = pd.read_csv(policy.environment_dir / f"{split}.csv")
        for scenario, values in original["environment_configs"].items():
            config = PortfolioEnvConfig(**values)
            frame = raw.copy()
            if scenario != "native":
                frame["spread_bps"] = config.spread_rate * 10_000
            if scenario == "zero":
                frame["funding_rate"] = 0.
            evaluation, metrics = evaluate_rl_model(model, frame, policy.feature_columns, config, deterministic=True, seed=11)
            if len(evaluation) != len(frame) - 1:
                raise ValueError("補充比較提前終止")
            name = f"{split}_{scenario}_sac_final.csv"
            evaluation.to_csv(root / name, index=False)
            rows.append({"split": split, "scenario": scenario, "policy": "sac_final", "path": name,
                "metrics": summarize(evaluation, metrics, config.initial_capital), "config": asdict(config)})
            print(name, metrics["total_return"], flush=True)
    write_json_atomic(root / "final_checkpoint_results.json", {"status": "complete", "results": rows, "live_eligible": False})
    build_artifact_manifest(root, [p for p in root.rglob("*") if p.is_file()])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    args = parser.parse_args()
    run(args.study)
