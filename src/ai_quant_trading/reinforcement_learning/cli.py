"""Step 9 PPO／SAC 訓練命令列入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_quant_trading.reinforcement_learning.training import train_rl_agent
from ai_quant_trading.reinforcement_learning.training_config import RLTrainingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="訓練 Step 9 PPO／SAC 投資組合模型")
    parser.add_argument("--environment", required=True, help="強化學習環境資料夾")
    parser.add_argument("--config", help="包含全部 RLTrainingConfig 欄位的 JSON")
    parser.add_argument("--algorithm", choices=["ppo", "sac"], help="覆蓋 JSON 演算法")
    parser.add_argument("--timesteps", type=int, help="覆蓋 JSON 訓練步數")
    parser.add_argument("--device", help="覆蓋 JSON 裝置，例如 auto、cpu、cuda:0")
    parser.add_argument("--resume", help="從 best_model、final_model 或 checkpoint ZIP 續跑")
    return parser


def _load_config(args: argparse.Namespace) -> RLTrainingConfig:
    payload: dict[str, object] = {}
    if args.config:
        payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.algorithm:
        payload["algorithm"] = args.algorithm
    if args.timesteps is not None:
        payload["total_timesteps"] = args.timesteps
    if args.device:
        payload["device"] = args.device
    if isinstance(payload.get("net_arch"), list):
        payload["net_arch"] = tuple(int(value) for value in payload["net_arch"])
    return RLTrainingConfig(**payload)


def main() -> int:
    args = build_parser().parse_args()
    config = _load_config(args)
    last_bucket = -1

    def report(payload: dict[str, object]) -> None:
        nonlocal last_bucket
        progress = float(payload.get("progress", 0))
        bucket = int(progress * 10)
        if bucket != last_bucket:
            print(
                f"訓練進度 {progress:.0%}："
                f"{int(payload.get('completed_timesteps', 0)):,}／"
                f"{int(payload.get('total_timesteps', 0)):,} 步"
            )
            last_bucket = bucket

    result = train_rl_agent(
        args.environment,
        config,
        progress_callback=report,
        resume_from=args.resume,
    )
    print(f"訓練完成：{result.paths.run_dir}")
    print(f"使用裝置：{result.resolved_device}")
    print(f"最佳模型：{result.selected_model}")
    print(f"測試報酬：{float(result.metrics['test']['total_return']):.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
