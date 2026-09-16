"""把訓練電腦匯出的 SAC/PPO 模型包匯回交易系統。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_quant_trading.training_portable import import_rl_model_package  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="匯入可信任的 SAC/PPO 模型 ZIP")
    parser.add_argument("archive", type=Path, help="訓練中心匯出的模型 ZIP")
    parser.add_argument(
        "--trust-local-model",
        action="store_true",
        help="確認模型包是由你控制的訓練電腦產生，並接受 SB3 反序列化風險",
    )
    args = parser.parse_args()
    if not args.trust_local_model:
        parser.error("匯入前必須確認來源並加入 --trust-local-model")
    training_dir = import_rl_model_package(
        args.archive,
        PROJECT_ROOT / "data" / "processed" / "rl" / "environments",
        trusted_local_artifact=True,
    )
    print(f"模型已匯入：{training_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
