"""從專案根目錄啟動 Step 9 PPO／SAC 訓練。"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_quant_trading.reinforcement_learning.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
