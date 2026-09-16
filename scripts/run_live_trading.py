"""Step 8 實盤交易執行腳本。"""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_quant_trading.live_trading.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
