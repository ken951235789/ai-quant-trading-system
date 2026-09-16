"""把 Step 2 的 OHLCV CSV 轉換成 Step 3 特徵資料集。"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_quant_trading.features.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
