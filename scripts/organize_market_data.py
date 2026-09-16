"""整理既有加密貨幣與美股 CSV 的資料夾及檔名。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_quant_trading.data_collection.organization import (  # noqa: E402
    apply_organization,
    plan_market_data_organization,
)


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="整理市場資料檔名；預設只預覽")
    parser.add_argument("--raw-dir", default=str(PROJECT_ROOT / "data" / "raw"))
    parser.add_argument("--processed-dir", default=str(PROJECT_ROOT / "data" / "processed"))
    parser.add_argument("--apply", action="store_true", help="實際搬移；未指定時只顯示計畫")
    args = parser.parse_args(arguments)

    actions = plan_market_data_organization(args.raw_dir, args.processed_dir)
    if not actions:
        print("資料名稱與路徑已符合標準，不需要搬移。")
        return 0
    for action in actions:
        print(f"[{action.data_type}] {action.source} -> {action.target}")
    if not args.apply:
        print(f"預覽完成，共 {len(actions)} 個檔案；加上 --apply 才會實際搬移。")
        return 0
    completed = apply_organization(actions, [args.raw_dir, args.processed_dir])
    print(f"整理完成，共搬移 {len(completed)} 個檔案。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
