"""建立或驗證模型與設定檔 SHA-256 清單。"""

from __future__ import annotations

import argparse
from pathlib import Path

from ai_quant_trading.operations.integrity import (
    build_artifact_manifest,
    verify_artifact_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="模型成品完整性工具")
    parser.add_argument("command", choices=["build", "verify"])
    parser.add_argument("directory")
    args = parser.parse_args()
    directory = Path(args.directory)
    if args.command == "build":
        print(f"已建立：{build_artifact_manifest(directory)}")
        return 0
    verified = verify_artifact_manifest(directory)
    print(f"完整性通過：{len(verified)} 個檔案")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
