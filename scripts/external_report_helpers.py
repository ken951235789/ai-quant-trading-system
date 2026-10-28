"""公開研究入口共用的報告工具，不依賴私人 Skills 或其他研究管線。"""

import importlib.util
from pathlib import Path


def load_helper(name):
    if name not in {"new_report", "validate_report"}:
        raise ValueError("不支援的報告工具")
    path = Path(__file__).resolve().parents[1] / "tools" / "research_reports" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"external_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
