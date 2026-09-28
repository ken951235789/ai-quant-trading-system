"""以白名單建立私人 Kaggle 事件研究包，不打包帳戶、模型或環境變數。"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SAFE_COLUMNS = ("timestamp", "symbol", "exchange", "interval", "open", "high", "low", "close", "volume")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def local_dependencies(source_root: Path, entry: str) -> list[Path]:
    """追蹤本地 Python 匯入及父套件，只收錄可定位的 .py，不複製整個專案。"""
    pending = [entry]
    visited: set[str] = set()
    paths: set[Path] = set()
    while pending:
        name = pending.pop()
        if name in visited or not name.startswith("ai_quant_trading"):
            continue
        visited.add(name)
        relative = Path(*name.split("."))
        path = source_root / relative.with_suffix(".py")
        package = False
        if not path.is_file():
            path = source_root / relative / "__init__.py"
            package = True
        if not path.is_file():
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(source_root.resolve()):
            raise ValueError("來源程式不可使用外部連結")
        paths.add(path)
        parts = name.split(".")
        pending.extend(".".join(parts[:index]) for index in range(1, len(parts)))
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        package_name = name if package else name.rpartition(".")[0]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                pending.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    base = importlib.util.resolve_name("." * node.level + base, package_name)
                pending.append(base)
                pending.extend(f"{base}.{alias.name}" for alias in node.names if alias.name != "*")
    return sorted(paths)


def check_source_text(path: Path) -> None:
    """阻擋常見硬編碼憑證格式；错误只列檔名，不輸出疑似密鑰。"""
    text = path.read_text(encoding="utf-8-sig")
    patterns = (
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        r"\b(?:ghp_|github_pat_|hf_)[A-Za-z0-9_]{20,}",
        r"\bKGAT_[A-Za-z0-9_\-]{16,}",
        r"\bAKIA[A-Z0-9]{16}\b",
        r"(?i)(?:api_key|api_secret|password|access_token)\s*[:=]\s*['\"][A-Za-z0-9/+\-]{24,}['\"]",
    )
    if any(re.search(pattern, text) for pattern in patterns):
        raise ValueError(f"來源疑似包含硬編碼憑證，禁止上傳：{path.name}")


def build(source: Path, config_path: Path, output: Path, username: str) -> dict:
    if output.exists():
        raise FileExistsError("輸出目錄已存在；請換新目錄，避免覆蓋或重複提交")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", username):
        raise ValueError("Kaggle 使用者名稱格式不合法")
    sys.path.insert(0, str(SRC))
    from ai_quant_trading.transformer.config import TemporalTransformerConfig, TransformerTrainingConfig
    from ai_quant_trading.transformer.strategy_events import validate_event_bars

    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["training"]["device"] = "cuda"
    TemporalTransformerConfig(**config["model"])
    training = TransformerTrainingConfig(**config["training"])
    if training.trading_target_mode != "strategy_event" or training.max_rows_per_source is not None:
        raise ValueError("本包只接受完整資料的 strategy_event 研究")
    initial_hash = sha256(source)
    frame = pd.read_csv(source, usecols=list(SAFE_COLUMNS))[list(SAFE_COLUMNS)]
    validate_event_bars(frame)
    if sha256(source) != initial_hash:
        raise ValueError("來源在讀取時變動，請先建立不變快照")
    output.mkdir(parents=True, exist_ok=False)
    bundle = output / "staging"
    (bundle / "data").mkdir(parents=True)
    market = bundle / "data/btc_15m.csv"
    frame.to_csv(market, index=False, encoding="utf-8")
    (bundle / "training_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    for path in local_dependencies(SRC, "ai_quant_trading.transformer.training"):
        check_source_text(path)
        destination = bundle / "src" / path.relative_to(SRC)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    files = {path.relative_to(bundle).as_posix(): {"sha256": sha256(path), "size": path.stat().st_size}
             for path in sorted(bundle.rglob("*")) if path.is_file()}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    manifest = {
        "schema_version": 1, "study": "btc_strategy_event_v1", "research_only": True,
        "created_at": datetime.now(timezone.utc).isoformat(), "git_revision": revision,
        "dirty_worktree": True, "configuration": "training_config.json", "files": files,
        "data": {"path": "data/btc_15m.csv", "rows": len(frame), "columns": list(SAFE_COLUMNS),
                 "start_utc": pd.to_datetime(frame.timestamp.iloc[0], utc=True).isoformat(),
                 "end_utc": pd.to_datetime(frame.timestamp.iloc[-1], utc=True).isoformat(),
                 "original_source_sha256": initial_hash, "sha256": sha256(market)},
        "policy": {"private_only": True, "live_eligible": False, "seed_selection_uses_test": False,
                   "test_is_pristine_holdout": False, "funding": "assumed_reserve_not_actual",
                   "tax": "unverified", "no_credentials_included_by_allowlist": True},
    }
    (bundle / "strategy_event_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    dataset = output / "dataset"
    kernel = output / "kernel"
    dataset.mkdir()
    kernel.mkdir()
    with ZipFile(dataset / "strategy_event_bundle.zip", "w", ZIP_DEFLATED) as archive:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(bundle).as_posix())
    dataset_id = f"{username}/btc-strategy-event-v1-input"
    kernel_id = f"{username}/btc-transformer-strategy-event-v1"
    (dataset / "dataset-metadata.json").write_text(json.dumps({
        "title": "BTC Strategy Event V1 Input", "id": dataset_id,
        "licenses": [{"name": "other"}], "description": "Private strategy-event research snapshot.",
    }, indent=2), encoding="utf-8")
    runner = ROOT / "scripts/kaggle_strategy_event_runner.py"
    check_source_text(runner)
    shutil.copy2(runner, kernel / "train_strategy_event.py")
    (kernel / "kernel-metadata.json").write_text(json.dumps({
        "id": kernel_id, "title": "BTC Transformer Strategy Event V1",
        "code_file": "train_strategy_event.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4", "dataset_sources": [dataset_id],
        "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }, indent=2), encoding="utf-8")
    return {"dataset_id": dataset_id, "kernel_id": kernel_id, "rows": len(frame),
            "package_files": len(files), "archive_bytes": (dataset / "strategy_event_bundle.zip").stat().st_size,
            "bundle_root": str(bundle.resolve()), "output": str(output.resolve())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/transformer_strategy_event.example.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--username", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.config, args.output, args.username),
                     ensure_ascii=False, indent=2))
