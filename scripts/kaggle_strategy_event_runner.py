"""Kaggle 私人 GPU 事件研究；同一份設定、資料雜湊與既有 Transformer 訓練器。"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import stat
import sys
from time import monotonic
import traceback
from zipfile import ZipFile


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_bundle(input_root: Path, destination: Path) -> tuple[Path, dict]:
    """接受 Kaggle 自動解壓或 ZIP；在匯入程式前驗證每個白名單檔案。"""
    manifests = list(input_root.rglob("strategy_event_manifest.json"))
    if len(manifests) > 1:
        raise ValueError("掛載到多份事件研究包")
    if manifests:
        root = manifests[0].parent.resolve()
    else:
        archives = list(input_root.rglob("strategy_event_bundle.zip"))
        if len(archives) != 1:
            raise FileNotFoundError("找不到唯一事件研究 ZIP 或 manifest")
        root = destination.resolve()
        root.mkdir(parents=True, exist_ok=False)
        with ZipFile(archives[0]) as archive:
            members = archive.infolist()
            if len(members) > 1500 or sum(item.file_size for item in members) > 200_000_000:
                raise ValueError("研究包超過檔案數或大小限制")
            if len({item.filename for item in members}) != len(members):
                raise ValueError("研究包包含重複檔名")
            for item in members:
                if ("\\" in item.filename or ":" in item.filename
                        or ".." in PurePosixPath(item.filename).parts
                        or not (root / item.filename).resolve().is_relative_to(root)
                        or stat.S_ISLNK(item.external_attr >> 16)):
                    raise ValueError("研究包包含不安全路徑或連結")
            archive.extractall(root)
    manifest = json.loads((root / "strategy_event_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("study") != "btc_strategy_event_v1" or not manifest.get("research_only"):
        raise ValueError("不支援的研究契約")
    files = manifest["files"]
    expected_files = set(files) | {"strategy_event_manifest.json"}
    observed_files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if expected_files != observed_files:
        raise ValueError("研究包有缺檔或未列入白名單的額外檔案")
    for relative, expected in files.items():
        path = root / relative
        if (path.is_symlink() or not path.resolve().is_relative_to(root)
                or path.stat().st_size != expected["size"] or sha256(path) != expected["sha256"]):
            raise ValueError(f"研究包完整性驗證失敗：{relative}")
    return root, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output-root", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--local-smoke", action="store_true", help="僅供本機封包隔離測試")
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "strategy_event_run_summary.json"
    progress_path = output / "strategy_event_progress.json"
    started = monotonic()
    try:
        root, manifest = prepare_bundle(args.input_root, output / "extracted_bundle")
        sys.path.insert(0, str(root / "src"))
        import numpy as np
        import pandas as pd
        import torch
        import ai_quant_trading.transformer.training as training_module
        from ai_quant_trading.transformer.config import TemporalTransformerConfig, TransformerTrainingConfig
        from ai_quant_trading.operations.integrity import build_artifact_manifest

        if not Path(training_module.__file__).resolve().is_relative_to(root):
            raise RuntimeError("誤載入封包之外的訓練程式，禁止繼續")
        payload = json.loads((root / "training_config.json").read_text(encoding="utf-8"))
        model = TemporalTransformerConfig(**payload["model"])
        config = TransformerTrainingConfig(**payload["training"])
        if config.trading_target_mode != "strategy_event":
            raise ValueError("這不是事件研究設定")
        if args.local_smoke:
            model = replace(model, sequence_length=32, d_model=16, n_layers=1,
                            feedforward_dim=32, horizon_adapter_dim=8)
            config = replace(config, epochs=1, max_rows_per_source=16000,
                             device="cpu", mixed_precision=False)
        elif not torch.cuda.is_available():
            raise RuntimeError("Kaggle 沒有 GPU；不默默改用 CPU 消耗執行時間")
        hardware = {"python": platform.python_version(), "torch": torch.__version__,
                    "numpy": np.__version__, "pandas": pd.__version__,
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        print("硬體：", json.dumps(hardware, ensure_ascii=False), flush=True)
        print("開始事件研究，完整資料筆數：", manifest["data"]["rows"], flush=True)
        write_json(progress_path, {"status": "preparing_data", "progress": 0,
                                   "updated_at": datetime.now(timezone.utc).isoformat()})
        last_message = [None]

        def progress(event: dict) -> None:
            write_json(progress_path, {**event, "wall_seconds": monotonic() - started,
                                       "updated_at": datetime.now(timezone.utc).isoformat()})
            key = (str(event.get("status")), int(float(event.get("progress", 0)) * 100))
            if key != last_message[0]:
                print(f"{key[0]} {key[1]}% epoch={event.get('epoch')}/{config.epochs} "
                      f"elapsed={monotonic()-started:.1f}s", flush=True)
                last_message[0] = key

        result = training_module.train_temporal_transformer(
            [root / "data/btc_15m.csv"], model, config, output / "strategy_event_results",
            run_name=f"btc_event_seed{config.seed}", progress_callback=progress,
        )
        summary = json.loads(result.summary_json.read_text(encoding="utf-8"))
        write_json(summary_path, {
            "schema_version": 1, "status": "complete", "research_only": True,
            "live_eligible": False, "smoke_only": args.local_smoke,
            "duration_seconds": monotonic() - started, "hardware": hardware,
            "data_manifest": manifest, "run_dir": str(result.run_dir),
            "training_summary": summary,
        })
        build_artifact_manifest(output, files=[summary_path, progress_path])
        print("STRATEGY_EVENT_RESEARCH_COMPLETE", flush=True)
        print(json.dumps(summary["test_metrics"], ensure_ascii=False), flush=True)
        return 0
    except Exception as error:
        write_json(summary_path, {"status": "failed", "live_eligible": False,
                                  "duration_seconds": monotonic() - started,
                                  "error": f"{type(error).__name__}: {error}",
                                  "traceback": traceback.format_exc()})
        write_json(progress_path, {"status": "failed", "error_type": type(error).__name__})
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
