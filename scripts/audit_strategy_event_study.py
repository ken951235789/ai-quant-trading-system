"""驗證配對研究的切分、標籤、設定與成品雜湊；不修改研究結果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_quant_trading.operations.integrity import sha256_file, verify_artifact_manifest  # noqa: E402
from ai_quant_trading.persistence import write_json_atomic  # noqa: E402
from ai_quant_trading.transformer.event_study import VARIANTS, choose_challenger, window_config  # noqa: E402
from ai_quant_trading.transformer.config import TransformerTrainingConfig  # noqa: E402


def audit(root: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("稽核輸出已存在，請使用新檔名")
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    records = json.loads((root / "records.json").read_text(encoding="utf-8"))
    study = json.loads((root / "study.json").read_text(encoding="utf-8"))
    if study["status"] != "complete" or len(records) != plan["max_runs"]:
        raise ValueError("研究尚未完整完成")
    if sha256_file(Path(plan["source"])) != plan["source_sha256"]:
        raise ValueError("來源資料雜湊與預先登記不符")
    for path, expected in plan["code_sha256"].items():
        if sha256_file(ROOT / path) != expected:
            raise ValueError(f"程式版本與訓練前不符：{path}")
    base = TransformerTrainingConfig(**plan["training"])
    windows = {row["fold"]: row for row in plan["windows"]}
    challenger = choose_challenger(records)
    if challenger != study["challenger"]:
        raise ValueError("挑戰組不符合原先 Selection 規則")
    expected_jobs = {(0, name, seed) for name in VARIANTS for seed in plan["seeds"]}
    expected_jobs.update((fold, name, seed) for fold in (1, 2, 3)
                         for name in ("A_legacy_smooth", challenger) for seed in plan["seeds"])
    actual_jobs = {(r["fold"], r["variant"], r["seed"]) for r in records}
    if expected_jobs != actual_jobs or len(actual_jobs) != len(records):
        raise ValueError("研究工作重複或有遺漏")
    references, targets = {}, {}
    verified, hashes = 0, {}
    for record in records:
        run = (root / record["run_dir"]).resolve()
        if not run.is_relative_to(root.resolve()):
            raise ValueError("成品超出研究目錄")
        verified += len(verify_artifact_manifest(run))
        summary = json.loads((run / "training.json").read_text(encoding="utf-8"))
        window = windows[record["fold"]]
        expected = window_config(base, window, record["variant"], record["seed"]).to_dict()
        if summary["training_config"] != expected or summary["live_eligible"]:
            raise ValueError("訓練設定不符合預先登記或模型錯誤標記可實盤")
        reference = summary["event_reference"]
        fold = record["fold"]
        if fold in references and reference != references[fold]:
            raise ValueError("相同窗口的 Train 參考分布不一致")
        references[fold] = reference
        for boundary in summary["calibration_protocol"]["boundaries"]:
            if boundary["calibration_last"] + boundary["purge_bars"] >= boundary["selection_first"]:
                raise ValueError("Calibration 標籤與 Selection 邊界重疊")
        for split in ("validation", "test"):
            path = run / f"{split}_predictions.csv"
            frame = pd.read_csv(path)
            if frame.timestamp_ns.duplicated().any():
                raise ValueError("同一模型有重複決策時間")
            lower = window["train_end"] if split == "validation" else window["validation_end"]
            upper = window["validation_end"] if split == "validation" else window["source_end"]
            if (frame.endpoint < lower + expected["embargo_bars"]).any() or (frame.event_exit_endpoint >= upper).any():
                raise ValueError("預測或成交標籤超出時間切分")
            columns = [name for name in frame if name in ("endpoint", "timestamp_ns")
                       or name.startswith(("event_", "actual_return_", "actual_tradeability_"))]
            target = frame[sorted(columns)].sort_values("timestamp_ns").reset_index(drop=True)
            key = (fold, split)
            if key in targets:
                pd.testing.assert_frame_equal(target, targets[key], check_exact=True)
            targets[key] = target
            if not np.isfinite(frame.select_dtypes(include="number").to_numpy()).all():
                raise ValueError("預測包含非有限數值")
            hashes[path.relative_to(root).as_posix()] = sha256_file(path)
    result = {"status": "pass", "runs": len(records), "verified_model_and_json_files": verified,
              "paired_targets_equal": True, "train_references_equal_within_window": True,
              "boundaries_valid": True, "selection_ignores_test": True, "source_unchanged": True,
              "code_matches_preregistration": True, "prediction_sha256": hashes,
              "scope": "驗證工程契約，不認證獲利或實盤資格"}
    write_json_atomic(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.study.resolve(), args.output.resolve())
    print(json.dumps({key: value for key, value in result.items() if key != "prediction_sha256"},
                     ensure_ascii=False, indent=2))
