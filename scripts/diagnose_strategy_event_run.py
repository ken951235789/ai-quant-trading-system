"""唯讀分析已完成事件模型；另存診斷，不修改舊產物、不選 Test 門檻。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_quant_trading.transformer.event_diagnostics import (  # noqa: E402
    diagnose_events, training_reference,
)
from ai_quant_trading.transformer.strategy_events import StrategyEventConfig  # noqa: E402


def analyze(run_dir: Path) -> dict:
    summary_path = run_dir / "training.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    training = summary["training_config"]
    if training["trading_target_mode"] != "strategy_event":
        raise ValueError("只接受策略事件研究，不混用其他模型的績效定義")
    balance = summary["training_class_balance"]
    reference = training_reference(summary["scaler"]["return_means"][0],
                                   balance["tradeability_positive"][0], balance["tradeability_total"][0])
    horizon = summary["model_config"]["return_horizons"][0]
    result = {"created_at": datetime.now(timezone.utc).isoformat(), "research_only": True,
              "thresholds_changed": False, "live_eligible": False, "splits": {}, "source_hashes": {}}
    for split in ("validation", "test"):
        path = run_dir / f"{split}_predictions.csv"
        predictions = pd.read_csv(path)
        result["splits"][split] = diagnose_events(predictions, horizon,
            StrategyEventConfig(**training["strategy_event_config"]),
            training["economic_minimum_edge_bps"], reference,
            slippage_bps_per_side=training["slippage_bps_per_side"],
            minimum_trades=training["economic_minimum_trades"])
    for path in (summary_path, run_dir / "validation_predictions.csv", run_dir / "test_predictions.csv"):
        with path.open("rb") as stream:
            result["source_hashes"][path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("診斷輸出已存在，請選新檔名以保留歷史")
    report = analyze(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"output": str(args.output),
                      "splits": {key: {"metrics": value["metrics"], "policies": value["policies"]}
                                 for key, value in report["splits"].items()}},
                     ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
