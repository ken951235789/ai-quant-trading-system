"""執行固定策略候選事件研究，預設只檢查資料；不提交雲端或連接交易帳戶。"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_quant_trading.transformer.config import (  # noqa: E402
    TemporalTransformerConfig, TransformerTrainingConfig,
)
from ai_quant_trading.transformer.dataset import prepare_transformer_datasets  # noqa: E402
from ai_quant_trading.transformer.training import train_temporal_transformer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Binance BTCUSDT 15m 原始 CSV")
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/transformer_strategy_event.example.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/strategy_event_research")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true", help="依設定開始研究訓練")
    mode.add_argument("--smoke", action="store_true", help="CPU 一輪小模型流程測試，不具績效意義")
    args = parser.parse_args()
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    model = TemporalTransformerConfig(**payload["model"])
    config = TransformerTrainingConfig(**payload["training"])
    if config.trading_target_mode != "strategy_event":
        raise ValueError("此指令只接受策略事件研究設定")
    if args.smoke:
        model = replace(model, sequence_length=32, d_model=16, n_layers=1,
                        feedforward_dim=32, horizon_adapter_dim=8)
        config = replace(config, epochs=1, max_rows_per_source=16_000, device="cpu",
                         mixed_precision=False)
    with args.source.open("rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    print("研究模式：不更動現有模型、SAC 或實盤設定。", flush=True)
    if not (args.train or args.smoke):
        data = prepare_transformer_datasets([args.source], model, config)
        print(json.dumps({"status": "preflight_only", "source_sha256": source_hash,
                          "samples": data.sample_counts, "features": data.scaler.feature_columns,
                          "contract": data.diagnostics["strategy_event_contract"]},
                         ensure_ascii=False, indent=2))
        return 0
    last_progress = [None]

    def progress(event: dict[str, object]) -> None:
        percent = int(float(event.get("progress", 0)) * 100)
        key = (percent, event.get("status"))
        if key != last_progress[0]:
            print(f"{percent:3d}% {event.get('status')} epoch={event.get('epoch')} "
                  f"elapsed={float(event.get('elapsed_seconds', 0)):.1f}s", flush=True)
            last_progress[0] = key

    result = train_temporal_transformer(
        [args.source], model, config, args.output,
        run_name="event_smoke" if args.smoke else "event_research", progress_callback=progress,
    )
    with args.source.open("rb") as stream:
        end_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "research_only": True,
                "smoke_only": args.smoke, "source": str(args.source.resolve()),
                "source_sha256": source_hash, "source_unchanged": source_hash == end_hash,
                "status": "complete" if source_hash == end_hash else "invalid_source_changed",
                "config": str(args.config.resolve()), "model": model.to_dict(),
                "training": config.to_dict(), "tax": "unverified", "live_eligible": False}
    (result.run_dir / "experiment.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    from ai_quant_trading.operations.integrity import build_artifact_manifest

    build_artifact_manifest(result.run_dir)
    if source_hash != end_hash:
        raise RuntimeError("來源 CSV 在執行中變動，本次結果不可使用，請用固定快照重新執行")
    print(json.dumps({"run_dir": str(result.run_dir), "metrics": result.metrics},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
