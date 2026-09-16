"""在 Kaggle T4 GPU 執行完整 Transformer V3 正式候選模型訓練。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from time import monotonic
import traceback
from zipfile import ZipFile


WORKING_ROOT = Path("/kaggle/working")
RESULT_ROOT = WORKING_ROOT / "transformer_v3_formal"
PROGRESS_PATH = WORKING_ROOT / "formal_progress.json"
SUMMARY_PATH = WORKING_ROOT / "formal_run_summary.json"


def _json_write(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare_bundle_root(destination: Path) -> Path:
    archives = list(Path("/kaggle/input").rglob("transformer_v3_formal_bundle.zip"))
    if archives:
        destination.mkdir(parents=True, exist_ok=True)
        with ZipFile(archives[0]) as handle:
            handle.extractall(destination)
        return destination
    manifests = list(Path("/kaggle/input").rglob("formal_manifest.json"))
    if not manifests:
        raise FileNotFoundError("Kaggle Input 找不到 Transformer V3 正式訓練包")
    return manifests[0].parent


def main() -> int:
    started = monotonic()
    try:
        root = _prepare_bundle_root(Path("/tmp/ai_quant_transformer_v3_formal"))
        sys.path.insert(0, str(root / "src"))

        import torch

        from ai_quant_trading.transformer.config import (
            TemporalTransformerConfig,
            TransformerTrainingConfig,
        )
        from ai_quant_trading.transformer.training import train_temporal_transformer

        if not torch.cuda.is_available():
            raise RuntimeError("Kaggle Session 沒有可用 GPU")
        gpu_name = torch.cuda.get_device_name(0)
        if "T4" not in gpu_name.upper():
            raise RuntimeError(f"正式訓練要求 Tesla T4，目前取得：{gpu_name}")

        manifest = json.loads((root / "formal_manifest.json").read_text(encoding="utf-8"))
        source = root / "data" / "transformer_v3_formal.csv"
        expected_hash = str(dict(manifest["data"])["sha256"])
        observed_hash = _sha256(source)
        if observed_hash != expected_hash:
            raise RuntimeError("正式訓練資料 SHA-256 驗證失敗")

        hardware = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": gpu_name,
            "gpu_count": torch.cuda.device_count(),
            "cpu_count": os.cpu_count(),
        }
        print("正式訓練硬體：", json.dumps(hardware, ensure_ascii=False), flush=True)

        model_config = TemporalTransformerConfig(
            input_features=256,
            sequence_length=96,
            d_model=96,
            n_heads=4,
            n_layers=3,
            feedforward_dim=192,
            dropout=0.10,
            latent_dim=16,
            return_horizons=(1, 5, 20),
            regime_classes=3,
            architecture_version=3,
            local_kernel_size=3,
            quantile_levels=(0.10, 0.50, 0.90),
            patch_size=4,
            patch_stride=2,
            volatility_regime_classes=3,
        )
        training_config = TransformerTrainingConfig(
            epochs=50,
            batch_size=128,
            learning_rate=3e-4,
            weight_decay=1e-4,
            warmup_ratio=0.05,
            gradient_clip=1.0,
            early_stopping_patience=8,
            mixed_precision=True,
            device="cuda",
            num_workers=2,
            cpu_threads=4,
            train_fraction=0.60,
            validation_fraction=0.20,
            seed=42,
            return_loss_weight=1.0,
            volatility_loss_weight=0.5,
            regime_loss_weight=0.25,
            direction_loss_weight=0.50,
            quantile_loss_weight=0.25,
            direction_threshold_bps=12.0,
            label_smoothing=0.02,
            fee_bps_per_side=4.0,
            slippage_bps_per_side=2.0,
            max_observed_spread_bps=50.0,
            edge_loss_weight=0.50,
            excursion_loss_weight=0.25,
            tradeability_loss_weight=0.25,
            volatility_regime_loss_weight=0.20,
            probability_calibration=True,
        )

        last_printed_percent = -1

        def progress(payload: dict[str, object]) -> None:
            nonlocal last_printed_percent
            event = {
                **payload,
                "updated_elapsed_seconds": monotonic() - started,
                "hardware": hardware,
                "data_end_at": dict(manifest["data"])["end_at"],
            }
            _json_write(PROGRESS_PATH, event)
            status = str(payload.get("status", ""))
            percent = int(float(payload.get("progress", 0.0)) * 100)
            should_print = (
                status in {"preparing", "validating", "complete"}
                or percent >= last_printed_percent + 5
            )
            if should_print:
                last_printed_percent = max(last_printed_percent, percent)
                metrics = dict(payload.get("metrics", {}))
                print(
                    f"V3 {status} {percent}% | epoch "
                    f"{payload.get('epoch', 0)}/{payload.get('epochs', 50)} | "
                    f"train={float(metrics.get('train_loss', 0.0)):.6f} | "
                    f"val={float(metrics.get('validation_loss', 0.0)):.6f}",
                    flush=True,
                )

        RESULT_ROOT.mkdir(parents=True, exist_ok=True)
        result = train_temporal_transformer(
            [source],
            model_config,
            training_config,
            RESULT_ROOT,
            run_name="btc_15m_mtf_v3_formal_seed42",
            progress_callback=progress,
        )
        training_summary = json.loads(result.summary_json.read_text(encoding="utf-8"))
        final_summary = {
            "schema_version": 1,
            "status": "complete",
            "candidate_only": True,
            "duration_seconds": monotonic() - started,
            "hardware": hardware,
            "data_manifest": manifest,
            "run_dir": str(result.run_dir),
            "model_path": str(result.model_path),
            "training_summary": training_summary,
        }
        _json_write(SUMMARY_PATH, final_summary)
        print("FORMAL_TRANSFORMER_V3_COMPLETE", flush=True)
        return 0
    except Exception as error:
        _json_write(
            SUMMARY_PATH,
            {
                "schema_version": 1,
                "status": "failed",
                "duration_seconds": monotonic() - started,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            },
        )
        traceback.print_exc()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
