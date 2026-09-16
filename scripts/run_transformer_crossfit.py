"""以 expanding／rolling cross-fit 產生可供 SAC 使用的 Transformer OOS 預測。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_quant_trading.transformer import (  # noqa: E402
    TemporalTransformerConfig,
    TransformerCrossFitConfig,
    TransformerTrainingConfig,
    run_transformer_crossfit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="建立因果正確、final holdout 封存的 Transformer cross-fit OOS 資料"
    )
    parser.add_argument("--source", required=True, help="BTC 15m 多週期特徵 CSV")
    parser.add_argument(
        "--output-dir",
        default="artifacts/transformer/crossfit",
        help="模型與 OOS 預測輸出資料夾",
    )
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--mode", choices=["expanding", "rolling"], default="expanding")
    parser.add_argument("--initial-train-fraction", type=float, default=0.40)
    parser.add_argument("--final-holdout-fraction", type=float, default=0.10)
    parser.add_argument("--minimum-fit-rows", type=int, default=35_000)
    parser.add_argument("--minimum-oos-rows", type=int, default=8_000)
    parser.add_argument("--minimum-final-holdout-rows", type=int, default=8_000)
    parser.add_argument("--rolling-train-rows", type=int)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    model = TemporalTransformerConfig(
        input_features=256,
        sequence_length=256,
        d_model=128,
        n_heads=8,
        n_layers=4,
        feedforward_dim=384,
        dropout=0.15,
        latent_dim=24,
        return_horizons=(1, 5, 20),
        patch_size=8,
        patch_stride=4,
        hierarchical_direction=True,
        horizon_adapter_dim=64,
    )
    training = TransformerTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=1.2e-4,
        weight_decay=5e-4,
        warmup_ratio=0.10,
        gradient_clip=0.75,
        early_stopping_patience=8,
        mixed_precision=True,
        device=args.device,
        num_workers=args.num_workers,
        cpu_threads=args.cpu_threads,
        train_fraction=0.60,
        validation_fraction=0.20,
        seed=args.seed,
        return_loss_weight=0.50,
        volatility_loss_weight=0.20,
        regime_loss_weight=0.10,
        direction_loss_weight=0.75,
        side_loss_weight=0.90,
        quantile_loss_weight=0.15,
        direction_threshold_bps=12.0,
        movement_threshold_bps=2.0,
        movement_atr_multiplier=0.10,
        label_smoothing=0.01,
        fee_bps_per_side=4.0,
        slippage_bps_per_side=2.0,
        edge_loss_weight=0.90,
        excursion_loss_weight=0.20,
        tradeability_loss_weight=0.75,
        volatility_regime_loss_weight=0.10,
        checkpoint_metric="hierarchical_skill_score",
    )
    crossfit = TransformerCrossFitConfig(
        folds=args.folds,
        mode=args.mode,
        initial_train_fraction=args.initial_train_fraction,
        final_holdout_fraction=args.final_holdout_fraction,
        minimum_fit_rows=args.minimum_fit_rows,
        minimum_oos_rows=args.minimum_oos_rows,
        minimum_final_holdout_rows=args.minimum_final_holdout_rows,
        rolling_train_rows=args.rolling_train_rows,
    )

    def progress(payload: dict[str, object]) -> None:
        print(
            f"[{payload.get('status')}] fold "
            f"{payload.get('fold', 0)}/{payload.get('folds', 0)} | "
            f"{float(payload.get('progress', 0.0)):.1%}",
            flush=True,
        )

    result = run_transformer_crossfit(
        Path(args.source),
        model,
        training,
        crossfit,
        Path(args.output_dir),
        progress_callback=progress,
    )
    print(f"OOS 預測：{result.predictions_csv}")
    print(f"Cross-fit 契約：{result.summary_json}")
    print(f"預測列數：{result.predicted_rows:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
