"""Transformer cross-fit 因果邊界與成品測試。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai_quant_trading.transformer import (
    TemporalTransformerConfig,
    TransformerCrossFitConfig,
    TransformerTrainingConfig,
    assemble_transformer_crossfit_predictions,
    build_transformer_crossfit_plan,
    load_transformer_crossfit_predictions,
    run_transformer_crossfit,
)


def _frame(rows: int = 120) -> pd.DataFrame:
    close = 100 + np.arange(rows, dtype=float) * 0.1
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=rows, freq="15min", tz="UTC"),
            "exchange": "binance_futures",
            "symbol": "BTC/USDT",
            "interval": "15m",
            "open": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": 100.0,
            "feature_a": np.linspace(-1, 1, rows),
        }
    )


def _config(mode: str = "expanding") -> TransformerCrossFitConfig:
    return TransformerCrossFitConfig(
        folds=3,
        mode=mode,  # type: ignore[arg-type]
        initial_train_fraction=0.40,
        final_holdout_fraction=0.10,
        minimum_fit_rows=30,
        minimum_oos_rows=10,
        minimum_final_holdout_rows=12,
        rolling_train_rows=36 if mode == "rolling" else None,
    )


def test_crossfit_plan_has_non_overlapping_oos_and_sealed_holdout() -> None:
    plan = build_transformer_crossfit_plan(_frame(), _config(), max_horizon_bars=20)

    assert plan.final_holdout_rows == 12
    assert plan.folds[0].fit_start == 0
    assert all(fold.fit_end == fold.oos_start for fold in plan.folds)
    assert all(
        left.oos_end == right.oos_start
        for left, right in zip(plan.folds, plan.folds[1:], strict=False)
    )
    assert plan.folds[-1].oos_end == plan.final_holdout_start


def test_rolling_crossfit_uses_bounded_training_window() -> None:
    plan = build_transformer_crossfit_plan(_frame(), _config("rolling"), max_horizon_bars=20)

    assert all(fold.fit_end - fold.fit_start <= 36 for fold in plan.folds)
    assert plan.folds[-1].fit_start > 0


def test_crossfit_assembler_rejects_prediction_from_fit_period() -> None:
    fold = build_transformer_crossfit_plan(_frame(), _config(), max_horizon_bars=20).folds[0]
    leaked = _frame().iloc[fold.fit_end - 1 : fold.oos_end].copy()
    leaked["transformer_available"] = 1.0

    with pytest.raises(ValueError, match="fit cutoff"):
        assemble_transformer_crossfit_predictions([(fold, leaked, "abc")])


def test_crossfit_runner_builds_only_oos_predictions(tmp_path: Path) -> None:
    source = tmp_path / "features.csv"
    _frame().to_csv(source, index=False)

    def trainer(
        _source: Path,
        fold: object,
        output: Path,
        _model: TemporalTransformerConfig,
        _training: TransformerTrainingConfig,
    ) -> Path:
        output.mkdir(parents=True, exist_ok=True)
        checkpoint = output / f"fold_{getattr(fold, 'fold')}.pt"
        checkpoint.write_bytes(b"checkpoint")
        return checkpoint

    def inferencer(_checkpoint: Path, context: pd.DataFrame, _fold: object) -> pd.DataFrame:
        result = context.copy()
        result["transformer_available"] = 1.0
        result["transformer_return_5"] = 0.001
        result["transformer_return_20"] = 0.002
        return result

    result = run_transformer_crossfit(
        source,
        TemporalTransformerConfig(
            input_features=8,
            sequence_length=8,
            d_model=16,
            n_heads=2,
            n_layers=1,
            feedforward_dim=32,
            latent_dim=4,
        ),
        TransformerTrainingConfig(epochs=1, batch_size=8, num_workers=0),
        _config(),
        tmp_path / "runs",
        trainer=trainer,
        inferencer=inferencer,
    )
    predictions = pd.read_csv(result.predictions_csv)
    loaded, metadata = load_transformer_crossfit_predictions(result.run_dir)

    assert result.predicted_rows == sum(
        fold.oos_end - fold.oos_start for fold in result.plan.folds
    )
    assert predictions["transformer_oos"].eq(1).all()
    assert "expected_return" in loaded
    assert loaded.attrs["expected_return_contract"]["horizon_bars"] == 5
    assert metadata["predictions_sha256"]
    assert pd.to_datetime(predictions["timestamp"], utc=True).max() < pd.Timestamp(
        result.plan.final_holdout_start_at
    )
