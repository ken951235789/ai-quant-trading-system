"""策略事件資料切分、模型保存與舊管線隔離的端到端測試。"""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from ai_quant_trading.transformer.config import TemporalTransformerConfig, TransformerTrainingConfig
from ai_quant_trading.transformer.dataset import prepare_transformer_datasets
from ai_quant_trading.transformer.inference import _load_checkpoint
from ai_quant_trading.transformer.training import _split_calibration, train_temporal_transformer


def configs():
    root = Path(__file__).resolve().parents[2]
    payload = json.loads((root / "configs/transformer_strategy_event.example.json").read_text())
    model = TemporalTransformerConfig(**payload["model"])
    training = TransformerTrainingConfig(**payload["training"])
    return (replace(model, d_model=16, sequence_length=8, n_layers=1,
                    feedforward_dim=32, horizon_adapter_dim=8),
            replace(training, epochs=1, device="cpu", mixed_precision=False,
                    strategy_event_config={**training.strategy_event_config, "warmup_hours": 200}))


def write_source(path):
    rng = np.random.default_rng(19)
    rows = 7000
    movement = 0.0005 * np.sin(np.arange(rows) / 160) + rng.normal(0, 0.0007, rows)
    close = 100 * np.exp(movement.cumsum())
    opening = np.r_[close[0], close[:-1]]
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=rows, freq="15min", tz="UTC"),
        "open": opening, "high": np.maximum(opening, close) * 1.0002,
        "low": np.minimum(opening, close) * 0.9998, "close": close,
        "volume": rng.uniform(100, 1000, rows), "symbol": "BTC/USDT",
        "exchange": "binance_futures", "interval": "15m",
        "future_cheat": np.roll(close, -10),
    })
    frame.to_csv(path, index=False)
    return path


def test_event_split_labels_and_training_smoke(tmp_path):
    source = write_source(tmp_path / "events.csv")
    model, config = configs()
    data = prepare_transformer_datasets([source], model, config)
    assert data.scaler.label_mode == "strategy_event_net_pnl_v1"
    assert len(data.scaler.feature_columns) <= 53
    assert not {"open", "high", "low", "close", "volume", "future_cheat", "event_atr"}.intersection(
        data.scaler.feature_columns)
    calibration, selection, protocol = _split_calibration(data, config)
    assert protocol["boundaries"][0]["purge_bars"] == 33
    assert calibration.references[-1][1] + 33 < selection.references[0][1]
    for dataset, upper in ((data.train, 4200), (data.validation, 5600), (data.test, 7000)):
        for series_index, endpoint in dataset.references:
            series = dataset.series[series_index]
            assert series.event_metadata["exit_endpoint"][endpoint] < upper
            actual = series.future_returns[endpoint, 0]
            assert series.tradeability[endpoint, 0] == (actual > config.direction_threshold_bps / 10000)
    progress = []
    result = train_temporal_transformer([source], model, config, tmp_path / "runs",
                                        progress_callback=progress.append)
    assert progress[-1]["status"] == "complete"
    payload = torch.load(result.model_path, map_location="cpu", weights_only=True)
    assert payload["artifact_kind"] == "strategy_event_research_v1"
    with pytest.raises(ValueError, match="研究模型"):
        _load_checkpoint(result.model_path, torch.device("cpu"))
    summary = json.loads(result.summary_json.read_text(encoding="utf-8"))
    assert summary["live_eligible"] is False
    predictions = pd.read_csv(result.run_dir / "test_predictions.csv")
    assert predictions["label_contract"].nunique() == 1
    assert predictions["source_key"].nunique() == 1
    assert "event_exit_reason_code" in predictions
    assert "raw_tradeability_brier_32" in result.metrics
    assert "calibrated_tradeability_brier_32" in result.metrics
    assert "regime_accuracy" not in result.metrics
    assert np.isfinite(list(result.metrics.values())).all()


def test_event_configuration_cannot_silently_repurpose_other_heads(tmp_path):
    model, config = configs()
    with pytest.raises(ValueError):
        replace(config, side_loss_weight=0.5)
    with pytest.raises(ValueError):
        replace(config, calibration_fraction=0)
    source = write_source(tmp_path / "invalid_horizon.csv")
    with pytest.raises(ValueError, match="return_horizons"):
        prepare_transformer_datasets([source], replace(model, return_horizons=(20,),
                                     timing_horizon=20, primary_horizon=20, regime_horizon=20), config)
