"""研究契約、時間隔離、品質閘門與固定訊號資金配置測試。"""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from stable_baselines3.common.env_checker import check_env

from ai_quant_trading.research.sizing import EventSizingEnv, SizingConfig, evaluate_sizing, fit_scaler, run_sizing_comparison
from ai_quant_trading.research.staged_training import assess_transformer, build_contract, chronological_windows, digest_contract, require_sizing_gate, split_oos
from ai_quant_trading.research import staged_training
from ai_quant_trading.transformer.config import TemporalTransformerConfig, TransformerTrainingConfig
from ai_quant_trading.transformer.strategy_events import StrategyEventConfig


def fixture_events(count=60):
    index = np.arange(count)
    return pd.DataFrame({"series_index": 0, "endpoint": index * 40,
        "event_entry_endpoint": index * 40 + 1, "event_exit_endpoint": index * 40 + 10,
        "event_side": np.where(index % 2, -1, 1), "causal_atr_fraction": .003 + index * .00001,
        "predicted_return_32": .004 + index * .00001, "predicted_return_q10_32": -.005,
        "predicted_return_q50_32": .002, "predicted_return_q90_32": .01,
        "tradeability_probability_32": .7, "actual_tradeability_32": index % 2,
        "actual_return_32": np.where(index % 2, .008, -.004)})


def configs():
    root = Path(__file__).resolve().parents[2]
    payload = json.loads((root / "configs/transformer_strategy_event_v2.example.json").read_text(encoding="utf-8"))
    return TemporalTransformerConfig(**payload["model"]), TransformerTrainingConfig(**payload["training"])


def test_contract_identifies_net_event_not_old_gross_horizon():
    model, training = configs()
    contract = build_contract(model, training, SizingConfig())
    assert contract["horizon_bars"] == 32
    assert not contract["expected_return_is_gross"]
    assert not contract["compatible_with_existing_live_models"]
    assert digest_contract(contract) != digest_contract({**contract, "horizon_bars": 20})
    with pytest.raises(ValueError, match="horizon"):
        build_contract(replace(model, return_horizons=(20,), timing_horizon=20, primary_horizon=20, regime_horizon=20), training, SizingConfig())


def test_chronological_windows_keep_tail_and_split_purges_exits():
    windows = chronological_windows(100_000)
    assert windows[0]["validation_end"] == 40_000
    assert windows[-1]["source_end"] == 90_000
    assert all(a["source_end"] == b["validation_end"] for a, b in zip(windows, windows[1:]))
    splits = split_oos(fixture_events(), 0, 2400, 32)
    assert splits["train"].event_exit_endpoint.max() < 1440
    assert splits["validation"].endpoint.min() >= 1472
    assert splits["validation"].event_exit_endpoint.max() < 1920
    assert splits["test"].endpoint.min() >= 1952


def test_gate_requires_all_registered_seeds():
    gates = {str(i): {"passed": True} for i in range(5)}
    require_sizing_gate(gates, 2)
    gates["4"]["passed"] = False
    with pytest.raises(ValueError, match="禁止"):
        require_sizing_gate(gates, 2)
    with pytest.raises(ValueError):
        require_sizing_gate({"1": {"passed": True}}, 1)


def test_transformer_gate_does_not_read_sac_test():
    _, training = configs()
    frame = fixture_events()
    verdict = assess_transformer({"train": frame.iloc[:30], "validation": frame.iloc[30:],
        "test": object()}, 32, StrategyEventConfig(), training, 50)
    assert not verdict["passed"]
    assert verdict["score_ranking"]["threshold_fit_on"] == "oos_train"
    assert not verdict["checks"]["minimum_filtered_trades"]


def test_future_outcomes_cannot_enter_sizing_observation():
    frame = fixture_events()
    scaler = fit_scaler(frame, 32)
    changed = frame.copy()
    changed["actual_return_32"] = .20
    changed["event_exit_price"] = 1e9
    a = EventSizingEnv(frame, 32, SizingConfig(), scaler)
    b = EventSizingEnv(changed, 32, SizingConfig(), scaler)
    np.testing.assert_equal(a.reset()[0], b.reset()[0])
    assert a.step(np.array([.5]))[1] != b.step(np.array([.5]))[1]


def test_fixed_and_volatility_use_same_events_and_accounting():
    frame = fixture_events()
    scaler = fit_scaler(frame, 32)
    config = SizingConfig()
    fixed, metrics = evaluate_sizing(frame, 32, config, scaler, "fixed")
    vol, _ = evaluate_sizing(frame, 32, config, scaler, "volatility")
    assert fixed.endpoint.equals(vol.endpoint)
    expected = config.initial_capital * np.prod(1 + config.fixed_fraction * frame.actual_return_32)
    assert fixed.equity.iloc[-1] == pytest.approx(expected)
    assert metrics["total_return"] == pytest.approx(expected / config.initial_capital - 1)
    check_env(EventSizingEnv(frame, 32, config, scaler), warn=False)


def test_risk_halt_is_persistent_and_does_not_change_schedule():
    frame = fixture_events()
    frame["actual_return_32"] = -.30
    config = SizingConfig(max_settled_drawdown=.05)
    env = EventSizingEnv(frame, 32, config, fit_scaler(frame, 32))
    first = env.step(np.array([1.]))[-1]
    second = env.step(np.array([1.]))[-1]
    assert first["halted"] and first["allocation"] == .2
    assert second["allocation"] == 0 and second["endpoint"] == 40
    assert second["equity"] == first["equity"]


@pytest.mark.parametrize("mutation", ["overlap", "nan", "future_entry", "volatility"])
def test_sizing_rejects_invalid_inputs(mutation):
    frame = fixture_events()
    scaler = fit_scaler(frame, 32)
    if mutation == "overlap":
        frame.loc[1, "endpoint"] = 3
    elif mutation == "nan":
        frame.loc[0, "actual_return_32"] = np.nan
    elif mutation == "future_entry":
        frame.loc[0, "event_entry_endpoint"] = -1
    else:
        frame.loc[0, "causal_atr_fraction"] = 0
    with pytest.raises(ValueError):
        EventSizingEnv(frame, 32, SizingConfig(), scaler)


def test_real_sac_training_smoke_is_not_profit_evidence(tmp_path):
    frame = fixture_events(120)
    splits = {"train": frame.iloc[:60], "validation": frame.iloc[60:90], "test": frame.iloc[90:]}
    result = run_sizing_comparison(splits, 32, SizingConfig(total_timesteps=40, seeds=(7,)), tmp_path / "smoke", "fixture_only")
    assert (tmp_path / "smoke/seed_7/best_model.zip").is_file()
    assert not result["incremental_value_passed"]
    assert result["stage4"] == "locked" and not result["live_eligible"]
    assert len(result["results"]) == 6


def test_orchestrator_relative_output_and_failed_gate_never_calls_sac(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    model_config = root / "configs/transformer_strategy_event_v2.example.json"
    config = {"transformer_config": str(model_config), "folds": 3,
              "seeds": [1, 2, 3, 4, 5], "minimum_trades": 50, "sizing": {"total_timesteps": 1}}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    count = 20_000
    bars = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=count, freq="15min", tz="UTC"),
        "open": 100., "high": 101., "low": 99., "close": 100., "volume": 1.,
        "symbol": "BTCUSDT", "exchange": "binance_futures", "interval": "15m"})
    bars.to_csv(tmp_path / "source.csv", index=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(staged_training, "prepare_event_frame", lambda frame, _: frame.assign(event_atr=.3))

    def fake_train(sources, model, training, output, **kwargs):
        raw = pd.read_csv(sources[0])
        cutoff = int(len(raw) * (training.train_fraction + training.validation_fraction))
        points = np.arange(cutoff + training.embargo_bars, len(raw) - 40, 40)
        predictions = fixture_events(len(points))
        predictions["endpoint"] = points
        predictions["event_entry_endpoint"] = points + 1
        predictions["event_exit_endpoint"] = points + 10
        predictions["timestamp_ns"] = pd.to_datetime(raw.timestamp.iloc[points], utc=True).astype("int64").to_numpy()
        run = (output / kwargs["run_name"]).resolve()
        run.mkdir(parents=True)
        (run / "best_model.pt").write_bytes(b"fixture not a real model")
        predictions.to_csv(run / "test_predictions.csv", index=False)
        return SimpleNamespace(run_dir=run, model_path=run / "best_model.pt")

    monkeypatch.setattr(staged_training, "train_temporal_transformer", fake_train)
    monkeypatch.setattr(staged_training, "run_sizing_comparison", lambda *args: pytest.fail("失敗的 gate 不得啟動 SAC"))
    result = staged_training.run_staged_training(Path("source.csv"), Path("config.json"), Path("relative_output"))
    assert result["transformer_runs"] == 15
    assert not result["transformer_gate_passed"] and not result["sac_test_evaluated"]
    assert result["sac_stage"] == "blocked_by_transformer_gate"
    assert (tmp_path / "relative_output/artifact_manifest.json").exists()
