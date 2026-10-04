"""新候選契約的因果、成本、模型隔離與舊引擎相容性。"""

from dataclasses import replace
import json
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import pytest

from ai_quant_trading.research.candidate_contract import (
    CandidateContract, FAMILIES, candidate_outcomes, prepare_candidate_frame, uniqueness_weights,
)
from ai_quant_trading.research.candidate_evaluation import apply_statistical_filter, fit_statistical_filter
from ai_quant_trading.research.event_attribution import attribute_events, attribution_summary
from ai_quant_trading.transformer.config import TemporalTransformerConfig, TransformerTrainingConfig
from ai_quant_trading.transformer.dataset import prepare_transformer_datasets
from ai_quant_trading.transformer.strategy_events import prepare_event_frame, replay_event
from ai_quant_trading.transformer.training import _split_calibration


def bars(n=14000):
    rng = np.random.default_rng(19)
    close = 100 * np.exp(np.cumsum(.0005 * np.sin(np.arange(n) / 160) + rng.normal(0, .0007, n)))
    opening = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"timestamp": pd.date_range("2023-01-01", periods=n, freq="15min", tz="UTC"),
        "open": opening, "high": np.maximum(opening, close) * 1.0002,
        "low": np.minimum(opening, close) * .9998, "close": close,
        "volume": rng.uniform(100, 1000, n), "symbol": "BTC/USDT",
        "exchange": "binance_futures", "interval": "15m"})


def configurations(contract):
    payload = json.loads((Path(__file__).parents[2] / "configs/transformer_strategy_event_v2.example.json").read_text())
    training = TransformerTrainingConfig(**payload["training"])
    return TemporalTransformerConfig(**payload["model"]), replace(training,
        strategy_event_config=contract.event_config().to_dict(), candidate_contract=contract.to_dict())


@pytest.mark.parametrize("family", FAMILIES)
def test_candidate_signal_and_features_are_causal(family):
    raw = bars()
    contract = CandidateContract(family=family, regime_exit="disabled")
    before = prepare_candidate_frame(raw, contract)
    changed = raw.copy()
    changed.loc[9003:, ["open", "high", "low", "close"]] *= 1.3
    changed.loc[9003:, "volume"] *= 8
    after = prepare_candidate_frame(changed, contract)
    pd.testing.assert_frame_equal(before.iloc[:9003], after.iloc[:9003])
    expected = raw.volume / raw.volume.shift(1).rolling(20).mean()
    np.testing.assert_allclose(before.mtf_15m_relative_volume_prev20, expected, equal_nan=True)
    assert "mtf_15m_relative_volume" not in before


def test_original_signals_remain_compatible():
    raw = bars()
    contract = CandidateContract()
    old = prepare_event_frame(raw, contract.event_config())
    new = prepare_candidate_frame(raw, contract)
    np.testing.assert_array_equal(old.candidate_side, new.candidate_side)
    np.testing.assert_allclose(old.event_atr, new.event_atr, equal_nan=True)


@pytest.mark.parametrize("family", FAMILIES)
def test_batch_labels_equal_reference(family):
    contract = CandidateContract(family=family, regime_exit="disabled")
    frame = prepare_candidate_frame(bars(), contract)
    # 另外強制均勻多空候選，確保零訊號家族也測到標籤引擎而非空測試。
    frame["candidate_side"] = 0
    frame.loc[5000:5200:20, "candidate_side"] = np.resize([1, -1], 11)
    outcomes = candidate_outcomes(frame, contract)
    assert len(outcomes) == 11
    for row in outcomes.itertuples():
        reference = replay_event(frame, row.endpoint, contract.event_config(),
            fee_bps_per_side=contract.fee_bps, slippage_bps_per_side=contract.slippage_bps,
            regime_exit=contract.regime_exit)
        for key, value in reference.items():
            if isinstance(value, str):
                assert getattr(row, key) == value
            else:
                assert getattr(row, key) == pytest.approx(value)


@pytest.mark.parametrize("side", [-1, 1])
def test_stop_first_cost_reconciliation_and_path_bounds(side):
    frame = bars(100)
    frame[["open", "close"]] = 100.
    frame["high"], frame["low"] = 101., 99.
    frame["event_atr"], frame["event_regime"], frame["candidate_side"] = 1., side, 0
    frame.loc[0, "candidate_side"] = side
    frame.loc[1, ["high", "low"]] = [110., 90.]
    contract = CandidateContract(regime_exit="disabled")
    outcomes = candidate_outcomes(frame, contract)
    assert outcomes.iloc[0].exit_reason == "stop"
    attributed = attribute_events(frame, outcomes, contract, "a" * 64)
    row = attributed.iloc[0]
    assert abs(row.reconciliation_error) < 1e-12
    assert row.mfe_confirmed_lower <= row.mfe_possible_upper
    assert row.mae_confirmed_lower <= row.mae_possible_upper
    assert row.exit_bar_intrabar_order_unknown
    assert row.post_stop_start_endpoint > row.exit_endpoint
    assert row.holding_minutes_lower == 0
    assert row.holding_minutes_upper == 15
    empty = attribution_summary(attributed.iloc[:0])
    assert empty["mean"] is None


def test_train_overlap_uniqueness():
    weights = uniqueness_weights(20, [0, 1], [3, 3])
    assert weights[0] == pytest.approx(2 / 3)
    assert weights[1] == pytest.approx(.5)
    assert weights[10] == 1
    with pytest.raises(ValueError):
        uniqueness_weights(20, [0], [20])


def test_training_contract_conflicts_rejected():
    contract = CandidateContract()
    _, training = configurations(contract)
    with pytest.raises(ValueError, match="不一致"):
        replace(training, fee_bps_per_side=1.)
    with pytest.raises(ValueError):
        CandidateContract(family="historical_best")
    with pytest.raises(ValueError):
        CandidateContract(slippage_bps=float("nan"))


def test_dataset_boundaries_weighting_and_train_only_scaler(tmp_path):
    path = tmp_path / "bars.csv"
    raw = bars()
    raw.to_csv(path, index=False)
    contract = CandidateContract()
    model, training = configurations(contract)
    first = prepare_transformer_datasets([path], model, training)
    calibration, selection, _ = _split_calibration(first, training)
    assert first.diagnostics["candidate_contract"]["contract_sha256"] == contract.digest
    assert min(first.train.sampling_weights) < 1
    assert all(w == 1 for w in first.test.sampling_weights)
    forbidden = ("future", "post_stop", "mfe", "mae", "label_end", "exit", "actual")
    assert not any(any(token in c for token in forbidden) for c in first.scaler.feature_columns)
    for subset, boundary in ((first.train, 8400), (calibration, selection.references[0][1]),
                             (selection, 11200), (first.test, 14000)):
        series = subset.series[0]
        assert all(series.event_metadata["label_end_endpoint"][p] < boundary for _, p in subset.references)
    raw.loc[8400:, ["open", "high", "low", "close"]] *= 1.5
    raw.to_csv(path, index=False)
    second = prepare_transformer_datasets([path], model, training)
    assert first.scaler.to_dict() == second.scaler.to_dict()
    assert first.train.sampling_weights == second.train.sampling_weights


def test_statistical_filter_uses_only_train_labels():
    outcomes = pd.DataFrame({"endpoint": [1, 2, 3], "side": [1, 1, -1],
        "entry_regime": [1, 1, -1], "net_return": [.01, -.02, 500.]})
    fitted = fit_statistical_filter(outcomes, [1, 2], 2)
    outcomes.loc[2, "net_return"] = -1000
    assert fitted == fit_statistical_filter(outcomes, [1, 2], 2)
    predictions = pd.DataFrame({"event_side": [-1], "event_entry_regime": [-1]})
    output = apply_statistical_filter(predictions, fitted, 32)
    assert output.predicted_return_32.iloc[0] == pytest.approx(-.005)


def test_full_factorial_plan_keeps_all_variants_and_contracts():
    root = Path(__file__).parents[2]
    spec = importlib.util.spec_from_file_location("candidate_runner", root / "scripts/run_candidate_training_research.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    plan = runner.research_plan([42, 137, 2026])
    assert plan["formal_runs"] == 135
    assert plan["future_holdout"]["state"].startswith("BLOCKED")
    payload = json.loads((root / "configs/transformer_strategy_event_v2.example.json").read_text())
    records = []
    for i, variant in enumerate(runner.VARIANTS):
        contract, model, training = runner.configurations(payload, "trend_pullback", variant, 42,
            {"source_end": 10000, "train_end": 6000, "validation_end": 8000})
        assert contract.to_dict() == training.candidate_contract
        assert model.d_model == 64 and model.n_layers == 2
        records.append({"family": "trend_pullback", "fold": 1, "seed": 42, "variant": variant,
            "test_prediction_metrics": {m: float(i) for m in (
                "event_return_skill", "event_probability_skill", "event_prediction_skill_score")}})
    paired = runner.paired_summary(records)
    assert paired["complete_pairs"] == 1
    assert paired["factorial_effects"][0]["context_effect"] == 2.
    assert paired["factorial_effects"][0]["mse_effect"] == 1.


def test_empty_and_corrupted_trade_csv_audit():
    root = Path(__file__).parents[2]
    spec = importlib.util.spec_from_file_location("candidate_audit", root / "scripts/audit_candidate_training_research.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fields = ["endpoint", "exit_endpoint", "gross_same_path_return", "spread_return",
              "slippage_return", "fee_return", "funding_return", "net_return"]
    assert module.verify_trade_table(pd.DataFrame(columns=fields), 4) == 0
    valid = pd.DataFrame([[1, 3, .01, .0001, .0004, .001, .0001, .0084]], columns=fields)
    assert module.verify_trade_table(valid, 4) == 1
    valid.loc[0, "net_return"] = np.nan
    with pytest.raises(ValueError, match="非有限"):
        module.verify_trade_table(valid, 4)


def test_cloud_artifact_paths_stay_inside_download_root(tmp_path):
    root = Path(__file__).parents[2]
    spec = importlib.util.spec_from_file_location("portable_candidate_audit", root / "scripts/audit_candidate_training_research.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for saved in ("/kaggle/working/study/runs/run_1", "C:\\old\\study\\run_1", "../../run_1"):
        assert module.portable_path(saved, tmp_path) == (tmp_path / "run_1").resolve()
    with pytest.raises(ValueError):
        module.portable_path("..", tmp_path)
