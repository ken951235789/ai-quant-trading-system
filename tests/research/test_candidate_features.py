"""特徵消融不改交易標籤；驗證成交來源、因果對齊與舊契約相容。"""

from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai_quant_trading.research.candidate_contract import CandidateContract, candidate_outcomes, prepare_candidate_frame
from ai_quant_trading.research.candidate_features import (
    FEATURE_SETS, FLOW_COLUMNS, _flow_features, _setup_features, enrich_flow_source, validate_flow,
)
from ai_quant_trading.transformer.config import TemporalTransformerConfig, TransformerTrainingConfig
from ai_quant_trading.transformer.dataset import prepare_transformer_datasets


ROOT = Path(__file__).parents[2]


def bars(n=14000):
    rng = np.random.default_rng(19)
    close = 100 * np.exp(np.cumsum(.0005 * np.sin(np.arange(n) / 160) + rng.normal(0, .0007, n)))
    opening = np.r_[close[0], close[:-1]]
    volume = rng.uniform(100, 1000, n)
    ratio = rng.uniform(.1, .9, n)
    return pd.DataFrame({"timestamp": pd.date_range("2023-01-01", periods=n, freq="15min", tz="UTC"),
        "open": opening, "high": np.maximum(opening, close) * 1.0002,
        "low": np.minimum(opening, close) * .9998, "close": close, "volume": volume,
        "symbol": "BTC/USDT", "exchange": "binance_futures", "interval": "15m",
        "quote_asset_volume": volume * close, "number_of_trades": rng.integers(100, 300, n),
        "taker_buy_base_volume": volume * ratio, "taker_buy_quote_volume": volume * ratio * close})


def runner():
    spec = importlib.util.spec_from_file_location("feature_runner", ROOT / "scripts/run_candidate_training_research.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("feature_set", FEATURE_SETS)
def test_features_future_perturbation(feature_set):
    raw = bars()
    contract = CandidateContract(feature_set=feature_set)
    before = prepare_candidate_frame(raw, contract)
    raw.loc[9003:, ["open", "high", "low", "close", "quote_asset_volume", "taker_buy_quote_volume"]] *= 1.5
    raw.loc[9003:, ["volume", "taker_buy_base_volume", "quote_asset_volume", "taker_buy_quote_volume"]] *= 2
    raw.loc[9003:, "number_of_trades"] *= 3
    after = prepare_candidate_frame(raw, contract)
    pd.testing.assert_frame_equal(before.iloc[:9003], after.iloc[:9003])
    if feature_set != "existing":
        assert "mtf_15m_context_trend_consensus" not in before
        assert "mtf_4h_ema50_200_spread" not in before
        assert "candidate_side" in before
    assert not any(c in before for c in FLOW_COLUMNS)


@pytest.mark.parametrize("family", ["original", "trend_pullback", "vwap_reversion"])
def test_all_feature_sets_share_signals_and_labels(family):
    raw = bars()
    base = CandidateContract(family=family)
    reference = prepare_candidate_frame(raw, base)
    expected = candidate_outcomes(reference, base)
    for feature_set in FEATURE_SETS[1:]:
        contract = replace(base, feature_set=feature_set)
        actual = prepare_candidate_frame(raw, contract)
        pd.testing.assert_frame_equal(reference[["event_atr", "event_regime", "candidate_side"]],
                                      actual[["event_atr", "event_regime", "candidate_side"]])
        pd.testing.assert_frame_equal(expected, candidate_outcomes(actual, contract))


def test_old_contract_hash_and_metadata_stay_compatible():
    saved = {"family": "original", "version": "btc_candidate_v1", "timeframe": "15m", "strict": False,
        "stop_atr": 2., "reward_r": 2., "holding_bars": 32, "cooldown_bars": 4, "regime_exit": "loss",
        "feature_profile": "context_v2", "feature_ablation": "none", "fee_bps": 5., "slippage_bps": 2.,
        "spread_bps": 1., "funding_reserve_bps": 1.}
    contract = CandidateContract(**saved)
    assert contract.to_dict() == saved
    assert contract.digest == hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()
    assert "feature_contract" not in contract.metadata()
    assert replace(contract, feature_set="compact").digest != contract.digest
    with pytest.raises(ValueError):
        CandidateContract(feature_set="compact", feature_profile="legacy_v1")


def test_strict_enrichment_does_not_change_price_or_accept_unknown_features():
    full = bars(200)
    raw = full.drop(columns=list(FLOW_COLUMNS))
    supplement = full.assign(future_profit=999.)
    enriched = enrich_flow_source(raw, supplement)
    pd.testing.assert_frame_equal(enriched[raw.columns], raw)
    assert "future_profit" not in enriched
    supplement.loc[2, "close"] *= 1.000001
    with pytest.raises(ValueError, match="OHLCV 不一致"):
        enrich_flow_source(raw, supplement)
    with pytest.raises(ValueError, match="時間戳"):
        enrich_flow_source(raw, full.iloc[1:])
    with pytest.raises(ValueError):
        enrich_flow_source(raw, pd.concat([full, full.iloc[[-1]]]))


@pytest.mark.parametrize("kind", ["missing", "nan", "excess", "fractional", "negative", "wrong_zero"])
def test_flow_data_errors_fail_closed(kind):
    full = bars(100)
    if kind == "missing":
        full = full.drop(columns="number_of_trades")
    elif kind == "nan":
        full.loc[0, "taker_buy_base_volume"] = np.nan
    elif kind == "excess":
        full.loc[0, "taker_buy_base_volume"] = full.volume.iloc[0] * 2
    elif kind == "fractional":
        full["number_of_trades"] = 2.5
    elif kind == "negative":
        full.loc[0, "quote_asset_volume"] = -1
    else:
        full.loc[0, "number_of_trades"] = 0
    with pytest.raises(ValueError):
        validate_flow(full)


def test_flow_alignment_zero_volume_and_previous_mean():
    full = bars(100)
    output = _flow_features(full)
    assert np.isnan(output.mtf_1h_flow_imbalance.iloc[2])
    expected = 2 * full.taker_buy_base_volume.iloc[:4].sum() / full.volume.iloc[:4].sum() - 1
    assert output.mtf_1h_flow_imbalance.iloc[3] == pytest.approx(expected)
    assert output.mtf_1h_flow_imbalance.iloc[6] == pytest.approx(expected)
    assert np.isnan(output.mtf_4h_flow_imbalance.iloc[14])
    expected4 = 2 * full.taker_buy_base_volume.iloc[:16].sum() / full.volume.iloc[:16].sum() - 1
    assert output.mtf_4h_flow_imbalance.iloc[15] == pytest.approx(expected4)
    assert output.mtf_15m_flow_trade_activity.iloc[20] == pytest.approx(
        full.number_of_trades.iloc[20] / full.number_of_trades.iloc[:20].mean())
    full.loc[20, ["volume", *FLOW_COLUMNS]] = 0
    output = _flow_features(full)
    assert output.mtf_15m_flow_imbalance.iloc[20] == 0
    assert output.mtf_15m_flow_has_trades.iloc[20] == 0
    assert np.isnan(output.mtf_15m_flow_average_trade_relative.iloc[20])


def test_setup_levels_and_ages_use_only_known_bars():
    frame = pd.DataFrame({"close": np.full(100, 100.), "high": np.full(100, 102.),
        "low": np.full(100, 97.), "event_atr": np.ones(100), "event_regime": np.ones(100),
        "mtf_15m_context_previous_day_high_distance_atr": np.full(100, -4.),
        "mtf_15m_context_previous_day_low_distance_atr": np.full(100, 6.)})
    frame.loc[99, "high"] = 110
    result = _setup_features(frame)
    assert result.mtf_15m_setup_long_room_atr.iloc[99] == 2
    assert result.mtf_15m_setup_short_room_atr.iloc[99] == 3
    assert result.mtf_15m_setup_since_up_break_capped96.iloc[99] == 1
    frame.loc[99, "close"] = 106
    frame.loc[99, "mtf_15m_context_previous_day_high_distance_atr"] = 2
    frame.loc[99, "mtf_15m_context_previous_day_low_distance_atr"] = 12
    result = _setup_features(frame)
    assert np.isnan(result.mtf_15m_setup_long_room_atr.iloc[99])
    assert result.mtf_15m_setup_long_level_known.iloc[99] == 0
    assert result.mtf_15m_setup_since_up_break_capped96.iloc[99] == 0


def test_combined_dataset_scaler_and_endpoints_are_train_only(tmp_path):
    raw = bars()
    path = tmp_path / "bars.csv"
    raw.to_csv(path, index=False)
    contract = CandidateContract(feature_set="compact_combined")
    payload = json.loads((ROOT / "configs/transformer_strategy_event_v2.example.json").read_text())
    model = TemporalTransformerConfig(**payload["model"])
    training = replace(TransformerTrainingConfig(**payload["training"]),
        strategy_event_config=contract.event_config().to_dict(), candidate_contract=contract.to_dict())
    data = prepare_transformer_datasets([path], model, training)
    old_contract = CandidateContract()
    baseline = prepare_transformer_datasets([path], model, replace(training, candidate_contract=old_contract.to_dict()))
    for split in ("train", "validation", "test"):
        assert getattr(data, split).references == getattr(baseline, split).references
    assert any("_flow_" in c for c in data.scaler.feature_columns)
    assert any("_setup_" in c for c in data.scaler.feature_columns)
    raw.loc[8400:, "taker_buy_base_volume"] = raw.loc[8400:, "volume"] * .99
    raw.loc[8400:, "number_of_trades"] *= 20
    raw.to_csv(path, index=False)
    changed = prepare_transformer_datasets([path], model, training)
    assert data.scaler.to_dict() == changed.scaler.to_dict()
    assert data.train.sampling_weights == changed.train.sampling_weights


def test_feature_matrix_and_sac_decision_are_not_profit_shortcuts():
    module = runner()
    plan = module.research_plan([42, 137, 2026], feature_study=True)
    assert plan["formal_runs"] == 135
    assert list(module.FEATURE_VARIANTS) == plan["variants"]
    payload = json.loads((ROOT / "configs/transformer_strategy_event_v2.example.json").read_text())
    records = []
    for i, variant in enumerate(module.FEATURE_VARIANTS):
        contract, model, training = module.configurations(payload, "original", variant, 42,
            {"source_end": 10000, "train_end": 6000, "validation_end": 8000}, True)
        assert contract.feature_set == module.FEATURE_VARIANTS[variant]
        assert training.epochs == 1 and model.d_model == 64 and training.cpu_threads == 2
        records.append({"name": variant, "family": "original", "fold": 0, "seed": 42, "variant": variant,
            "test_prediction_metrics": {m: float(i) for m in (
                "event_return_skill", "event_probability_skill", "event_prediction_skill_score")},
            "test_policies": {p: {"trades": 0, "mean": None} for p in ("transformer_base", "transformer_stress")}})
    assert module.paired_summary(records)["complete_feature_pairs"] == 1
    decision = module.assess_sac_readiness(records, True)
    assert decision["status"] == "BLOCKED" and not decision["live_eligible"]
    for r in records:
        for p in r["test_policies"].values():
            p.update(trades=1000, mean=.02)
    assert module.assess_sac_readiness(records, False)["status"] == "BLOCKED"


def test_preregistered_limited_matrix_plan(tmp_path):
    module = runner()
    source = tmp_path / "bars.csv"
    bars(20000).to_csv(source, index=False)
    output = tmp_path / "plan"
    assert module.main(["--source", str(source), "--output", str(output), "--feature-study",
        "--variants", "F_existing", "F_compact_combined", "--folds", "3"]) == 0
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    assert plan["formal_runs"] == 18 and plan["full_matrix_runs"] == 135
    assert plan["selected_folds"] == [3]
    assert plan["mode"] == "plan" and not plan["live_eligible"]


def test_paired_loss_and_exit_contracts_are_separate(tmp_path):
    module = runner()
    payload = json.loads((ROOT / "configs/transformer_strategy_event_v2.example.json").read_text())
    window = {"source_end": 10000, "train_end": 6000, "validation_end": 8000}
    old, model, mse = module.configurations(payload, "original", "F_compact_combined", 42, window, True)
    paired, same_model, smooth = module.configurations(payload, "original", "F_compact_combined_smooth_l1", 42, window, True)
    assert old == paired and model == same_model
    assert mse.return_loss_kind == "mse" and smooth.return_loss_kind == "smooth_l1"
    different, _, _ = module.configurations(payload, "original", "F_compact_combined", 42, window, True, 1.0)
    assert different.digest != old.digest
    assert different.event_config().target_atr == 2.
    source = tmp_path / "bars.csv"
    bars(20000).to_csv(source, index=False)
    output = tmp_path / "paired_plan"
    assert module.main(["--source", str(source), "--output", str(output), "--feature-study",
        "--variants", "F_existing", "F_compact_combined", "--folds", "3", "--paired-loss"]) == 0
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    assert plan["formal_runs"] == 36 and plan["paired_loss"]
    assert "Smooth L1" in plan["feature_group_ablation"]
    assert plan["reward_r"] == 2 and not plan["dynamic_exit_enabled"]
    with pytest.raises(ValueError, match="分開"):
        module.main(["--source", str(source), "--output", str(output), "--feature-study",
                     "--paired-loss", "--reward-r", "1"])
