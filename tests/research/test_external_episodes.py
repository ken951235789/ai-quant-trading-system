"""持倉完整性、帳務守恆、未來資料隔離與訓練用途閘門。"""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from ai_quant_trading.research.external_episodes import (
    order_fills,
    pair_episodes,
    funding_evidence,
    prepare_training,
    split_for,
    market_features,
    mark_to_market,
    overlap_metadata,
    FEATURES,
    SAMPLE_COLUMNS,
    load_episode_dataset,
)
from ai_quant_trading.operations.integrity import build_artifact_manifest
from ai_quant_trading.persistence import write_json_atomic
from ai_quant_trading.research.external_traders import write_csv


def fill(t, start, size, side, price, pnl=0, fee=1, tid=None):
    signed = size if side == "B" else -size
    return dict(
        time_ms=str(t),
        start_position=str(start),
        end_position=str(start + signed),
        size=str(size),
        execution_side=side,
        price=str(price),
        closed_pnl=str(pnl),
        fee=str(fee),
        fee_token="USDC",
        coin="BTC",
        order_id=str(tid or t),
        trade_id=str(tid or t),
    )


def bars(n=500):
    t = np.arange(n, dtype=np.int64) * 900000
    c = 100 + np.sin(np.arange(n) / 10) + np.arange(n) * 0.01
    return pd.DataFrame(
        dict(
            time_ms=t,
            open_time_utc=pd.to_datetime(t, unit="ms", utc=True),
            open=c - 0.1,
            high=c + 0.5,
            low=c - 0.5,
            close=c,
            volume=np.ones(n) * 100,
        )
    )


def test_same_millisecond_reordered_by_position_not_id():
    rows = [
        fill(1, 1, 2, "B", 100, tid=1),
        fill(1, 0, 1, "B", 100, tid=9),
        fill(2, 3, 3, "A", 110, pnl=30),
    ]
    ordered, audit = order_fills(rows)
    assert [r["trade_id"] for r in ordered] == ["9", "1", "2"]
    assert audit["quarantined_fills"] == 0
    episodes, _, _ = pair_episodes(rows, "a")
    assert episodes[0]["accounting_reconciled"] and episodes[0]["pnl_before_funding_usdc"] == 27


def test_reversal_splits_two_episodes_and_fee():
    rows = [
        fill(1, 0, 1, "B", 100),
        fill(2, 1, 3, "A", 110, pnl=10, fee=3),
        fill(3, -2, 2, "B", 105, pnl=10, fee=2),
    ]
    e, legs, _ = pair_episodes(rows, "a")
    assert len(e) == 2 and [x["side"] for x in e] == [1, -1]
    assert [x["fee_usdc"] for x in e] == [2, 4]
    assert all(x["accounting_reconciled"] for x in e)
    assert [x["pnl_before_funding_usdc"] for x in e] == [8, 6]
    assert len(legs) == 4


def test_scale_in_and_out_cashflow():
    rows = [
        fill(1, 0, 1, "B", 100),
        fill(2, 1, 1, "B", 120),
        fill(3, 2, 1, "A", 115, pnl=5),
        fill(4, 1, 1, "A", 130, pnl=20),
    ]
    e, _, _ = pair_episodes(rows, "a")
    assert len(e) == 1 and e[0]["cashflow_pnl_usdc"] == 25
    assert e[0]["pnl_before_funding_usdc"] == 21
    assert e[0]["opening_notional_usdc"] == 220


def test_fee_rebate_not_forced_positive():
    e, _, _ = pair_episodes(
        [fill(1, 0, 1, "B", 100, fee=-1), fill(2, 1, 1, "A", 101, pnl=1, fee=-1)], "a"
    )
    assert e[0]["pnl_before_funding_usdc"] == 3


def test_unexplained_pnl_difference_rejected_not_hidden_cost():
    e, _, _ = pair_episodes([fill(1, 0, 1, "B", 100), fill(2, 1, 1, "A", 110, pnl=10.01)], "a")
    assert e[0]["complete"] and not e[0]["accounting_reconciled"]
    assert e[0]["return_before_funding"] is None


def test_left_and_right_censoring():
    e, _, _ = pair_episodes([fill(1, 1, 1, "A", 110, pnl=10), fill(2, 0, 1, "B", 100)], "a")
    assert e[0]["left_censored"] and not e[0]["complete"]
    assert e[1]["right_censored"] and not e[1]["complete"]


def test_ambiguous_chain_quarantined():
    rows = [fill(1, 0, 1, "B", 100), fill(2, 1, 1, "B", 100, tid=2), fill(2, 1, 2, "B", 100, tid=3)]
    ordered, audit = order_fills(rows)
    assert len(ordered) == 1 and audit["issue"] == "ambiguous_chain"


def test_position_gap_quarantined():
    ordered, audit = order_fills([fill(1, 0, 1, "B", 100), fill(2, 2, 1, "A", 110)])
    assert len(ordered) == 1 and audit["issue"] == "position_gap"


def test_duplicate_rejected():
    row = fill(1, 0, 1, "B", 100)
    with pytest.raises(ValueError, match="重複"):
        order_fills([row, row])


def test_aggregated_funding_never_zero_filled():
    e, _, _ = pair_episodes([fill(1, 0, 1, "B", 100), fill(2, 1, 1, "A", 110, pnl=10)], "a")
    audit = funding_evidence(e, [dict(delta=dict(nSamples=24, usdc="-4"))])
    assert audit["aggregated_records"] == 1 and e[0]["funding_usdc"] is None
    assert e[0]["net_pnl_usdc"] is None and not e[0]["full_cost_training_eligible"]


def test_empty_funding_not_proven_zero():
    e, _, _ = pair_episodes([fill(1, 0, 1, "B", 100), fill(2, 1, 1, "A", 110, pnl=10)], "a")
    funding_evidence(e, [])
    assert e[0]["funding_status"] == "coverage_unverified" and e[0]["net_pnl_usdc"] is None


@pytest.mark.parametrize(
    "start,end,expected",
    [
        (1, 59, "train"),
        (59, 60, "purged_boundary"),
        (60, 79, "validation"),
        (79, 80, "purged_boundary"),
        (80, 99, "research_test"),
    ],
)
def test_boundary_purge(start, end, expected):
    assert split_for(start, end, [0, 60, 80, 100]) == expected


def test_future_perturbation_does_not_change_features():
    b = bars()
    original, available = market_features(b)
    modified = b.copy()
    modified.loc[300:, ["open", "high", "low", "close", "volume"]] *= 2
    new, _ = market_features(modified)
    np.testing.assert_allclose(original[:300], new[:300], equal_nan=True)
    assert available[0] == 900000


def test_sequence_is_closed_before_fill():
    b = bars()
    e, _, _ = pair_episodes(
        [fill(200 * 900000, 0, 1, "B", 100), fill(210 * 900000, 1, 1, "A", 110, pnl=10)], "a"
    )
    funding_evidence(e, [])
    x, m, _ = prepare_training(e, b, 0, 500 * 900000)
    assert x.shape == (1, 64, len(FEATURES))
    assert m[0]["feature_available_ms"] == 199 * 900000
    assert m[0]["target_return_before_funding"] == 0.08


def test_outcome_changes_do_not_change_input():
    b = bars()
    e, _, _ = pair_episodes(
        [fill(200 * 900000 + 1, 0, 1, "B", 100), fill(210 * 900000, 1, 1, "A", 110, pnl=10)], "a"
    )
    funding_evidence(e, [])
    x, m, _ = prepare_training(e, b, 0, 500 * 900000)
    changed = deepcopy(e)
    changed[0]["return_before_funding"] = -0.5
    x2, m2, _ = prepare_training(changed, b, 0, 500 * 900000)
    np.testing.assert_array_equal(x, x2)
    assert m[0]["target_return_before_funding"] != m2[0]["target_return_before_funding"]


def test_overlapping_labels_weight():
    episodes = [
        dict(episode_id="a", entry_ms=0, exit_ms=10, dataset_eligible=True),
        dict(episode_id="b", entry_ms=0, exit_ms=10, dataset_eligible=True),
    ]
    overlap_metadata(episodes)
    assert episodes[0]["uniqueness_weight"] == 0.5
    assert episodes[0]["overlap_group"] == episodes[1]["overlap_group"]


def test_mark_to_market_reconciles_and_includes_unrealized():
    b = bars(10)
    e, legs, _ = pair_episodes(
        [fill(1, 0, 1, "B", 100), fill(5 * 900000, 1, 1, "A", 110, pnl=10)], "a"
    )
    trace = mark_to_market(e[0], legs, b)
    assert len(trace) > 2 and any(r["position"] != 0 for r in trace)
    assert trace[-1]["pnl_before_funding_usdc"] == 8


def package(tmp_path, empty=False):
    e, _, _ = pair_episodes(
        [fill(200 * 900000, 0, 1, "B", 100), fill(210 * 900000, 1, 1, "A", 110, pnl=10)], "a"
    )
    funding_evidence(e, [])
    x, m, bounds = prepare_training([] if empty else e, bars(), 0, 500 * 900000)
    write_csv(tmp_path / "samples.csv", m, SAMPLE_COLUMNS)
    np.savez_compressed(tmp_path / "sequences.npz", x=x)
    write_json_atomic(
        tmp_path / "dataset_contract.json",
        dict(feature_columns=FEATURES, sequence_length=64, split_boundaries_ms=bounds),
    )
    build_artifact_manifest(tmp_path, files=[p for p in tmp_path.iterdir() if p.is_file()])


def test_loader_requires_explicit_incomplete_cost_consent(tmp_path):
    package(tmp_path)
    with pytest.raises(ValueError, match="完整淨成本"):
        load_episode_dataset(tmp_path)
    result = load_episode_dataset(
        tmp_path, target="return_before_funding", allow_missing_funding=True
    )
    assert result["x"].shape == (1, 64, 16) and result["y"][0] == np.float32(0.08)


def test_empty_valid_dataset_can_be_reported(tmp_path):
    package(tmp_path, empty=True)
    result = load_episode_dataset(
        tmp_path, target="return_before_funding", allow_missing_funding=True
    )
    assert result["x"].shape == (0, 64, 16) and len(result["y"]) == 0


def test_tampered_file_rejected(tmp_path):
    package(tmp_path)
    with (tmp_path / "samples.csv").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError):
        load_episode_dataset(tmp_path, target="return_before_funding", allow_missing_funding=True)
