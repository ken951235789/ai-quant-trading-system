"""低換手研究的因果時間、成本、風險單位、選擇隔離及完整編排測試。"""

import numpy as np
import pandas as pd
import pytest

from ai_quant_trading.backtesting.candidate_optimization import (
    COSTS, RULES, audit_research, enrich_trades, forward_gate, hourly_entries, measure,
    prepare_rule, research_windows, run_research, select_development,
)
from ai_quant_trading.backtesting.candidate_research import replay_window
from ai_quant_trading.operations.integrity import verify_artifact_manifest
from ai_quant_trading.transformer.strategy_events import StrategyEventConfig, prepare_event_frame


def bars(rows=5000, flat=False):
    close = np.full(rows, 100.) if flat else 100 + np.arange(rows) * .01
    opening = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=rows, freq="15min", tz="UTC"),
        "open": opening, "high": np.maximum(opening, close) + .01,
        "low": np.minimum(opening, close) - .01, "close": close, "volume": 10.,
        "symbol": "BTCUSDT", "exchange": "binance_futures", "interval": "15m"})


@pytest.mark.parametrize("entry", ["hourly_breakout", "hourly_pullback"])
def test_known_hourly_signals_include_both_sides(entry):
    hourly = pd.DataFrame({"open": 100., "high": 100.1, "low": 99.9, "close": 100.}, index=range(70))
    hourly.loc[29, "close"] = 99.
    hourly.loc[30, ["close", "high"]] = [101., 101.1]
    hourly.loc[49, "close"] = 101.
    hourly.loc[50, ["close", "low"]] = [99., 98.9]
    signals = hourly_entries(hourly, entry)
    assert signals.iloc[30] == 1
    assert signals.iloc[50] == -1
    with pytest.raises(ValueError):
        hourly_entries(hourly, "unregistered")


def test_future_prices_do_not_change_past_signals_or_atr():
    original = bars()
    changed = original.copy()
    changed.loc[4502:, ["open", "high", "low", "close"]] *= 2
    a = prepare_event_frame(original, StrategyEventConfig())
    b = prepare_event_frame(changed, StrategyEventConfig())
    for rule in RULES:
        before, after = prepare_rule(a, rule), prepare_rule(b, rule)
        for column in ("candidate_side", "event_atr", "event_regime"):
            pd.testing.assert_series_equal(before[column].iloc[:4502], after[column].iloc[:4502])


def test_hourly_signals_only_at_close_and_use_closed_hour_atr():
    prepared = prepare_event_frame(bars(), StrategyEventConfig())
    result = prepare_rule(prepared, RULES[1])
    points = result.loc[result.candidate_side.ne(0)]
    assert len(points) > 0
    assert (points.timestamp + pd.Timedelta(minutes=15)).dt.minute.eq(0).all()
    assert result.candidate_side.iloc[:4000].eq(0).all()
    expected = prepared.mtf_1h_atr_pct.iloc[4003] * prepared.close.iloc[4003]
    assert result.event_atr.iloc[4003] == pytest.approx(expected)
    assert result.event_atr.iloc[4004] == pytest.approx(expected)
    assert prepare_rule(prepared.assign(event_regime=-1), RULES[1]).candidate_side.eq(0).all()


def test_equal_entry_factor_has_identical_candidates_and_atr():
    prepared = prepare_event_frame(bars(), StrategyEventConfig())
    for a, b in ((1, 2), (3, 4)):
        left, right = prepare_rule(prepared, RULES[a]), prepare_rule(prepared, RULES[b])
        pd.testing.assert_frame_equal(left, right)


def test_shared_execution_costs_and_risk_unit():
    frame = bars(300, flat=True).assign(candidate_side=1, event_regime=1, event_atr=1.)
    zero = enrich_trades(replay_window(frame, RULES[1], COSTS["zero"], 0, len(frame)), frame)
    paid = enrich_trades(replay_window(frame, RULES[1], COSTS["base"], 0, len(frame)), frame)
    assert zero.endpoint.tolist() == paid.endpoint.tolist()
    assert (paid.net_return < zero.net_return).all()
    assert (paid.exit_endpoint == paid.endpoint + 65).all()
    np.testing.assert_allclose(paid.net_r, paid.net_return / paid.planned_stop_fraction)
    result = measure(paid, {"start": 0, "stop": len(frame)})
    assert result["net_r"]["mean"] < 0 and result["mean_cost_to_stop"] > 0
    assert result["trades_per_30_days"] == pytest.approx(len(paid) / 300 * 96 * 30)


def fake_records():
    return [{"window": "development", "rule": r.name, "scenario": c,
             "metrics": {"trades": 100, "mean": -.001, "ci_low": -.002,
                         "subperiods": [{"trades": 30, "mean": -.001}] * 3}}
            for r in RULES for c in COSTS]


def test_development_selection_ignores_forward_results_and_insufficient_samples():
    records = fake_records()
    assert select_development(records)["selected"] is None
    for row in records:
        if row["rule"] == RULES[1].name:
            row["metrics"].update(mean=.003, ci_low=.001,
                                  subperiods=[{"trades": 30, "mean": .002}] * 3)
    selected = select_development(records)
    assert selected["selected"] == RULES[1].name
    assert select_development(records + [{"window": "forward_1", "metrics": object()}]) == selected
    records[3]["metrics"]["trades"] = 10
    records[4]["metrics"]["trades"] = 10
    assert select_development(records)["selected"] is None
    with pytest.raises(ValueError, match="缺漏"):
        select_development(records[:-1])


def test_forward_cannot_promote_a_rule_not_selected_in_development():
    assert not forward_gate([], [], {"selected": None})["passed"]
    pooled = [{"rule": RULES[1].name, "scenario": cost, "metrics":
               {"trades": 120, "ci_low": .001, "mean": .002}} for cost in COSTS]
    records = [{"rule": RULES[1].name, "scenario": "base", "window": f"forward_{i}",
                "metrics": {"trades": 40, "mean": .002}} for i in range(1, 4)]
    assert forward_gate(records, pooled, {"selected": RULES[1].name})["passed"]
    records[-1]["metrics"]["mean"] = -.001
    assert not forward_gate(records, pooled, {"selected": RULES[1].name})["passed"]


def test_boundaries_purge_and_preserve_tail():
    windows = research_windows(100_000)
    assert windows[0]["stop"] == 60000
    assert windows[1]["start"] == 60064
    assert windows[-1]["stop"] == 90000
    with pytest.raises(ValueError):
        research_windows(100)


def test_empty_market_full_run_stays_blocked_and_refuses_overwrite(tmp_path):
    source = tmp_path / "bars.csv"
    bars(20_000, flat=True).to_csv(source, index=False)
    output = tmp_path / "study"
    result = run_research(source, output)
    assert len(result["results"]) == 60 and len(result["pooled_forward"]) == 15
    assert result["selection"]["selected"] is None
    assert result["next_stage"] == "blocked_by_rule_quality" and not result["live_eligible"]
    assert "research.json" in verify_artifact_manifest(output)
    assert audit_research(output)["all_trades_replayed"]
    with pytest.raises(FileExistsError):
        run_research(source, output)
