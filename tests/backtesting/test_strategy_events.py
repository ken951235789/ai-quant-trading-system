"""固定事件策略的因果資料與成交順序測試。"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from ai_quant_trading.transformer.strategy_events import (
    StrategyEventConfig, prepare_event_frame, replay_event, validate_event_bars,
    evaluate_event_predictions,
)


def bars(rows=40):
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01 07:30", periods=rows, freq="15min", tz="UTC"),
        "open": 100., "high": 100.2, "low": 99.8, "close": 100., "volume": 10.,
        "symbol": "BTC/USDT", "exchange": "binance_futures", "interval": "15m",
        "candidate_side": 1, "event_regime": 1, "event_atr": 1.,
    })


def replay(frame, **kwargs):
    config = StrategyEventConfig(spread_bps=0, funding_reserve_bps_per_settlement=0, **kwargs)
    return replay_event(frame, 0, config, fee_bps_per_side=0, slippage_bps_per_side=0)


@pytest.mark.parametrize("side,expected", [(1, 98), (-1, 102)])
def test_same_bar_stop_has_priority_for_both_sides(side, expected):
    frame = bars()
    frame[["candidate_side", "event_regime"]] = side
    frame.loc[1, ["high", "low"]] = [105, 95]
    result = replay(frame)
    assert result["exit_reason"] == "stop"
    assert result["exit_price"] == expected
    assert result["exit_endpoint"] == 1
    assert result["net_return"] == pytest.approx(-0.02)


def test_gap_stop_next_open_time_and_regime_exit():
    frame = bars()
    frame.loc[2, ["open", "low"]] = [95, 94]
    assert replay(frame)["exit_price"] == 95
    frame = bars()
    assert replay(frame)["exit_endpoint"] == 33
    frame.loc[3, "event_regime"] = 0
    result = replay(frame)
    assert result["exit_endpoint"] == 4
    assert result["exit_reason"] == "regime"


def test_costs_and_funding_are_applied_to_every_exit():
    frame = bars()
    config = StrategyEventConfig(max_holding_bars=2, spread_bps=2,
                                funding_reserve_bps_per_settlement=1)
    result = replay_event(frame, 0, config, fee_bps_per_side=5, slippage_bps_per_side=2)
    entry, exit_price = 100 * 1.0003, 100 * 0.9997
    assert result["funding_return"] == pytest.approx(0.0001)
    assert result["fee_return"] == pytest.approx(0.0005 * (1 + exit_price / entry))
    assert result["net_return"] == pytest.approx(
        exit_price / entry - 1 - result["fee_return"] - 0.0001,
    )
    frame.loc[1, "low"] = 97
    stopped = replay_event(frame, 0, config, fee_bps_per_side=5, slippage_bps_per_side=2)
    assert stopped["funding_return"] == 0  # 07:45 棒內離場，不收 08:00 結算。


def test_bars_fail_closed_on_missing_duplicate_open_or_spot_data():
    frame = bars()
    validate_event_bars(frame)
    for invalid in (frame.drop(index=3), pd.concat([frame, frame.tail(1)]),
                    frame.assign(exchange="binance"), frame.assign(interval="5m")):
        with pytest.raises(ValueError):
            validate_event_bars(invalid)
    with pytest.raises(ValueError):
        validate_event_bars(frame.assign(timestamp=pd.date_range(
            pd.Timestamp.now(tz="UTC").ceil("15min"), periods=len(frame), freq="15min")))


def test_higher_timeframe_features_do_not_see_future_bars():
    frame = bars(4000)
    rng = np.random.default_rng(7)
    frame["close"] = 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.001, len(frame))))
    frame["open"] = frame["close"].shift(1).fillna(frame["close"].iloc[0])
    frame["high"] = frame[["open", "close"]].max(axis=1) * 1.0005
    frame["low"] = frame[["open", "close"]].min(axis=1) * 0.9995
    frame["volume"] = rng.uniform(1, 10, len(frame))
    config = StrategyEventConfig(warmup_hours=200)
    original = prepare_event_frame(frame, config)
    changed = frame.copy()
    changed.loc[3500:, ["open", "high", "low", "close"]] *= 2
    altered = prepare_event_frame(changed, config)
    pd.testing.assert_frame_equal(original.iloc[:3500], altered.iloc[:3500])
    assert (original.loc[:799, "candidate_side"] == 0).all()
    assert len([name for name in original if name.startswith("mtf_")]) == 50


def test_filter_and_baseline_obey_position_and_cooldown():
    frame = pd.DataFrame({"series_index": [0] * 4, "endpoint": [1, 2, 8, 15],
                          "event_exit_endpoint": [5, 6, 10, 17],
                          "tradeability_probability_32": [0.1, 0.9, 0.9, 0.9],
                          "predicted_return_32": [0.01] * 4,
                          "actual_return_32": [-0.01, 0.02, 0.03, 0.04]})
    metrics = evaluate_event_predictions(frame, 32, StrategyEventConfig(), 2)
    assert metrics["event_baseline_trades"] == 2
    assert metrics["event_filtered_trades"] == 2
    assert metrics["event_filtered_notional_return_sum"] == pytest.approx(0.06)
    assert metrics["event_baseline_notional_return_sum"] == pytest.approx(0.03)
    frame["tradeability_probability_32"] = 0
    assert evaluate_event_predictions(frame, 32, StrategyEventConfig(), 2)[
        "event_filtered_has_trades"] == 0
    frame.loc[0, "tradeability_probability_32"] = np.nan
    with pytest.raises(ValueError):
        evaluate_event_predictions(frame, 32, StrategyEventConfig(), 2)


def test_invalid_config_and_no_execution_bar_are_rejected():
    with pytest.raises(ValueError):
        StrategyEventConfig(stop_atr=float("nan"))
    with pytest.raises(ValueError):
        replay(bars(33))
    with pytest.raises(ValueError):
        replace(StrategyEventConfig(), spread_bps=-1)


def test_research_exit_modes_preserve_original_default_and_only_use_previous_close():
    frame = bars()
    frame.loc[3, "event_regime"] = 0
    frame.loc[6:, "event_regime"] = -1
    config = StrategyEventConfig(spread_bps=0, funding_reserve_bps_per_settlement=0)
    def run(mode):
        return replay_event(frame, 0, config, fee_bps_per_side=0,
                            slippage_bps_per_side=0, regime_exit=mode)
    assert run("loss")["exit_endpoint"] == 4
    assert run("opposite")["exit_endpoint"] == 7
    assert run("disabled")["exit_endpoint"] == 33
    with pytest.raises(ValueError):
        run("unknown")


def test_context_features_are_causal_and_use_only_completed_previous_day():
    frame = bars(4000)
    frame["volume"] = np.linspace(1, 2, len(frame))
    config = StrategyEventConfig(warmup_hours=200, feature_profile="context_v2")
    original = prepare_event_frame(frame, config)
    columns = [name for name in original if "context_" in name]
    assert len(columns) == 20
    changed = frame.copy()
    changed.loc[3500:, ["open", "high", "low", "close"]] *= 2
    changed.loc[3500:, "volume"] *= 5
    altered = prepare_event_frame(changed, config)
    pd.testing.assert_frame_equal(original.iloc[:3500], altered.iloc[:3500])
    # 開始日只有部分 K 棒，隔天不能把不完整日範圍當完整前日。
    partial_next_day = original.timestamp.dt.date == pd.Timestamp("2024-01-02").date()
    assert original.loc[partial_next_day, "mtf_15m_context_previous_day_high_distance_atr"].isna().all()
    complete_previous_day = original.timestamp.dt.date == pd.Timestamp("2024-01-03").date()
    assert original.loc[complete_previous_day, "mtf_15m_context_previous_day_high_distance_atr"].notna().all()
