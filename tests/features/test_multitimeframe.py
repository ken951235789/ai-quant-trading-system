"""多週期 K 線因果對齊與批次輸出的測試。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai_quant_trading.features.indicators import FEATURE_COLUMNS
from ai_quant_trading.features.multitimeframe import (
    MULTITIMEFRAME_FEATURE_GROUPS,
    build_multitimeframe_feature_datasets,
    build_multitimeframe_frame,
    multitimeframe_numeric_columns,
)
from ai_quant_trading.market_clock import BTC_MULTITIMEFRAME_INTERVALS


def _feature_frame(
    interval: str,
    timestamps: pd.DatetimeIndex,
    markers: np.ndarray,
    *,
    symbol: str = "BTC/USDT",
) -> pd.DataFrame:
    close = 100 + np.arange(len(timestamps), dtype=float)
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": symbol,
            "exchange": "binance",
            "interval": interval,
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1_000.0 + np.arange(len(timestamps), dtype=float) ** 2,
        }
    )
    feature_values = pd.DataFrame(
        1.0,
        index=frame.index,
        columns=FEATURE_COLUMNS,
    )
    frame = pd.concat([frame, feature_values], axis=1)
    frame["return_1"] = markers
    frame["rsi_14"] = 50.0
    frame["ema_20"] = close
    frame["sma_200"] = close
    frame["macd_histogram"] = 0.1
    frame["atr_14"] = 1.0
    frame["bb_width_20"] = 0.05
    frame["volume_change"] = 0.1
    frame["breakout_20"] = 0.0
    return frame


def test_alignment_never_uses_unclosed_higher_timeframe() -> None:
    decision_times = pd.date_range("2025-01-02 05:45", periods=20, freq="15min", tz="UTC")
    hourly_times = pd.date_range("2025-01-01 13:00", periods=22, freq="h", tz="UTC")
    minute_times = pd.date_range("2025-01-02 10:00", periods=45, freq="min", tz="UTC")
    frames = {
        "1m": _feature_frame("1m", minute_times, np.arange(45) / 1_000),
        "15m": _feature_frame("15m", decision_times, np.arange(20) / 100),
        "1h": _feature_frame(
            "1h",
            hourly_times,
            np.array([*np.linspace(0.01, 0.08, 20), 0.09, 9.99]),
        ),
    }

    fused, coverage = build_multitimeframe_frame(frames, "15m")

    # 10:30～10:45 的決策不能使用 10:00～11:00 尚未收盤的 1h K 線。
    assert fused.iloc[-1]["mtf_1h_return_1"] == 0.09
    # 同一決策可使用 10:44～10:45 已收盤的最後一根 1m K 線。
    assert fused.iloc[-1]["mtf_1m_return_1"] == 0.044
    assert fused.iloc[-1]["mtf_1h_available"] == 1.0
    assert all(0.0 < value <= 1.0 for value in coverage.values())


def test_batch_builder_writes_csv_and_causal_manifest(tmp_path: Path) -> None:
    minute = _feature_frame(
        "1m",
        pd.date_range("2025-01-01", periods=300, freq="min", tz="UTC"),
        np.linspace(0.0, 0.01, 300),
    )
    fifteen = _feature_frame(
        "15m",
        pd.date_range("2025-01-01", periods=20, freq="15min", tz="UTC"),
        np.linspace(0.0, 0.01, 20),
    )
    one = tmp_path / "btc_1m.csv"
    fifteen_path = tmp_path / "btc_15m.csv"
    minute.to_csv(one, index=False)
    fifteen.to_csv(fifteen_path, index=False)

    artifacts = build_multitimeframe_feature_datasets(
        [one, fifteen_path], "15m", tmp_path / "processed"
    )

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.output_path.exists()
    assert artifact.manifest_path.exists()
    saved = pd.read_csv(artifact.output_path)
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    assert saved["mtf_source_intervals"].eq("1m|15m").all()
    assert "mtf_1m_return_1" in saved
    assert manifest["causal_alignment"] == "source_close_time <= decision_close_time"
    assert set(manifest["feature_groups"]) == set(MULTITIMEFRAME_FEATURE_GROUPS)


def test_btc_bundle_contains_all_standard_timeframes_and_indicator_groups() -> None:
    frequencies = {
        "1m": "1min",
        "3m": "3min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "4h": "4h",
        "12h": "12h",
        "1d": "1D",
    }
    end = pd.Timestamp("2025-06-30 00:00", tz="UTC")
    frames = {
        interval: _feature_frame(
            interval,
            pd.date_range(end=end, periods=260, freq=frequencies[interval]),
            np.linspace(0.0, 0.02, 260),
        )
        for interval in BTC_MULTITIMEFRAME_INTERVALS
    }

    fused, coverage = build_multitimeframe_frame(frames, "5m")

    assert fused.iloc[-1]["mtf_source_intervals"] == "|".join(
        BTC_MULTITIMEFRAME_INTERVALS
    )
    assert set(coverage) == set(BTC_MULTITIMEFRAME_INTERVALS)
    expected_per_interval = sum(
        len(features) for features in MULTITIMEFRAME_FEATURE_GROUPS.values()
    ) + 2
    context_columns = [
        column for column in fused if column.startswith("mtf_context_")
    ]
    assert len(multitimeframe_numeric_columns(fused)) == (
        len(BTC_MULTITIMEFRAME_INTERVALS) * expected_per_interval
        + len(context_columns)
    )
    for interval in BTC_MULTITIMEFRAME_INTERVALS:
        assert fused.iloc[-1][f"mtf_{interval}_available"] == 1.0
        for features in MULTITIMEFRAME_FEATURE_GROUPS.values():
            for feature in features:
                assert f"mtf_{interval}_{feature}" in fused
