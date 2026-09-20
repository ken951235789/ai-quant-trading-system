"""精簡 RL 欄位與 expected_return 中繼資料的回歸測試。"""

import json

import numpy as np
import pandas as pd
import pytest

from ai_quant_trading.features import build_feature_dataset
from ai_quant_trading.features.market_context import MARKET_CONTEXT_COLUMNS
from ai_quant_trading.features.multitimeframe import MULTITIMEFRAME_FEATURE_NAMES
from ai_quant_trading.market_clock import BTC_MULTITIMEFRAME_INTERVALS
from ai_quant_trading.reinforcement_learning.config import PortfolioEnvConfig, RLSplitConfig
from ai_quant_trading.reinforcement_learning.dataset import prepare_rl_dataset
from ai_quant_trading.reinforcement_learning.feature_contract import (
    attach_expected_return,
    build_expected_return_contract,
    resolve_expected_return_contract,
    select_compact_short_term_features,
)
from ai_quant_trading.reinforcement_learning.storage import save_rl_environment
from ai_quant_trading.reinforcement_learning.universal import (
    add_universal_rl_features,
    build_compact_short_term_frame,
)
from ai_quant_trading.transformer.dataset import _select_feature_columns


ALL_TIMEFRAMES = ("1m", "3m", "5m", "15m", "30m", "1h", "4h", "12h", "1d")


def compact_market_frame(intervals=ALL_TIMEFRAMES, rows=400):
    close = 100 + np.arange(rows) * 0.02 + np.sin(np.arange(rows) / 6)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="15min", tz="UTC"),
            "exchange": "binance_futures",
            "symbol": "BTC/USDT",
            "interval": "15m",
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 10 + np.arange(rows) % 17,
        }
    )
    frame = build_feature_dataset(frame, drop_na=False)
    values = {}
    for interval in intervals:
        for name in MULTITIMEFRAME_FEATURE_NAMES:
            values[f"mtf_{interval}_{name}"] = np.sin(np.arange(rows) / 7) * 0.1
        values[f"mtf_{interval}_available"] = 1.0
        values[f"mtf_{interval}_age_ratio"] = 0.0
        values[f"mtf_{interval}_rsi_14"] = 0.6
    frame = pd.concat([frame, pd.DataFrame(values, index=frame.index)], axis=1)
    frame["transformer_return_5"] = 0.003
    frame["transformer_return_20"] = -0.02
    frame["transformer_return_48"] = 0.01
    frame["transformer_available"] = 1.0
    return frame


@pytest.mark.parametrize("intervals", [ALL_TIMEFRAMES, BTC_MULTITIMEFRAME_INTERVALS])
def test_compact_features_have_balanced_scales_without_duplicate_15m(intervals) -> None:
    frame = add_universal_rl_features(compact_market_frame(intervals))
    columns = select_compact_short_term_features(frame)
    assert 80 <= len(columns) <= 112
    assert not {"u_rsi_14", "u_macd_pct", "u_return_1", "u_short_reversal_entry"}.intersection(
        columns
    )
    assert all(f"u_{column}" in columns for column in MARKET_CONTEXT_COLUMNS)
    assert "u_transformer_return_48" in columns
    assert "u_transformer_tradeability_48" in columns
    counts = [
        sum(column.startswith(f"u_mtf_{interval}_") for column in columns) for interval in intervals
    ]
    assert max(counts) - min(counts) <= 1
    assert all(f"u_mtf_{interval}_ema_20_50_atr" in columns for interval in intervals)


def test_transformer_budget_balances_indicator_types_and_timeframes() -> None:
    frame = compact_market_frame()
    columns = _select_feature_columns([frame], 256)
    assert len(columns) == 256
    assert not {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "obv",
        "vwap",
        "macd",
        "atr_14",
        "bb_upper_20",
        "bb_lower_20",
        "bb_middle_20",
    }.intersection(columns)
    counts = [
        sum(column.startswith(f"mtf_{interval}_") for column in columns)
        for interval in ALL_TIMEFRAMES
    ]
    assert max(counts) - min(counts) <= 2
    for interval in ALL_TIMEFRAMES:
        for name in (
            "ema_20_gap",
            "return_1",
            "atr_pct",
            "volume_change",
            "candle_body_pct",
            "smc_swing_high_distance_atr",
            "funding_rate",
            "trend_regime",
        ):
            assert f"mtf_{interval}_{name}" in columns
    shuffled = frame.loc[:, list(reversed(frame.columns))]
    assert _select_feature_columns([shuffled], 256) == columns


def test_transformer_small_budget_is_not_exceeded() -> None:
    frame = compact_market_frame(rows=220)
    assert len(_select_feature_columns([frame], 1)) == 1


def test_compact_builder_matches_full_conversion_and_retains_contract() -> None:
    source = attach_expected_return(compact_market_frame(), build_expected_return_contract(5))
    full = add_universal_rl_features(source)
    columns = select_compact_short_term_features(full)
    compact, compact_columns = build_compact_short_term_frame(source)
    assert compact_columns == columns
    pd.testing.assert_frame_equal(compact[columns], full[columns])
    assert compact.attrs["expected_return_contract"]["horizon_bars"] == 5
    assert compact.attrs["rl_feature_contract"]["feature_columns"] == columns
    assert sorted(column for column in compact if column.startswith("u_")) == sorted(columns)
    assert compact.expected_return.eq(0.003).all()


def test_compact_schema_does_not_change_with_first_row_warmup_missing_values() -> None:
    source = compact_market_frame()
    _, expected = build_compact_short_term_frame(source)
    fields = [column for column in source if column.startswith("mtf_")]
    source.loc[0, fields] = np.nan
    compact, actual = build_compact_short_term_frame(source)
    assert actual == expected
    source["mtf_5m_funding_rate"] = np.nan
    compact, actual = build_compact_short_term_frame(source)
    assert actual == expected
    assert compact.u_mtf_5m_funding_rate.eq(0).all()


def test_expected_return_contract_survives_transformations_and_storage(tmp_path) -> None:
    frame = attach_expected_return(compact_market_frame(), build_expected_return_contract(5))
    frame = add_universal_rl_features(frame)
    frame["expected_return"] = -0.02
    columns = select_compact_short_term_features(frame)
    dataset = prepare_rl_dataset(frame, columns, RLSplitConfig(min_rows_per_split=10))
    assert dataset.frame.expected_return.eq(0.003).all()
    assert dataset.expected_return_contract["horizon_bars"] == 5
    assert dataset.frame.transformer_return_5.eq(0.003).all()
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"fixture")
    artifact = save_rl_environment(
        dataset,
        PortfolioEnvConfig(),
        tmp_path / "environments",
        transformer_checkpoint=checkpoint,
        transformer_provenance={"fixture": True},
    )
    metadata = json.loads(artifact.metadata_json.read_text(encoding="utf-8"))
    assert (
        metadata["ai_context"]["expected_return_contract"]["source_column"]
        == "transformer_return_5"
    )
    assert metadata["feature_contract"]["version"] == "btc_short_compact_v3"
    assert metadata["feature_transform"] == "universal_ratios"


def test_expected_return_contract_rejects_wrong_source_and_units() -> None:
    assert resolve_expected_return_contract({})["horizon_bars"] == 20
    contract = build_expected_return_contract(5)
    with pytest.raises(ValueError, match="來源、單位或方向"):
        resolve_expected_return_contract(
            {"expected_return_contract": {**contract, "source_column": "transformer_return_20"}}
        )
    with pytest.raises(ValueError, match="來源不存在"):
        attach_expected_return(pd.DataFrame({"expected_return": [0.01]}), contract)


def test_new_transformer_environment_requires_explicit_expected_return_contract(tmp_path) -> None:
    frame = add_universal_rl_features(compact_market_frame())
    columns = select_compact_short_term_features(frame)
    dataset = prepare_rl_dataset(frame, columns, RLSplitConfig(min_rows_per_split=10))
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"fixture")
    with pytest.raises(ValueError, match="明確指定 expected_return_contract"):
        save_rl_environment(
            dataset,
            PortfolioEnvConfig(),
            tmp_path / "env",
            transformer_checkpoint=checkpoint,
            transformer_provenance={"fixture": True},
        )


def test_transformer_signal_entropy_retains_opposing_horizons() -> None:
    frame = compact_market_frame()
    for horizon, side in ((5, "up"), (20, "down"), (48, "neutral")):
        for name in ("down", "neutral", "up"):
            frame[f"transformer_{name}_probability_{horizon}"] = float(name == side)
    result = add_universal_rl_features(frame)
    np.testing.assert_allclose(result.u_transformer_signal_entropy, 1.0)
    frame["transformer_available"] = 0.0
    assert add_universal_rl_features(frame).u_transformer_signal_entropy.eq(0.0).all()
