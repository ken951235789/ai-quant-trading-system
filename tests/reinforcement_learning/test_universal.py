"""通用多市場 RL 特徵、切分、環境與成品測試。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ai_quant_trading.features import build_feature_dataset
from ai_quant_trading.reinforcement_learning import (
    PortfolioEnvConfig,
    RLSplitConfig,
    UniversalPortfolioTradingEnv,
    add_universal_rl_features,
    prepare_universal_rl_dataset,
    save_universal_rl_environment,
)
from ai_quant_trading.reinforcement_learning.universal import (
    MULTIMODAL_RL_FEATURE_COLUMNS,
    UNIVERSAL_RL_FEATURE_COLUMNS,
)


def make_universal_market(
    symbol: str,
    exchange: str,
    *,
    scale: float = 1.0,
    phase: float = 0.0,
    rows: int = 620,
) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    close = scale * (100 + index * 0.04 + np.sin(index / 8 + phase) * 4)
    raw = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=rows, freq="D", tz="UTC"),
            "symbol": symbol,
            "exchange": exchange,
            "interval": "1d",
            "open": close * 0.998,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1000 + index,
        }
    )
    return build_feature_dataset(raw, annualization_periods=365, drop_na=True)


def make_universal_frames() -> dict[str, pd.DataFrame]:
    return {
        "AAPL": make_universal_market("AAPL", "yahoo_finance"),
        "BTC/USDT": make_universal_market("BTC/USDT", "binance", phase=1.1),
    }


def test_universal_features_are_price_scale_independent() -> None:
    base = add_universal_rl_features(make_universal_market("AAPL", "yahoo_finance"))
    scaled = add_universal_rl_features(
        make_universal_market("AAPL", "yahoo_finance", scale=100)
    )
    columns = [column for column in base.columns if column.startswith("u_")]

    np.testing.assert_allclose(
        base[columns].to_numpy(dtype=float),
        scaled[columns].to_numpy(dtype=float),
        rtol=1e-8,
        atol=1e-8,
    )


def test_multimodal_features_are_safe_when_models_are_unavailable() -> None:
    result = add_universal_rl_features(
        make_universal_market("BTC/USDT", "binance")
    )

    assert set(MULTIMODAL_RL_FEATURE_COLUMNS).issubset(result.columns)
    assert result["u_finbert_available"].eq(0).all()
    assert result["u_transformer_available"].eq(0).all()
    assert result["u_transformer_consensus_signal"].eq(0).all()
    assert result["u_transformer_no_trade_pressure"].eq(0).all()
    assert np.isfinite(result[MULTIMODAL_RL_FEATURE_COLUMNS].to_numpy()).all()


def test_binance_futures_is_recognized_as_crypto() -> None:
    result = add_universal_rl_features(
        make_universal_market("BTC/USDT", "binance_futures")
    )

    assert result["u_is_crypto"].eq(1).all()


def test_multimodal_features_preserve_valid_context() -> None:
    frame = make_universal_market("AAPL", "yahoo_finance")
    frame["finbert_sentiment"] = 0.7
    frame["finbert_positive"] = 0.8
    frame["finbert_negative"] = 0.1
    frame["finbert_confidence"] = 0.9
    frame["finbert_news_count"] = 3
    frame["finbert_sentiment_change"] = 0.2
    frame["finbert_hours_since_news"] = 2
    frame["finbert_available"] = 1
    frame["transformer_return_5"] = 0.04
    frame["transformer_available"] = 1
    frame["transformer_uncertainty"] = 0.2
    for horizon, side_edge, tradeability in (
        (1, 0.2, 0.5),
        (5, 0.6, 0.8),
        (20, 0.8, 0.9),
    ):
        frame[f"transformer_side_down_probability_{horizon}"] = (
            1 - side_edge
        ) / 2
        frame[f"transformer_side_up_probability_{horizon}"] = (
            1 + side_edge
        ) / 2
        frame[f"transformer_tradeability_{horizon}"] = tradeability

    result = add_universal_rl_features(frame)

    assert result["u_finbert_sentiment"].eq(0.7).all()
    assert result["u_finbert_available"].eq(1).all()
    assert result["u_transformer_return_5"].eq(0.04).all()
    assert result["u_transformer_available"].eq(1).all()
    np.testing.assert_allclose(result["u_transformer_trend_signal"], 0.576)
    np.testing.assert_allclose(result["u_transformer_setup_signal"], 0.384)
    np.testing.assert_allclose(result["u_transformer_execution_signal"], 0.08)
    np.testing.assert_allclose(result["u_transformer_consensus_signal"], 0.4344)


def test_multitimeframe_features_are_added_to_ppo_observation() -> None:
    frames = make_universal_frames()
    for frame in frames.values():
        frame["mtf_15m_return_1"] = frame["return_1"]
        frame["mtf_15m_rsi_14"] = frame["rsi_14"] / 100
        frame["mtf_15m_available"] = 1.0
        frame["mtf_15m_age_ratio"] = 0.0
        frame["mtf_decision_interval"] = "1d"
        frame["mtf_source_intervals"] = "15m|1d"
        frame["mtf_schema_version"] = 1

    dataset = prepare_universal_rl_dataset(
        frames,
        RLSplitConfig(0.6, 0.2, min_rows_per_split=20),
    )

    assert "u_mtf_15m_return_1" in dataset.feature_columns
    assert "u_mtf_15m_available" in dataset.feature_columns
    assert all(
        np.isfinite(market.frame[dataset.feature_columns].to_numpy()).all()
        for market in dataset.markets.values()
    )


def test_pipeline_can_disable_multimodal_context() -> None:
    frames = make_universal_frames()
    for frame in frames.values():
        frame["finbert_sentiment"] = 0.8
        frame["finbert_available"] = 1
        frame["transformer_return_5"] = 0.1
        frame["transformer_available"] = 1

    dataset = prepare_universal_rl_dataset(
        frames,
        RLSplitConfig(0.6, 0.2, min_rows_per_split=20),
        use_finbert=False,
        use_transformer=False,
    )

    combined = pd.concat(
        [market.frame for market in dataset.markets.values()],
        ignore_index=True,
    )
    ai_columns = [
        column
        for column in dataset.feature_columns
        if "finbert" in column or "transformer" in column
    ]
    assert combined[ai_columns].eq(0).all().all()


def test_universal_dataset_uses_joint_train_only_normalization() -> None:
    dataset = prepare_universal_rl_dataset(
        make_universal_frames(),
        RLSplitConfig(0.6, 0.2, min_rows_per_split=20),
    )
    combined_train = pd.concat(
        [market.train[dataset.feature_columns] for market in dataset.markets.values()],
        ignore_index=True,
    )

    assert len(dataset.markets) == 2
    assert dataset.observation_size == len(dataset.feature_columns) + 5
    assert np.allclose(combined_train.mean(), 0, atol=1e-7)
    assert np.isfinite(combined_train.to_numpy()).all()


def test_universal_feature_rebuild_is_idempotent() -> None:
    source = make_universal_frames()["AAPL"]
    first = add_universal_rl_features(source)
    second = add_universal_rl_features(first)

    assert not second.columns.duplicated().any()
    pd.testing.assert_frame_equal(
        first[UNIVERSAL_RL_FEATURE_COLUMNS],
        second[UNIVERSAL_RL_FEATURE_COLUMNS],
    )


def test_universal_environment_never_crosses_market_inside_episode() -> None:
    dataset = prepare_universal_rl_dataset(
        make_universal_frames(),
        RLSplitConfig(0.6, 0.2, min_rows_per_split=20),
    )
    env = UniversalPortfolioTradingEnv(
        {name: market.train for name, market in dataset.markets.items()},
        dataset.feature_columns,
        PortfolioEnvConfig(episode_length=8),
    )

    observation, info = env.reset(seed=42, options={"market": "AAPL"})
    observed_markets = {info["market"]}
    for _ in range(8):
        observation, _, terminated, truncated, info = env.step(np.array([0.25]))
        observed_markets.add(info["market"])
        if terminated or truncated:
            break

    assert observed_markets == {"AAPL"}
    assert env.observation_space.contains(observation)


def test_saves_universal_market_files_and_metadata(tmp_path) -> None:
    dataset = prepare_universal_rl_dataset(
        make_universal_frames(),
        RLSplitConfig(0.6, 0.2, min_rows_per_split=20),
    )
    paths = save_universal_rl_environment(dataset, PortfolioEnvConfig(), tmp_path)
    metadata = json.loads(paths.metadata_json.read_text(encoding="utf-8"))

    assert metadata["environment_kind"] == "universal"
    assert metadata["source"]["market_count"] == 2
    assert metadata["normalization"]["fit_on"] == "all_markets_train_only"
    for values in metadata["markets"].values():
        for relative in values["files"].values():
            assert (paths.run_dir / relative).exists()


def test_saves_runtime_ai_provenance_and_bundles_transformer(tmp_path) -> None:
    frames = make_universal_frames()
    for frame in frames.values():
        frame["transformer_return_20"] = .01
        frame["transformer_available"] = 1.
    dataset = prepare_universal_rl_dataset(
        frames,
        RLSplitConfig(0.6, 0.2, min_rows_per_split=20),
    )
    checkpoint = tmp_path / "source_model.pt"
    checkpoint.write_bytes(b"trusted local checkpoint")
    scored_news = tmp_path / "finbert_news_latest.csv"
    scored_news.write_text("published_at,finbert_sentiment\n", encoding="utf-8")

    paths = save_universal_rl_environment(
        dataset,
        PortfolioEnvConfig(),
        tmp_path / "environments",
        transformer_checkpoint=checkpoint,
        transformer_provenance={
            "policy": "after_transformer_validation_and_calibration",
            "safe_rl_start_exclusive": "2025-01-01T00:00:00+00:00",
        },
        finbert_scored_news_path="data/processed/sentiment/finbert_news_latest.csv",
        finbert_config={"aggregation_window_hours": 24},
    )
    metadata = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
    context = metadata["ai_context"]

    assert context["runtime_ready"] is True
    assert context["expected_return_contract"]["horizon_bars"] == 20
    assert context["transformer_checkpoint"] == "ai/transformer_model.pt"
    assert context["transformer_provenance"]["policy"] == (
        "after_transformer_validation_and_calibration"
    )
    assert context["finbert_config"]["aggregation_window_hours"] == 24
    assert (paths.run_dir / context["transformer_checkpoint"]).read_bytes() == (
        checkpoint.read_bytes()
    )


def test_refuses_transformer_environment_without_time_provenance(tmp_path) -> None:
    dataset = prepare_universal_rl_dataset(
        make_universal_frames(),
        RLSplitConfig(0.6, 0.2, min_rows_per_split=20),
    )
    checkpoint = tmp_path / "source_model.pt"
    checkpoint.write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="時間隔離來源證明"):
        save_universal_rl_environment(
            dataset,
            PortfolioEnvConfig(),
            tmp_path / "environments",
            transformer_checkpoint=checkpoint,
        )


def test_saves_crossfit_oos_environment_before_runtime_checkpoint_exists(tmp_path) -> None:
    frames = make_universal_frames()
    for frame in frames.values():
        frame["transformer_return_20"] = 0.01
        frame["transformer_available"] = 1.0
        frame["transformer_oos"] = 1
    dataset = prepare_universal_rl_dataset(
        frames,
        RLSplitConfig(0.6, 0.2, min_rows_per_split=20),
    )
    crossfit = tmp_path / "crossfit.json"
    crossfit.write_text(
        json.dumps(
            {
                "status": "complete",
                "plan": {"final_holdout_sealed": True},
            }
        ),
        encoding="utf-8",
    )

    paths = save_universal_rl_environment(
        dataset,
        PortfolioEnvConfig(),
        tmp_path / "environments",
        transformer_crossfit_summary=crossfit,
    )
    context = json.loads(paths.metadata_json.read_text(encoding="utf-8"))["ai_context"]

    assert context["runtime_ready"] is False
    assert context["transformer_crossfit_summary"] == "ai/transformer_crossfit.json"
    assert context["expected_return_contract"]["horizon_bars"] == 20
