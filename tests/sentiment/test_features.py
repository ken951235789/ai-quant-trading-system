"""FinBERT 新聞因果聚合測試。"""

from __future__ import annotations

import pandas as pd
import pytest

from ai_quant_trading.sentiment import FinBERTConfig, aggregate_finbert_features


def test_news_after_bar_is_not_used() -> None:
    market = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01 10:00Z", "2026-01-01 11:00Z", "2026-01-01 12:00Z"]
            ),
            "symbol": ["AAPL"] * 3,
        }
    )
    news = pd.DataFrame(
        {
            "published_at": pd.to_datetime(["2026-01-01 10:30Z"]),
            "collected_at": pd.to_datetime(["2026-01-01 10:45Z"]),
            "symbol": ["AAPL"],
            "finbert_sentiment": [0.8],
            "finbert_positive": [0.9],
            "finbert_negative": [0.1],
            "finbert_confidence": [0.9],
        }
    )

    result = aggregate_finbert_features(
        market,
        news,
        FinBERTConfig(aggregation_window_hours=24, decay_half_life_hours=8),
    )

    assert result.loc[0, "finbert_available"] == 0
    assert result.loc[1, "finbert_available"] == 1
    assert result.loc[1, "finbert_news_count"] == 1
    assert result.loc[1, "finbert_sentiment"] == 0.8


def test_news_symbol_is_filtered() -> None:
    market = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01 12:00Z"]),
            "symbol": ["BTC/USDT"],
        }
    )
    news = pd.DataFrame(
        {
            "published_at": pd.to_datetime(["2026-01-01 11:00Z"]),
            "collected_at": pd.to_datetime(["2026-01-01 11:01Z"]),
            "symbol": ["AAPL"],
            "finbert_sentiment": [0.8],
            "finbert_positive": [0.9],
            "finbert_negative": [0.1],
            "finbert_confidence": [0.9],
        }
    )

    result = aggregate_finbert_features(market, news)

    assert result.loc[0, "finbert_available"] == 0


def test_marketwide_crypto_news_is_not_applied_to_equity() -> None:
    market = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01 12:00Z"]),
            "symbol": ["AAPL"],
            "exchange": ["yahoo_finance"],
        }
    )
    news = pd.DataFrame(
        {
            "published_at": pd.to_datetime(["2026-01-01 11:00Z"]),
            "collected_at": pd.to_datetime(["2026-01-01 11:01Z"]),
            "symbol": [""],
            "asset_class": ["crypto"],
            "finbert_sentiment": [0.8],
            "finbert_positive": [0.9],
            "finbert_negative": [0.1],
            "finbert_confidence": [0.9],
        }
    )

    result = aggregate_finbert_features(market, news)

    assert result.loc[0, "finbert_available"] == 0


def test_news_is_not_available_before_collection_time() -> None:
    market = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01 11:00Z", "2026-01-01 12:00Z"]),
            "symbol": ["BTC/USDT", "BTC/USDT"],
        }
    )
    news = pd.DataFrame(
        {
            "published_at": pd.to_datetime(["2026-01-01 10:00Z"]),
            "collected_at": pd.to_datetime(["2026-01-01 11:30Z"]),
            "symbol": ["BTC/USDT"],
            "finbert_sentiment": [0.8],
            "finbert_positive": [0.9],
            "finbert_negative": [0.1],
            "finbert_confidence": [0.9],
        }
    )

    result = aggregate_finbert_features(market, news)

    assert result.loc[0, "finbert_available"] == 0
    assert result.loc[1, "finbert_available"] == 1


def test_news_without_availability_time_fails_closed() -> None:
    market = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01 12:00Z"]),
            "symbol": ["BTC/USDT"],
        }
    )
    news = pd.DataFrame(
        {
            "published_at": pd.to_datetime(["2026-01-01 10:00Z"]),
            "symbol": ["BTC/USDT"],
            "finbert_sentiment": [0.8],
            "finbert_positive": [0.9],
            "finbert_negative": [0.1],
            "finbert_confidence": [0.9],
        }
    )

    with pytest.raises(ValueError, match="collected_at 或 available_at"):
        aggregate_finbert_features(market, news)
