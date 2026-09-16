"""將已評分新聞以因果方式對齊市場 K 線。"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ai_quant_trading.data_collection.assets import infer_asset_class
from ai_quant_trading.sentiment.config import FinBERTConfig


FINBERT_CONTEXT_COLUMNS = [
    "finbert_sentiment",
    "finbert_positive",
    "finbert_negative",
    "finbert_confidence",
    "finbert_news_count",
    "finbert_sentiment_change",
    "finbert_hours_since_news",
    "finbert_available",
]

_SCORED_COLUMNS = [
    "finbert_sentiment",
    "finbert_positive",
    "finbert_negative",
    "finbert_confidence",
]


def _empty_context(rows: int, window_hours: int) -> pd.DataFrame:
    result = pd.DataFrame(0.0, index=range(rows), columns=FINBERT_CONTEXT_COLUMNS)
    result["finbert_hours_since_news"] = float(window_hours)
    return result


def aggregate_finbert_features(
    market: pd.DataFrame,
    scored_news: pd.DataFrame,
    config: FinBERTConfig | None = None,
) -> pd.DataFrame:
    """只聚合 K 線時間當下已被系統取得的新聞，避免 Look Ahead Bias。"""
    settings = config or FinBERTConfig()
    if "timestamp" not in market:
        raise ValueError("市場資料缺少 timestamp 欄位")
    required_news = ["published_at", *_SCORED_COLUMNS]
    missing = [column for column in required_news if column not in scored_news]
    if missing:
        raise ValueError(f"FinBERT 新聞資料缺少欄位：{missing}")

    result = market.drop(
        columns=[column for column in FINBERT_CONTEXT_COLUMNS if column in market],
        errors="ignore",
    ).reset_index(drop=True)
    market_time = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
    if market_time.isna().any():
        raise ValueError("市場資料 timestamp 含無法解析的值")
    news = scored_news.copy()
    news["published_at"] = pd.to_datetime(news["published_at"], utc=True, errors="coerce")
    availability_column = "available_at" if "available_at" in news else "collected_at"
    if availability_column not in news:
        raise ValueError(
            "FinBERT 新聞資料缺少 collected_at 或 available_at，不可只用 published_at 進行歷史對齊"
        )
    news["_available_at"] = pd.to_datetime(
        news[availability_column],
        utc=True,
        errors="coerce",
    )
    # 供應商時間可能早於實際抓取時間；模型只能在兩者都已發生後看到新聞。
    news["_available_at"] = news[["published_at", "_available_at"]].max(axis=1)
    news[_SCORED_COLUMNS] = news[_SCORED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    news = news.dropna(subset=["published_at", "_available_at", *_SCORED_COLUMNS])
    if "symbol" in news and "symbol" in result:
        market_symbols = set(result["symbol"].dropna().astype(str))
        news_symbol = news["symbol"].fillna("").astype(str)
        news = news[news_symbol.isin(market_symbols) | news_symbol.eq("")]
    if "asset_class" in news:
        exchange = str(result.iloc[0].get("exchange", "")) if not result.empty else ""
        symbol = str(result.iloc[0].get("symbol", "")) if not result.empty else ""
        market_asset_class = infer_asset_class(exchange, symbol)
        news_asset_class = news["asset_class"].fillna("").astype(str)
        news = news[news_asset_class.eq(market_asset_class) | news_asset_class.eq("")]
    news = news.sort_values("_available_at").reset_index(drop=True)
    context = _empty_context(len(result), settings.aggregation_window_hours)
    if news.empty:
        return pd.concat([result, context], axis=1)

    window = pd.Timedelta(hours=settings.aggregation_window_hours)
    half_life = settings.decay_half_life_hours
    news_time = pd.DatetimeIndex(news["_available_at"])
    previous_sentiment = 0.0
    for row_index, timestamp in enumerate(market_time):
        start = news_time.searchsorted(timestamp - window, side="right")
        end = news_time.searchsorted(timestamp, side="right")
        eligible = news.iloc[start:end]
        if eligible.empty:
            previous_sentiment = 0.0
            continue
        age_hours = ((timestamp - eligible["_available_at"]).dt.total_seconds() / 3600).to_numpy(
            dtype=float
        )
        weights = np.exp(-math.log(2) * age_hours / half_life)
        weights = weights / weights.sum()
        sentiment = float(np.dot(eligible["finbert_sentiment"], weights))
        context.loc[row_index, "finbert_sentiment"] = sentiment
        context.loc[row_index, "finbert_positive"] = float(
            np.dot(eligible["finbert_positive"], weights)
        )
        context.loc[row_index, "finbert_negative"] = float(
            np.dot(eligible["finbert_negative"], weights)
        )
        context.loc[row_index, "finbert_confidence"] = float(
            np.dot(eligible["finbert_confidence"], weights)
        )
        context.loc[row_index, "finbert_news_count"] = float(len(eligible))
        context.loc[row_index, "finbert_sentiment_change"] = sentiment - previous_sentiment
        context.loc[row_index, "finbert_hours_since_news"] = float(age_hours.min())
        context.loc[row_index, "finbert_available"] = 1.0
        previous_sentiment = sentiment
    return pd.concat([result, context], axis=1)
