"""FinBERT 新聞情緒分析與因果特徵聚合。"""

from ai_quant_trading.sentiment.config import FinBERTConfig
from ai_quant_trading.sentiment.features import (
    FINBERT_CONTEXT_COLUMNS,
    aggregate_finbert_features,
)
from ai_quant_trading.sentiment.finbert import (
    FinBERTAnalyzer,
    FinBERTBackendStatus,
    finbert_backend_status,
)
from ai_quant_trading.sentiment.news import (
    FINBERT_SCORE_COLUMNS,
    NEWS_COLUMNS,
    CoinDeskNewsClient,
    NewsCollectionError,
    NewsCollectionResult,
    YahooFinanceNewsClient,
    collect_market_news,
    load_news_history,
    load_scored_news,
    news_history_path,
    parse_rss_news,
    recent_news,
    save_news_history,
    save_scored_news,
)

__all__ = [
    "FINBERT_CONTEXT_COLUMNS",
    "FinBERTAnalyzer",
    "FinBERTBackendStatus",
    "FinBERTConfig",
    "FINBERT_SCORE_COLUMNS",
    "NEWS_COLUMNS",
    "CoinDeskNewsClient",
    "NewsCollectionError",
    "NewsCollectionResult",
    "YahooFinanceNewsClient",
    "aggregate_finbert_features",
    "collect_market_news",
    "finbert_backend_status",
    "load_news_history",
    "load_scored_news",
    "news_history_path",
    "parse_rss_news",
    "recent_news",
    "save_news_history",
    "save_scored_news",
]
