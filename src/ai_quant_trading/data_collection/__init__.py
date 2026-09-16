"""市場資料收集模組。"""

from ai_quant_trading.data_collection.assets import (
    CRYPTO_PRESETS,
    AssetDefinition,
    asset_definition,
    asset_display_name,
    infer_asset_class,
    normalize_symbol,
)
from ai_quant_trading.data_collection.binance import BinanceSpotClient
from ai_quant_trading.data_collection.binance_stream import (
    BinanceBookTickerUpdate,
    BinanceKlineUpdate,
    BinanceRealtimeStream,
    BinanceStreamConfig,
    parse_binance_stream_message,
)
from ai_quant_trading.data_collection.binance_futures import (
    BinanceFuturesPublicClient,
    attach_derivatives_context,
)
from ai_quant_trading.data_collection.collectors import (
    DataCollectionResult,
    collect_binance_data,
    collect_binance_futures_data,
    collect_binance_futures_timeframe_bundle,
    collect_binance_timeframe_bundle,
    collect_yahoo_finance_data,
)
from ai_quant_trading.data_collection.equity_universe import (
    US_EQUITY_PRESETS,
    preset_symbols_text,
)
from ai_quant_trading.data_collection.yahoo_finance import YahooFinanceClient

__all__ = [
    "CRYPTO_PRESETS",
    "AssetDefinition",
    "BinanceSpotClient",
    "BinanceBookTickerUpdate",
    "BinanceKlineUpdate",
    "BinanceRealtimeStream",
    "BinanceStreamConfig",
    "BinanceFuturesPublicClient",
    "DataCollectionResult",
    "YahooFinanceClient",
    "US_EQUITY_PRESETS",
    "asset_definition",
    "asset_display_name",
    "attach_derivatives_context",
    "collect_binance_data",
    "collect_binance_futures_data",
    "collect_binance_futures_timeframe_bundle",
    "collect_binance_timeframe_bundle",
    "collect_yahoo_finance_data",
    "infer_asset_class",
    "normalize_symbol",
    "preset_symbols_text",
    "parse_binance_stream_message",
]
