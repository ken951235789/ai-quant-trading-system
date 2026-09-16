"""由安全設定與環境變數建立實盤交易閘門。"""

from __future__ import annotations

from pathlib import Path

import requests

from ai_quant_trading.live_trading.binance_client import BinancePrivateClient
from ai_quant_trading.live_trading.binance_futures_client import (
    BinanceFuturesPrivateClient,
)
from ai_quant_trading.live_trading.config import LiveTradingConfig
from ai_quant_trading.live_trading.credentials import load_binance_credentials
from ai_quant_trading.live_trading.gateway import LiveTradingGateway
from ai_quant_trading.live_trading.futures_gateway import FuturesTradingGateway


def build_live_gateway(
    config: LiveTradingConfig,
    *,
    env_path: str | Path | None = None,
    session: requests.Session | None = None,
) -> LiveTradingGateway | FuturesTradingGateway:
    """載入指定環境憑證，建立不會自動連線的 Gateway。"""
    credentials = load_binance_credentials(config.environment, env_path)
    client_type = (
        BinanceFuturesPrivateClient
        if config.market_type == "usd_m_futures"
        else BinancePrivateClient
    )
    client = client_type(
        credentials,
        config.base_url,
        recv_window_ms=config.recv_window_ms,
        session=session,
    )
    if config.market_type == "usd_m_futures":
        return FuturesTradingGateway(client, config)  # type: ignore[arg-type]
    return LiveTradingGateway(client, config)  # type: ignore[arg-type]
