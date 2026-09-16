"""從官方公開來源建立可搜尋的加密貨幣與美股清單。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from io import StringIO
import re

import requests

from ai_quant_trading.data_collection.assets import (
    CRYPTO_PRESETS,
    US_EQUITY_PRESETS,
    asset_display_name,
)
from ai_quant_trading.data_collection.binance import BinanceApiError, BinanceSpotClient
from ai_quant_trading.data_collection.http_client import create_secure_session, require_session


NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"


@dataclass(frozen=True, slots=True)
class AssetUniverse:
    """介面使用的代碼、顯示名稱與清單來源。"""

    symbols: tuple[str, ...]
    labels: dict[str, str]
    source: str
    warning: str | None = None


class NasdaqSymbolDirectoryError(RuntimeError):
    """Nasdaq Symbol Directory 下載或解析失敗。"""


@dataclass(slots=True)
class NasdaqSymbolDirectoryClient:
    """讀取 Nasdaq 官方每日更新的美國掛牌證券清單。"""

    timeout: int = 30
    session: requests.Session | None = None

    def __post_init__(self) -> None:
        if self.session is None:
            self.session = create_secure_session()

    def _download(self, url: str) -> str:
        session = require_session(self.session)
        try:
            response = session.get(url, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise NasdaqSymbolDirectoryError(f"Nasdaq 美股清單下載失敗：{exc}") from exc
        return response.text

    @staticmethod
    def _parse(text: str, symbol_column: str) -> dict[str, str]:
        records: dict[str, str] = {}
        for row in csv.DictReader(StringIO(text), delimiter="|"):
            raw_symbol = str(row.get(symbol_column, "")).strip().upper()
            if not raw_symbol or raw_symbol.startswith("FILE CREATION TIME"):
                continue
            if str(row.get("Test Issue", "N")).strip().upper() == "Y":
                continue
            # Yahoo Finance 以連字號表示 BRK.B 這類類股代碼。
            symbol = raw_symbol.replace(".", "-")
            if not re.fullmatch(r"[A-Z0-9-]{1,15}", symbol):
                continue
            name = str(row.get("Security Name", symbol)).strip() or symbol
            records[symbol] = name
        return records

    def fetch_symbols(self) -> dict[str, str]:
        """合併 Nasdaq 與其他美國交易所清單，並排除測試證券。"""
        listed = self._parse(self._download(NASDAQ_LISTED_URL), "Symbol")
        other = self._parse(self._download(OTHER_LISTED_URL), "ACT Symbol")
        listed.update(other)
        if not listed:
            raise NasdaqSymbolDirectoryError("Nasdaq 官方清單沒有可用的美股代碼")
        return dict(sorted(listed.items()))


def _fallback_crypto_symbols() -> list[str]:
    return sorted({symbol for values in CRYPTO_PRESETS.values() for symbol in values})


def _fallback_equity_symbols() -> list[str]:
    return sorted({symbol for values in US_EQUITY_PRESETS.values() for symbol in values})


def load_crypto_universe(client: BinanceSpotClient | None = None) -> AssetUniverse:
    """優先使用 Binance 即時清單，失敗時退回內建常用交易對。"""
    try:
        symbols = (client or BinanceSpotClient()).fetch_spot_symbols("USDT")
        if not symbols:
            raise BinanceApiError("Binance 沒有回傳可交易的 USDT 現貨")
        return AssetUniverse(
            tuple(symbols),
            {symbol: asset_display_name(symbol, "crypto") for symbol in symbols},
            "Binance 即時現貨清單",
        )
    except (BinanceApiError, requests.RequestException, ValueError) as exc:
        symbols = _fallback_crypto_symbols()
        return AssetUniverse(
            tuple(symbols),
            {symbol: asset_display_name(symbol, "crypto") for symbol in symbols},
            "內建備援清單",
            f"Binance 完整清單暫時無法更新：{exc}",
        )


def load_us_equity_universe(
    client: NasdaqSymbolDirectoryClient | None = None,
) -> AssetUniverse:
    """優先使用 Nasdaq 官方完整清單，失敗時退回內建常用美股。"""
    try:
        labels = (client or NasdaqSymbolDirectoryClient()).fetch_symbols()
        return AssetUniverse(
            tuple(labels),
            {symbol: f"{symbol}｜{name}" for symbol, name in labels.items()},
            "Nasdaq 官方 Symbol Directory",
        )
    except (NasdaqSymbolDirectoryError, requests.RequestException, ValueError) as exc:
        symbols = _fallback_equity_symbols()
        return AssetUniverse(
            tuple(symbols),
            {symbol: asset_display_name(symbol, "us_equity") for symbol in symbols},
            "內建備援清單",
            f"Nasdaq 完整清單暫時無法更新：{exc}",
        )
