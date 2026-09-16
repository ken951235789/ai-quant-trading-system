"""官方資產清單與備援行為測試。"""

from ai_quant_trading.data_collection.binance import BinanceApiError
from ai_quant_trading.data_collection.universe import (
    NasdaqSymbolDirectoryClient,
    load_crypto_universe,
    load_us_equity_universe,
)


class FakeBinanceUniverseClient:
    def fetch_spot_symbols(self, quote_asset: str):
        assert quote_asset == "USDT"
        return ["BTC/USDT", "ETH/USDT"]


class FailedBinanceUniverseClient:
    def fetch_spot_symbols(self, quote_asset: str):
        raise BinanceApiError("暫時無法連線")


class FakeNasdaqUniverseClient:
    def fetch_symbols(self):
        return {"AAPL": "Apple Inc.", "BRK-B": "Berkshire Hathaway Class B"}


def test_loads_complete_crypto_choices_from_client() -> None:
    universe = load_crypto_universe(FakeBinanceUniverseClient())
    assert universe.symbols == ("BTC/USDT", "ETH/USDT")
    assert universe.warning is None


def test_crypto_failure_uses_builtin_fallback() -> None:
    universe = load_crypto_universe(FailedBinanceUniverseClient())
    assert "BTC/USDT" in universe.symbols
    assert universe.warning is not None


def test_loads_us_symbols_and_names() -> None:
    universe = load_us_equity_universe(FakeNasdaqUniverseClient())
    assert universe.symbols == ("AAPL", "BRK-B")
    assert "Apple" in universe.labels["AAPL"]


def test_nasdaq_parser_excludes_test_issues_and_normalizes_class_symbol() -> None:
    text = (
        "Symbol|Security Name|Test Issue|ETF\n"
        "AAPL|Apple Inc.|N|N\n"
        "BRK.B|Berkshire Hathaway|N|N\n"
        "ZTEST|Test Security|Y|N\n"
        "File Creation Time: 20260716||||\n"
    )

    parsed = NasdaqSymbolDirectoryClient._parse(text, "Symbol")

    assert parsed == {"AAPL": "Apple Inc.", "BRK-B": "Berkshire Hathaway"}
