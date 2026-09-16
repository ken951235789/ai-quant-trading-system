"""統一管理加密貨幣與美股的代碼、名稱及資料分類。"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal


AssetClass = Literal["crypto", "us_equity"]


@dataclass(frozen=True, slots=True)
class AssetDefinition:
    """單一研究標的的標準代碼與中英文名稱。"""

    symbol: str
    asset_class: AssetClass
    chinese_name: str
    english_name: str

    @property
    def display_name(self) -> str:
        return f"{self.symbol}｜{self.chinese_name}（{self.english_name}）"


CRYPTO_PRESETS: dict[str, tuple[str, ...]] = {
    "長期專家": (
        "BTC/USDT",
        "ETH/USDT",
        "SOL/USDT",
        "BNB/USDT",
        "LINK/USDT",
    ),
    "短期專家": (
        "BTC/USDT",
        "ETH/USDT",
        "SOL/USDT",
        "BNB/USDT",
        "XRP/USDT",
        "DOGE/USDT",
    ),
    "主要幣種": ("BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"),
    "大型公鏈": ("ETH/USDT", "SOL/USDT", "ADA/USDT", "AVAX/USDT", "DOT/USDT"),
    "其他常用": ("XRP/USDT", "DOGE/USDT", "LINK/USDT", "LTC/USDT", "BCH/USDT"),
    "DeFi": ("UNI/USDT", "AAVE/USDT", "LINK/USDT"),
}

US_EQUITY_PRESETS: dict[str, tuple[str, ...]] = {
    "長期專家": (
        "SPY",
        "QQQ",
        "MSFT",
        "AAPL",
        "JPM",
        "XOM",
        "GOOGL",
        "AMZN",
        "META",
        "BRK-B",
    ),
    "短期專家": (
        "SPY",
        "QQQ",
        "NVDA",
        "TSLA",
        "AMD",
        "AAPL",
        "META",
        "AMZN",
        "COIN",
    ),
    "大型科技": ("AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "NFLX"),
    "半導體": ("NVDA", "AMD", "AVGO", "TSM", "ASML", "INTC", "QCOM", "MU"),
    "金融": ("JPM", "BAC", "WFC", "GS", "MS", "V", "MA"),
    "醫療": ("LLY", "UNH", "JNJ", "ABBV", "MRK", "PFE"),
    "消費": ("COST", "WMT", "HD", "MCD", "NKE", "SBUX"),
    "常用 ETF": ("SPY", "QQQ", "DIA", "IWM", "VTI", "VOO", "XLK", "XLF", "XLE", "GLD", "TLT"),
}


_CRYPTO_NAMES = {
    "BTC": ("比特幣", "Bitcoin"),
    "ETH": ("以太幣", "Ethereum"),
    "BNB": ("幣安幣", "BNB"),
    "SOL": ("索拉納", "Solana"),
    "XRP": ("瑞波幣", "XRP"),
    "ADA": ("艾達幣", "Cardano"),
    "DOGE": ("狗狗幣", "Dogecoin"),
    "AVAX": ("雪崩幣", "Avalanche"),
    "LINK": ("鏈結幣", "Chainlink"),
    "DOT": ("波卡幣", "Polkadot"),
    "LTC": ("萊特幣", "Litecoin"),
    "BCH": ("比特幣現金", "Bitcoin Cash"),
    "UNI": ("Uniswap", "Uniswap"),
    "AAVE": ("Aave", "Aave"),
    "USDT": ("泰達幣", "Tether"),
    "USDC": ("美元穩定幣", "USD Coin"),
}

_US_EQUITY_NAMES = {
    "AAPL": ("蘋果", "Apple"),
    "MSFT": ("微軟", "Microsoft"),
    "NVDA": ("輝達", "NVIDIA"),
    "GOOGL": ("Alphabet", "Alphabet"),
    "AMZN": ("亞馬遜", "Amazon"),
    "META": ("Meta", "Meta Platforms"),
    "TSLA": ("特斯拉", "Tesla"),
    "NFLX": ("網飛", "Netflix"),
    "AMD": ("超微", "AMD"),
    "AVGO": ("博通", "Broadcom"),
    "TSM": ("台積電 ADR", "Taiwan Semiconductor ADR"),
    "ASML": ("艾司摩爾", "ASML"),
    "INTC": ("英特爾", "Intel"),
    "QCOM": ("高通", "Qualcomm"),
    "MU": ("美光", "Micron"),
    "JPM": ("摩根大通", "JPMorgan Chase"),
    "XOM": ("埃克森美孚", "Exxon Mobil"),
    "BRK-B": ("波克夏海瑟威 B 股", "Berkshire Hathaway Class B"),
    "COIN": ("Coinbase", "Coinbase Global"),
    "BAC": ("美國銀行", "Bank of America"),
    "WFC": ("富國銀行", "Wells Fargo"),
    "GS": ("高盛", "Goldman Sachs"),
    "MS": ("摩根士丹利", "Morgan Stanley"),
    "V": ("Visa", "Visa"),
    "MA": ("萬事達卡", "Mastercard"),
    "LLY": ("禮來", "Eli Lilly"),
    "UNH": ("聯合健康", "UnitedHealth"),
    "JNJ": ("嬌生", "Johnson & Johnson"),
    "ABBV": ("艾伯維", "AbbVie"),
    "MRK": ("默沙東", "Merck"),
    "PFE": ("輝瑞", "Pfizer"),
    "COST": ("好市多", "Costco"),
    "WMT": ("沃爾瑪", "Walmart"),
    "HD": ("家得寶", "Home Depot"),
    "MCD": ("麥當勞", "McDonald's"),
    "NKE": ("耐吉", "Nike"),
    "SBUX": ("星巴克", "Starbucks"),
    "SPY": ("標普 500 ETF", "SPDR S&P 500 ETF"),
    "QQQ": ("那斯達克 100 ETF", "Invesco QQQ"),
    "DIA": ("道瓊 ETF", "SPDR Dow Jones ETF"),
    "IWM": ("羅素 2000 ETF", "iShares Russell 2000 ETF"),
    "VTI": ("美國全市場 ETF", "Vanguard Total Stock Market ETF"),
    "VOO": ("標普 500 ETF", "Vanguard S&P 500 ETF"),
    "XLK": ("科技類股 ETF", "Technology Select Sector SPDR"),
    "XLF": ("金融類股 ETF", "Financial Select Sector SPDR"),
    "XLE": ("能源類股 ETF", "Energy Select Sector SPDR"),
    "GLD": ("黃金 ETF", "SPDR Gold Shares"),
    "TLT": ("美國長期公債 ETF", "iShares 20+ Year Treasury Bond ETF"),
}

_QUOTE_ASSETS = ("USDT", "USDC", "BUSD", "BTC", "ETH")


def normalize_crypto_symbol(symbol: str) -> str:
    """將 BTCUSDT、BTC-USDT 等輸入統一成 BTC/USDT。"""
    compact = re.sub(r"[\s/_-]+", "", symbol).upper()
    for quote in _QUOTE_ASSETS:
        if compact.endswith(quote) and len(compact) > len(quote):
            return f"{compact[:-len(quote)]}/{quote}"
    raise ValueError(f"無法辨識加密貨幣交易對：{symbol}")


def normalize_us_equity_symbol(symbol: str) -> str:
    """統一美股代碼大小寫，保留類股代碼可能使用的點號或連字號。"""
    normalized = symbol.strip().upper()
    if not re.fullmatch(r"[A-Z0-9.-]{1,15}", normalized):
        raise ValueError(f"美股代碼格式不正確：{symbol}")
    return normalized


def infer_asset_class(exchange: str | None = None, symbol: str | None = None) -> AssetClass:
    """依資料來源優先判斷資產類別，必要時再由交易對格式判斷。"""
    source = (exchange or "").strip().lower()
    if source in {"binance", "binance_futures", "bybit"}:
        return "crypto"
    if source in {"yahoo", "yahoo_finance", "alpaca", "ibkr"}:
        return "us_equity"
    value = (symbol or "").strip().upper()
    return "crypto" if any(separator in value for separator in ("/", "-")) else "us_equity"


def normalize_symbol(symbol: str, asset_class: AssetClass) -> str:
    """依資產類別回傳研究系統使用的標準代碼。"""
    return (
        normalize_crypto_symbol(symbol)
        if asset_class == "crypto"
        else normalize_us_equity_symbol(symbol)
    )


def asset_definition(symbol: str, asset_class: AssetClass | None = None) -> AssetDefinition:
    """取得標準資產資料；未知代碼仍保留可讀的預設名稱。"""
    resolved_class = asset_class or infer_asset_class(symbol=symbol)
    normalized = normalize_symbol(symbol, resolved_class)
    if resolved_class == "crypto":
        base, quote = normalized.split("/", maxsplit=1)
        base_zh, base_en = _CRYPTO_NAMES.get(base, (base, base))
        quote_zh, quote_en = _CRYPTO_NAMES.get(quote, (quote, quote))
        return AssetDefinition(
            normalized,
            resolved_class,
            f"{base_zh}／{quote_zh}",
            f"{base_en} / {quote_en}",
        )
    chinese, english = _US_EQUITY_NAMES.get(normalized, (normalized, normalized))
    return AssetDefinition(normalized, resolved_class, chinese, english)


def asset_display_name(symbol: str, asset_class: AssetClass | None = None) -> str:
    """回傳 Dashboard 使用的代碼加中英文名稱。"""
    return asset_definition(symbol, asset_class).display_name


def asset_class_folder(asset_class: AssetClass) -> str:
    """回傳資料夾使用的穩定資產類別名稱。"""
    return "crypto" if asset_class == "crypto" else "us_equity"


def market_file_symbol_slug(symbol: str, asset_class: AssetClass) -> str:
    """回傳易讀且適合 Windows 檔名的資產代碼。"""
    normalized = normalize_symbol(symbol, asset_class)
    return normalized.replace("/", "-").replace(".", "-")
