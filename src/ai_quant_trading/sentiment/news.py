"""從固定公開來源收集新聞，並以單一 CSV 合併去重。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from hashlib import sha256
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
import re
from uuid import uuid4

from defusedxml import ElementTree
from defusedxml.ElementTree import ParseError
import pandas as pd
import requests

from ai_quant_trading.data_collection.assets import normalize_us_equity_symbol
from ai_quant_trading.data_collection.http_client import create_secure_session, require_session


COINDESK_RSS_URL = "https://www.coindesk.com/arc/outboundfeeds/rss/"
YAHOO_FINANCE_SEARCH_URL = "https://query1.finance.yahoo.com/v1/finance/search"
NEWS_COLUMNS = [
    "news_id",
    "published_at",
    "symbol",
    "asset_class",
    "provider",
    "source",
    "title",
    "text",
    "url",
    "collected_at",
]
FINBERT_SCORE_COLUMNS = [
    "finbert_positive",
    "finbert_negative",
    "finbert_neutral",
    "finbert_sentiment",
    "finbert_confidence",
    "finbert_confident",
]
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 AIQuantTradingSystem/0.1 (local research application)"
}
MAX_RESPONSE_BYTES = 5 * 1024 * 1024
CRYPTO_NEWS_ALIASES = {
    "BTC": ("BTC", "BITCOIN"),
    "ETH": ("ETH", "ETHEREUM"),
    "BNB": ("BNB", "BINANCE COIN"),
    "DOGE": ("DOGE", "DOGECOIN"),
    "SOL": ("SOL", "SOLANA"),
    "XRP": ("XRP", "RIPPLE"),
}


class NewsCollectionError(RuntimeError):
    """新聞下載或解析失敗。"""


@dataclass(frozen=True, slots=True)
class NewsCollectionResult:
    """一次新聞更新的資料與個別來源錯誤。"""

    frame: pd.DataFrame
    errors: tuple[str, ...] = ()


class _PlainTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        normalized = " ".join(data.split())
        if normalized:
            self.parts.append(normalized)


def _plain_text(value: object, *, max_length: int) -> str:
    parser = _PlainTextParser()
    parser.feed(unescape(str(value or "")))
    parser.close()
    return " ".join(parser.parts)[:max_length].strip()


def _published_at(value: object) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError("新聞缺少發布時間")
    try:
        parsed = parsedate_to_datetime(text)
    except (TypeError, ValueError, OverflowError):
        parsed = pd.Timestamp(text).to_pydatetime()
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _stable_news_id(source: str, url: str, title: str, published_at: str) -> str:
    identity = "\n".join([source, url, title, published_at]).encode("utf-8")
    return sha256(identity).hexdigest()


def _empty_news_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=NEWS_COLUMNS)


def _filter_crypto_news(
    frame: pd.DataFrame,
    crypto_symbols: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """只保留文字明確提到指定幣種的新聞，避免混入無關市場內容。"""
    if frame.empty:
        return frame.copy()

    requested: dict[str, str] = {}
    for symbol in crypto_symbols:
        normalized = str(symbol).strip().upper()
        base_asset = normalized.split("/", maxsplit=1)[0]
        if base_asset:
            requested.setdefault(base_asset, normalized)

    rows: list[pd.Series] = []
    for _, row in frame.iterrows():
        searchable = " ".join(
            str(row.get(column, "")) for column in ("title", "text", "url")
        ).upper()
        for base_asset, symbol in requested.items():
            aliases = CRYPTO_NEWS_ALIASES.get(base_asset, (base_asset,))
            if any(
                re.search(rf"(?<![A-Z0-9]){re.escape(alias)}(?![A-Z0-9])", searchable)
                for alias in aliases
            ):
                matched = row.copy()
                matched["symbol"] = symbol
                rows.append(matched)
                break
    if not rows:
        return _empty_news_frame()
    return pd.DataFrame(rows, columns=NEWS_COLUMNS).reset_index(drop=True)


def parse_rss_news(
    content: bytes,
    *,
    provider: str,
    asset_class: str,
    symbol: str = "",
    limit: int = 100,
) -> pd.DataFrame:
    """解析受信任 RSS 回應，輸出統一新聞欄位。"""
    if not content:
        raise NewsCollectionError(f"{provider} RSS 沒有內容")
    if len(content) > MAX_RESPONSE_BYTES:
        raise NewsCollectionError(f"{provider} RSS 超過允許大小")
    try:
        root = ElementTree.fromstring(content)
    except ParseError as exc:
        raise NewsCollectionError(f"{provider} RSS 格式錯誤：{exc}") from exc

    collected_at = _iso_utc(datetime.now(timezone.utc))
    rows: list[dict[str, object]] = []
    for item in root.findall(".//item")[:limit]:
        title = _plain_text(item.findtext("title"), max_length=1000)
        description = _plain_text(item.findtext("description"), max_length=10_000)
        url = str(item.findtext("link") or "").strip()[:2048]
        source = _plain_text(item.findtext("source") or provider, max_length=200)
        try:
            published = _iso_utc(_published_at(item.findtext("pubDate")))
        except (ValueError, TypeError, OverflowError):
            continue
        if not title:
            continue
        rows.append(
            {
                "news_id": _stable_news_id(provider, url, title, published),
                "published_at": published,
                "symbol": symbol,
                "asset_class": asset_class,
                "provider": provider,
                "source": source or provider,
                "title": title,
                "text": description,
                "url": url,
                "collected_at": collected_at,
            }
        )
    return pd.DataFrame(rows, columns=NEWS_COLUMNS)


@dataclass(slots=True)
class CoinDeskNewsClient:
    """使用 CoinDesk 官方 RSS 收集加密貨幣市場新聞。"""

    session: requests.Session | None = None
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.session is None:
            self.session = create_secure_session()

    def fetch(self, limit: int = 100) -> pd.DataFrame:
        session = require_session(self.session)
        try:
            response = session.get(
                COINDESK_RSS_URL,
                headers=REQUEST_HEADERS,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise NewsCollectionError(f"CoinDesk 新聞下載失敗：{exc}") from exc
        return parse_rss_news(
            response.content,
            provider="CoinDesk",
            asset_class="crypto",
            limit=limit,
        )


@dataclass(slots=True)
class YahooFinanceNewsClient:
    """使用 Yahoo Finance 搜尋端點收集單一美股的相關新聞。"""

    session: requests.Session | None = None
    timeout_seconds: float = 20.0

    def __post_init__(self) -> None:
        if self.session is None:
            self.session = create_secure_session()

    def fetch(self, symbol: str, limit: int = 10) -> pd.DataFrame:
        normalized_symbol = normalize_us_equity_symbol(symbol)
        session = require_session(self.session)
        try:
            response = session.get(
                YAHOO_FINANCE_SEARCH_URL,
                params={
                    "q": normalized_symbol,
                    "quotesCount": 1,
                    "newsCount": max(1, min(int(limit), 50)),
                },
                headers=REQUEST_HEADERS,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise NewsCollectionError(
                f"Yahoo Finance {normalized_symbol} 新聞下載失敗：{exc}"
            ) from exc

        collected_at = _iso_utc(datetime.now(timezone.utc))
        rows: list[dict[str, object]] = []
        for item in list(payload.get("news") or [])[:limit]:
            if not isinstance(item, dict):
                continue
            related = {str(value).upper() for value in item.get("relatedTickers") or []}
            if related and normalized_symbol not in related:
                continue
            title = _plain_text(item.get("title"), max_length=1000)
            if not title or item.get("providerPublishTime") is None:
                continue
            try:
                published_dt = datetime.fromtimestamp(
                    int(item["providerPublishTime"]),
                    tz=timezone.utc,
                )
            except (TypeError, ValueError, OSError, OverflowError):
                continue
            published = _iso_utc(published_dt)
            canonical = item.get("canonicalUrl") or {}
            canonical_url = canonical.get("url") if isinstance(canonical, dict) else ""
            url = str(item.get("link") or canonical_url or "")
            url = url.strip()[:2048]
            source = _plain_text(item.get("publisher") or "Yahoo Finance", max_length=200)
            rows.append(
                {
                    "news_id": _stable_news_id(
                        "Yahoo Finance",
                        url,
                        title,
                        published,
                    ),
                    "published_at": published,
                    "symbol": normalized_symbol,
                    "asset_class": "us_equity",
                    "provider": "Yahoo Finance",
                    "source": source or "Yahoo Finance",
                    "title": title,
                    "text": "",
                    "url": url,
                    "collected_at": collected_at,
                }
            )
        return pd.DataFrame(rows, columns=NEWS_COLUMNS)


def collect_market_news(
    *,
    crypto_symbols: list[str] | tuple[str, ...] = (),
    equity_symbols: list[str] | tuple[str, ...] = (),
    coindesk_client: CoinDeskNewsClient | None = None,
    yahoo_client: YahooFinanceNewsClient | None = None,
    yahoo_limit_per_symbol: int = 10,
) -> NewsCollectionResult:
    """依本機追蹤市場收集新聞；單一來源失敗不取消其他結果。"""
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    if crypto_symbols:
        try:
            crypto_news = (coindesk_client or CoinDeskNewsClient()).fetch()
            frames.append(_filter_crypto_news(crypto_news, crypto_symbols))
        except NewsCollectionError as exc:
            errors.append(str(exc))
    equity_client = yahoo_client or YahooFinanceNewsClient()
    for symbol in dict.fromkeys(equity_symbols):
        try:
            frames.append(equity_client.fetch(symbol, yahoo_limit_per_symbol))
        except NewsCollectionError as exc:
            errors.append(str(exc))
    usable = [frame for frame in frames if not frame.empty]
    frame = (
        pd.concat(usable, ignore_index=True).drop_duplicates("news_id", keep="last")
        if usable
        else _empty_news_frame()
    )
    return NewsCollectionResult(frame.reset_index(drop=True), tuple(errors))


def news_history_path(raw_dir: str | Path) -> Path:
    """回傳自動新聞的固定 Raw CSV 路徑。"""
    return Path(raw_dir) / "news" / "news_latest.csv"


def load_news_history(path: str | Path) -> pd.DataFrame:
    """讀取新聞歷史；檔案不存在時回傳空表。"""
    source = Path(path)
    if not source.exists():
        return _empty_news_frame()
    try:
        frame = pd.read_csv(source)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise NewsCollectionError(f"新聞 CSV 無法讀取：{source.name}") from exc
    for column in NEWS_COLUMNS:
        if column not in frame:
            frame[column] = ""
    return frame[NEWS_COLUMNS]


def save_news_history(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    retention_days: int = 3650,
    max_rows: int = 100_000,
) -> Path:
    """合併新聞並覆寫固定檔案，避免每次啟動產生新 CSV。"""
    if retention_days <= 0 or max_rows <= 0:
        raise ValueError("新聞保留天數與最大筆數必須大於 0")
    target = Path(path)
    existing = load_news_history(target)
    incoming = frame.copy()
    for column in NEWS_COLUMNS:
        if column not in incoming:
            incoming[column] = ""
    merged = pd.concat([existing, incoming[NEWS_COLUMNS]], ignore_index=True)
    merged["published_at"] = pd.to_datetime(
        merged["published_at"], utc=True, errors="coerce"
    )
    merged = merged.dropna(subset=["published_at"])
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=retention_days)
    merged = merged[merged["published_at"] >= cutoff]
    merged = merged.drop_duplicates("news_id", keep="last").sort_values("published_at")
    merged = merged.tail(max_rows).reset_index(drop=True)
    merged["published_at"] = merged["published_at"].map(
        lambda value: pd.Timestamp(value).isoformat().replace("+00:00", "Z")
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.{uuid4().hex}.tmp")
    try:
        merged[NEWS_COLUMNS].to_csv(temporary, index=False, encoding="utf-8")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def load_scored_news(path: str | Path) -> pd.DataFrame:
    """讀取已評分新聞；欄位不完整時忽略該檔案。"""
    source = Path(path)
    columns = [*NEWS_COLUMNS, *FINBERT_SCORE_COLUMNS]
    if not source.exists():
        return pd.DataFrame(columns=columns)
    try:
        frame = pd.read_csv(source)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise NewsCollectionError(f"FinBERT 新聞 CSV 無法讀取：{source.name}") from exc
    if any(column not in frame for column in columns):
        return pd.DataFrame(columns=columns)
    return frame[columns]


def save_scored_news(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    retention_days: int = 3650,
    max_rows: int = 100_000,
) -> Path:
    """合併 FinBERT 結果並覆寫固定 CSV。"""
    if retention_days <= 0 or max_rows <= 0:
        raise ValueError("新聞保留天數與最大筆數必須大於 0")
    target = Path(path)
    columns = [*NEWS_COLUMNS, *FINBERT_SCORE_COLUMNS]
    existing = load_scored_news(target)
    incoming = frame.copy()
    missing = [column for column in columns if column not in incoming]
    if missing:
        raise ValueError(f"FinBERT 新聞缺少欄位：{missing}")
    merged = pd.concat([existing, incoming[columns]], ignore_index=True)
    merged["published_at"] = pd.to_datetime(
        merged["published_at"], utc=True, errors="coerce"
    )
    merged = merged.dropna(subset=["published_at"])
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=retention_days)
    merged = merged[merged["published_at"] >= cutoff]
    merged = merged.drop_duplicates("news_id", keep="last").sort_values("published_at")
    merged = merged.tail(max_rows).reset_index(drop=True)
    merged["published_at"] = merged["published_at"].map(
        lambda value: pd.Timestamp(value).isoformat().replace("+00:00", "Z")
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.{uuid4().hex}.tmp")
    try:
        merged[columns].to_csv(temporary, index=False, encoding="utf-8")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def recent_news(frame: pd.DataFrame, *, days: int = 7) -> pd.DataFrame:
    """保留最近數日新聞，供介面快速預覽。"""
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    published = pd.to_datetime(result["published_at"], utc=True, errors="coerce")
    cutoff = pd.Timestamp.now(tz="UTC") - timedelta(days=days)
    return result[published >= cutoff].reset_index(drop=True)
