"""自動新聞收集、去重與市場分類測試。"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from ai_quant_trading.sentiment import (
    collect_market_news,
    load_news_history,
    parse_rss_news,
    save_news_history,
)


def _rss(now: datetime) -> bytes:
    published = now.strftime("%a, %d %b %Y %H:%M:%S GMT")
    return f"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel><item>
      <title>Bitcoin rises after market update</title>
      <description><![CDATA[<p>Institutional demand improved.</p>]]></description>
      <link>https://example.com/article</link>
      <pubDate>{published}</pubDate>
      <source>Example Desk</source>
    </item></channel></rss>""".encode()


def test_parse_rss_news_normalizes_html_and_metadata() -> None:
    frame = parse_rss_news(
        _rss(datetime.now(timezone.utc)),
        provider="CoinDesk",
        asset_class="crypto",
    )

    assert len(frame) == 1
    assert frame.loc[0, "asset_class"] == "crypto"
    assert frame.loc[0, "text"] == "Institutional demand improved."
    assert len(frame.loc[0, "news_id"]) == 64


def test_save_news_history_overwrites_and_deduplicates(tmp_path) -> None:
    frame = parse_rss_news(
        _rss(datetime.now(timezone.utc)),
        provider="CoinDesk",
        asset_class="crypto",
    )
    path = tmp_path / "news" / "news_latest.csv"

    save_news_history(frame, path)
    save_news_history(frame, path)
    restored = load_news_history(path)

    assert len(restored) == 1
    assert list(path.parent.glob("news_*.csv")) == [path]


def test_news_history_discards_expired_rows(tmp_path) -> None:
    frame = parse_rss_news(
        _rss(datetime(2020, 1, 1, tzinfo=timezone.utc)),
        provider="CoinDesk",
        asset_class="crypto",
    )
    path = tmp_path / "news_latest.csv"

    save_news_history(frame, path, retention_days=30)

    assert pd.read_csv(path).empty


def test_collect_market_news_keeps_only_requested_crypto_symbol() -> None:
    now = datetime.now(timezone.utc)
    bitcoin = parse_rss_news(
        _rss(now),
        provider="CoinDesk",
        asset_class="crypto",
    )
    unrelated = bitcoin.copy()
    unrelated["news_id"] = "unrelated"
    unrelated["title"] = "Solana ecosystem activity increases"
    unrelated["text"] = "Developers released a network update."
    unrelated["url"] = "https://example.com/solana"

    class FakeCoinDeskClient:
        def fetch(self) -> pd.DataFrame:
            return pd.concat([bitcoin, unrelated], ignore_index=True)

    result = collect_market_news(
        crypto_symbols=["BTC/USDT"],
        coindesk_client=FakeCoinDeskClient(),  # type: ignore[arg-type]
    )

    assert len(result.frame) == 1
    assert result.frame.loc[0, "symbol"] == "BTC/USDT"
    assert "Bitcoin" in result.frame.loc[0, "title"]
