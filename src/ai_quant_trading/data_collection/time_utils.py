"""資料收集用的時間轉換工具。"""

from __future__ import annotations

from datetime import UTC, datetime


def parse_datetime_to_utc_ms(value: str | datetime | None) -> int | None:
    """把日期或時間字串轉成 Binance API 使用的 UTC 毫秒時間戳。"""
    if value is None:
        return None

    if isinstance(value, datetime):
        dt = value
    else:
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    else:
        dt = dt.astimezone(UTC)

    return int(dt.timestamp() * 1000)


def utc_ms_to_iso(value: int) -> str:
    """把 UTC 毫秒時間戳轉成人類可讀且適合寫入 CSV 的 ISO 字串。"""
    return datetime.fromtimestamp(value / 1000, tz=UTC).isoformat(
        timespec="milliseconds"
    ).replace("+00:00", "Z")


def utc_now_iso() -> str:
    """回傳目前 UTC 時間，作為資料收集時間戳。"""
    return datetime.now(tz=UTC).isoformat(timespec="seconds").replace("+00:00", "Z")

