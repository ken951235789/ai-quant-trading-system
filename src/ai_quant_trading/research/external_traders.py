"""公開交易資料的唯讀下載與隔離驗證；不連接交易或訓練入口。"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import requests

from ai_quant_trading.data_collection.http_client import (
    NativeTruststoreAdapter,
    create_secure_ssl_context,
)
from ai_quant_trading.persistence import write_json_atomic

INFO = "https://api.hyperliquid.xyz/info"
LEADERBOARD = "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard"
DOC_ROOT = "https://hyperliquid.gitbook.io/hyperliquid-docs/"
DOCUMENTS = {
    "info": DOC_ROOT + "for-developers/api/info-endpoint.md",
    "perpetuals": DOC_ROOT + "for-developers/api/info-endpoint/perpetuals.md",
    "limits": DOC_ROOT + "for-developers/api/rate-limits-and-user-limits.md",
    "history": DOC_ROOT + "historical-data.md",
}
READ_TYPES = {
    "userFillsByTime",
    "userFunding",
    "userNonFundingLedgerUpdates",
    "clearinghouseState",
    "candleSnapshot",
}


def utc(ms=None):
    dt = (
        datetime.now(timezone.utc)
        if ms is None
        else datetime.fromtimestamp(ms / 1000, timezone.utc)
    )
    return dt.isoformat()


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def number(value):
    result = Decimal(str(value).strip())
    if not result.is_finite():
        raise ValueError("非有限數字")
    return result


class PublicClient:
    """只允許指定公開網址與唯讀類型；快取內容及請求均核對雜湊。"""

    def __init__(self, folder, interval=2.0, budget=180):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.interval, self.budget = interval, budget
        self.attempts, self.last = 0, 0.0
        self.session = requests.Session()
        # 不自動載入使用者的 .netrc 帳密，也不沿用瀏覽器登入狀態。
        self.session.trust_env = False
        self.session.mount(
            "https://", NativeTruststoreAdapter(create_secure_ssl_context(), max_retries=0)
        )
        self.session.headers["User-Agent"] = "AIQuantPublicResearch/1.0"

    def fetch(self, url, payload=None, limit=12 * 1024 * 1024):
        if url not in {INFO, LEADERBOARD, *DOCUMENTS.values()}:
            raise ValueError("非允許的公開來源")
        if url == INFO:
            if not isinstance(payload, dict) or payload.get("type") not in READ_TYPES:
                raise ValueError("非唯讀請求")
        elif payload is not None:
            raise ValueError("來源不接受 POST")
        for attempt in range(3):
            if self.attempts >= self.budget:
                raise RuntimeError("下載預算已用完")
            time.sleep(max(0, self.interval - (time.monotonic() - self.last)))
            self.last = time.monotonic()
            self.attempts += 1
            try:
                with self.session.request(
                    "POST" if payload else "GET",
                    url,
                    json=payload,
                    stream=True,
                    timeout=(10, 40),
                    allow_redirects=False,
                ) as response:
                    if response.status_code == 429 or response.status_code >= 500:
                        raise requests.ConnectionError("可重試的公開服務錯誤")
                    if response.status_code != 200:
                        raise ValueError(f"公開服務 HTTP {response.status_code}")
                    chunks, size, started = [], 0, time.monotonic()
                    for chunk in response.iter_content(65536):
                        size += len(chunk)
                        if size > limit or time.monotonic() - started > 120:
                            raise ValueError("公開回應超過大小或時間上限")
                        chunks.append(chunk)
                    body = b"".join(chunks)
                    meta = dict(
                        url=url,
                        received_at_utc=utc(),
                        bytes=len(body),
                        sha256=hashlib.sha256(body).hexdigest(),
                        server_date=response.headers.get("Date"),
                    )
                    return body, meta
            except (requests.Timeout, requests.ConnectionError):
                if attempt == 2:
                    raise RuntimeError("公開服務連線失敗，已達重試上限") from None
                time.sleep(2**attempt)
        raise RuntimeError("無回應")

    def info(self, payload):
        key = digest(payload)
        path = self.folder / f"{key}.json.gz"
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                entry = json.load(stream)
            if entry["request"] != payload or digest(entry["data"]) != entry["data_sha256"]:
                raise ValueError("快取驗證失敗")
            return entry["data"]
        body, meta = self.fetch(INFO, payload)
        data = json.loads(body)
        entry = dict(request=payload, data=data, data_sha256=digest(data), **meta)
        # x 模式防止覆寫；中斷檔會在下次讀取時失敗，不能悄悄改掉來源。
        with gzip.open(path, "xt", encoding="utf-8") as stream:
            json.dump(entry, stream, ensure_ascii=False, allow_nan=False)
        return data


def select_accounts(data, count=3):
    groups = {"positive": {}, "negative": {}}
    for row in data["leaderboardRows"]:
        address = str(row.get("ethAddress", "")).lower()
        if not re.fullmatch(r"0x[0-9a-f]{40}", address):
            continue
        windows = dict(row.get("windowPerformances", []))
        month = windows.get("month")
        if not month:
            continue
        try:
            pnl, volume = number(month["pnl"]), number(month["vlm"])
        except (KeyError, ValueError, ArithmeticError):
            continue
        if volume <= 0 or pnl == 0:
            continue
        group = "positive" if pnl > 0 else "negative"
        groups[group][address] = dict(address=address, stratum=group, reported_month_pnl=str(pnl))
    selected = []
    for group, rows in groups.items():
        ordered = sorted(
            rows, key=lambda a: hashlib.sha256(("external_traders_v1" + a).encode()).hexdigest()
        )
        for address in ordered[:count]:
            selected.append(dict(alias=f"public_{len(selected) + 1:02d}", **rows[address]))
    return selected, {group: len(rows) for group, rows in groups.items()}


def history(client, kind, address, start, end, max_pages=8):
    """含首尾時間分頁；重複邊界去重，無法推進時標示而不是跳過毫秒。"""
    cursor, seen, warnings = start, {}, []
    cap = 2000 if kind == "userFillsByTime" else 500
    for page in range(max_pages):
        rows = client.info(dict(type=kind, user=address, startTime=cursor, endTime=end))
        if not isinstance(rows, list):
            raise ValueError("歷史回應不是列表")
        for row in rows:
            timestamp = int(row["time"])
            if not cursor <= timestamp <= end:
                raise ValueError("歷史回應超出請求時間")
            key = (
                (row["coin"], str(row["tid"]), str(row["oid"]))
                if kind == "userFillsByTime"
                else (
                    timestamp,
                    row["hash"],
                    row.get("delta", {}).get("coin"),
                    row.get("delta", {}).get("type"),
                )
            )
            if key in seen and digest(seen[key]) != digest(row):
                raise ValueError("同一紀錄出現衝突內容")
            seen[key] = row
        if not rows or len(rows) < cap:
            break
        latest = max(int(row["time"]) for row in rows)
        if latest <= cursor:
            warnings.append("timestamp_boundary_unresolved")
            break
        cursor = latest
    else:
        warnings.append("page_limit")
    if kind == "userFillsByTime" and len(seen) >= 10000:
        warnings.append("recent_10000_limit")
    return sorted(seen.values(), key=lambda x: int(x["time"])), dict(
        pages=page + 1,
        rows=len(seen),
        warnings=warnings,
        lifetime_complete=False,
        requested_window_complete="unverified",
    )


def normalize_fills(rows, alias):
    result = []
    for row in rows:
        if row["side"] not in {"B", "A"}:
            raise ValueError("未知買賣方向")
        px, size, start = (number(row[k]) for k in ("px", "sz", "startPosition"))
        fee, pnl = number(row["fee"]), number(row["closedPnl"])
        if px <= 0 or size <= 0:
            raise ValueError("非正成交價或數量")
        result.append(
            dict(
                alias=alias,
                venue="Hyperliquid",
                coin=row["coin"],
                time_ms=int(row["time"]),
                timestamp_utc=utc(int(row["time"])),
                execution_side=row["side"],
                direction=row["dir"],
                price=str(px),
                size=str(size),
                start_position=str(start),
                end_position=str(start + (size if row["side"] == "B" else -size)),
                closed_pnl=str(pnl),
                fee=str(fee),
                fee_token=row["feeToken"].strip(),
                builder_fee_included=str(number(row["builderFee"]))
                if "builderFee" in row
                else None,
                crossed=row["crossed"],
                order_id=str(row["oid"]),
                trade_id=str(row["tid"]),
                training_eligible=False,
            )
        )
    return result


def position_checks(rows):
    ordered = sorted(rows, key=lambda r: r["time_ms"])
    gaps = ambiguous = 0
    for prev, row in zip(ordered, ordered[1:]):
        if prev["time_ms"] == row["time_ms"]:
            ambiguous += 1
        elif number(prev["end_position"]) != number(row["start_position"]):
            gaps += 1
    return dict(
        position_gaps=gaps,
        simultaneous_fill_pairs=ambiguous,
        left_open=bool(ordered and number(ordered[0]["start_position"]) != 0),
        right_open=bool(ordered and number(ordered[-1]["end_position"]) != 0),
    )


def normalize_funding(rows, alias):
    return [
        dict(
            alias=alias,
            time_ms=int(r["time"]),
            timestamp_utc=utc(int(r["time"])),
            coin=r["delta"]["coin"],
            signed_cashflow_usdc=str(number(r["delta"]["usdc"])),
            funding_rate=str(number(r["delta"]["fundingRate"])),
            position_size=str(number(r["delta"]["szi"])),
        )
        for r in rows
    ]


def normalize_candles(rows, start, end):
    result, seen = [], set()
    for row in sorted(rows, key=lambda r: int(r["t"])):
        t, close = int(row["t"]), int(row["T"])
        if t < start or close >= end:
            continue
        if row["s"] != "BTC" or row["i"] != "15m" or close != t + 900000 - 1 or t % 900000:
            raise ValueError("K線契約不符")
        if t in seen:
            raise ValueError("K線時間重複")
        seen.add(t)
        o, h, low, c, v = (number(row[k]) for k in ("o", "h", "l", "c", "v"))
        if not (0 < low <= min(o, c) <= max(o, c) <= h) or v < 0:
            raise ValueError("K線價格或成交量不合法")
        result.append(
            dict(
                open_time_utc=utc(t),
                close_time_utc=utc(close),
                open=str(o),
                high=str(h),
                low=str(low),
                close=str(c),
                volume=str(v),
                time_ms=t,
            )
        )
    expected = (end - start) // 900000
    return result, dict(
        expected_bars=expected,
        received_bars=len(result),
        missing_bars=expected - len(result),
        closed_only=True,
    )


def write_csv(path, rows, columns=None):
    """外部文字以純文字保存，避免試算表公式注入；數值字串不加前綴。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not columns:
        return

    def safe(value):
        if isinstance(value, str) and value and value[:1] in "=+-@":
            try:
                number(value)
            except (ValueError, ArithmeticError):
                return "'" + value
        return value

    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns or list(rows[0]))
        writer.writeheader()
        writer.writerows({k: safe(v) for k, v in r.items()} for r in rows)


def save_progress(folder, stage, **extra):
    write_json_atomic(
        Path(folder) / "progress.json", dict(stage=stage, updated_at_utc=utc(), **extra)
    )
