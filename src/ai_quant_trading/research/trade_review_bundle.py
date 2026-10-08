"""桌面版的唯讀逐筆研究資料包，不含模型權重、密鑰或交易帳戶。"""

import gzip
import json
from pathlib import Path

import numpy as np

from ai_quant_trading.operations.integrity import build_artifact_manifest, verify_artifact_manifest

MAX_BYTES = 64 * 1024 * 1024
FILENAME = "review.json.gz"


def validate_payload(payload):
    """拒絕過大、越界或非有限資料，避免圖表錯置或載入不可信程式。"""
    bars, rows, schedules = payload["bars"], payload["rows"], payload["schedules"]
    if not 0 < len(bars) <= 200000 or len(rows) > 30000 or len(schedules) > 3000:
        raise ValueError("逐筆資料包大小超出限制")
    values = np.asarray(bars, dtype=float)
    if values.shape != (len(bars), 5) or not np.isfinite(values).all():
        raise ValueError("K 棒格式不正確")
    if (np.diff(values[:, 0]) <= 0).any() or (values[:, 1:] <= 0).any():
        raise ValueError("K 棒時間或價格不合法")
    offset = payload["offset"]
    if not isinstance(offset, int) or offset < 0:
        raise ValueError("K 棒索引不合法")
    for schedule in schedules:
        for trade in schedule:
            if not offset <= trade["entry"] <= trade["exit"] < offset + len(bars):
                raise ValueError("成交不在 K 棒範圍內")
            if trade["side"] not in (-1, 1):
                raise ValueError("方向不合法")
            keys = ("entry_price", "exit_price", "stop", "target", "planned_rr", "net_r",
                    "net_return", "price_return", "risk_fraction", "gross", "spread", "slippage", "fee", "funding")
            if not np.isfinite([trade[k] for k in keys]).all():
                raise ValueError("交易數值不是有限值")
    for row in rows:
        if row["family"] not in {"original", "trend_pullback", "vwap_reversion"}:
            raise ValueError("未知策略")
        if row["policy"] not in {"no_ai", "transformer"}:
            raise ValueError("未知篩選規則")
        if not isinstance(row["schedule"], int) or not 0 <= row["schedule"] < len(schedules):
            raise ValueError("排程索引不合法")
        if row["trades"] != len(schedules[row["schedule"]]):
            raise ValueError("成交數與排程不符")
        if row["contract_sha256"] not in payload["contracts"]:
            raise ValueError("缺少成交契約")
    return payload


def export_review_bundle(payload, target: Path, *, source_sha256: str, label: str):
    """另建小型可攜包，保留原始回測與完整資料來源。"""
    validate_payload(payload)
    if target.exists():
        raise FileExistsError("不覆寫既有逐筆研究包")
    target.mkdir(parents=True)
    body = {"schema_version": 1, "label": label, "research_only": True,
            "live_eligible": False, "source_sha256": source_sha256, "payload": payload}
    raw = json.dumps(body, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(raw) > MAX_BYTES:
        raise ValueError("資料包超出解壓大小限制")
    (target / FILENAME).write_bytes(gzip.compress(raw, compresslevel=6, mtime=0))
    build_artifact_manifest(target, files=[target / FILENAME])
    return target


def load_review_bundle(root: Path):
    """先驗雜湊再限量解壓，只解析 JSON，不載入 checkpoint 或 pickle。"""
    if (root / FILENAME).stat().st_size > MAX_BYTES:
        raise ValueError("壓縮資料包過大")
    if FILENAME not in verify_artifact_manifest(root):
        raise ValueError("研究包缺少完整性紀錄")
    with gzip.open(root / FILENAME, "rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("資料包解壓大小超限")
    body = json.loads(raw)
    if body["schema_version"] != 1 or body["research_only"] is not True or body["live_eligible"] is not False:
        raise ValueError("不支援的研究資料包")
    validate_payload(body["payload"])
    return body
