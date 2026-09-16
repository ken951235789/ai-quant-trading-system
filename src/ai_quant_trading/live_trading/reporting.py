"""交易營運的每日報告與未解決狀態統計。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pandas as pd

from ai_quant_trading.live_trading.audit import verify_audit_log
from ai_quant_trading.live_trading.storage import (
    CYCLE_COLUMNS,
    ORDER_COLUMNS,
    SNAPSHOT_COLUMNS,
    LiveTradingPaths,
    emergency_halt_reason,
    load_positions,
    read_live_csv,
)
from ai_quant_trading.persistence import write_json_atomic


def _boolean_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].astype(str).str.lower().isin({"true", "1", "yes"})


def unresolved_order_count(orders: pd.DataFrame) -> int:
    """相同 clientOrderId 只看最後狀態，避免已對帳訂單永久卡住驗收。"""
    if orders.empty:
        return 0
    frame = orders.copy()
    if "client_order_id" not in frame or "status" not in frame:
        return 0
    frame["timestamp"] = pd.to_datetime(frame.get("timestamp"), utc=True, errors="coerce")
    frame = frame.sort_values("timestamp").drop_duplicates("client_order_id", keep="last")
    executed = ~_boolean_series(frame, "validated_only")
    unresolved = (
        frame["status"]
        .fillna("")
        .astype(str)
        .str.upper()
        .isin({"", "UNKNOWN", "PENDING", "PENDING_NEW"})
    )
    return int((executed & unresolved).sum())


@dataclass(frozen=True, slots=True)
class DailyTradingReport:
    """可供 UI、JSON 與 Markdown 共用的每日營運摘要。"""

    report_date: str
    environment: str
    status: str
    opening_equity: float | None
    closing_equity: float | None
    pnl: float
    return_rate: float
    maximum_drawdown: float
    decision_cycles: int
    submitted_orders: int
    validation_orders: int
    unresolved_orders: int
    active_positions: int
    unprotected_positions: int
    emergency_halt: str | None
    audit_valid: bool
    audit_events: int
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        return payload


def build_daily_trading_report(
    paths: LiveTradingPaths,
    *,
    report_date: date | None = None,
) -> DailyTradingReport:
    """只使用已落盤資料產生報告，因此不需要交易所 API。"""
    selected_day = report_date or datetime.now(timezone.utc).date()
    day_start = datetime.combine(selected_day, time.min, tzinfo=timezone.utc)
    day_end = day_start + timedelta(days=1)
    snapshots = read_live_csv(
        paths.account_snapshots_csv,
        SNAPSHOT_COLUMNS,
        since=day_start,
        until=day_end,
    )
    cycles = read_live_csv(
        paths.cycles_csv,
        CYCLE_COLUMNS,
        since=day_start,
        until=day_end,
    )
    orders = read_live_csv(
        paths.orders_csv,
        ORDER_COLUMNS,
        since=day_start,
        until=day_end,
    )

    opening: float | None = None
    closing: float | None = None
    pnl = 0.0
    return_rate = 0.0
    maximum_drawdown = 0.0
    if not snapshots.empty:
        equity = pd.to_numeric(snapshots["estimated_equity"], errors="coerce").dropna()
        equity = equity.loc[equity > 0]
        if not equity.empty:
            opening = float(equity.iloc[0])
            closing = float(equity.iloc[-1])
            pnl = closing - opening
            return_rate = pnl / opening if opening > 0 else 0.0
            peaks = equity.cummax()
            maximum_drawdown = float((equity / peaks - 1).min())

    submitted = 0
    validated = 0
    if not orders.empty:
        validation_mask = _boolean_series(orders, "validated_only")
        validated = int(validation_mask.sum())
        submitted = int((~validation_mask).sum())

    positions = load_positions(paths)
    unprotected = sum(
        position.protection_status not in {"active", "disabled"} for position in positions.values()
    )
    audit = verify_audit_log(paths.audit_jsonl)
    halt = emergency_halt_reason(paths)
    unresolved = unresolved_order_count(read_live_csv(paths.orders_csv, ORDER_COLUMNS))
    reasons: list[str] = []
    if halt:
        reasons.append(f"緊急停機：{halt}")
    if not audit.valid:
        reasons.extend(audit.reasons)
    if unresolved:
        reasons.append(f"有 {unresolved} 筆實際訂單狀態尚未確認")
    if unprotected:
        reasons.append(f"有 {unprotected} 個受管理部位尚未完成交易所保護")
    if snapshots.empty:
        reasons.append("當日尚無帳戶快照")
    return DailyTradingReport(
        report_date=selected_day.isoformat(),
        environment=paths.environment_dir.name,
        status="attention" if reasons else "normal",
        opening_equity=opening,
        closing_equity=closing,
        pnl=pnl,
        return_rate=return_rate,
        maximum_drawdown=maximum_drawdown,
        decision_cycles=len(cycles),
        submitted_orders=submitted,
        validation_orders=validated,
        unresolved_orders=unresolved,
        active_positions=len(positions),
        unprotected_positions=unprotected,
        emergency_halt=halt,
        audit_valid=audit.valid,
        audit_events=audit.event_count,
        reasons=tuple(reasons),
    )


def _markdown(report: DailyTradingReport) -> str:
    reasons = "\n".join(f"- {reason}" for reason in report.reasons) or "- 無"
    opening = "-" if report.opening_equity is None else f"{report.opening_equity:,.2f}"
    closing = "-" if report.closing_equity is None else f"{report.closing_equity:,.2f}"
    return f"""# 每日交易營運報告

- 日期（UTC）：{report.report_date}
- 環境：{report.environment}
- 狀態：{report.status}
- 期初資產：{opening}
- 期末資產：{closing}
- 損益：{report.pnl:,.2f}（{report.return_rate:.2%}）
- 最大回撤：{report.maximum_drawdown:.2%}
- 決策輪次：{report.decision_cycles}
- 實際訂單：{report.submitted_orders}
- 驗證訂單：{report.validation_orders}
- 未確認訂單：{report.unresolved_orders}
- 受管理部位：{report.active_positions}
- 未完成保護部位：{report.unprotected_positions}
- 稽核鏈：{"正常" if report.audit_valid else "異常"}（{report.audit_events} 筆）

## 注意事項

{reasons}
"""


def save_daily_trading_report(
    paths: LiveTradingPaths,
    *,
    report_date: date | None = None,
) -> tuple[Path, Path, DailyTradingReport]:
    """保存 JSON 與中文 Markdown，重複產生同一天會覆蓋舊報告。"""
    report = build_daily_trading_report(paths, report_date=report_date)
    json_path = paths.reports_dir / f"{report.report_date}.json"
    markdown_path = paths.reports_dir / f"{report.report_date}.md"
    write_json_atomic(json_path, report.to_dict())
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    return json_path, markdown_path, report
