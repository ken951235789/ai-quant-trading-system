"""Live 前的 Testnet 證據與操作安全檢查。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

from ai_quant_trading.live_trading.storage import (
    CYCLE_COLUMNS,
    ORDER_COLUMNS,
    SNAPSHOT_COLUMNS,
    emergency_halt_reason,
    live_trading_paths,
    load_positions,
    read_live_csv,
)
from ai_quant_trading.live_trading.audit import verify_audit_log
from ai_quant_trading.live_trading.reporting import unresolved_order_count


@dataclass(frozen=True, slots=True)
class TestnetEvidence:
    """Testnet 是否已累積足夠、跨越實際時間的成功輪次。"""

    eligible: bool
    successful_cycles: int
    observed_days: float
    minimum_cycles: int
    minimum_days: int
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OperationalEvidence:
    """Testnet／Live 是否具備可持續運行與事故追查條件。"""

    eligible: bool
    latest_snapshot_age_minutes: float | None
    unresolved_orders: int
    unprotected_positions: int
    audit_valid: bool
    audit_events: int
    reasons: tuple[str, ...]


def assess_testnet_evidence(
    storage_root: str | Path,
    *,
    model_dir: str | Path | None = None,
    symbol: str | None = None,
    minimum_cycles: int = 30,
    minimum_days: int = 7,
    market_type: str | None = None,
) -> TestnetEvidence:
    """只計算 Testnet 實際執行且安全完成的輪次。"""
    paths = live_trading_paths(storage_root, "testnet")
    cycles = read_live_csv(paths.cycles_csv, CYCLE_COLUMNS)
    if cycles.empty:
        successful = cycles
    else:
        successful = cycles.loc[
            cycles["execution_mode"].astype(str).eq("execute")
            & cycles["status"].astype(str).isin(
                ["submitted", "hold", "no_position"]
            )
        ].copy()
        if market_type is not None and "market_type" in successful:
            successful = successful.loc[
                successful["market_type"].astype(str).eq(market_type)
            ]
        if model_dir is not None:
            successful = successful.loc[
                successful["model_dir"].astype(str).eq(str(Path(model_dir)))
            ]
        if symbol is not None:
            successful = successful.loc[
                successful["symbol"].astype(str).eq(str(symbol))
            ]
    count = len(successful)
    observed_days = 0.0
    if count >= 2:
        times = pd.to_datetime(successful["timestamp"], utc=True, errors="coerce").dropna()
        if len(times) >= 2:
            observed_days = float((times.max() - times.min()).total_seconds() / 86_400)
    reasons: list[str] = []
    if count < minimum_cycles:
        reasons.append(f"Testnet 成功輪次 {count}，至少需要 {minimum_cycles} 次")
    if observed_days < minimum_days:
        reasons.append(
            f"Testnet 觀察期間 {observed_days:.1f} 天，至少需要 {minimum_days} 天"
        )
    halt = emergency_halt_reason(paths)
    if halt:
        reasons.append(f"Testnet 緊急停機尚未解除：{halt}")
    return TestnetEvidence(
        not reasons,
        count,
        observed_days,
        minimum_cycles,
        minimum_days,
        tuple(reasons),
    )


def assess_operational_evidence(
    storage_root: str | Path,
    environment: str = "testnet",
    *,
    maximum_snapshot_age_minutes: float = 30.0,
    now: datetime | pd.Timestamp | None = None,
    market_type: str | None = None,
) -> OperationalEvidence:
    """檢查快照時效、訂單對帳、保護單、停機狀態與稽核完整性。"""
    if maximum_snapshot_age_minutes <= 0:
        raise ValueError("maximum_snapshot_age_minutes 必須大於 0")
    paths = live_trading_paths(storage_root, environment)  # type: ignore[arg-type]
    reasons: list[str] = []
    snapshots = read_live_csv(paths.account_snapshots_csv, SNAPSHOT_COLUMNS)
    if market_type is not None and "market_type" in snapshots:
        snapshots = snapshots.loc[snapshots["market_type"].astype(str).eq(market_type)]
    age: float | None = None
    if snapshots.empty:
        reasons.append("尚無帳戶快照")
    else:
        timestamps = pd.to_datetime(snapshots["timestamp"], utc=True, errors="coerce").dropna()
        if timestamps.empty:
            reasons.append("帳戶快照時間無法解析")
        else:
            current = pd.Timestamp(now or datetime.now(timezone.utc))
            current = (
                current.tz_localize("UTC")
                if current.tzinfo is None
                else current.tz_convert("UTC")
            )
            age = max(float((current - timestamps.max()).total_seconds() / 60), 0.0)
            if age > maximum_snapshot_age_minutes:
                reasons.append(
                    f"最後帳戶快照已過 {age:.1f} 分鐘，上限為 "
                    f"{maximum_snapshot_age_minutes:.1f} 分鐘"
                )

    orders = read_live_csv(paths.orders_csv, ORDER_COLUMNS)
    if market_type is not None and "market_type" in orders:
        orders = orders.loc[orders["market_type"].astype(str).eq(market_type)]
    unresolved = unresolved_order_count(orders)
    if unresolved:
        reasons.append(f"有 {unresolved} 筆實際訂單狀態尚未確認")
    positions = load_positions(paths)
    if market_type is not None:
        positions = {
            key: position
            for key, position in positions.items()
            if position.market_type == market_type
        }
    unprotected = sum(
        position.protection_status not in {"active", "disabled"}
        for position in positions.values()
    )
    if unprotected:
        reasons.append(f"有 {unprotected} 個受管理部位尚未完成保護")
    halt = emergency_halt_reason(paths)
    if halt:
        reasons.append(f"緊急停機尚未解除：{halt}")
    audit = verify_audit_log(paths.audit_jsonl)
    if not audit.valid:
        reasons.extend(audit.reasons)
    if audit.event_count == 0:
        reasons.append("尚無交易稽核事件")
    return OperationalEvidence(
        eligible=not reasons,
        latest_snapshot_age_minutes=age,
        unresolved_orders=unresolved,
        unprotected_positions=unprotected,
        audit_valid=audit.valid,
        audit_events=audit.event_count,
        reasons=tuple(reasons),
    )


def ensure_live_operational_readiness(
    storage_root: str | Path,
    *,
    model_dir: str | Path,
    symbol: str,
    market_type: str | None = None,
) -> None:
    """Live 送單前強制通過 Testnet 運行證據檢查。"""
    evidence = assess_testnet_evidence(
        storage_root,
        model_dir=model_dir,
        symbol=symbol,
        market_type=market_type,
    )
    operations = assess_operational_evidence(
        storage_root,
        "testnet",
        market_type=market_type,
    )
    reasons = [*evidence.reasons, *operations.reasons]
    if reasons:
        raise ValueError("尚未達到 Live 操作門檻：" + "；".join(reasons))
