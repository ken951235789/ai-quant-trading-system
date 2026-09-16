"""記錄外部入出金，並計算不受資金移轉扭曲的回撤。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import pandas as pd


@dataclass(frozen=True, slots=True)
class FlowAdjustedEquity:
    """以外部現金流修正後的績效指數與回撤。"""

    index_value: float
    peak_index: float
    drawdown: float
    net_external_flow: float


def flow_adjusted_equity(
    snapshots: pd.DataFrame,
    flows: pd.DataFrame,
    *,
    current_timestamp: str | pd.Timestamp,
    current_equity: float,
) -> FlowAdjustedEquity:
    """用時間加權報酬鏈結各期，入金不算獲利、出金不算虧損。"""
    if not math.isfinite(float(current_equity)) or current_equity <= 0:
        raise ValueError("current_equity 必須是大於 0 的有限數值")
    current_time = pd.Timestamp(current_timestamp)
    current_time = (
        current_time.tz_localize("UTC")
        if current_time.tzinfo is None
        else current_time.tz_convert("UTC")
    )
    history = snapshots.copy()
    if not history.empty and {"timestamp", "managed_equity"}.issubset(history.columns):
        history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True, errors="coerce")
        history["managed_equity"] = pd.to_numeric(
            history["managed_equity"], errors="coerce"
        )
        history = history.dropna(subset=["timestamp", "managed_equity"])
        history = history.loc[history["managed_equity"] > 0]
        history = history.loc[history["timestamp"] < current_time]
        history = history.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    else:
        history = pd.DataFrame(columns=["timestamp", "managed_equity"])
    current_row = pd.DataFrame(
        [{"timestamp": current_time, "managed_equity": float(current_equity)}]
    )
    history = (
        current_row
        if history.empty
        else pd.concat(
            [history[["timestamp", "managed_equity"]], current_row],
            ignore_index=True,
        )
    )

    cash_flows = flows.copy()
    if not cash_flows.empty and {"timestamp", "amount"}.issubset(cash_flows.columns):
        cash_flows["timestamp"] = pd.to_datetime(
            cash_flows["timestamp"], utc=True, errors="coerce"
        )
        cash_flows["amount"] = pd.to_numeric(cash_flows["amount"], errors="coerce")
        cash_flows = cash_flows.dropna(subset=["timestamp", "amount"])
        cash_flows = cash_flows.loc[cash_flows["timestamp"] <= current_time]
    else:
        cash_flows = pd.DataFrame(columns=["timestamp", "amount"])

    index_value = 1.0
    peak_index = 1.0
    previous_time: pd.Timestamp | None = None
    previous_equity: float | None = None
    net_external_flow = 0.0
    for row in history.itertuples(index=False):
        timestamp = pd.Timestamp(row.timestamp)
        equity = float(row.managed_equity)
        if previous_time is None or previous_equity is None:
            previous_time = timestamp
            previous_equity = equity
            continue
        period_flows = cash_flows.loc[
            (cash_flows["timestamp"] > previous_time)
            & (cash_flows["timestamp"] <= timestamp),
            "amount",
        ]
        period_flow = float(period_flows.sum()) if not period_flows.empty else 0.0
        net_external_flow += period_flow
        period_return = (equity - period_flow) / previous_equity - 1.0
        if not math.isfinite(period_return) or period_return <= -1.0:
            index_value = 0.0
        else:
            index_value *= 1.0 + period_return
        peak_index = max(peak_index, index_value)
        previous_time = timestamp
        previous_equity = equity
    drawdown = index_value / peak_index - 1.0 if peak_index > 0 else -1.0
    return FlowAdjustedEquity(
        index_value=float(index_value),
        peak_index=float(peak_index),
        drawdown=float(drawdown),
        net_external_flow=float(net_external_flow),
    )


def validate_capital_flow(
    *,
    amount: float,
    flow_type: str,
    reference: str,
) -> None:
    """外部資金異動必須有方向、類型與可追查識別碼。"""
    if not math.isfinite(float(amount)) or abs(float(amount)) <= 1e-12:
        raise ValueError("資金異動金額不可為 0，且必須是有限數值")
    normalized = flow_type.strip().lower()
    if normalized not in {"deposit", "withdrawal"}:
        raise ValueError("flow_type 只支援 deposit 或 withdrawal")
    if normalized == "deposit" and amount < 0:
        raise ValueError("deposit 金額必須為正")
    if normalized == "withdrawal" and amount > 0:
        raise ValueError("withdrawal 金額必須為負")
    if not reference.strip():
        raise ValueError("資金異動必須提供不可重複的 reference")


def capital_flow_reference_exists(path: str | Path, reference: str) -> bool:
    """避免同一筆交易所轉帳被重複記錄。"""
    source = Path(path)
    if not source.exists() or source.stat().st_size == 0:
        return False
    frame = pd.read_csv(source, usecols=lambda column: column == "reference")
    return bool(frame.get("reference", pd.Series(dtype=str)).astype(str).eq(reference).any())


def record_capital_flow(
    storage_root: str | Path,
    environment: str,
    *,
    amount: float,
    flow_type: str,
    reference: str,
    asset: str = "USDT",
    note: str = "",
    timestamp: str | pd.Timestamp | None = None,
) -> Path:
    """保存人工核對過的交易所入出金；相同 reference 不可重複。"""
    from ai_quant_trading.live_trading.audit import append_audit_event
    from ai_quant_trading.live_trading.storage import (
        CAPITAL_FLOW_COLUMNS,
        append_csv_row,
        live_trading_paths,
        read_live_csv,
    )

    validate_capital_flow(amount=amount, flow_type=flow_type, reference=reference)
    if environment not in {"testnet", "demo", "live"}:
        raise ValueError("environment 只支援 testnet、demo 或 live")
    if asset.strip().upper() != "USDT":
        raise ValueError("目前 BTC USD-M 帳務只接受以 USDT 計價的外部資金異動")
    paths = live_trading_paths(storage_root, environment)  # type: ignore[arg-type]
    existing = read_live_csv(paths.capital_flows_csv, CAPITAL_FLOW_COLUMNS)
    duplicate = (
        not existing.empty
        and existing.get("reference", pd.Series(dtype=str)).astype(str).eq(reference).any()
    )
    if duplicate:
        raise ValueError(f"資金異動 reference 已存在：{reference}")
    occurred_at = pd.Timestamp(timestamp or pd.Timestamp.now(tz="UTC"))
    occurred_at = (
        occurred_at.tz_localize("UTC")
        if occurred_at.tzinfo is None
        else occurred_at.tz_convert("UTC")
    )
    append_csv_row(
        paths.capital_flows_csv,
        CAPITAL_FLOW_COLUMNS,
        {
            "timestamp": occurred_at.isoformat(),
            "environment": environment,
            "asset": asset.strip().upper(),
            "amount": float(amount),
            "flow_type": flow_type.strip().lower(),
            "reference": reference.strip(),
            "note": note.strip(),
        },
    )
    append_audit_event(
        paths.audit_jsonl,
        "capital_flow_recorded",
        {
            "environment": environment,
            "asset": asset.strip().upper(),
            "amount": float(amount),
            "flow_type": flow_type.strip().lower(),
            "reference": reference.strip(),
        },
        severity="warning",
    )
    return paths.capital_flows_csv
