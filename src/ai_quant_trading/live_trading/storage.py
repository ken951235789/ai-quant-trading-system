"""實盤交易的訊號、快照、訂單與本地持倉紀錄。"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import pandas as pd

from ai_quant_trading.database.config import DatabaseSettings
from ai_quant_trading.persistence import read_csv_snapshot
from ai_quant_trading.live_trading.audit import _acquire_lock

from ai_quant_trading.live_trading.config import TradingEnvironment
from ai_quant_trading.persistence import write_json_atomic


ORDER_COLUMNS = [
    "timestamp",
    "environment",
    "market_type",
    "symbol",
    "side",
    "position_side",
    "reduce_only",
    "quantity",
    "reference_price",
    "estimated_notional",
    "client_order_id",
    "validated_only",
    "recovered_after_unknown",
    "status",
    "order_id",
    "executed_quantity",
    "cumulative_quote_quantity",
    "stop_loss",
    "take_profit",
]

CYCLE_COLUMNS = [
    "timestamp",
    "environment",
    "market_type",
    "model_dir",
    "signal_time",
    "symbol",
    "target_fraction",
    "leverage",
    "required_leverage",
    "leverage_evidence_cap",
    "leverage_win_probability",
    "leverage_risk_reward",
    "leverage_net_expectancy",
    "leverage_reason",
    "policy_signal",
    "effective_signal",
    "reason",
    "execution_mode",
    "status",
    "client_order_id",
    "message",
]

SNAPSHOT_COLUMNS = [
    "timestamp",
    "environment",
    "market_type",
    "symbol",
    "base_asset",
    "quote_asset",
    "base_free",
    "base_locked",
    "quote_free",
    "quote_locked",
    "price",
    "estimated_equity",
    "can_trade",
    "wallet_balance",
    "available_balance",
    "margin_balance",
    "position_quantity",
    "entry_price",
    "unrealized_pnl",
    "liquidation_price",
    "leverage",
    "margin_type",
]

PROTECTION_COLUMNS = [
    "timestamp",
    "environment",
    "market_type",
    "symbol",
    "quantity",
    "take_profit_price",
    "stop_price",
    "list_client_order_id",
    "order_list_id",
    "list_status_type",
    "list_order_status",
    "recovered_after_unknown",
    "position_side",
    "stop_client_algo_id",
    "take_profit_client_algo_id",
    "stop_status",
    "take_profit_status",
    "message",
]

RL_CONTEXT_COLUMNS = [
    "timestamp",
    "symbol",
    "managed_equity",
    "cash_ratio",
    "position_ratio",
    "drawdown",
    "flow_adjusted_index",
    "net_external_flow",
]

CAPITAL_FLOW_COLUMNS = [
    "timestamp",
    "environment",
    "asset",
    "amount",
    "flow_type",
    "reference",
    "note",
]

_DATABASE_RECORD_TYPES = {
    "orders.csv": "orders",
    "cycles.csv": "cycles",
    "account_snapshots.csv": "account_snapshots",
    "rl_context.csv": "rl_context",
    "protections.csv": "protections",
    "capital_flows.csv": "capital_flows",
}


def _postgres_repository():
    """只在明確啟用 PostgreSQL 時載入 SQL 套件，file 模式保持輕量。"""
    settings = DatabaseSettings.from_environment()
    if not settings.enabled:
        return None
    from ai_quant_trading.database.repository import repository_from_settings

    return repository_from_settings(settings)


def _database_record_type(path: Path) -> str | None:
    return _DATABASE_RECORD_TYPES.get(path.name.lower())


@dataclass(frozen=True, slots=True)
class LiveTradingPaths:
    """單一 Binance 環境的持久化檔案。"""

    environment_dir: Path
    orders_csv: Path
    cycles_csv: Path
    account_snapshots_csv: Path
    rl_context_csv: Path
    protections_csv: Path
    capital_flows_csv: Path
    positions_json: Path
    notifications_log: Path
    emergency_halt_json: Path
    audit_jsonl: Path
    reports_dir: Path


@dataclass(frozen=True, slots=True)
class LivePosition:
    """由本系統建立並負責管理的現貨或永續合約部位。"""

    symbol: str
    quantity: float
    entry_price: float
    stop_loss: float
    take_profit: float | None
    entry_order_id: str
    opened_at: str
    updated_at: str
    protection_list_client_order_id: str | None = None
    protection_order_list_id: int | None = None
    protection_status: str = "none"
    market_type: str = "spot"
    side: str = "LONG"
    leverage: int = 1
    margin_type: str = "NONE"
    stop_client_algo_id: str | None = None
    take_profit_client_algo_id: str | None = None

    @property
    def signed_quantity(self) -> float:
        return -abs(self.quantity) if self.side.upper() == "SHORT" else abs(self.quantity)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LivePosition":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in payload.items() if key in allowed})


def live_trading_paths(
    root_dir: str | Path,
    environment: TradingEnvironment,
) -> LiveTradingPaths:
    """建立環境隔離的實盤紀錄路徑。"""
    if environment not in {"testnet", "demo", "live"}:
        raise ValueError("不支援的交易環境")
    environment_dir = Path(root_dir).resolve() / environment
    return LiveTradingPaths(
        environment_dir,
        environment_dir / "orders.csv",
        environment_dir / "cycles.csv",
        environment_dir / "account_snapshots.csv",
        environment_dir / "rl_context.csv",
        environment_dir / "protections.csv",
        environment_dir / "capital_flows.csv",
        environment_dir / "positions.json",
        environment_dir / "notifications.log",
        environment_dir / "emergency_halt.json",
        environment_dir / "audit.jsonl",
        environment_dir / "reports",
    )


def append_csv_row(path: Path, columns: list[str], row: dict[str, object]) -> None:
    """加鎖後依固定欄位追加 UTF-8 CSV；欄位升級時先安全遷移舊檔。"""
    repository = _postgres_repository()
    record_type = _database_record_type(path)
    if repository is not None and record_type is not None:
        repository.append_record(
            record_type,
            path.parent.name,
            {column: row.get(column) for column in columns},
        )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    descriptor = _acquire_lock(lock_path)
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        has_content = path.exists() and path.stat().st_size > 0
        if has_content:
            with path.open("r", newline="", encoding="utf-8-sig") as stream:
                existing_columns = next(csv.reader(stream), [])
            if existing_columns != columns:
                frame = pd.read_csv(path)
                for column in columns:
                    if column not in frame.columns:
                        frame[column] = None
                descriptor_tmp, temporary_name = tempfile.mkstemp(
                    dir=path.parent,
                    prefix=f".{path.name}.",
                    suffix=".tmp",
                    text=True,
                )
                os.close(descriptor_tmp)
                temporary = Path(temporary_name)
                try:
                    frame.reindex(columns=columns).to_csv(
                        temporary,
                        index=False,
                        encoding="utf-8",
                    )
                    for attempt in range(12):
                        try:
                            os.replace(temporary, path)
                            break
                        except PermissionError:
                            if attempt == 11:
                                raise
                            time.sleep(min(0.02 * (2**attempt), 0.25))
                finally:
                    temporary.unlink(missing_ok=True)
        has_content = path.exists() and path.stat().st_size > 0
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            if not has_content:
                writer.writeheader()
            writer.writerow({column: row.get(column) for column in columns})
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def read_live_csv(
    path: str | Path,
    columns: list[str],
    *,
    limit: int | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
) -> pd.DataFrame:
    """讀取實盤 CSV；尚無紀錄時回傳空表。"""
    if limit is not None and limit <= 0:
        raise ValueError("limit 必須大於 0")
    source = Path(path)
    repository = _postgres_repository()
    record_type = _database_record_type(source)
    if repository is not None and record_type is not None:
        records = repository.read_records(
            record_type,
            source.parent.name,
            limit=limit,
            since=since,
            until=until,
        )
        if not records:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(records).reindex(columns=columns)
    if not source.exists() or source.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    frame = read_csv_snapshot(source)
    if (since is not None or until is not None) and "timestamp" in frame:
        timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        if since is not None:
            frame = frame.loc[timestamps >= pd.Timestamp(since)]
            timestamps = timestamps.loc[frame.index]
        if until is not None:
            frame = frame.loc[timestamps < pd.Timestamp(until)]
    if limit is not None:
        frame = frame.tail(limit)
    return frame


def load_positions(paths: LiveTradingPaths) -> dict[str, LivePosition]:
    """載入由本系統管理的部位，不把帳戶其他資產誤認成策略部位。"""
    repository = _postgres_repository()
    if repository is not None:
        payload = repository.load_positions(paths.environment_dir.name)
        return {symbol: LivePosition.from_dict(dict(values)) for symbol, values in payload.items()}
    if not paths.positions_json.exists():
        return {}
    payload = json.loads(paths.positions_json.read_text(encoding="utf-8"))
    return {symbol: LivePosition.from_dict(dict(values)) for symbol, values in payload.items()}


def save_positions(paths: LiveTradingPaths, positions: dict[str, LivePosition]) -> None:
    """原子更新本地持倉 JSON。"""
    repository = _postgres_repository()
    if repository is not None:
        repository.save_positions(
            paths.environment_dir.name,
            {symbol: asdict(position) for symbol, position in positions.items()},
        )
        return
    write_json_atomic(
        paths.positions_json,
        {symbol: asdict(position) for symbol, position in positions.items()},
    )


def cycle_was_executed(
    paths: LiveTradingPaths,
    model_dir: str,
    symbol: str,
    signal_time: str,
) -> bool:
    """只把已要求實際送單的成功 cycle 視為完成。"""
    frame = read_live_csv(paths.cycles_csv, CYCLE_COLUMNS)
    if frame.empty:
        return False
    matches = (
        (frame["model_dir"].astype(str) == str(model_dir))
        & (frame["symbol"].astype(str) == symbol)
        & (frame["signal_time"].astype(str) == signal_time)
        & (frame["execution_mode"].astype(str) == "execute")
        & (frame["status"].astype(str).isin(["submitted", "hold", "no_position"]))
    )
    return bool(matches.any())


def activate_emergency_halt(paths: LiveTradingPaths, reason: str) -> None:
    """原子建立緊急停機狀態，阻止後續新增部位。"""
    previous = emergency_halt_reason(paths)
    write_json_atomic(
        paths.emergency_halt_json,
        {"active": True, "reason": reason},
    )
    if previous != reason:
        from ai_quant_trading.live_trading.audit import append_audit_event

        append_audit_event(
            paths.audit_jsonl,
            "emergency_halt_activated",
            {"reason": reason},
            severity="critical",
        )


def clear_emergency_halt(paths: LiveTradingPaths) -> None:
    """由明確的人工操作或演練清理程序解除停機。"""
    previous = emergency_halt_reason(paths)
    paths.emergency_halt_json.unlink(missing_ok=True)
    if previous is not None:
        from ai_quant_trading.live_trading.audit import append_audit_event

        append_audit_event(
            paths.audit_jsonl,
            "emergency_halt_cleared",
            {"previous_reason": previous},
            severity="warning",
        )


def emergency_halt_reason(paths: LiveTradingPaths) -> str | None:
    """回傳停機原因；檔案不存在時表示可繼續。"""
    if not paths.emergency_halt_json.exists():
        return None
    try:
        payload = json.loads(paths.emergency_halt_json.read_text(encoding="utf-8"))
        return str(payload.get("reason") or "緊急停機已啟用")
    except (OSError, ValueError, TypeError):
        return "緊急停機檔無法讀取"
