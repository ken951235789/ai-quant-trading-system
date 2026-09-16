"""模擬帳戶 JSON 與 CSV 紀錄儲存。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
from typing import Iterable

import pandas as pd

from ai_quant_trading.persistence import read_csv_snapshot

from ai_quant_trading.persistence import write_json_atomic

from ai_quant_trading.paper_trading.state import PaperAccountState


ACCOUNT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

ORDER_COLUMNS = [
    "timestamp",
    "side",
    "reason",
    "signal_time",
    "target_fraction",
    "leverage",
    "market_open",
    "fill_price",
    "quantity",
    "notional",
    "fee",
    "cash_after",
]

TRADE_COLUMNS = [
    "entry_time",
    "exit_time",
    "entry_price",
    "exit_price",
    "quantity",
    "entry_fee",
    "exit_fee",
    "total_fee",
    "gross_pnl",
    "net_pnl",
    "return_pct",
    "holding_periods",
    "exit_reason",
    "stop_loss",
    "take_profit",
    "risk_budget",
    "leverage",
]

PREDICTION_COLUMNS = [
    "timestamp",
    "symbol",
    "close",
    "model_target_fraction",
    "transformer_target_fraction",
    "threshold_target_fraction",
    "approved_target_fraction",
    "target_fraction",
    "action_signal",
    "finbert_sentiment",
    "finbert_confidence",
    "finbert_available",
    "transformer_trend",
    "transformer_return_1",
    "transformer_return_5",
    "transformer_return_20",
    "transformer_volatility",
    "transformer_bull_probability",
    "transformer_bear_probability",
    "transformer_uncertainty",
    "transformer_probability_calibrated",
    "market_regime",
    "regime_risk_multiplier",
    "capital_multiplier",
    "confidence_multiplier",
    "volatility_multiplier",
    "model_input_drifted",
    "model_input_severe_drift",
    "model_input_extreme_fraction",
    "model_input_drift_score",
    "model_input_drift_multiplier",
    "target_risk_multiplier",
    "portfolio_gross_exposure",
    "portfolio_net_exposure",
    "portfolio_halted",
    "dynamic_leverage_enabled",
    "selected_leverage",
    "required_leverage",
    "leverage_evidence_cap",
    "leverage_win_probability",
    "leverage_risk_reward",
    "leverage_net_expectancy",
    "leverage_reason",
    "decision_guard_reason",
    "pending_reason",
]

PERFORMANCE_COLUMNS = [
    "timestamp",
    "open",
    "close",
    "executed_signal",
    "policy_signal",
    "cash",
    "quantity",
    "position_value",
    "equity",
    "total_return",
    "drawdown",
    "short_carry_cost",
    "funding_paid",
    "margin_used",
    "leverage",
    "realized_pnl",
    "daily_return",
    "consecutive_losses",
    "risk_halted",
    "skipped_bars",
]

POSITION_COLUMNS = [
    "timestamp",
    "symbol",
    "quantity",
    "entry_price",
    "market_price",
    "market_value",
    "unrealized_pnl",
    "stop_loss",
    "take_profit",
    "leverage",
]


@dataclass(frozen=True, slots=True)
class PaperAccountPaths:
    """單一模擬帳戶的所有持久化檔案。"""

    account_dir: Path
    account_json: Path
    orders_csv: Path
    trades_csv: Path
    predictions_csv: Path
    performance_csv: Path
    positions_csv: Path


def validate_account_id(account_id: str) -> str:
    """限制帳戶名稱，避免寫入模擬交易根目錄以外。"""
    normalized = account_id.strip()
    if not ACCOUNT_ID_PATTERN.fullmatch(normalized):
        raise ValueError("account_id 只能使用英數字、底線與連字號，長度最多 64")
    return normalized


def account_paths(root_dir: str | Path, account_id: str) -> PaperAccountPaths:
    """建立經過驗證的帳戶路徑。"""
    account_id = validate_account_id(account_id)
    account_dir = Path(root_dir).resolve() / account_id
    return PaperAccountPaths(
        account_dir=account_dir,
        account_json=account_dir / "account.json",
        orders_csv=account_dir / "orders.csv",
        trades_csv=account_dir / "trades.csv",
        predictions_csv=account_dir / "predictions.csv",
        performance_csv=account_dir / "performance.csv",
        positions_csv=account_dir / "positions.csv",
    )


def save_account_state(paths: PaperAccountPaths, state: PaperAccountState) -> None:
    """以暫存檔取代 account.json，避免中途中斷留下半份 JSON。"""
    write_json_atomic(paths.account_json, state.to_dict())


def load_account_state(paths: PaperAccountPaths) -> PaperAccountState:
    """讀取既有帳戶狀態。"""
    if not paths.account_json.exists():
        raise FileNotFoundError(f"找不到模擬帳戶：{paths.account_dir.name}")
    payload = json.loads(paths.account_json.read_text(encoding="utf-8"))
    return PaperAccountState.from_dict(payload)


def append_csv_row(path: Path, columns: list[str], row: dict[str, object]) -> None:
    """依固定欄位追加一筆 UTF-8 CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    has_content = path.exists() and path.stat().st_size > 0
    if has_content:
        with path.open("r", newline="", encoding="utf-8") as stream:
            existing_columns = next(csv.reader(stream), [])
        if existing_columns != columns:
            # 新版本增加紀錄欄位時，自動保留舊資料並補齊空欄，避免追加後錯位。
            existing = pd.read_csv(path)
            migrated = existing.reindex(columns=columns)
            temporary = path.with_suffix(path.suffix + ".tmp")
            migrated.to_csv(temporary, index=False, encoding="utf-8")
            temporary.replace(path)
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        if not has_content:
            writer.writeheader()
        writer.writerow({column: row.get(column) for column in columns})


def read_account_csv(path: str | Path, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """載入帳戶 CSV；檔案尚未產生時回傳空表。"""
    source = Path(path)
    if not source.exists() or source.stat().st_size == 0:
        return pd.DataFrame(columns=list(columns or []))
    return read_csv_snapshot(source)


def list_paper_accounts(root_dir: str | Path) -> list[PaperAccountPaths]:
    """列出已有 account.json 的模擬帳戶，最近更新者優先。"""
    root = Path(root_dir)
    if not root.exists():
        return []
    paths: list[PaperAccountPaths] = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        try:
            paths.append(account_paths(root, path.name))
        except ValueError:
            # 手動放入的不合法資料夾不應讓整個 Dashboard 無法開啟。
            continue
    valid = [path for path in paths if path.account_json.exists()]
    return sorted(valid, key=lambda path: path.account_json.stat().st_mtime, reverse=True)


def delete_paper_account(root_dir: str | Path, account_id: str) -> None:
    """刪除經過驗證且確實位於 paper_trading 根目錄內的帳戶。"""
    paths = account_paths(root_dir, account_id)
    root = Path(root_dir).resolve()
    if paths.account_dir.parent != root:
        raise ValueError("帳戶路徑不在 paper_trading 根目錄內")
    if not paths.account_dir.exists():
        return
    shutil.rmtree(paths.account_dir)
