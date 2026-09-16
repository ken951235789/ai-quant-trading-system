"""交易機器人監控頁共用的績效與狀態計算。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

from ai_quant_trading.automation import AutomationStatus


@dataclass(frozen=True, slots=True)
class ProfitMetrics:
    """以一致口徑呈現帳戶目前與分期損益。"""

    current_equity: float
    today_pnl: float
    today_return: float
    week_pnl: float
    week_return: float
    cumulative_pnl: float
    cumulative_return: float
    maximum_drawdown: float
    trade_count: int
    win_rate: float


@dataclass(frozen=True, slots=True)
class RobotHealth:
    """將 runner、風控與模型狀態整理成監控頁可讀文字。"""

    status: str
    severity: Literal["normal", "warning", "error", "idle"]
    heartbeat_age_minutes: float | None
    message: str


@dataclass(frozen=True, slots=True)
class PositionExposure:
    """整理目前持倉實際使用本金、名目部位與帳戶曝險。"""

    direction: Literal["多單", "空單", "空手"]
    capital_used: float
    position_notional: float
    account_exposure: float
    leverage: float


def calculate_position_exposure(
    *,
    quantity: float,
    entry_price: float | None,
    market_price: float | None,
    equity: float,
    leverage: float = 1.0,
) -> PositionExposure:
    """依實際數量計算本金與曝險；目前現貨／模擬帳戶使用 1 倍槓桿。"""
    if leverage <= 0:
        raise ValueError("leverage 必須大於 0")
    if abs(quantity) <= 1e-12 or entry_price is None or entry_price <= 0:
        return PositionExposure("空手", 0.0, 0.0, 0.0, 0.0)

    direction: Literal["多單", "空單", "空手"] = "多單" if quantity > 0 else "空單"
    valid_market_price = (
        float(market_price) if market_price is not None and market_price > 0 else float(entry_price)
    )
    entry_notional = abs(float(quantity)) * float(entry_price)
    position_notional = abs(float(quantity)) * valid_market_price
    capital_used = entry_notional / leverage
    account_exposure = position_notional / equity if equity > 0 else 0.0
    return PositionExposure(
        direction=direction,
        capital_used=capital_used,
        position_notional=position_notional,
        account_exposure=account_exposure,
        leverage=leverage,
    )


def prepare_equity_curve(
    frame: pd.DataFrame,
    *,
    equity_column: str = "equity",
    initial_equity: float,
) -> pd.DataFrame:
    """清理資產序列並計算累計損益與回撤。"""
    if initial_equity <= 0:
        raise ValueError("initial_equity 必須大於 0")
    if frame.empty or not {"timestamp", equity_column}.issubset(frame.columns):
        return pd.DataFrame(
            columns=[
                "timestamp",
                "equity",
                "cumulative_pnl",
                "cumulative_return",
                "drawdown",
            ]
        )
    result = frame[["timestamp", equity_column]].copy()
    result["timestamp"] = pd.to_datetime(
        result["timestamp"], utc=True, errors="coerce", format="mixed"
    )
    result["equity"] = pd.to_numeric(result[equity_column], errors="coerce")
    result = (
        result.dropna(subset=["timestamp", "equity"])
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )
    if result.empty:
        return result.reindex(
            columns=[
                "timestamp",
                "equity",
                "cumulative_pnl",
                "cumulative_return",
                "drawdown",
            ]
        )
    result["cumulative_pnl"] = result["equity"] - initial_equity
    result["cumulative_return"] = result["equity"] / initial_equity - 1
    running_peak = result["equity"].cummax().clip(lower=initial_equity)
    result["drawdown"] = result["equity"] / running_peak - 1
    return result


def _period_pnl(
    curve: pd.DataFrame,
    boundary_utc: pd.Timestamp,
    current_equity: float,
    initial_equity: float,
) -> tuple[float, float]:
    """計算邊界之後的損益；沒有新資料時回傳零。"""
    if curve.empty:
        return 0.0, 0.0
    latest_timestamp = pd.Timestamp(curve.iloc[-1]["timestamp"])
    if latest_timestamp < boundary_utc:
        return 0.0, 0.0
    previous = curve[curve["timestamp"] < boundary_utc]
    baseline = float(previous.iloc[-1]["equity"]) if not previous.empty else initial_equity
    pnl = current_equity - baseline
    return pnl, pnl / baseline if baseline else 0.0


def calculate_profit_metrics(
    performance: pd.DataFrame,
    *,
    initial_equity: float,
    trades: pd.DataFrame | None = None,
    now: datetime | pd.Timestamp | None = None,
    timezone_name: str = "Asia/Taipei",
    equity_column: str = "equity",
) -> tuple[ProfitMetrics, pd.DataFrame]:
    """依台北日曆日與週一切分今日、本週及累計績效。"""
    curve = prepare_equity_curve(
        performance,
        equity_column=equity_column,
        initial_equity=initial_equity,
    )
    current_equity = float(curve.iloc[-1]["equity"]) if not curve.empty else initial_equity
    now_timestamp = pd.Timestamp.now(tz=timezone_name) if now is None else pd.Timestamp(now)
    if now_timestamp.tzinfo is None:
        now_timestamp = now_timestamp.tz_localize(timezone_name)
    else:
        now_timestamp = now_timestamp.tz_convert(timezone_name)
    day_start = now_timestamp.normalize()
    week_start = day_start - pd.Timedelta(days=day_start.weekday())
    today_pnl, today_return = _period_pnl(
        curve,
        day_start.tz_convert("UTC"),
        current_equity,
        initial_equity,
    )
    week_pnl, week_return = _period_pnl(
        curve,
        week_start.tz_convert("UTC"),
        current_equity,
        initial_equity,
    )
    cumulative_pnl = current_equity - initial_equity
    cumulative_return = cumulative_pnl / initial_equity
    maximum_drawdown = float(curve["drawdown"].min()) if not curve.empty else 0.0
    valid_trades = trades if trades is not None else pd.DataFrame()
    net_pnl = (
        pd.to_numeric(valid_trades.get("net_pnl"), errors="coerce").dropna()
        if "net_pnl" in valid_trades
        else pd.Series(dtype="float64")
    )
    trade_count = len(net_pnl)
    win_rate = float((net_pnl > 0).mean()) if trade_count else 0.0
    return (
        ProfitMetrics(
            current_equity=current_equity,
            today_pnl=today_pnl,
            today_return=today_return,
            week_pnl=week_pnl,
            week_return=week_return,
            cumulative_pnl=cumulative_pnl,
            cumulative_return=cumulative_return,
            maximum_drawdown=maximum_drawdown,
            trade_count=trade_count,
            win_rate=win_rate,
        ),
        curve,
    )


def assess_robot_health(
    automation: AutomationStatus,
    *,
    risk_halted: bool = False,
    model_available: bool = True,
    now: datetime | pd.Timestamp | None = None,
    heartbeat_timeout_minutes: float = 5.0,
    activity_at: str | datetime | pd.Timestamp | None = None,
) -> RobotHealth:
    """依風控、模型、程序與最新背景活動判斷機器人目前狀態。"""
    if risk_halted:
        return RobotHealth("風控停機", "error", None, "最大回撤或風控條件已觸發")
    if not model_available:
        return RobotHealth("模型遺失", "error", None, "找不到此機器人的模型成品")
    heartbeat_age: float | None = None
    valid_activity = [
        pd.Timestamp(value)
        for item in (automation.heartbeat_at, activity_at)
        if item is not None
        for value in [pd.to_datetime(item, utc=True, errors="coerce")]
        if pd.notna(value)
    ]
    if valid_activity:
        heartbeat = max(valid_activity)
        if pd.notna(heartbeat):
            current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
            if current.tzinfo is None:
                current = current.tz_localize("UTC")
            else:
                current = current.tz_convert("UTC")
            heartbeat_age = max(
                0.0,
                (current - heartbeat).total_seconds() / 60,
            )
    if automation.state == "running":
        if heartbeat_age is not None and heartbeat_age > heartbeat_timeout_minutes:
            return RobotHealth(
                "心跳逾時",
                "warning",
                heartbeat_age,
                automation.last_message,
            )
        return RobotHealth(
            "運行中",
            "normal",
            heartbeat_age,
            automation.last_message,
        )
    if automation.state == "failed":
        return RobotHealth(
            "執行錯誤",
            "error",
            heartbeat_age,
            automation.last_error or automation.last_message,
        )
    if automation.state == "completed":
        return RobotHealth(
            "已完成",
            "idle",
            heartbeat_age,
            automation.last_message,
        )
    if automation.state == "stopped":
        return RobotHealth(
            "已停止",
            "idle",
            heartbeat_age,
            automation.last_message,
        )
    return RobotHealth("未啟動", "idle", heartbeat_age, automation.last_message)


def signed_money(value: float) -> str:
    """統一監控頁的正負損益格式。"""
    return f"{value:+,.2f}"


def finite_or_zero(value: object) -> float:
    """將 CSV 中的非有限數值安全轉成零。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if np.isfinite(number) else 0.0
