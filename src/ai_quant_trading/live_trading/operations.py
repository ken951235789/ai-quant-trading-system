"""模型外部的即時營運風控，故障時只允許降低風險。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math

import pandas as pd


@dataclass(frozen=True, slots=True)
class OperationalRiskLimits:
    """不由模型修改的帳戶級硬限制。"""

    maximum_daily_loss: float = 0.01
    maximum_drawdown: float = 0.08
    warning_fraction: float = 0.75

    def __post_init__(self) -> None:
        if not 0 < self.maximum_daily_loss <= 1:
            raise ValueError("maximum_daily_loss 必須介於 0 與 1")
        if not 0 < self.maximum_drawdown <= 1:
            raise ValueError("maximum_drawdown 必須介於 0 與 1")
        if not 0 < self.warning_fraction < 1:
            raise ValueError("warning_fraction 必須介於 0 與 1")


@dataclass(frozen=True, slots=True)
class OperationalRiskAssessment:
    """營運風控對新增風險與既有部位的處置。"""

    allow_new_risk: bool
    force_reduce: bool
    hard_halt: bool
    daily_return: float
    drawdown: float
    current_equity: float | None
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "allow_new_risk": self.allow_new_risk,
            "force_reduce": self.force_reduce,
            "hard_halt": self.hard_halt,
            "daily_return": self.daily_return,
            "drawdown": self.drawdown,
            "current_equity": self.current_equity,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
        }


def assess_operational_risk(
    snapshots: pd.DataFrame,
    *,
    current_equity: float | None = None,
    has_open_position: bool = False,
    can_trade: bool = True,
    emergency_halt: str | None = None,
    limits: OperationalRiskLimits | None = None,
    now: datetime | pd.Timestamp | None = None,
) -> OperationalRiskAssessment:
    """用實際資產曲線計算單日損失與全期回撤，不參考模型 Reward。"""
    limits = limits or OperationalRiskLimits()
    current_time = pd.Timestamp(now or datetime.now(timezone.utc))
    current_time = (
        current_time.tz_localize("UTC")
        if current_time.tzinfo is None
        else current_time.tz_convert("UTC")
    )
    history = snapshots.copy()
    if not history.empty and {"timestamp", "estimated_equity"}.issubset(history.columns):
        history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True, errors="coerce")
        history["estimated_equity"] = pd.to_numeric(
            history["estimated_equity"], errors="coerce"
        )
        history = history.dropna(subset=["timestamp", "estimated_equity"])
        history = history.loc[history["estimated_equity"] > 0].sort_values("timestamp")
    else:
        history = pd.DataFrame(columns=["timestamp", "estimated_equity"])

    equity = float(current_equity) if current_equity is not None else None
    if equity is not None and (not math.isfinite(equity) or equity <= 0):
        equity = None
    if equity is None and not history.empty:
        equity = float(history.iloc[-1]["estimated_equity"])

    daily_return = 0.0
    drawdown = 0.0
    if equity is not None and not history.empty:
        peak = max(float(history["estimated_equity"].max()), equity)
        drawdown = equity / peak - 1 if peak > 0 else 0.0
        same_day = history.loc[history["timestamp"].dt.date == current_time.date()]
        if not same_day.empty:
            opening_equity = float(same_day.iloc[0]["estimated_equity"])
            daily_return = equity / opening_equity - 1 if opening_equity > 0 else 0.0

    reasons: list[str] = []
    warnings: list[str] = []
    hard_halt = False
    if not can_trade:
        reasons.append("交易所帳戶目前不可交易")
    if emergency_halt:
        reasons.append(f"緊急停機：{emergency_halt}")
    if equity is None:
        reasons.append("缺少有效帳戶資產快照")
    if daily_return <= -limits.maximum_daily_loss:
        reasons.append(
            f"單日損失 {daily_return:.2%} 已達上限 {limits.maximum_daily_loss:.2%}"
        )
        hard_halt = True
    elif daily_return <= -limits.maximum_daily_loss * limits.warning_fraction:
        warnings.append(f"單日損失接近上限：{daily_return:.2%}")
    if drawdown <= -limits.maximum_drawdown:
        reasons.append(f"帳戶回撤 {drawdown:.2%} 已達上限 {limits.maximum_drawdown:.2%}")
        hard_halt = True
    elif drawdown <= -limits.maximum_drawdown * limits.warning_fraction:
        warnings.append(f"帳戶回撤接近上限：{drawdown:.2%}")
    return OperationalRiskAssessment(
        allow_new_risk=not reasons,
        force_reduce=hard_halt and has_open_position,
        hard_halt=hard_halt,
        daily_return=daily_return,
        drawdown=drawdown,
        current_equity=equity,
        reasons=tuple(reasons),
        warnings=tuple(warnings),
    )
