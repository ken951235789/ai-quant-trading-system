"""依風險預算計算部位、停損與停利價格。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ai_quant_trading.risk.config import RiskConfig


PositionSide = Literal["long", "short"]


@dataclass(frozen=True, slots=True)
class PositionSizeResult:
    """單次進場所允許的部位計算結果。"""

    quantity: float
    risk_budget: float
    stop_distance: float
    position_notional: float


def calculate_stop_loss(
    entry_price: float,
    config: RiskConfig,
    atr: float | None = None,
    side: PositionSide = "long",
) -> float:
    """依固定百分比或前一期 ATR 計算停損價。"""
    if entry_price <= 0:
        raise ValueError("entry_price 必須大於 0")
    if side not in {"long", "short"}:
        raise ValueError("side 必須是 long 或 short")
    direction = 1 if side == "long" else -1
    if config.stop_loss_mode == "fixed":
        stop_loss = entry_price * (1 - direction * config.fixed_stop_loss_pct)
    else:
        if atr is None or atr <= 0:
            raise ValueError("ATR 停損需要大於 0 的 atr_14")
        stop_loss = entry_price - direction * atr * config.atr_multiplier
    if stop_loss <= 0 or (side == "long" and stop_loss >= entry_price):
        raise ValueError("多頭停損價必須介於 0 與進場價之間")
    if side == "short" and stop_loss <= entry_price:
        raise ValueError("空頭停損價必須高於進場價")
    return stop_loss


def calculate_take_profit(
    entry_price: float,
    config: RiskConfig,
    side: PositionSide = "long",
) -> float | None:
    """依進場價計算停利價；None 代表不啟用停利。"""
    if config.take_profit_pct is None:
        return None
    if side not in {"long", "short"}:
        raise ValueError("side 必須是 long 或 short")
    direction = 1 if side == "long" else -1
    return entry_price * (1 + direction * config.take_profit_pct)


def calculate_position_size(
    *,
    equity: float,
    cash: float,
    entry_price: float,
    stop_loss: float,
    fee_rate: float,
    max_risk_per_trade: float,
    max_position_fraction: float,
    side: PositionSide = "long",
) -> PositionSizeResult:
    """取風險限制部位與資金可負擔部位中較小者。"""
    if equity <= 0 or cash <= 0:
        return PositionSizeResult(0.0, 0.0, 0.0, 0.0)
    valid_stop = (
        0 < stop_loss < entry_price
        if side == "long"
        else stop_loss > entry_price
    )
    if entry_price <= 0 or not valid_stop:
        raise ValueError("進場價與停損價不合法")
    if not 0 <= fee_rate < 1:
        raise ValueError("fee_rate 必須介於 0 與 1 之間")
    if not 0 < max_risk_per_trade <= 1:
        raise ValueError("max_risk_per_trade 必須介於 0 與 1 之間")
    if not 0 < max_position_fraction <= 1:
        raise ValueError("max_position_fraction 必須介於 0 與 1 之間")

    stop_distance = abs(entry_price - stop_loss)
    risk_budget = equity * max_risk_per_trade
    risk_limited_quantity = risk_budget / stop_distance
    allocation_budget = min(cash, equity * max_position_fraction)
    affordable_quantity = allocation_budget / (entry_price * (1 + fee_rate))
    quantity = max(0.0, min(risk_limited_quantity, affordable_quantity))
    return PositionSizeResult(
        quantity=quantity,
        risk_budget=risk_budget,
        stop_distance=stop_distance,
        position_notional=quantity * entry_price,
    )
