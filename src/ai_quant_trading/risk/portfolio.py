"""跨帳戶與長短期策略共用的投資組合硬性風控。"""

from __future__ import annotations

from dataclasses import dataclass

import math
import pandas as pd


@dataclass(frozen=True, slots=True)
class PortfolioRiskConfig:
    """限制整體曝險、策略資金與不同時間尺度的虧損。"""

    max_gross_exposure: float = 0.90
    max_net_exposure: float = 0.70
    max_long_exposure: float = 0.80
    max_short_exposure: float = 0.20
    max_strategy_exposure: float = 0.60
    max_asset_exposure: float = 0.20
    max_daily_loss: float = 0.02
    max_weekly_loss: float = 0.05
    max_monthly_loss: float = 0.10

    def __post_init__(self) -> None:
        fields = (
            self.max_gross_exposure,
            self.max_net_exposure,
            self.max_long_exposure,
            self.max_short_exposure,
            self.max_strategy_exposure,
            self.max_asset_exposure,
            self.max_daily_loss,
            self.max_weekly_loss,
            self.max_monthly_loss,
        )
        if any(not 0 < value <= 1 for value in fields):
            raise ValueError("投資組合風控比例必須大於 0 且不超過 1")
        if self.max_long_exposure + self.max_short_exposure < self.max_gross_exposure:
            raise ValueError("多頭與空頭上限合計不可小於總曝險上限")


@dataclass(frozen=True, slots=True)
class PortfolioExposure:
    """單一模擬帳戶目前占用的有方向市場曝險。"""

    account_id: str
    strategy: str
    asset_group: str
    equity: float
    market_value: float

    def __post_init__(self) -> None:
        if self.equity <= 0:
            raise ValueError("帳戶 equity 必須大於 0")


@dataclass(frozen=True, slots=True)
class PortfolioRiskDecision:
    """投資組合層級對單一帳戶目標部位的裁決。"""

    proposed_target: float
    approved_target: float
    gross_exposure: float
    net_exposure: float
    halted: bool
    reasons: tuple[str, ...]


def calculate_period_returns(
    history: pd.DataFrame,
    *,
    current_timestamp: str | pd.Timestamp,
    current_value: float,
    timestamp_column: str = "timestamp",
    value_column: str = "equity",
) -> tuple[float, float, float]:
    """以期間開始前最後一筆淨值為基準，計算日、週、月報酬。"""
    now = pd.Timestamp(current_timestamp)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    value = float(current_value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("current_value 必須是大於 0 的有限數值")
    if history.empty or not {timestamp_column, value_column}.issubset(history.columns):
        return 0.0, 0.0, 0.0
    frame = history[[timestamp_column, value_column]].copy()
    frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce")
    frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
    frame = frame.dropna().loc[lambda item: item[value_column] > 0]
    frame = frame.loc[frame[timestamp_column] <= now].sort_values(timestamp_column)
    current = pd.DataFrame({timestamp_column: [now], value_column: [value]})
    frame = pd.concat([frame, current], ignore_index=True).drop_duplicates(
        timestamp_column, keep="last"
    )

    day_start = now.normalize()
    week_start = day_start - pd.Timedelta(days=day_start.weekday())
    month_start = pd.Timestamp(year=now.year, month=now.month, day=1, tz="UTC")

    def period_return(start: pd.Timestamp) -> float:
        before = frame.loc[frame[timestamp_column] < start]
        inside = frame.loc[frame[timestamp_column] >= start]
        if not before.empty:
            baseline = float(before.iloc[-1][value_column])
        elif not inside.empty:
            baseline = float(inside.iloc[0][value_column])
        else:
            return 0.0
        return value / baseline - 1.0 if baseline > 0 else 0.0

    return period_return(day_start), period_return(week_start), period_return(month_start)


def expert_portfolio_risk(kind: str) -> PortfolioRiskConfig:
    """回傳長期與短期專家不同的投資組合上限。"""
    if kind == "long_term":
        return PortfolioRiskConfig(
            max_gross_exposure=0.70,
            max_net_exposure=0.60,
            max_long_exposure=0.70,
            max_short_exposure=0.10,
            max_strategy_exposure=0.55,
            max_asset_exposure=0.15,
            max_daily_loss=0.015,
            max_weekly_loss=0.04,
            max_monthly_loss=0.08,
        )
    if kind == "short_term":
        return PortfolioRiskConfig(
            max_gross_exposure=0.50,
            max_net_exposure=0.50,
            max_long_exposure=0.50,
            max_short_exposure=0.50,
            max_strategy_exposure=0.50,
            max_asset_exposure=0.50,
            max_daily_loss=0.01,
            max_weekly_loss=0.025,
            max_monthly_loss=0.05,
        )
    return PortfolioRiskConfig()


def govern_portfolio_target(
    *,
    account_id: str,
    strategy: str,
    asset_group: str,
    account_equity: float,
    current_target: float,
    proposed_target: float,
    exposures: list[PortfolioExposure],
    config: PortfolioRiskConfig,
    daily_return: float = 0.0,
    weekly_return: float = 0.0,
    monthly_return: float = 0.0,
) -> PortfolioRiskDecision:
    """先允許減碼，再以整體、方向、策略與標的上限裁切加碼。"""
    if account_equity <= 0:
        raise ValueError("account_equity 必須大於 0")
    reasons: list[str] = []
    loss_limits = (
        (daily_return, config.max_daily_loss, "單日"),
        (weekly_return, config.max_weekly_loss, "單週"),
        (monthly_return, config.max_monthly_loss, "單月"),
    )
    for value, limit, label in loss_limits:
        if value <= -limit:
            reasons.append(f"{label}虧損 {abs(value):.2%} 已達 {limit:.2%} 上限")
    others = [item for item in exposures if item.account_id != account_id]
    known_accounts = {item.account_id for item in exposures}
    total_equity = sum(item.equity for item in exposures)
    if account_id not in known_accounts:
        total_equity += account_equity
    total_equity = max(total_equity, account_equity)
    other_values = [item.market_value for item in others]
    other_gross = sum(abs(value) for value in other_values)
    other_net = sum(other_values)
    gross_fraction = (other_gross + abs(current_target * account_equity)) / total_equity
    net_fraction = (other_net + current_target * account_equity) / total_equity
    if reasons:
        return PortfolioRiskDecision(
            proposed_target,
            0.0,
            gross_fraction,
            net_fraction,
            True,
            tuple(reasons),
        )

    # 任何朝零方向的交易都應放行，讓超限投資組合能自行降風險。
    same_direction = proposed_target * current_target >= 0
    if same_direction and abs(proposed_target) <= abs(current_target):
        return PortfolioRiskDecision(
            proposed_target,
            proposed_target,
            gross_fraction,
            net_fraction,
            False,
            (),
        )

    sign = 1.0 if proposed_target >= 0 else -1.0
    requested_value = abs(proposed_target) * account_equity
    strategy_gross = sum(
        abs(item.market_value) for item in others if item.strategy == strategy
    )
    asset_gross = sum(
        abs(item.market_value) for item in others if item.asset_group == asset_group
    )
    limits = {
        "總曝險": config.max_gross_exposure * total_equity - other_gross,
        "策略曝險": config.max_strategy_exposure * total_equity - strategy_gross,
        "單一標的曝險": config.max_asset_exposure * total_equity - asset_gross,
    }
    if sign > 0:
        other_long = sum(max(item.market_value, 0.0) for item in others)
        limits["多頭曝險"] = config.max_long_exposure * total_equity - other_long
        limits["淨曝險"] = config.max_net_exposure * total_equity - other_net
    else:
        other_short = sum(abs(min(item.market_value, 0.0)) for item in others)
        limits["空頭曝險"] = config.max_short_exposure * total_equity - other_short
        limits["淨曝險"] = config.max_net_exposure * total_equity + other_net
    allowed_value = max(min([requested_value, *limits.values()]), 0.0)
    for label, limit in limits.items():
        if requested_value > max(limit, 0.0) + 1e-9:
            reasons.append(f"已套用{label}上限")
    approved = sign * allowed_value / account_equity
    new_gross = (other_gross + allowed_value) / total_equity
    new_net = (other_net + sign * allowed_value) / total_equity
    return PortfolioRiskDecision(
        proposed_target,
        approved,
        new_gross,
        new_net,
        False,
        tuple(reasons),
    )
