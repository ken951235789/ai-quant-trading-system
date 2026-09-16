"""模型外部的目標持倉風控與雙專家資金分配。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TargetRiskDecision:
    """Risk Governor 對模型目標持倉的核准結果。"""

    proposed_target: float
    approved_target: float
    current_position: float
    risk_multiplier: float
    halted: bool
    reasons: tuple[str, ...]


def govern_target_position(
    *,
    proposed_target: float,
    current_position: float,
    drawdown: float,
    holding_bars: int,
    max_position_fraction: float,
    allow_short: bool = False,
    max_short_fraction: float = 0.0,
    hard_drawdown_limit: float,
    soft_drawdown_limit: float | None = None,
    soft_drawdown_multiplier: float = 0.5,
    drawdown_curve_exponent: float | None = None,
    rebalance_deadband: float = 0.0,
    minimum_holding_bars: int = 0,
) -> TargetRiskDecision:
    """先套回撤與持有期限制，再建立不交易區間以降低無效換手。"""
    values = [proposed_target, current_position, drawdown]
    if any(not isinstance(value, (int, float)) for value in values):
        raise TypeError("目標持倉、目前持倉與回撤必須是數值")
    if not 0 < max_position_fraction <= 1:
        raise ValueError("max_position_fraction 必須大於 0 且不超過 1")
    if not 0 <= max_short_fraction <= 1:
        raise ValueError("max_short_fraction 必須介於 0 與 1 之間")
    if allow_short and max_short_fraction <= 0:
        raise ValueError("啟用作空時 max_short_fraction 必須大於 0")
    if not 0 < hard_drawdown_limit <= 1:
        raise ValueError("hard_drawdown_limit 必須大於 0 且不超過 1")
    if soft_drawdown_limit is not None and not 0 < soft_drawdown_limit < hard_drawdown_limit:
        raise ValueError("soft_drawdown_limit 必須小於硬性最大回撤")
    if not 0 < soft_drawdown_multiplier <= 1:
        raise ValueError("soft_drawdown_multiplier 必須大於 0 且不超過 1")
    if drawdown_curve_exponent is not None and drawdown_curve_exponent <= 0:
        raise ValueError("drawdown_curve_exponent 必須大於 0 或設為 None")
    if not 0 <= rebalance_deadband < 1 or minimum_holding_bars < 0:
        raise ValueError("調倉死區或最短持有期設定不合法")

    short_limit = max_short_fraction if allow_short else 0.0

    def clip_target(value: float, multiplier: float = 1.0) -> float:
        return min(
            max(float(value), -short_limit * multiplier),
            max_position_fraction * multiplier,
        )

    proposed = clip_target(float(proposed_target))
    current = min(max(float(current_position), -short_limit), max_position_fraction)
    drawdown_depth = max(-float(drawdown), 0.0)
    reasons: list[str] = []

    if drawdown_depth >= hard_drawdown_limit:
        return TargetRiskDecision(proposed, 0.0, current, 0.0, True, ("hard_drawdown",))

    multiplier = 1.0
    soft_drawdown_triggered = False
    if soft_drawdown_limit is not None and drawdown_depth >= soft_drawdown_limit:
        soft_drawdown_triggered = True
        multiplier = soft_drawdown_multiplier
        if drawdown_curve_exponent is not None:
            remaining = max(
                (hard_drawdown_limit - drawdown_depth)
                / (hard_drawdown_limit - soft_drawdown_limit),
                0.0,
            )
            multiplier *= remaining**drawdown_curve_exponent
        proposed = clip_target(proposed, multiplier)
        reasons.append(
            "adaptive_drawdown_reduction"
            if drawdown_curve_exponent is not None
            else "soft_drawdown_reduction"
        )

    # 最短持有期只能阻止加碼或直接反手，不能阻止減倉與平倉。
    # 反手要求先降到零，下一根再由完整決策鏈重新確認新方向。
    if holding_bars < minimum_holding_bars and abs(current) > 1e-12:
        if proposed * current < 0:
            proposed = 0.0
            reasons.append("minimum_holding_blocks_reversal")
        elif abs(proposed) > abs(current) and not soft_drawdown_triggered:
            proposed = current
            reasons.append("minimum_holding_blocks_addition")

    reducing = proposed * current >= 0 and abs(proposed) <= abs(current) + 1e-12
    if abs(proposed - current) < rebalance_deadband and not reducing:
        proposed = current
        reasons.append("rebalance_deadband")

    return TargetRiskDecision(
        float(proposed_target),
        clip_target(proposed),
        current,
        multiplier,
        False,
        tuple(reasons),
    )


@dataclass(frozen=True, slots=True)
class ExpertCapitalAllocation:
    """長期、短期與現金準備金的資金配置。"""

    long_term_fraction: float = 0.70
    short_term_fraction: float = 0.20
    reserve_fraction: float = 0.10

    def __post_init__(self) -> None:
        fractions = (
            self.long_term_fraction,
            self.short_term_fraction,
            self.reserve_fraction,
        )
        if any(value < 0 for value in fractions):
            raise ValueError("專家資金比例不可小於 0")
        if abs(sum(fractions) - 1.0) > 1e-9:
            raise ValueError("長期、短期與準備金比例合計必須為 100%")

    def allocate(self, total_equity: float) -> dict[str, float]:
        """依風險預算切分資金，不依最近一次報酬追逐模型。"""
        if total_equity <= 0:
            raise ValueError("total_equity 必須大於 0")
        return {
            "long_term": total_equity * self.long_term_fraction,
            "short_term": total_equity * self.short_term_fraction,
            "reserve": total_equity * self.reserve_fraction,
        }
