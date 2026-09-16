"""依據模型證據與帳戶風險選擇逐倉槓桿。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from ai_quant_trading.trading.contracts import ModelForecast


@dataclass(frozen=True, slots=True)
class DynamicLeverageConfig:
    """動態槓桿門檻；模型只能在這些硬限制內降低或提高槓桿。"""

    enabled: bool = False
    min_leverage: int = 1
    max_leverage: int = 3
    uncalibrated_max_leverage: int = 1
    leverage_2_min_probability: float = 0.58
    leverage_3_min_probability: float = 0.65
    leverage_2_min_risk_reward: float = 1.25
    leverage_3_min_risk_reward: float = 1.75
    leverage_2_max_uncertainty: float = 0.45
    leverage_3_max_uncertainty: float = 0.30
    leverage_2_max_drawdown: float = 0.05
    leverage_3_max_drawdown: float = 0.025
    leverage_2_max_spread_bps: float = 8.0
    leverage_3_max_spread_bps: float = 4.0
    leverage_2_max_volatility: float = 0.03
    leverage_3_max_volatility: float = 0.02
    require_positive_expectancy: bool = True

    def __post_init__(self) -> None:
        if not 1 <= self.min_leverage <= self.max_leverage <= 3:
            raise ValueError("動態槓桿必須符合 1 <= min_leverage <= max_leverage <= 3")
        if not self.min_leverage <= self.uncalibrated_max_leverage <= self.max_leverage:
            raise ValueError("uncalibrated_max_leverage 必須位於動態槓桿範圍內")
        for name, value in (
            ("leverage_2_min_probability", self.leverage_2_min_probability),
            ("leverage_3_min_probability", self.leverage_3_min_probability),
            ("leverage_2_max_uncertainty", self.leverage_2_max_uncertainty),
            ("leverage_3_max_uncertainty", self.leverage_3_max_uncertainty),
            ("leverage_2_max_drawdown", self.leverage_2_max_drawdown),
            ("leverage_3_max_drawdown", self.leverage_3_max_drawdown),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} 必須位於 0 到 1")
        if self.leverage_3_min_probability < self.leverage_2_min_probability:
            raise ValueError("3x 的最低勝率不可低於 2x")
        if self.leverage_3_min_risk_reward < self.leverage_2_min_risk_reward:
            raise ValueError("3x 的最低損益比不可低於 2x")
        if self.leverage_3_max_uncertainty > self.leverage_2_max_uncertainty:
            raise ValueError("3x 的不確定度上限不可高於 2x")
        if self.leverage_3_max_drawdown > self.leverage_2_max_drawdown:
            raise ValueError("3x 的回撤上限不可高於 2x")
        for name, value in (
            ("leverage_2_min_risk_reward", self.leverage_2_min_risk_reward),
            ("leverage_3_min_risk_reward", self.leverage_3_min_risk_reward),
            ("leverage_2_max_spread_bps", self.leverage_2_max_spread_bps),
            ("leverage_3_max_spread_bps", self.leverage_3_max_spread_bps),
            ("leverage_2_max_volatility", self.leverage_2_max_volatility),
            ("leverage_3_max_volatility", self.leverage_3_max_volatility),
        ):
            if float(value) <= 0:
                raise ValueError(f"{name} 必須大於 0")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DynamicLeverageDecision:
    """可供交易、稽核與 UI 使用的完整槓桿決策。"""

    enabled: bool
    selected_leverage: int
    required_leverage: int
    evidence_cap: int
    proposed_target: float
    approved_target: float
    win_probability: float
    risk_reward_ratio: float
    net_expectancy: float
    probability_calibrated: bool
    reasons: tuple[str, ...] = ()

    @property
    def reason_text(self) -> str:
        return "；".join(self.reasons) if self.reasons else "符合動態槓桿條件"

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["reasons"] = self.reason_text
        return result


def _bounded_number(value: float, lower: float, upper: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        return lower
    return min(max(numeric, lower), upper)


def _required_leverage(target: float, max_margin_fraction: float) -> int:
    if abs(target) <= 1e-12:
        return 1
    return max(1, int(math.ceil(abs(target) / max(max_margin_fraction, 1e-12) - 1e-12)))


def _is_reducing_risk(proposed_target: float, current_position: float) -> bool:
    if abs(proposed_target) <= 1e-12:
        return True
    if abs(current_position) <= 1e-12:
        return False
    same_direction = math.copysign(1.0, proposed_target) == math.copysign(1.0, current_position)
    return same_direction and abs(proposed_target) <= abs(current_position) + 1e-12


def select_dynamic_leverage(
    *,
    proposed_target: float,
    current_position: float,
    current_leverage: int,
    max_margin_fraction: float,
    stop_distance_fraction: float,
    take_profit_fraction: float,
    fee_rate: float,
    slippage_rate: float,
    forecast: ModelForecast | None,
    drawdown: float,
    spread_bps: float = 0.0,
    probability_calibrated: bool = False,
    config: DynamicLeverageConfig | None = None,
) -> DynamicLeverageDecision:
    """選擇 1x 到 3x，並在證據不足時同步縮小目標部位。"""
    settings = config or DynamicLeverageConfig()
    proposed = _bounded_number(proposed_target, -1.0, 1.0)
    current = _bounded_number(current_position, -1.0, 1.0)
    active_leverage = int(_bounded_number(current_leverage, 1, 3))
    margin_fraction = _bounded_number(max_margin_fraction, 1e-6, 1.0)
    required = _required_leverage(proposed, margin_fraction)

    if not settings.enabled:
        selected = min(max(active_leverage, settings.min_leverage), settings.max_leverage)
        approved = math.copysign(
            min(abs(proposed), margin_fraction * selected), proposed
        ) if abs(proposed) > 1e-12 else 0.0
        return DynamicLeverageDecision(
            enabled=False,
            selected_leverage=selected,
            required_leverage=required,
            evidence_cap=selected,
            proposed_target=proposed,
            approved_target=approved,
            win_probability=0.0,
            risk_reward_ratio=0.0,
            net_expectancy=0.0,
            probability_calibrated=False,
            reasons=("動態槓桿未啟用，沿用帳戶固定槓桿",),
        )

    if abs(proposed) <= 1e-12 and abs(current) <= 1e-12:
        return DynamicLeverageDecision(
            enabled=True,
            selected_leverage=settings.min_leverage,
            required_leverage=1,
            evidence_cap=settings.min_leverage,
            proposed_target=0.0,
            approved_target=0.0,
            win_probability=0.0,
            risk_reward_ratio=0.0,
            net_expectancy=0.0,
            probability_calibrated=probability_calibrated,
            reasons=("目前空手且沒有新目標，維持最低槓桿",),
        )

    # 平倉與減倉不應被提高槓桿的證據門檻阻擋。
    if _is_reducing_risk(proposed, current):
        selected = min(max(active_leverage, settings.min_leverage), settings.max_leverage)
        return DynamicLeverageDecision(
            enabled=True,
            selected_leverage=selected,
            required_leverage=_required_leverage(proposed, margin_fraction),
            evidence_cap=selected,
            proposed_target=proposed,
            approved_target=proposed,
            win_probability=0.0,
            risk_reward_ratio=0.0,
            net_expectancy=0.0,
            probability_calibrated=probability_calibrated,
            reasons=("減倉或平倉沿用目前槓桿，不阻擋降低風險",),
        )

    round_trip_cost = 2.0 * (
        _bounded_number(fee_rate, 0.0, 1.0)
        + _bounded_number(slippage_rate, 0.0, 1.0)
    )
    risk = _bounded_number(stop_distance_fraction, 0.0, 1.0) + round_trip_cost
    reward = max(_bounded_number(take_profit_fraction, 0.0, 1.0) - round_trip_cost, 0.0)
    risk_reward = reward / risk if risk > 1e-12 else 0.0
    market_drawdown = _bounded_number(drawdown, 0.0, 1.0)
    spread = _bounded_number(spread_bps, 0.0, 1_000_000.0)

    success_probability_available = False
    if forecast is None or not forecast.available:
        evidence_cap = settings.min_leverage
        win_probability = 0.0
        uncertainty = 1.0
        volatility = float("inf")
        reasons = ["Transformer 預測不可用，槓桿限制為最低值"]
    else:
        success_probability = (
            forecast.long_success_probability
            if proposed > 0
            else forecast.short_success_probability
        )
        if success_probability is None:
            win_probability = 0.0
            uncertainty = forecast.uncertainty
            volatility = forecast.volatility
            evidence_cap = settings.min_leverage
            reasons = [
                "缺少停利先於停損的樣本外校準機率，槓桿限制為最低值"
            ]
        else:
            success_probability_available = True
            win_probability = success_probability
            uncertainty = forecast.uncertainty
            volatility = forecast.volatility
            evidence_cap = settings.max_leverage
            reasons = []
        probability_calibrated = bool(
            success_probability_available
            and forecast.success_probability_calibrated
            and probability_calibrated
        )
        if not probability_calibrated:
            evidence_cap = min(evidence_cap, settings.uncalibrated_max_leverage)
            reasons.append("交易成功機率未校準，限制槓桿上限")

        tier_requirements = (
            (
                3,
                settings.leverage_3_min_probability,
                settings.leverage_3_min_risk_reward,
                settings.leverage_3_max_uncertainty,
                settings.leverage_3_max_drawdown,
                settings.leverage_3_max_spread_bps,
                settings.leverage_3_max_volatility,
            ),
            (
                2,
                settings.leverage_2_min_probability,
                settings.leverage_2_min_risk_reward,
                settings.leverage_2_max_uncertainty,
                settings.leverage_2_max_drawdown,
                settings.leverage_2_max_spread_bps,
                settings.leverage_2_max_volatility,
            ),
        )
        qualified = settings.min_leverage
        for tier, min_probability, min_rr, max_uncertainty, max_dd, max_spread, max_vol in tier_requirements:
            if tier > settings.max_leverage or tier > evidence_cap:
                continue
            checks = (
                win_probability >= min_probability,
                risk_reward >= min_rr,
                uncertainty <= max_uncertainty,
                market_drawdown <= max_dd,
                spread <= max_spread,
                volatility <= max_vol,
            )
            if all(checks):
                qualified = tier
                break
        evidence_cap = min(evidence_cap, max(qualified, settings.min_leverage))

    expectancy = (
        win_probability * reward - (1.0 - win_probability) * risk
        if success_probability_available
        else 0.0
    )
    if (
        settings.require_positive_expectancy
        and success_probability_available
        and expectancy <= 0
    ):
        reasons.append("扣除成本後期望值不為正，拒絕新增風險")
        return DynamicLeverageDecision(
            enabled=True,
            selected_leverage=settings.min_leverage,
            required_leverage=required,
            evidence_cap=evidence_cap,
            proposed_target=proposed,
            approved_target=0.0,
            win_probability=win_probability,
            risk_reward_ratio=risk_reward,
            net_expectancy=expectancy,
            probability_calibrated=probability_calibrated,
            reasons=tuple(dict.fromkeys(reasons)),
        )

    selected = min(max(required, settings.min_leverage), evidence_cap, settings.max_leverage)
    approved_magnitude = min(abs(proposed), margin_fraction * selected)
    approved = math.copysign(approved_magnitude, proposed)
    if required > evidence_cap:
        reasons.append(f"模型證據只允許 {evidence_cap}x，目標部位已縮小")
    if selected > settings.min_leverage:
        reasons.append(f"校準交易成功率、損益比與市場風險符合 {selected}x 門檻")
    elif not reasons:
        reasons.append("證據僅符合最低槓桿門檻")
    return DynamicLeverageDecision(
        enabled=True,
        selected_leverage=selected,
        required_leverage=required,
        evidence_cap=evidence_cap,
        proposed_target=proposed,
        approved_target=approved,
        win_probability=win_probability,
        risk_reward_ratio=risk_reward,
        net_expectancy=expectancy,
        probability_calibrated=probability_calibrated,
        reasons=tuple(dict.fromkeys(reasons)),
    )
