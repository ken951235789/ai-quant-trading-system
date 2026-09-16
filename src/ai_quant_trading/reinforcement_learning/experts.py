"""長期與短期交易專家的可重現環境預設。"""

from __future__ import annotations

from dataclasses import dataclass

from ai_quant_trading.reinforcement_learning.config import ExpertKind, PortfolioEnvConfig
from ai_quant_trading.risk import PortfolioRiskConfig, RiskConfig, expert_portfolio_risk


@dataclass(frozen=True, slots=True)
class TradingExpertProfile:
    """將投資週期、Reward、部位限制與模擬風控綁成同一份設定。"""

    kind: ExpertKind
    label: str
    recommended_algorithm: str
    allowed_intervals: tuple[str, ...]
    environment: PortfolioEnvConfig
    paper_risk: RiskConfig
    portfolio_risk: PortfolioRiskConfig
    capital_fraction: float
    reserve_fraction: float


def build_expert_profile(
    kind: ExpertKind,
    *,
    initial_capital: float = 50_000.0,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
) -> TradingExpertProfile:
    """回傳經風險尺度區分的長期或短期專家研究起始值。"""
    if kind == "long_term":
        environment = PortfolioEnvConfig(
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            max_position_fraction=0.15,
            drawdown_penalty=1.5,
            turnover_penalty=0.01,
            max_drawdown_limit=0.12,
            episode_length=512,
            random_start=True,
            holding_period_reference=20,
            expert_kind=kind,
            rebalance_deadband=0.05,
            minimum_holding_bars=5,
            soft_drawdown_limit=0.08,
            soft_drawdown_multiplier=0.5,
            downside_penalty=0.25,
            concentration_penalty=0.002,
        )
        risk = RiskConfig(
            max_risk_per_trade=0.005,
            stop_loss_mode="atr",
            atr_multiplier=3.0,
            take_profit_pct=None,
            max_drawdown_limit=0.12,
            max_position_fraction=0.15,
        )
        return TradingExpertProfile(
            kind,
            "長期交易專家",
            "ppo",
            ("1d", "1wk"),
            environment,
            risk,
            expert_portfolio_risk(kind),
            0.70,
            0.10,
        )
    if kind == "short_term":
        # 淨值報酬已扣真實成本；Reward 只保留小幅換手正則化，避免重複懲罰到空手。
        turnover_regularization = 0.25 * (
            fee_rate + slippage_rate + 0.0002 / 2
        )
        environment = PortfolioEnvConfig(
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            max_position_fraction=0.50,
            allow_short=True,
            max_short_fraction=0.50,
            short_borrow_rate_annual=0.0,
            drawdown_penalty=2.0,
            turnover_penalty=turnover_regularization,
            max_drawdown_limit=0.10,
            episode_length=2_880,
            random_start=True,
            holding_period_reference=96,
            expert_kind=kind,
            rebalance_deadband=0.05,
            minimum_holding_bars=1,
            soft_drawdown_limit=0.06,
            soft_drawdown_multiplier=0.5,
            downside_penalty=0.75,
            concentration_penalty=0.005,
            execution_mode="perpetual",
            normalized_action_space=True,
            include_risk_context=True,
            include_trade_plan_context=True,
            neutral_action_threshold=0.10,
            action_semantics="hold_close_target",
            hold_action_threshold=0.03,
            close_action_threshold=0.10,
            leverage=2.0,
            max_leverage=3.0,
            max_margin_fraction=0.20,
            risk_per_trade=0.0025,
            disaster_stop_atr_multiplier=1.5,
            minimum_stop_distance=0.004,
            maximum_stop_distance=0.015,
            maintenance_margin_rate=0.005,
            liquidation_fee_rate=0.005,
            daily_loss_limit=0.01,
            max_consecutive_losses=3,
            spread_rate=0.0002,
            max_spread_bps=5.0,
            max_slippage_bps=5.0,
            max_atr_fraction=0.05,
            minimum_gross_target_cost_multiple=1.5,
            minimum_net_risk_reward=0.75,
            take_profit_distance=0.015,
            initial_capital_randomization=0.10,
            slippage_randomization=0.0001,
        )
        risk = RiskConfig(
            max_risk_per_trade=0.0025,
            stop_loss_mode="atr",
            atr_multiplier=1.5,
            take_profit_pct=0.015,
            max_drawdown_limit=0.10,
            max_position_fraction=0.50,
        )
        return TradingExpertProfile(
            kind,
            "短期交易專家",
            "sac",
            ("15m",),
            environment,
            risk,
            expert_portfolio_risk(kind),
            0.20,
            0.10,
        )
    return TradingExpertProfile(
        "general",
        "一般研究模型",
        "ppo",
        ("1m", "5m", "15m", "30m", "60m", "1h", "4h", "1d", "1wk"),
        PortfolioEnvConfig(
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
        ),
        RiskConfig(),
        expert_portfolio_risk("general"),
        0.90,
        0.10,
    )


def validate_expert_interval(kind: ExpertKind, interval: str) -> None:
    """防止拿日線長期設定去訓練分鐘資料，或反向混用。"""
    profile = build_expert_profile(kind)
    if kind != "general" and interval.lower() not in profile.allowed_intervals:
        allowed = "、".join(profile.allowed_intervals)
        raise ValueError(f"{profile.label}不支援 {interval}；建議週期為 {allowed}")
