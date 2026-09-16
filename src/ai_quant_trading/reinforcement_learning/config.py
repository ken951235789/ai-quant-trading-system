"""Step 9 強化學習資料切分與交易環境參數。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


ExpertKind = Literal["general", "long_term", "short_term"]
ExecutionMode = Literal["spot", "perpetual"]
ActionSemantics = Literal["legacy_target", "hold_close_target"]


@dataclass(frozen=True, slots=True)
class RLSplitConfig:
    """依時間順序切分強化學習訓練、驗證與測試資料。"""

    train_fraction: float = 0.6
    validation_fraction: float = 0.2
    min_rows_per_split: int = 20

    def __post_init__(self) -> None:
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction 必須介於 0 與 1 之間")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction 必須介於 0 與 1 之間")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("訓練比例加驗證比例必須小於 1")
        if self.min_rows_per_split <= 0:
            raise ValueError("min_rows_per_split 必須大於 0")

    @property
    def test_fraction(self) -> float:
        return 1 - self.train_fraction - self.validation_fraction

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PortfolioEnvConfig:
    """單一標的多空 Portfolio Environment 參數。"""

    initial_capital: float = 1000.0
    fee_rate: float = 0.001
    slippage_rate: float = 0.0005
    max_position_fraction: float = 1.0
    allow_short: bool = False
    max_short_fraction: float = 0.0
    short_borrow_rate_annual: float = 0.0
    drawdown_penalty: float = 0.5
    turnover_penalty: float = 0.001
    max_drawdown_limit: float = 0.30
    reward_scale: float = 1.0
    episode_length: int | None = None
    random_start: bool = False
    holding_period_reference: int = 8
    include_position_context: bool = True
    expert_kind: ExpertKind = "general"
    rebalance_deadband: float = 0.0
    minimum_holding_bars: int = 0
    soft_drawdown_limit: float | None = None
    soft_drawdown_multiplier: float = 0.5
    drawdown_curve_exponent: float | None = 1.0
    downside_penalty: float = 0.0
    concentration_penalty: float = 0.0
    risk_termination_penalty: float = 1.0
    execution_mode: ExecutionMode = "spot"
    normalized_action_space: bool = False
    include_risk_context: bool = False
    include_trade_plan_context: bool = False
    neutral_action_threshold: float = 0.0
    action_semantics: ActionSemantics = "legacy_target"
    hold_action_threshold: float = 0.03
    close_action_threshold: float = 0.10
    leverage: float = 1.0
    max_leverage: float = 3.0
    max_margin_fraction: float = 1.0
    risk_per_trade: float = 1.0
    disaster_stop_atr_multiplier: float = 1.5
    minimum_stop_distance: float = 0.0025
    maximum_stop_distance: float = 0.03
    maintenance_margin_rate: float = 0.005
    liquidation_fee_rate: float = 0.005
    daily_loss_limit: float = 1.0
    max_consecutive_losses: int = 0
    spread_rate: float = 0.0
    max_spread_bps: float | None = None
    max_slippage_bps: float | None = None
    max_atr_fraction: float | None = None
    funding_interval_hours: float = 8.0
    minimum_gross_target_cost_multiple: float = 0.0
    minimum_net_risk_reward: float = 0.0
    take_profit_distance: float | None = None
    initial_capital_randomization: float = 0.0
    slippage_randomization: float = 0.0

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital 必須大於 0")
        if not 0 <= self.fee_rate < 1:
            raise ValueError("fee_rate 必須介於 0 與 1 之間")
        if not 0 <= self.slippage_rate < 1:
            raise ValueError("slippage_rate 必須介於 0 與 1 之間")
        if not 0 < self.max_position_fraction <= 1:
            raise ValueError("max_position_fraction 必須大於 0 且不超過 1")
        if not 0 <= self.max_short_fraction <= 1:
            raise ValueError("max_short_fraction 必須介於 0 與 1 之間")
        if self.allow_short and self.max_short_fraction <= 0:
            raise ValueError("啟用作空時 max_short_fraction 必須大於 0")
        if not self.allow_short and self.max_short_fraction != 0:
            raise ValueError("未啟用作空時 max_short_fraction 必須為 0")
        if self.short_borrow_rate_annual < 0:
            raise ValueError("short_borrow_rate_annual 不可小於 0")
        if self.drawdown_penalty < 0 or self.turnover_penalty < 0:
            raise ValueError("Reward 懲罰係數不可小於 0")
        if not 0 < self.max_drawdown_limit <= 1:
            raise ValueError("max_drawdown_limit 必須介於 0 與 1 之間")
        if self.reward_scale <= 0:
            raise ValueError("reward_scale 必須大於 0")
        if self.episode_length is not None and self.episode_length <= 0:
            raise ValueError("episode_length 必須大於 0，或設為 None")
        if self.holding_period_reference <= 0:
            raise ValueError("holding_period_reference 必須大於 0")
        if self.expert_kind not in {"general", "long_term", "short_term"}:
            raise ValueError("expert_kind 只支援 general、long_term 或 short_term")
        if not 0 <= self.rebalance_deadband < 1:
            raise ValueError("rebalance_deadband 必須介於 0（含）與 1（不含）之間")
        if self.minimum_holding_bars < 0:
            raise ValueError("minimum_holding_bars 不可小於 0")
        if self.soft_drawdown_limit is not None:
            if not 0 < self.soft_drawdown_limit < self.max_drawdown_limit:
                raise ValueError("soft_drawdown_limit 必須大於 0 且小於硬性最大回撤")
        if not 0 < self.soft_drawdown_multiplier <= 1:
            raise ValueError("soft_drawdown_multiplier 必須大於 0 且不超過 1")
        if self.drawdown_curve_exponent is not None and self.drawdown_curve_exponent <= 0:
            raise ValueError("drawdown_curve_exponent 必須大於 0 或設為 None")
        if self.downside_penalty < 0 or self.concentration_penalty < 0:
            raise ValueError("下行與集中度懲罰不可小於 0")

        if self.risk_termination_penalty < 0:
            raise ValueError("風控終止懲罰不可小於 0")
        if self.execution_mode not in {"spot", "perpetual"}:
            raise ValueError("execution_mode 只支援 spot 或 perpetual")
        if not 0 <= self.neutral_action_threshold < 1:
            raise ValueError("neutral_action_threshold 必須介於 0（含）與 1（不含）")
        if self.action_semantics not in {"legacy_target", "hold_close_target"}:
            raise ValueError("action_semantics 只支援 legacy_target 或 hold_close_target")
        if not 0 <= self.hold_action_threshold < self.close_action_threshold < 1:
            raise ValueError("SAC 動作門檻必須符合 0 <= 續抱門檻 < 平倉門檻 < 1")
        if self.leverage <= 0 or self.max_leverage <= 0:
            raise ValueError("leverage 與 max_leverage 必須大於 0")
        if self.leverage > self.max_leverage:
            raise ValueError("leverage 不可高於 max_leverage")
        if not 0 < self.max_margin_fraction <= 1:
            raise ValueError("max_margin_fraction 必須大於 0 且不超過 1")
        if not 0 < self.risk_per_trade <= 1:
            raise ValueError("risk_per_trade 必須大於 0 且不超過 1")
        if self.disaster_stop_atr_multiplier <= 0:
            raise ValueError("disaster_stop_atr_multiplier 必須大於 0")
        if not 0 < self.minimum_stop_distance <= self.maximum_stop_distance < 1:
            raise ValueError("停損距離必須符合 0 < 最小值 <= 最大值 < 1")
        if not 0 <= self.maintenance_margin_rate < 1:
            raise ValueError("maintenance_margin_rate 必須介於 0（含）與 1（不含）")
        if not 0 <= self.liquidation_fee_rate < 1:
            raise ValueError("liquidation_fee_rate 必須介於 0（含）與 1（不含）")
        if not 0 < self.daily_loss_limit <= 1:
            raise ValueError("daily_loss_limit 必須大於 0 且不超過 1")
        if self.max_consecutive_losses < 0:
            raise ValueError("max_consecutive_losses 不可小於 0")
        if not 0 <= self.spread_rate < 1:
            raise ValueError("spread_rate 必須介於 0（含）與 1（不含）")
        if self.max_spread_bps is not None and self.max_spread_bps <= 0:
            raise ValueError("max_spread_bps 必須大於 0 或設為 None")
        if self.max_slippage_bps is not None and self.max_slippage_bps <= 0:
            raise ValueError("max_slippage_bps 必須大於 0 或設為 None")
        if self.max_atr_fraction is not None and not 0 < self.max_atr_fraction < 1:
            raise ValueError("max_atr_fraction 必須介於 0 與 1 之間或設為 None")
        if self.funding_interval_hours <= 0:
            raise ValueError("funding_interval_hours 必須大於 0")
        if self.minimum_gross_target_cost_multiple < 0:
            raise ValueError("minimum_gross_target_cost_multiple 不可小於 0")
        if self.minimum_net_risk_reward < 0:
            raise ValueError("minimum_net_risk_reward 不可小於 0")
        if self.take_profit_distance is not None and not 0 < self.take_profit_distance < 1:
            raise ValueError("take_profit_distance 必須介於 0 與 1，或設為 None")
        if not 0 <= self.initial_capital_randomization < 1:
            raise ValueError("initial_capital_randomization 必須介於 0（含）與 1（不含）")
        if self.slippage_randomization < 0:
            raise ValueError("slippage_randomization 不可小於 0")
        if self.slippage_rate + self.slippage_randomization >= 1:
            raise ValueError("slippage_rate 加上 slippage_randomization 必須小於 1")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
