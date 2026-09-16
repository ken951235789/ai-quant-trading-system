"""SAC 動作、Reward 與進出場閘門的可重現消融方案。"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ai_quant_trading.reinforcement_learning.config import PortfolioEnvConfig


@dataclass(frozen=True, slots=True)
class SACEnvironmentAblation:
    """一組只能用驗證集比較、不可偷看 final holdout 的環境設定。"""

    name: str
    changed_component: str
    description: str
    environment: PortfolioEnvConfig


def build_sac_environment_ablations(
    base: PortfolioEnvConfig,
) -> tuple[SACEnvironmentAblation, ...]:
    """建立單變因消融矩陣，分辨空手究竟來自動作、Reward 或硬閘門。"""
    if not base.allow_short or not base.normalized_action_space:
        raise ValueError("SAC 多空消融需要 allow_short 與 normalized_action_space")

    hold_close = replace(
        base,
        action_semantics="hold_close_target",
        hold_action_threshold=0.03,
        close_action_threshold=max(base.neutral_action_threshold, 0.10),
    )
    # 真實手續費、滑價與點差已經反映在資產淨值；這裡只留很小的換手正則化。
    one_way_cost = base.fee_rate + base.slippage_rate + base.spread_rate / 2
    cost_aligned_turnover = min(base.turnover_penalty, one_way_cost * 0.25)
    return (
        SACEnvironmentAblation(
            "legacy_control",
            "control",
            "保留目前模型的目標部位語意與所有 Reward／進場閘門。",
            base,
        ),
        SACEnvironmentAblation(
            "hold_close_semantics",
            "action_semantics",
            "只把空手等待、續抱與主動平倉拆開，其餘設定完全不變。",
            hold_close,
        ),
        SACEnvironmentAblation(
            "no_extra_turnover_penalty",
            "turnover_reward",
            "保留真實交易成本，但移除 Reward 內重複的額外換手懲罰。",
            replace(hold_close, turnover_penalty=0.0),
        ),
        SACEnvironmentAblation(
            "cost_aligned_turnover_penalty",
            "turnover_reward",
            "把額外換手正則化限制為單邊執行成本的四分之一。",
            replace(hold_close, turnover_penalty=cost_aligned_turnover),
        ),
        SACEnvironmentAblation(
            "pnl_only_reward",
            "reward_shape",
            "僅保留扣除真實成本後的淨值報酬與硬性風控終止，用來檢查 Reward 是否過度懲罰。",
            replace(
                hold_close,
                turnover_penalty=0.0,
                drawdown_penalty=0.0,
                downside_penalty=0.0,
                concentration_penalty=0.0,
            ),
        ),
        SACEnvironmentAblation(
            "entry_exit_filters_off",
            "entry_exit_gate",
            "關閉毛利成本倍數、損益比與調倉死區，只用來量測硬閘門造成的空手比例。",
            replace(
                hold_close,
                minimum_gross_target_cost_multiple=0.0,
                minimum_net_risk_reward=0.0,
                rebalance_deadband=0.0,
                minimum_holding_bars=0,
            ),
        ),
        SACEnvironmentAblation(
            "relaxed_entry_exit_filters",
            "entry_exit_gate",
            "保留成本與損益比概念，但採較寬鬆候選門檻供驗證集比較。",
            replace(
                hold_close,
                minimum_gross_target_cost_multiple=min(
                    base.minimum_gross_target_cost_multiple,
                    1.50,
                ),
                minimum_net_risk_reward=min(base.minimum_net_risk_reward, 0.75),
                turnover_penalty=cost_aligned_turnover,
            ),
        ),
    )
