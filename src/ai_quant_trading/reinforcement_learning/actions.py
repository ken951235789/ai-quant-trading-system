"""SAC 連續動作在訓練、回測與執行期共用的唯一映射契約。"""

from __future__ import annotations

import numpy as np

from ai_quant_trading.reinforcement_learning.config import PortfolioEnvConfig


def map_continuous_action(
    action_value: float,
    config: PortfolioEnvConfig,
    *,
    current_position: float = 0.0,
) -> tuple[float, str]:
    """回傳目標曝險與可稽核意圖，避免訓練和實盤各自解讀 SAC 動作。"""
    if not np.isfinite(action_value):
        return 0.0, "INVALID_FLATTEN"
    short_limit = config.max_short_fraction if config.allow_short else 0.0
    if config.normalized_action_space:
        normalized = float(np.clip(action_value, -1.0 if config.allow_short else 0.0, 1.0))
        directional_target = (
            normalized * config.max_position_fraction
            if normalized >= 0
            else normalized * config.max_short_fraction
        )
    else:
        normalized = float(np.clip(action_value, -short_limit, config.max_position_fraction))
        directional_target = normalized

    if config.action_semantics == "continuous_target":
        target = float(directional_target)
        current = float(np.clip(current_position, -short_limit, config.max_position_fraction))
        # 只抑制小於實際再平衡成本的微調；不再把一整段 SAC 動作硬切成空手。
        if abs(target - current) <= config.rebalance_deadband:
            return current, "HOLD_POSITION" if abs(current) > 1e-12 else "WAIT_FLAT"
        if abs(target) <= 1e-12:
            return 0.0, "CLOSE_POSITION" if abs(current) > 1e-12 else "WAIT_FLAT"
        return target, "TARGET_LONG" if target > 0 else "TARGET_SHORT"

    if config.action_semantics == "legacy_target":
        if abs(normalized) <= config.neutral_action_threshold:
            return 0.0, "FLAT"
        return (
            float(directional_target),
            "TARGET_LONG" if directional_target > 0 else "TARGET_SHORT",
        )

    magnitude = abs(normalized)
    if magnitude <= config.hold_action_threshold:
        target = float(np.clip(current_position, -short_limit, config.max_position_fraction))
        return target, "HOLD_POSITION" if abs(target) > 1e-12 else "WAIT_FLAT"
    if magnitude <= config.close_action_threshold:
        return 0.0, "CLOSE_POSITION" if abs(current_position) > 1e-12 else "WAIT_FLAT"
    return (
        float(directional_target),
        "TARGET_LONG" if directional_target > 0 else "TARGET_SHORT",
    )
