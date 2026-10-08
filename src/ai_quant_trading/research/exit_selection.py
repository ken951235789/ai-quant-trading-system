"""用不同退出契約的合法模型預測選擇停利；不讀實際未來收益。"""

from dataclasses import dataclass
import math

import pandas as pd

from ai_quant_trading.research.candidate_contract import CandidateContract


@dataclass(frozen=True)
class ExitForecast:
    contract: CandidateContract
    candidate_id: str
    side: int
    decision_at: str
    model_information_end_at: str
    checkpoint_sha256: str
    predicted_net_return: float
    success_probability: float
    calibrated: bool
    research_quality_passed: bool = False


def choose_exit(forecasts, *, minimum_probability=.55, minimum_edge_bps=2., require_quality=True):
    """僅選固定停利倍率，不擴大停損或槓桿；所有 R 都須獨立訓練同契約標籤。"""
    if not .55 <= minimum_probability <= 1 or minimum_edge_bps < 2 or not math.isfinite(minimum_edge_bps):
        raise ValueError("此版本禁止降低既有機率與成本後門檻")
    if not forecasts:
        return {"action": "hold", "reason": "no_exit_models", "live_eligible": False}
    reference = forecasts[0]
    common = reference.contract.to_dict()
    common.pop("reward_r")
    seen, eligible = set(), []
    for item in forecasts:
        current = item.contract.to_dict()
        reward_r = current.pop("reward_r")
        if current != common or reward_r in seen or item.candidate_id != reference.candidate_id or item.side != reference.side or item.decision_at != reference.decision_at:
            raise ValueError("退出模型必須對應同一候選、方向、成本及特徵，只允許 reward_r 不同")
        seen.add(reward_r)
        if item.side not in {-1, 1} or len(item.checkpoint_sha256) != 64 or any(c not in "0123456789abcdef" for c in item.checkpoint_sha256):
            raise ValueError("退出模型來源不完整")
        decision, known = pd.Timestamp(item.decision_at), pd.Timestamp(item.model_information_end_at)
        if decision.tzinfo is None or known.tzinfo is None or known >= decision:
            raise ValueError("退出分數不是合法的樣本外預測")
        if not math.isfinite(item.predicted_net_return) or not 0 <= item.success_probability <= 1:
            raise ValueError("退出預測含不合法數值")
        if not item.calibrated or (require_quality and not item.research_quality_passed):
            continue
        if item.success_probability >= minimum_probability and item.predicted_net_return > minimum_edge_bps / 10000:
            eligible.append(item)
    if len(seen) < 2:
        return {"action": "hold", "reason": "need_multiple_exit_contracts", "live_eligible": False}
    if not eligible:
        return {"action": "hold", "reason": "no_qualified_positive_edge", "live_eligible": False}
    chosen = max(eligible, key=lambda item: (item.predicted_net_return, -item.contract.reward_r))
    return {"action": "research_exit_proposal", "reward_r": chosen.contract.reward_r,
            "stop_atr": chosen.contract.stop_atr, "holding_bars": chosen.contract.holding_bars,
            "checkpoint_sha256": chosen.checkpoint_sha256, "contract_sha256": chosen.contract.digest,
            "expected_net_return": chosen.predicted_net_return, "live_eligible": False,
            "reason": "same_candidate_max_calibrated_expected_net_return"}
