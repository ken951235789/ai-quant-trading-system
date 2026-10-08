"""停利選擇契約、樣本外時間與既有風險閘門。"""

from dataclasses import replace

import pytest

from ai_quant_trading.research.candidate_contract import CandidateContract
from ai_quant_trading.research.exit_selection import ExitForecast, choose_exit


def models():
    base = ExitForecast(CandidateContract(reward_r=1.), "candidate", 1,
        "2026-02-01T00:00:00Z", "2026-01-31T00:00:00Z", "a"*64, .001, .6, True, True)
    return [base, replace(base, contract=CandidateContract(reward_r=3.), predicted_net_return=.002)]


def test_exit_uses_predicted_not_realized_and_preserves_stop():
    result = choose_exit(models())
    assert result["reward_r"] == 3 and result["stop_atr"] == 2
    assert not result["live_eligible"]


def test_exit_missing_quality_holds():
    assert choose_exit([replace(m, research_quality_passed=False) for m in models()])["action"] == "hold"
    assert choose_exit([replace(m, success_probability=.54) for m in models()])["action"] == "hold"


def test_exit_rejects_incompatible_cost_or_future_training():
    one, two = models()
    with pytest.raises(ValueError):
        choose_exit([one, replace(two, contract=CandidateContract(fee_bps=4))])
    with pytest.raises(ValueError):
        choose_exit([one, replace(two, model_information_end_at=two.decision_at)])
    with pytest.raises(ValueError):
        choose_exit(models(), minimum_probability=.5)
