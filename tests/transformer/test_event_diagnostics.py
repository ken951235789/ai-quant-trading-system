"""驗證事件診斷不從 Test 擬合基準，也不以未來答案決定進場。"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from ai_quant_trading.transformer.config import TemporalTransformerConfig, TransformerTrainingConfig
from ai_quant_trading.transformer.event_diagnostics import diagnose_events, prediction_skill, training_reference
from ai_quant_trading.transformer.strategy_events import StrategyEventConfig, select_event_trades
from ai_quant_trading.transformer.training import _task_loss


def predictions():
    return pd.DataFrame({"series_index": [0] * 4, "endpoint": [0, 10, 20, 30],
                         "event_exit_endpoint": [2, 12, 22, 32],
                         "predicted_return_32": [0.01, -0.02, 0.01, 0.01],
                         "actual_return_32": [0.02, -0.01, -0.02, 0.03],
                         "tradeability_probability_32": [0.3, 0.3, 0.8, 0.8],
                         "actual_tradeability_32": [1, 0, 0, 1]})


def test_reference_baselines_are_frozen_train_values():
    frame = predictions()
    reference = training_reference(0.1, 1, 4)
    result = prediction_skill(frame, 32, reference)
    assert result["event_train_constant_mse"] == pytest.approx(((frame.actual_return_32 - 0.1) ** 2).mean())
    assert result["event_train_constant_brier"] == pytest.approx(((frame.actual_tradeability_32 - 0.25) ** 2).mean())
    assert reference["return_mean"] == 0.1
    constant = frame.assign(predicted_return_32=0.1, tradeability_probability_32=0.25)
    assert prediction_skill(constant, 32, reference)["event_prediction_skill_score"] == pytest.approx(0)


def test_rejections_and_future_label_invariance():
    frame = predictions()
    selected, reasons = select_event_trades(frame, 32, StrategyEventConfig(), 2, filtered=True)
    assert list(selected.endpoint) == [20, 30]
    assert reasons == {"position_or_cooldown": 0, "probability_only": 1, "return_only": 0, "both": 1}
    changed = frame.assign(actual_return_32=[5, 3, -7, 9])
    assert list(select_event_trades(changed, 32, StrategyEventConfig(), 2, filtered=True)[0].endpoint) == [20, 30]


def test_no_trade_is_not_pass_or_zero_expectancy_evidence():
    frame = predictions().assign(tradeability_probability_32=0.1)
    report = diagnose_events(frame, 32, StrategyEventConfig(), 2, training_reference(0, 1, 2),
                             slippage_bps_per_side=2)
    assert report["policies"]["filtered"]["trades"] == 0
    assert report["policies"]["filtered"]["mean_net_return"] is None
    assert report["research_gate"]["passed"] is False
    assert report["live_eligible"] is False


def test_invalid_reference_and_prediction_rejected():
    with pytest.raises(ValueError):
        training_reference(0, 1, 0)
    with pytest.raises(ValueError):
        prediction_skill(predictions().assign(actual_return_32=np.inf), 32, training_reference(0, 1, 2))
    with pytest.raises(ValueError):
        TransformerTrainingConfig(return_loss_kind="not_supported")
    with pytest.raises(ValueError):
        TransformerTrainingConfig(checkpoint_metric="event_prediction_skill_score")


def test_cost_decomposition_for_long_and_short_uses_same_fills():
    frame = predictions()
    side = np.array([1, -1, 1, -1])
    friction = 0.00025
    entry = 100 * (1 + side * friction)
    exit_fill = np.array([102, 98, 101, 99]) * (1 - side * friction)
    fee = 0.0005 * (1 + exit_fill / entry)
    funding = 0.0001
    net = side * (exit_fill / entry - 1) - fee - funding
    frame = frame.assign(event_entry_price=entry, event_exit_price=exit_fill,
                         event_fee_return=fee, event_funding_return=funding,
                         event_side=side, actual_return_32=net, actual_tradeability_32=(net > 0).astype(int))
    report = diagnose_events(frame, 32, StrategyEventConfig(), 2, training_reference(0, 1, 2),
                             slippage_bps_per_side=2)
    baseline = report["policies"]["baseline"]
    assert baseline["gross_same_fills_mean"] == pytest.approx(0.015)
    assert baseline["gross_same_fills_mean"] - baseline["mean_net_return"] == pytest.approx(
        baseline["fee_mean"] + baseline["funding_reserve_mean"] + baseline["slippage_and_spread_mean"])
    with pytest.raises(ValueError):
        diagnose_events(frame.assign(event_side=0), 32, StrategyEventConfig(), 2,
                        training_reference(0, 1, 2), slippage_bps_per_side=2)


def test_mean_loss_is_stationary_at_mean_not_robust_center():
    config = TransformerTrainingConfig(return_loss_kind="mse")
    model = TemporalTransformerConfig(return_horizons=(32,))
    mean = torch.zeros((4, 1), requires_grad=True)
    output = {"future_returns": mean, "volatility": torch.zeros(4), "regime_logits": torch.zeros(4, 3)}
    batch = {"future_returns": torch.tensor([[-1.], [-1.], [-1.], [3.]]),
             "volatility": torch.zeros(4), "regime": torch.zeros(4, dtype=torch.long)}
    _, losses = _task_loss(output, batch, config, model)
    losses["return_loss"].backward()
    assert mean.grad.sum().item() == pytest.approx(0)
    mean.grad.zero_()
    _, losses = _task_loss(output, batch, replace(config, return_loss_kind="smooth_l1"), model)
    losses["return_loss"].backward()
    assert mean.grad.sum().item() > 0
    assert TransformerTrainingConfig().return_loss_kind == "smooth_l1"
