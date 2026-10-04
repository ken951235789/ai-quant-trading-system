"""事件模型診斷：區分預測能力、拒單原因與成交成本，不用 Test 重選門檻。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ai_quant_trading.transformer.economics import block_confidence_interval
from ai_quant_trading.transformer.strategy_events import (
    StrategyEventConfig, evaluate_event_predictions, select_event_trades,
)


def training_reference(return_mean: float, positives: float, total: float) -> dict[str, float]:
    """參考值只能由 Train 標籤建立，不能以 Validation／Test 的平均值冒充基準。"""
    if not np.isfinite([return_mean, positives, total]).all() or total <= 0 or not 0 <= positives <= total:
        raise ValueError("事件 Train 基準不合法")
    return {"return_mean": float(return_mean), "positive_probability": float(positives / total),
            "train_samples": float(total)}


def prediction_skill(predictions: pd.DataFrame, horizon: int,
                     reference: dict[str, float]) -> dict[str, float]:
    """同一評估區間比較模型與事先凍結的 Train 常數；負 skill 代表比常數差。"""
    columns = [f"predicted_return_{horizon}", f"actual_return_{horizon}",
               f"tradeability_probability_{horizon}", f"actual_tradeability_{horizon}"]
    if set(columns).difference(predictions) or predictions.empty:
        raise ValueError("事件預測能力診斷缺少有效樣本")
    data = predictions[columns].to_numpy(dtype=float)
    if (not np.isfinite(data).all() or not np.isin(data[:, 3], [0, 1]).all()
            or ((data[:, 2] < 0) | (data[:, 2] > 1)).any()):
        raise ValueError("事件預測能力診斷含不合法數值")
    predicted, actual, probability, positive = data.T
    mean, prior = reference["return_mean"], reference["positive_probability"]
    if not np.isfinite([mean, prior]).all() or not 0 <= prior <= 1:
        raise ValueError("事件 Train 參考值不合法")
    mse = float(np.mean((predicted - actual) ** 2))
    baseline_mse = float(np.mean((mean - actual) ** 2))
    brier = float(np.mean((probability - positive) ** 2))
    baseline_brier = float(np.mean((prior - positive) ** 2))

    def skill(loss: float, baseline: float) -> float:
        # 退化區段沒有可辨識優勢，不能讓 0/0 變成選模的正收益證據。
        return 1 - loss / baseline if baseline > 1e-12 else (0.0 if loss <= 1e-12 else -1.0)

    return_skill, probability_skill = skill(mse, baseline_mse), skill(brier, baseline_brier)
    rank_actual, rank_predicted = pd.Series(actual).rank(), pd.Series(predicted).rank()
    rank_correlation = rank_actual.corr(rank_predicted) if rank_actual.nunique() > 1 and rank_predicted.nunique() > 1 else 0.0
    correlation = float(np.corrcoef(predicted, actual)[0, 1]) if np.std(predicted) > 1e-12 and np.std(actual) > 1e-12 else 0.0
    return {
        "event_return_mse": mse, "event_train_constant_mse": baseline_mse,
        "event_return_skill": return_skill, "event_probability_brier": brier,
        "event_train_constant_brier": baseline_brier, "event_probability_skill": probability_skill,
        "event_prediction_skill_score": 0.5 * (return_skill + probability_skill),
        "event_return_correlation": correlation,
        "event_return_rank_correlation": float(rank_correlation),
        "event_predict_always_no_trade_accuracy": float((positive == 0).mean()),
        "event_probability_min": float(probability.min()),
        "event_probability_max": float(probability.max()),
        "event_predicted_return_min": float(predicted.min()),
        "event_predicted_return_max": float(predicted.max()),
    }


def _trade_statistics(trades: pd.DataFrame, horizon: int) -> dict[str, object]:
    values = trades[f"actual_return_{horizon}"].to_numpy(dtype=float)
    lower, upper = block_confidence_interval(values)
    return {"trades": len(values), "mean_net_return": float(values.mean()) if len(values) else None,
            "net_expectancy_ci_low": lower if len(values) >= 2 else None,
            "net_expectancy_ci_high": upper if len(values) >= 2 else None,
            "win_rate": float((values > 0).mean()) if len(values) else None}


def diagnose_events(predictions: pd.DataFrame, horizon: int, config: StrategyEventConfig,
                    minimum_edge_bps: float, reference: dict[str, float], *,
                    slippage_bps_per_side: float, minimum_trades: int = 30) -> dict[str, object]:
    """事後描述不是可即時執行的分位選股；不修改任何交易條件。"""
    if not np.isfinite(minimum_trades) or minimum_trades < 2 or int(minimum_trades) != minimum_trades:
        raise ValueError("最低交易樣本數必須為至少 2 的整數")
    if not np.isfinite(slippage_bps_per_side) or slippage_bps_per_side < 0:
        raise ValueError("滑價假設必須為有限且非負的數值")
    metrics = evaluate_event_predictions(predictions, horizon, config, minimum_edge_bps)
    metrics.update(prediction_skill(predictions, horizon, reference))
    policies = {}
    for name in ("baseline", "filtered"):
        trades, reasons = select_event_trades(predictions, horizon, config, minimum_edge_bps,
                                              filtered=name == "filtered")
        values = _trade_statistics(trades, horizon)
        values["rejections"] = reasons
        cost_columns = {"event_entry_price", "event_exit_price", "event_fee_return",
                        "event_funding_return", "event_side"}
        if len(trades) and cost_columns.issubset(trades):
            if (not np.isfinite(trades[sorted(cost_columns)].to_numpy(dtype=float)).all()
                    or not trades["event_side"].isin([-1, 1]).all()
                    or (trades[["event_entry_price", "event_exit_price"]] <= 0).any().any()
                    or (trades[["event_fee_return", "event_funding_return"]] < 0).any().any()):
                raise ValueError("成交成本拆解含不合法價格、方向或成本準備金")
            friction = (slippage_bps_per_side + config.spread_bps / 2) / 10_000
            if not np.isfinite(friction) or not 0 <= friction < 1:
                raise ValueError("摩擦成本參數不合法")
            side = trades["event_side"].to_numpy()
            entry_reference = trades["event_entry_price"].to_numpy() / (1 + side * friction)
            exit_reference = trades["event_exit_price"].to_numpy() / (1 - side * friction)
            gross = side * (exit_reference / entry_reference - 1)
            net = trades[f"actual_return_{horizon}"].to_numpy()
            fees = trades["event_fee_return"].to_numpy()
            funding = trades["event_funding_return"].to_numpy()
            values.update({"gross_same_fills_mean": float(gross.mean()),
                           "fee_mean": float(fees.mean()), "funding_reserve_mean": float(funding.mean()),
                           "slippage_and_spread_mean": float((gross - net - fees - funding).mean()),
                           "extra_6bps_net_mean": float(net.mean() - 0.0006)})
        policies[name] = values
    filtered = policies["filtered"]
    checks = {
        "minimum_filtered_trades": filtered["trades"] >= minimum_trades,
        "positive_filtered_expectancy_lower_bound": filtered["net_expectancy_ci_low"] is not None and filtered["net_expectancy_ci_low"] > 0,
        "return_beats_train_constant": metrics["event_return_skill"] > 0,
        "probability_beats_train_constant": metrics["event_probability_skill"] > 0,
    }
    descriptive = []
    frame = predictions.copy()
    # 排序組只用來描述已完成研究，不能回頭拿測試分位當成進場門檻。
    frame["score_quintile"] = pd.qcut(frame[f"predicted_return_{horizon}"].rank(method="first"),
                                      min(5, len(frame)), labels=False)
    if "timestamp_ns" in frame:
        frame["utc_month"] = pd.to_datetime(frame["timestamp_ns"], utc=True).dt.strftime("%Y-%m")
    for column in ("score_quintile", "event_side", "event_exit_reason_code", "utc_month"):
        if column not in frame:
            continue
        for value, group in frame.groupby(column, sort=True):
            descriptive.append({"group": column, "bucket": str(value), "candidates": len(group),
                                "mean_net_return": float(group[f"actual_return_{horizon}"].mean())})
    return {"schema_version": 1, "reference_source": "train_only", "reference": reference,
            "metrics": metrics, "policies": policies,
            "research_gate": {"passed": all(checks.values()), "checks": checks},
            "candidate_groups_descriptive_only": descriptive, "live_eligible": False,
            "scope": "候選群組可能重疊；policies 才是不重疊成交。成本拆解固定原成交路徑，不是零成本重跑策略。不是帳戶或稅後報酬。"}
