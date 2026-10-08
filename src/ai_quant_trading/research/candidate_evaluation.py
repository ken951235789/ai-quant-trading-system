"""固定候選的三方比較；統計濾網只在 Train 擬合，測試不能調門檻。"""

from dataclasses import replace

import numpy as np

from ai_quant_trading.research.candidate_contract import candidate_outcomes
from ai_quant_trading.research.event_attribution import attribute_events, attribution_summary
from ai_quant_trading.transformer.economics import block_confidence_interval
from ai_quant_trading.transformer.strategy_events import select_event_trades


def fit_statistical_filter(outcomes, train_points, threshold_bps):
    """固定方向 × 已知小時趨勢分組，以 20 個先驗樣本收縮，避免稀少組過度極端。"""
    train = outcomes.set_index("endpoint").loc[list(train_points)].copy()
    if train.empty or train.net_return.isna().any():
        raise ValueError("統計濾網缺少 Train 標籤")
    train["positive"] = train.net_return > threshold_bps / 10000
    mean, probability = float(train.net_return.mean()), float(train.positive.mean())
    cells = {}
    for key, group in train.groupby(["side", "entry_regime"]):
        n = len(group)
        cells[f"{int(key[0])}:{int(key[1])}"] = {"n": n,
            "mean": float((group.net_return.sum() + 20 * mean) / (n + 20)),
            "probability": float((group.positive.sum() + 20 * probability) / (n + 20))}
    return {"method": "train_only_side_regime_shrink20_v1", "train_samples": len(train),
            "train_last_endpoint": int(max(train_points)), "mean": mean,
            "probability": probability, "cells": cells}


def apply_statistical_filter(predictions, fitted, horizon):
    result = predictions.copy()
    cells = [fitted["cells"].get(f"{int(s)}:{int(r)}", fitted)
             for s, r in zip(result.event_side, result.event_entry_regime)]
    result[f"predicted_return_{horizon}"] = [c["mean"] for c in cells]
    result[f"tradeability_probability_{horizon}"] = [c["probability"] for c in cells]
    return result


def score_diagnostics(predictions, horizon, edges):
    """分數邊界由 Selection 凍結；可靠度使用事前固定的十等寬機率區間。"""
    actual = predictions[f"actual_return_{horizon}"].to_numpy()
    predicted = predictions[f"predicted_return_{horizon}"].to_numpy()
    probability = predictions[f"tradeability_probability_{horizon}"].to_numpy()
    positive = predictions[f"actual_tradeability_{horizon}"].to_numpy()
    reliability, bins = [], []
    for bucket in range(10):
        mask = np.minimum((probability * 10).astype(int), 9) == bucket
        if mask.any():
            reliability.append({"bin": bucket, "n": int(mask.sum()),
                "predicted": float(probability[mask].mean()), "observed": float(positive[mask].mean())})
    assigned = np.searchsorted(np.asarray(edges), predicted, side="right")
    for bucket in range(len(edges) + 1):
        mask = assigned == bucket
        if mask.any():
            bins.append({"bin": bucket, "n_overlapping_candidates": int(mask.sum()),
                         "mean_net": float(actual[mask].mean())})
    lower = predictions[f"predicted_return_q10_{horizon}"].to_numpy()
    upper = predictions[f"predicted_return_q90_{horizon}"].to_numpy()
    return {"selection_score_boundaries": list(edges), "score_groups": bins,
        "reliability": reliability,
        "ece_10bins": float(sum(r["n"] * abs(r["predicted"] - r["observed"]) for r in reliability) / len(actual)),
        "quantile_10_90_coverage": float(((actual >= lower) & (actual <= upper)).mean()),
        "quantile_crossing_fraction": float((lower > upper).mean()),
        "scope": "候選可能重疊；分組與校準是描述性診斷，不等於獨立交易或成本後獲利"}


def compare_execution_policies(frame, predictions, fitted, contract, source_sha256,
                               minimum_edge_bps, output_dir, *, gate_ablation=False):
    """零、基本與壓力成本各自重播；訊號分數凍結，重新排程持倉與冷卻。"""
    output_dir.mkdir(parents=True, exist_ok=False)
    horizon = contract.holding_bars
    scenarios = {"zero": replace(contract, fee_bps=0., slippage_bps=0., spread_bps=0., funding_reserve_bps=0.),
                 "base": contract, "stress": replace(contract, slippage_bps=contract.slippage_bps + 3.)}
    records = {}
    for name, scenario in scenarios.items():
        outcomes = candidate_outcomes(frame, scenario)
        attributed = attribute_events(frame, outcomes, scenario, source_sha256)
        lookup = outcomes.set_index("endpoint")
        available = lookup.loc[predictions.endpoint.astype(int)]
        current = predictions.copy()
        current["event_exit_endpoint"] = available.exit_endpoint.to_numpy()
        current[f"actual_return_{horizon}"] = available.net_return.to_numpy()
        for field in ("entry_price", "exit_price", "fee_return", "funding_return"):
            current[f"event_{field}"] = available[field].to_numpy()
        policies = {"no_ai": current, "statistical": apply_statistical_filter(current, fitted, horizon),
                    "transformer": current}
        if gate_ablation:
            policies.update(transformer_probability=current, transformer_return=current)
        for policy, values in policies.items():
            mode = {"transformer_probability": "probability", "transformer_return": "return"}.get(policy)
            selected, reasons = select_event_trades(values, horizon, scenario.event_config(),
                minimum_edge_bps, filtered=policy != "no_ai", filter_mode=mode)
            trades = attributed.set_index("endpoint", drop=False).loc[selected.endpoint.astype(int)].copy()
            trades.to_csv(output_dir / f"{policy}_{name}_trades.csv", index=False)
            summary = attribution_summary(trades)
            returns = trades.net_return.to_numpy()
            # 少量成交的區塊抽樣容易退化；八筆也只是描述區間的最低計算量。
            ci = block_confidence_interval(returns) if len(returns) >= 8 else (None, None)
            summary.update(ci_low=ci[0], ci_high=ci[1], rejections=reasons,
                scenario=name, policy=policy, assumption="market_orders_no_maker_fill_claim",
                account_equity_validated=False)
            records[f"{policy}_{name}"] = summary
    return records
