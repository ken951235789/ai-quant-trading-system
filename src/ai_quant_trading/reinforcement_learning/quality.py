"""RL 模型進入 Testnet 與 Live 前的樣本外品質門檻。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True, slots=True)
class RLQualityReport:
    """RL 訓練成品是否達到最低自動交易研究門檻。"""

    eligible: bool
    validation_return: float | None
    test_return: float | None
    max_drawdown: float | None
    test_steps: int
    test_trades: int
    positive_market_ratio: float | None
    test_sharpe_ratio: float | None
    fee_to_initial_capital: float | None
    action_saturation_ratio: float | None
    test_profit_factor: float | None
    test_expectancy: float | None
    liquidation_count: int
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RLEvaluationQualityReport:
    """單一不可調參評估區間是否達到最低研究門檻。"""

    eligible: bool
    total_return: float | None
    max_drawdown: float | None
    steps: int
    trades: int
    sharpe_ratio: float | None
    fee_to_initial_capital: float | None
    action_saturation_ratio: float | None
    profit_factor: float | None
    expectancy: float | None
    liquidation_count: int
    reasons: tuple[str, ...]


def _finite_metric(metrics: dict[str, Any], name: str) -> float | None:
    value = metrics.get(name)
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def assess_rl_evaluation_quality(
    metrics: dict[str, Any],
    *,
    expert_kind: str = "general",
    min_steps: int = 100,
    min_trades: int = 5,
    max_drawdown: float | None = None,
    min_sharpe_ratio: float = 0.0,
    max_fee_to_initial_capital: float = 0.02,
    max_action_saturation_ratio: float = 0.98,
    min_profit_factor: float = 1.20,
) -> RLEvaluationQualityReport:
    """檢查 final holdout 等不可回頭調參的單一樣本外區間。"""
    if max_drawdown is None:
        max_drawdown = {
            "long_term": 0.12,
            "short_term": 0.06,
        }.get(expert_kind, 0.15)
    total_return = _finite_metric(metrics, "total_return")
    observed_drawdown = _finite_metric(metrics, "max_drawdown")
    sharpe_ratio = _finite_metric(metrics, "sharpe_ratio")
    fee_ratio = _finite_metric(metrics, "fee_to_initial_capital")
    saturation = _finite_metric(metrics, "action_saturation_ratio")
    profit_factor = _finite_metric(metrics, "profit_factor")
    expectancy = _finite_metric(metrics, "expectancy")
    steps = int(metrics.get("steps") or 0)
    trades = int(metrics.get("trades") or 0)
    liquidation_count = int(metrics.get("liquidation_count") or 0)
    reasons: list[str] = []
    if total_return is None or total_return <= 0:
        reasons.append("Final holdout 報酬必須大於 0")
    if observed_drawdown is None or observed_drawdown > max_drawdown:
        text = "N/A" if observed_drawdown is None else f"{observed_drawdown:.2%}"
        reasons.append(f"Final holdout 最大回撤 {text}，上限為 {max_drawdown:.0%}")
    if steps < min_steps:
        reasons.append(f"Final holdout 僅 {steps} 步，至少需要 {min_steps} 步")
    if trades < min_trades:
        reasons.append(f"Final holdout 僅 {trades} 次調倉，至少需要 {min_trades} 次")
    if sharpe_ratio is None or sharpe_ratio <= min_sharpe_ratio:
        text = "N/A" if sharpe_ratio is None else f"{sharpe_ratio:.2f}"
        reasons.append(f"Final holdout Sharpe {text}，必須大於 {min_sharpe_ratio:.2f}")
    if fee_ratio is None or fee_ratio > max_fee_to_initial_capital:
        text = "N/A" if fee_ratio is None else f"{fee_ratio:.2%}"
        reasons.append(
            f"Final holdout 交易成本占本金 {text}，上限為 {max_fee_to_initial_capital:.0%}"
        )
    if saturation is None or saturation > max_action_saturation_ratio:
        text = "N/A" if saturation is None else f"{saturation:.1%}"
        reasons.append(f"Final holdout 動作邊界比例 {text}，策略可能已退化")
    if profit_factor is None or profit_factor <= min_profit_factor:
        text = "N/A" if profit_factor is None else f"{profit_factor:.2f}"
        reasons.append(f"Final holdout Profit Factor {text}，必須大於 {min_profit_factor:.2f}")
    if expectancy is None or expectancy <= 0:
        text = "N/A" if expectancy is None else f"{expectancy:.4f}"
        reasons.append(f"Final holdout Expectancy {text}，必須大於 0")
    if liquidation_count > 0:
        reasons.append(f"Final holdout 發生 {liquidation_count} 次強平，門檻必須為 0")
    return RLEvaluationQualityReport(
        not reasons,
        total_return,
        observed_drawdown,
        steps,
        trades,
        sharpe_ratio,
        fee_ratio,
        saturation,
        profit_factor,
        expectancy,
        liquidation_count,
        tuple(reasons),
    )


def assess_rl_training_quality(
    metadata: dict[str, Any],
    *,
    min_test_steps: int = 100,
    min_test_trades: int = 5,
    max_test_drawdown: float | None = None,
    min_positive_market_ratio: float = 0.60,
    min_test_sharpe_ratio: float = 0.0,
    max_fee_to_initial_capital: float = 0.02,
    max_action_saturation_ratio: float = 0.98,
    min_test_profit_factor: float = 1.20,
) -> RLQualityReport:
    """同時檢查驗證、測試與通用模型的跨市場一致性。"""
    metrics = dict(metadata.get("metrics", {}))
    expert_kind = str(dict(metadata.get("environment_expert", {})).get("kind", "general"))
    if max_test_drawdown is None:
        max_test_drawdown = {
            "long_term": 0.12,
            "short_term": 0.10,
        }.get(expert_kind, 0.15)
    validation = dict(metrics.get("validation", {}))
    test = dict(metrics.get("test", {}))
    validation_return = _finite_metric(validation, "total_return")
    test_return = _finite_metric(test, "total_return")
    max_drawdown = _finite_metric(test, "max_drawdown")
    positive_market_ratio = _finite_metric(test, "positive_market_ratio")
    test_sharpe_ratio = _finite_metric(test, "sharpe_ratio")
    fee_to_initial_capital = _finite_metric(test, "fee_to_initial_capital")
    action_saturation_ratio = _finite_metric(test, "action_saturation_ratio")
    test_profit_factor = _finite_metric(test, "profit_factor")
    test_expectancy = _finite_metric(test, "expectancy")
    liquidation_count = int(test.get("liquidation_count") or 0)
    test_steps = int(test.get("steps") or 0)
    test_trades = int(test.get("trades") or 0)
    reasons: list[str] = []
    if metadata.get("status") != "complete":
        reasons.append("RL 訓練狀態不是 complete")
    if validation_return is None or validation_return <= 0:
        reasons.append("驗證集報酬必須大於 0")
    if test_return is None or test_return <= 0:
        reasons.append("測試集報酬必須大於 0")
    if max_drawdown is None or max_drawdown > max_test_drawdown:
        text = "N/A" if max_drawdown is None else f"{max_drawdown:.2%}"
        reasons.append(f"測試最大回撤 {text}，上限為 {max_test_drawdown:.0%}")
    if test_steps < min_test_steps:
        reasons.append(f"測試僅 {test_steps} 步，至少需要 {min_test_steps} 步")
    if test_trades < min_test_trades:
        reasons.append(f"測試僅 {test_trades} 次調倉，至少需要 {min_test_trades} 次")
    if positive_market_ratio is not None and positive_market_ratio < min_positive_market_ratio:
        reasons.append(
            f"正報酬市場比例 {positive_market_ratio:.0%}，至少需要 {min_positive_market_ratio:.0%}"
        )
    if test_sharpe_ratio is not None and test_sharpe_ratio <= min_test_sharpe_ratio:
        reasons.append(f"測試 Sharpe {test_sharpe_ratio:.2f}，必須大於 {min_test_sharpe_ratio:.2f}")
    if fee_to_initial_capital is not None and fee_to_initial_capital > max_fee_to_initial_capital:
        reasons.append(
            f"測試交易成本占本金 {fee_to_initial_capital:.2%}，上限為 "
            f"{max_fee_to_initial_capital:.0%}"
        )
    if (
        action_saturation_ratio is not None
        and action_saturation_ratio > max_action_saturation_ratio
    ):
        reasons.append(f"測試期動作有 {action_saturation_ratio:.1%} 卡在倉位邊界，策略可能已退化")
    if test_profit_factor is None or test_profit_factor <= min_test_profit_factor:
        text = "N/A" if test_profit_factor is None else f"{test_profit_factor:.2f}"
        reasons.append(f"樣本外 Profit Factor {text}，必須大於 {min_test_profit_factor:.2f}")
    if test_expectancy is None or test_expectancy <= 0:
        text = "N/A" if test_expectancy is None else f"{test_expectancy:.4f}"
        reasons.append(f"樣本外 Expectancy {text}，必須大於 0")
    if liquidation_count > 0:
        reasons.append(f"測試期發生 {liquidation_count} 次強平，實盤門檻必須為 0")
    ai_context = dict(metadata.get("environment_ai_context", {}))
    if ai_context:
        if not bool(ai_context.get("runtime_ready", False)):
            reasons.append("AI 特徵成品不完整，模擬／實盤無法重建訓練 observation")
        aggregate = dict(dict(ai_context.get("coverage", {})).get("aggregate", {}))
        if bool(ai_context.get("finbert_enabled", False)):
            finbert_coverage = float(aggregate.get("finbert", 0.0) or 0.0)
            finbert_config = dict(ai_context.get("finbert_config", {}))
            minimum_coverage = float(finbert_config.get("minimum_training_coverage", 0.30) or 0.30)
            if finbert_coverage < minimum_coverage:
                reasons.append(
                    f"FinBERT 歷史覆蓋僅 {finbert_coverage:.2%}，"
                    f"實盤至少需要 {minimum_coverage:.0%}"
                )
        if bool(ai_context.get("transformer_enabled", False)):
            transformer_coverage = float(aggregate.get("transformer", 0.0) or 0.0)
            if transformer_coverage < 0.50:
                reasons.append(f"Transformer 特徵覆蓋僅 {transformer_coverage:.2%}，至少需要 50%")
    return RLQualityReport(
        not reasons,
        validation_return,
        test_return,
        max_drawdown,
        test_steps,
        test_trades,
        positive_market_ratio,
        test_sharpe_ratio,
        fee_to_initial_capital,
        action_saturation_ratio,
        test_profit_factor,
        test_expectancy,
        liquidation_count,
        tuple(reasons),
    )


def ensure_rl_execution_eligible(metadata: dict[str, Any]) -> None:
    """Live 送單前強制 RL 訓練成品通過品質門檻。"""
    report = assess_rl_training_quality(metadata)
    if not report.eligible:
        raise ValueError("RL 模型未達實盤品質門檻：" + "；".join(report.reasons))
