"""長短期 RL 品質門檻測試。"""

from ai_quant_trading.reinforcement_learning import (
    assess_rl_evaluation_quality,
    assess_rl_training_quality,
)


def _metadata(kind: str, drawdown: float) -> dict[str, object]:
    return {
        "status": "complete",
        "environment_expert": {"kind": kind},
        "metrics": {
            "validation": {"total_return": 0.02},
            "test": {
                "total_return": 0.03,
                "max_drawdown": drawdown,
                "steps": 200,
                "trades": 10,
                "positive_market_ratio": 0.7,
                "sharpe_ratio": 0.8,
                "fee_to_initial_capital": 0.005,
                "profit_factor": 1.5,
                "expectancy": 1.0,
                "liquidation_count": 0,
            },
        },
    }


def test_short_term_quality_uses_ten_percent_drawdown_limit() -> None:
    report = assess_rl_training_quality(_metadata("short_term", 0.11))

    assert not report.eligible
    assert any("上限為 10%" in reason for reason in report.reasons)


def test_long_term_quality_uses_twelve_percent_drawdown_limit() -> None:
    report = assess_rl_training_quality(_metadata("long_term", 0.10))

    assert report.eligible


def test_final_holdout_quality_uses_strict_short_term_limit() -> None:
    metrics = _metadata("short_term", 0.07)["metrics"]["test"]

    report = assess_rl_evaluation_quality(metrics, expert_kind="short_term")

    assert not report.eligible
    assert any("上限為 6%" in reason for reason in report.reasons)
