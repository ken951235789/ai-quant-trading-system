"""跨長短期策略的投資組合風控測試。"""

import pandas as pd
import pytest

from ai_quant_trading.risk import (
    PortfolioExposure,
    calculate_period_returns,
    expert_portfolio_risk,
    govern_portfolio_target,
)


def test_period_returns_use_last_value_before_each_boundary() -> None:
    history = pd.DataFrame(
        {
            "timestamp": [
                "2026-08-31T23:55:00Z",
                "2026-09-01T12:00:00Z",
                "2026-09-02T00:00:00Z",
            ],
            "equity": [100.0, 105.0, 102.0],
        }
    )

    daily, weekly, monthly = calculate_period_returns(
        history,
        current_timestamp="2026-09-02T12:00:00Z",
        current_value=99.0,
    )

    assert daily == pytest.approx(99.0 / 105.0 - 1.0)
    assert weekly == pytest.approx(-0.01)
    assert monthly == pytest.approx(-0.01)


def test_btc_short_term_profile_allows_spec_maximum_notional() -> None:
    config = expert_portfolio_risk("short_term")
    exposures = [
        PortfolioExposure("long", "long_term", "AAPL", 40_000, 20_000),
        PortfolioExposure("short-a", "short_term", "BTC/USDT", 10_000, -5_000),
    ]

    decision = govern_portfolio_target(
        account_id="short-b",
        strategy="short_term",
        asset_group="ETH/USDT",
        account_equity=10_000,
        current_target=0.0,
        proposed_target=-0.50,
        exposures=exposures,
        config=config,
    )

    assert decision.approved_target == pytest.approx(-0.50)
    assert decision.reasons == ()


def test_portfolio_governor_always_allows_risk_reduction() -> None:
    decision = govern_portfolio_target(
        account_id="account",
        strategy="long_term",
        asset_group="AAPL",
        account_equity=10_000,
        current_target=0.20,
        proposed_target=0.05,
        exposures=[PortfolioExposure("account", "long_term", "AAPL", 10_000, 2_000)],
        config=expert_portfolio_risk("long_term"),
    )

    assert decision.approved_target == pytest.approx(0.05)


def test_portfolio_loss_limit_flattens_position() -> None:
    decision = govern_portfolio_target(
        account_id="account",
        strategy="short_term",
        asset_group="BTC/USDT",
        account_equity=10_000,
        current_target=-0.05,
        proposed_target=-0.08,
        exposures=[],
        config=expert_portfolio_risk("short_term"),
        daily_return=-0.011,
    )

    assert decision.halted
    assert decision.approved_target == 0.0
