"""帳戶級即時營運風控測試。"""

from __future__ import annotations

import pandas as pd

from ai_quant_trading.live_trading.operations import (
    OperationalRiskLimits,
    assess_operational_risk,
)


def _snapshots() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": ["2026-08-23T00:00:00Z", "2026-08-23T01:00:00Z"],
            "estimated_equity": [1_000.0, 995.0],
        }
    )


def test_daily_loss_hard_limit_blocks_entry_and_forces_existing_position_reduction() -> None:
    limits = OperationalRiskLimits(maximum_daily_loss=0.01, maximum_drawdown=0.08)

    flat = assess_operational_risk(
        _snapshots(),
        current_equity=985.0,
        has_open_position=False,
        limits=limits,
        now=pd.Timestamp("2026-08-23T02:00:00Z"),
    )
    invested = assess_operational_risk(
        _snapshots(),
        current_equity=985.0,
        has_open_position=True,
        limits=limits,
        now=pd.Timestamp("2026-08-23T02:00:00Z"),
    )

    assert flat.hard_halt and not flat.allow_new_risk and not flat.force_reduce
    assert invested.hard_halt and invested.force_reduce


def test_manual_halt_blocks_new_risk_without_forcing_position_exit() -> None:
    result = assess_operational_risk(
        _snapshots(),
        current_equity=1_000.0,
        has_open_position=True,
        emergency_halt="人工檢查",
        now=pd.Timestamp("2026-08-23T02:00:00Z"),
    )

    assert not result.allow_new_risk
    assert not result.force_reduce
    assert not result.hard_halt
