"""入出金不能扭曲實盤績效與回撤測試。"""

from __future__ import annotations

import pandas as pd
import pytest

from ai_quant_trading.live_trading.capital_flows import (
    flow_adjusted_equity,
    record_capital_flow,
)


def test_deposit_is_not_counted_as_profit() -> None:
    snapshots = pd.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"],
            "managed_equity": [1_000.0, 1_100.0],
        }
    )
    flows = pd.DataFrame(
        {
            "timestamp": ["2026-01-02T12:00:00Z"],
            "amount": [500.0],
        }
    )

    result = flow_adjusted_equity(
        snapshots,
        flows,
        current_timestamp="2026-01-03T00:00:00Z",
        current_equity=1_600.0,
    )

    assert result.index_value == pytest.approx(1.10)
    assert result.drawdown == pytest.approx(0.0)
    assert result.net_external_flow == pytest.approx(500.0)


def test_loss_after_deposit_creates_real_drawdown() -> None:
    snapshots = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01T00:00:00Z",
                "2026-01-02T00:00:00Z",
                "2026-01-03T00:00:00Z",
            ],
            "managed_equity": [1_000.0, 1_100.0, 1_600.0],
        }
    )
    flows = pd.DataFrame(
        {"timestamp": ["2026-01-02T12:00:00Z"], "amount": [500.0]}
    )

    result = flow_adjusted_equity(
        snapshots,
        flows,
        current_timestamp="2026-01-04T00:00:00Z",
        current_equity=1_500.0,
    )

    assert result.drawdown == pytest.approx(-0.0625)


def test_capital_flow_reference_cannot_be_reused(tmp_path) -> None:
    record_capital_flow(
        tmp_path,
        "testnet",
        amount=500.0,
        flow_type="deposit",
        reference="transfer-001",
        timestamp="2026-01-01T00:00:00Z",
    )

    with pytest.raises(ValueError, match="已存在"):
        record_capital_flow(
            tmp_path,
            "testnet",
            amount=500.0,
            flow_type="deposit",
            reference="transfer-001",
        )
