"""Live 前 Testnet 運行證據檢查。"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ai_quant_trading.live_trading.readiness import (
    assess_testnet_evidence,
    ensure_live_operational_readiness,
)
from ai_quant_trading.live_trading.storage import (
    CYCLE_COLUMNS,
    append_csv_row,
    live_trading_paths,
)


def test_live_requires_testnet_cycles_and_elapsed_days(tmp_path) -> None:
    report = assess_testnet_evidence(tmp_path)
    assert not report.eligible
    with pytest.raises(ValueError, match="Live 操作門檻"):
        ensure_live_operational_readiness(
            tmp_path,
            model_dir="models/current",
            symbol="BTC/USDT",
        )


def test_testnet_evidence_accepts_successful_cycles_over_time(tmp_path) -> None:
    paths = live_trading_paths(tmp_path, "testnet")
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for index in range(30):
        append_csv_row(
            paths.cycles_csv,
            CYCLE_COLUMNS,
            {
                "timestamp": (start + timedelta(hours=6 * index)).isoformat(),
                "model_dir": str(Path("models/current")),
                "symbol": "BTC/USDT",
                "execution_mode": "execute",
                "status": "hold",
            },
        )

    report = assess_testnet_evidence(
        tmp_path,
        model_dir="models/current",
        symbol="BTC/USDT",
    )
    assert report.eligible
    assert report.successful_cycles == 30
    assert report.observed_days >= 7

    other_market = assess_testnet_evidence(
        tmp_path,
        model_dir="models/current",
        symbol="ETH/USDT",
    )
    assert not other_market.eligible
