"""逐棒浮動權益與舊成交成本的一致性。"""

import pandas as pd
import pytest

from ai_quant_trading.research.account_replay import replay_mark_to_market


def test_unrealized_drawdown_is_not_hidden_by_winning_exit():
    frame = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=4, freq="15min", tz="UTC"),
                          "open": [100]*4, "high": [101, 102, 103, 110], "low": [99, 99, 80, 100], "close": [100, 100, 90, 110]})
    trades = pd.DataFrame([dict(endpoint=0, entry_endpoint=1, exit_endpoint=3, side=1,
        entry_price=100., exit_price=110., net_return=.1, reason_code=4)])
    path, closed, report = replay_mark_to_market(frame, trades, allocation=1, fee_bps=0, funding_reserve_bps=0)
    assert report["total_return"] == pytest.approx(.1)
    assert report["close_max_drawdown"] == pytest.approx(.1)
    assert report["adverse_drawdown_bound"] == pytest.approx(.2)
    assert closed.net_pnl.iloc[0] == pytest.approx(100)
    assert not report["liquidation_validated"]
    with pytest.raises(ValueError):
        replay_mark_to_market(frame, trades, allocation=3)
    with pytest.raises(ValueError, match="成本帳務"):
        replay_mark_to_market(frame, trades, fee_bps=5)
