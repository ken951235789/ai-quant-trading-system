"""逐筆图表必须遵守原始成交帳務與時間语意。"""

import json

import pandas as pd
import pytest

from ai_quant_trading.research.trade_review import contained_file, review_trades, script_json


def fixture(side=1, reason="target"):
    bars = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=10, freq="15min", tz="UTC")})
    entry, stop, target = 100., 100. - side * 2, 100. + side * 4
    exit_ = target
    price_return = side * (exit_ / entry - 1)
    trades = pd.DataFrame([dict(endpoint=1, entry_endpoint=2, exit_endpoint=4, side=side,
        entry_price=entry, exit_price=exit_, stop_price=stop, target_price=target,
        gross_same_path_return=price_return+.0004+.0001, spread_return=.0001,
        slippage_return=.0004, fee_return=.001, funding_return=.0001,
        net_return=price_return-.001-.0001, source_sha256="source", exit_reason=reason,
        entry_at=bars.timestamp.iloc[2], signal_at=bars.timestamp.iloc[2],
        label_end_at=bars.timestamp.iloc[5])])
    return bars, trades


@pytest.mark.parametrize("side", [1, -1])
def test_long_short_and_cost_not_double_counted(side):
    bars, trades = fixture(side)
    result = review_trades(bars, trades, "source")[0]
    assert result["planned_rr"] == pytest.approx(2)
    assert result["net_r"] == pytest.approx(1.945)
    assert result["net_return"] == pytest.approx(.0389)
    assert result["intrabar"] is True
    assert (result["holding_min"], result["holding_max"]) == (30, 45)


@pytest.mark.parametrize("reason", ["time", "regime"])
def test_open_exits_have_exact_bar_time(reason):
    bars, trades = fixture(reason=reason)
    row = review_trades(bars, trades, "source")[0]
    assert not row["intrabar"]
    assert row["holding_min"] == row["holding_max"] == 30


def test_no_trade_does_not_invent_payoff():
    bars, trades = fixture()
    assert review_trades(bars, trades.iloc[:0], "source") == []


@pytest.mark.parametrize("field,value", [
    ("source_sha256", "wrong"), ("entry_endpoint", 3), ("exit_endpoint", 20),
    ("entry_endpoint", 2.5), ("side", 0), ("stop_price", 102), ("net_return", .9),
    ("gross_same_path_return", .9), ("entry_at", "2025-01-01T00:00:00Z"),
    ("target_price", float("nan")), ("exit_reason", "invented"),
])
def test_invalid_inputs_fail_closed(field, value):
    bars, trades = fixture()
    trades[field] = value
    with pytest.raises((ValueError, AssertionError)):
        review_trades(bars, trades, "source")


def test_source_future_changes_do_not_change_trade_metrics():
    bars, trades = fixture()
    original = review_trades(bars, trades, "source")
    bars["close"] = 1.
    bars.loc[6:, "close"] = 1000000.
    assert review_trades(bars, trades, "source") == original


def test_duplicate_bar_time_rejected():
    bars, trades = fixture()
    bars.loc[1, "timestamp"] = bars.timestamp.iloc[0]
    with pytest.raises(ValueError, match="時間戳"):
        review_trades(bars, trades, "source")


def test_path_escape_rejected(tmp_path):
    with pytest.raises(ValueError):
        contained_file(tmp_path, "../secret.csv")


def test_json_cannot_close_script():
    value = {"x": "</script><script>alert(1)</script>"}
    encoded = script_json(value)
    assert "<" not in encoded
    assert json.loads(encoded) == value


def test_static_chart_context_and_preserve_existing_file(tmp_path, monkeypatch):
    """模型成交不可誤標成無 AI；靜態圖也不能覆寫舊研究。"""
    from matplotlib.figure import Figure
    from ai_quant_trading.research.trade_review import render_example_png

    bars, trades = fixture()
    for column, value in (("open", 100.), ("high", 105.), ("low", 97.), ("close", 102.)):
        bars[column] = value
    trade = review_trades(bars, trades, "source")[0]
    labels = []
    original = Figure.savefig

    def capture(self, *args, **kwargs):
        labels.append(self._suptitle.get_text())
        labels.append(self.axes[0].get_title(loc="left"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    output = tmp_path / "trade.png"
    render_example_png(bars, trade, output, context_label="Transformer 32% seed 137",
                       trade_label="Test 第 1 筆")
    assert "Transformer 32% seed 137" in labels[0]
    assert "無 AI" not in labels[0]
    assert "Test 第 1 筆" in labels[1]
    assert output.read_bytes().startswith(b"\x89PNG")
    with pytest.raises(FileExistsError):
        render_example_png(bars, trade, output)
