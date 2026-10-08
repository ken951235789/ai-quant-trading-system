"""圖文交易 UI 的空資料、未產圖與實現損益狀態。"""

from streamlit.testing.v1 import AppTest

from tests.trading.test_journal import setup_journal


def app_for(root):
    return AppTest.from_string(f"""
from pathlib import Path
from ai_quant_trading.dashboard.trade_journal_page import render_trade_journal
render_trade_journal(Path({str(root)!r}))
""", default_timeout=30)


def test_empty_journal_page(tmp_path):
    app = app_for(tmp_path).run()
    assert not app.exception
    assert "尚無" in app.info[0].value


def test_journal_pending_and_closed_page(tmp_path):
    journal, state, row = setup_journal(tmp_path)
    app = app_for(journal.root).run()
    assert not app.exception
    assert "待產生" in app.warning[0].value
    journal.closed(state, row, quantity=2., entry_fee=.1, exit_fee=.1, carry=0.,
                   gross=8., net=7.8, reason="take_profit", fill_price=104.)
    journal.render_pending()
    rows = journal.records()
    app.run()
    record_id = next(r["id"] for r in rows if r["kind"] == "close")
    app.selectbox[0].set_value(record_id).run()
    assert not app.exception
    assert [m.value for m in app.metric] == ["+7.8000", "2.00:1", "+1.950R"]
    assert len(app.get("image")) == 1
    assert "12:15" not in app.selectbox[0].options[0]
    assert "08:15" in app.selectbox[0].options[0]
