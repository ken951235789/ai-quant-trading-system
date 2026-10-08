"""桌面逐筆回放的主題、空成交及切換狀態測試。"""

from copy import deepcopy
import gzip

import pytest
from streamlit.testing.v1 import AppTest

from ai_quant_trading.dashboard.trade_review_page import make_trade_review_figure
from ai_quant_trading.operations.integrity import build_artifact_manifest
from ai_quant_trading.research.trade_review_bundle import export_review_bundle, load_review_bundle


def sample():
    trade = dict(entry=2, exit=4, side=1, entry_price=100., exit_price=103., stop=98., target=104.,
        planned_rr=2., net_r=1.45, net_return=.029, price_return=.03, risk_fraction=.02,
        gross=.0305, spread=.0001, slippage=.0004, fee=.001, funding=0., reason="time")
    row = dict(family="original", policy="no_ai", split="test", scenario="base", variant="F_existing",
        seed=42, threshold_pct=-1, contract_sha256="test", trades=1, actual_win_rate=1.,
        payoff_ratio=None, average_win=.029, average_loss=None, mean_net=.029, schedule=0)
    no_trade = dict(row, policy="transformer", variant="F_compact_combined", threshold_pct=55,
        trades=0, actual_win_rate=None, average_win=None, mean_net=None, schedule=1)
    active = dict(no_trade, threshold_pct=0, trades=1, actual_win_rate=1., average_win=.029, mean_net=.029, schedule=0)
    return {"offset": 0, "bars": [[1735689600000+i*900000,100,105,97,101] for i in range(10)],
        "rows": [row,no_trade,active], "schedules": [[trade],[]],
        "windows": {"original|test":"2025-01-01"},
        "contracts": {"test":dict(stop_atr=2., reward_r=2., holding_bars=32, regime_exit="loss",
                                 fee_bps=5., slippage_bps=2., spread_bps=1., funding_reserve_bps=1.)}}


@pytest.mark.parametrize("dark", [False, True])
def test_figure_has_real_markers_and_fixed_levels(dark):
    payload=sample()
    fig=make_trade_review_figure(payload,payload['schedules'][0][0],dark_mode=dark)
    assert fig.data[0].type=="candlestick"
    assert fig.data[2].y[0]==100
    assert fig.data[3].y[0]==103
    assert fig.layout.shapes[1].y0==98
    assert str(fig.data[2].x[0]).startswith("2025-01-01 08:30")
    assert fig.layout.yaxis.range[0] < 97
    assert fig.layout.yaxis.range[1] > 105
    fig.to_json()


def test_bundle_tampering_and_size_fail_closed(tmp_path,monkeypatch):
    root=tmp_path/'bundle'
    export_review_bundle(sample(),root,source_sha256='source',label='測試')
    assert load_review_bundle(root)['payload']['rows'][0]['trades']==1
    path=root/'review.json.gz'
    original=path.read_bytes()
    path.write_bytes(original+b'tampered')
    with pytest.raises(ValueError):
        load_review_bundle(root)
    path.write_bytes(gzip.compress(b' '*1000))
    build_artifact_manifest(root,files=[path])
    monkeypatch.setattr('ai_quant_trading.research.trade_review_bundle.MAX_BYTES',500)
    with pytest.raises(ValueError,match='解壓'):
        load_review_bundle(root)


def test_bundle_refuses_invalid_trade(tmp_path):
    payload=deepcopy(sample())
    payload['schedules'][0][0]['exit']=100
    with pytest.raises(ValueError,match='範圍'):
        export_review_bundle(payload,tmp_path/'bad',source_sha256='source',label='測試')


def test_streamlit_selections_and_empty_state(tmp_path):
    export_review_bundle(sample(),tmp_path/'data/processed/research_reviews/test',source_sha256='source',label='測試')
    script=f"""
from pathlib import Path
import streamlit as st
from ai_quant_trading.dashboard.trade_review_page import render_trade_review_page
dark = st.toggle('深色模式',key='dark_mode')
render_trade_review_page(Path({str(tmp_path)!r}), dark)
"""
    app=AppTest.from_string(script,default_timeout=30).run()
    assert not app.exception
    assert len(app.get('plotly_chart'))==1
    app.toggle(key='dark_mode').set_value(True).run()
    assert not app.exception
    assert app.selectbox(key='review_trade_index').value==0
    app.selectbox(key='review_policy').set_value('transformer').run()
    assert not app.exception
    assert len(app.get('plotly_chart'))==0
    assert '沒有成交' in app.warning[0].value
    app.number_input(key='review_threshold').set_value(0).run()
    assert not app.exception
    assert len(app.get('plotly_chart'))==1


def test_empty_research_folder(tmp_path):
    script=f"""
from pathlib import Path
from ai_quant_trading.dashboard.trade_review_page import render_trade_review_page
render_trade_review_page(Path({str(tmp_path)!r}), False)
"""
    app=AppTest.from_string(script).run()
    assert not app.exception
    assert '尚未匯入' in app.info[0].value
