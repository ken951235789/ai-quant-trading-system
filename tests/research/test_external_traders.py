"""公開來源資料隔離、費用語意及分頁的離線測試。"""

from copy import deepcopy
import csv

import pytest

from ai_quant_trading.research.external_traders import (
    PublicClient,
    history,
    normalize_fills,
    normalize_candles,
    normalize_funding,
    position_checks,
    select_accounts,
    write_csv,
)


def fill(t=1, tid=1, **kwargs):
    return dict(
        coin="BTC",
        time=t,
        tid=tid,
        oid=tid,
        side="B",
        px="100",
        sz="2",
        startPosition="0",
        dir="Open Long",
        closedPnl="0",
        fee="0.1",
        feeToken=" USDC",
        crossed=True,
        builderFee="0.02",
        **kwargs,
    )


def test_fee_includes_builder_and_side_is_not_position():
    raw = fill()
    raw.update(side="A", startPosition="2", dir="Close Long", closedPnl="6")
    row = normalize_fills([raw], "public_01")[0]
    assert row["fee"] == "0.1" and row["builder_fee_included"] == "0.02"
    assert row["end_position"] == "0" and row["direction"] == "Close Long"
    assert row["fee_token"] == "USDC" and row["training_eligible"] is False


@pytest.mark.parametrize(
    "key,value", [("px", "NaN"), ("sz", "0"), ("fee", "Infinity"), ("side", "X")]
)
def test_bad_fill_rejected(key, value):
    raw = fill()
    raw[key] = value
    with pytest.raises((ValueError, ArithmeticError)):
        normalize_fills([raw], "x")


def test_missing_fee_not_zero():
    raw = fill()
    del raw["fee"]
    with pytest.raises(KeyError):
        normalize_fills([raw], "x")


class Fake:
    def __init__(self, pages):
        self.pages = iter(pages)
        self.requests = []

    def info(self, payload):
        self.requests.append(payload)
        return next(self.pages)


def test_inclusive_cursor_deduplication():
    page = [fill(t=i, tid=i) for i in range(2000)]
    client = Fake([page, [page[-1], fill(t=2000, tid=2000)]])
    rows, audit = history(client, "userFillsByTime", "public", 0, 3000)
    assert len(rows) == 2001 and client.requests[1]["startTime"] == 1999
    assert audit["lifetime_complete"] is False


def test_conflicting_duplicate_rejected():
    first = fill()
    second = dict(first, fee="0.2")
    with pytest.raises(ValueError, match="衝突"):
        history(Fake([[first, second]]), "userFillsByTime", "x", 0, 100)


def test_same_millisecond_pagination_does_not_skip():
    page = [fill(t=0, tid=i) for i in range(2000)]
    rows, audit = history(Fake([page]), "userFillsByTime", "x", 0, 100)
    assert len(rows) == 2000 and "timestamp_boundary_unresolved" in audit["warnings"]


def test_out_of_window_rejected():
    with pytest.raises(ValueError, match="時間"):
        history(Fake([[fill(t=999)]]), "userFillsByTime", "x", 0, 100)


def test_position_censoring_and_gaps():
    a, b = fill(), fill(t=2, tid=2)
    b["startPosition"] = "3"
    audit = position_checks(normalize_fills([a, b], "x"))
    assert audit["position_gaps"] == 1 and audit["right_open"] is True


def test_candles_closed_only_and_missing_bars():
    row = dict(t=0, T=899999, s="BTC", i="15m", o="100", h="102", l="99", c="101", v="3")
    future = dict(row, t=1800000, T=2699999)
    rows, audit = normalize_candles([row, future], 0, 1800000)
    assert len(rows) == 1 and audit["missing_bars"] == 1
    future["h"] = "9999"
    assert normalize_candles([row, future], 0, 1800000)[0] == rows


def test_funding_sign_preserved():
    row = dict(time=1, delta=dict(coin="BTC", usdc="-2.5", fundingRate=".001", szi="2"))
    assert normalize_funding([row], "x")[0]["signed_cashflow_usdc"] == "-2.5"


def test_selection_balanced_deterministic_not_top_profit():
    data = dict(
        leaderboardRows=[
            dict(
                ethAddress=f"0x{i:040x}",
                windowPerformances=[["month", dict(pnl=str(i if i % 2 else -i), vlm="100")]],
            )
            for i in range(1, 31)
        ]
    )
    selected, strata = select_accounts(data)
    reversed_data = deepcopy(data)
    reversed_data["leaderboardRows"].reverse()
    assert select_accounts(reversed_data)[0] == selected
    assert len(selected) == 6 and strata == {"positive": 15, "negative": 15}


def test_network_allowlist_and_read_only(tmp_path):
    client = PublicClient(tmp_path)
    with pytest.raises(ValueError):
        client.fetch("https://evil.example")
    with pytest.raises(ValueError):
        client.fetch("https://api.hyperliquid.xyz/info", {"type": "order"})
    assert client.attempts == 0 and not client.session.trust_env


def test_csv_formula_injection_not_numeric_negative(tmp_path):
    path = tmp_path / "rows.csv"
    write_csv(path, [dict(text="=cmd()", numeric="-2.3")])
    with path.open(encoding="utf-8-sig") as stream:
        row = next(csv.DictReader(stream))
    assert row == dict(text="'=cmd()", numeric="-2.3")


def test_csv_empty_text_remains_empty(tmp_path):
    path = tmp_path / "empty.csv"
    write_csv(path, [dict(text="")])
    with path.open(encoding="utf-8-sig") as stream:
        assert next(csv.DictReader(stream))["text"] == ""
