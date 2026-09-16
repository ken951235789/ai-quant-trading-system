"""宏觀、SEC 與 Order Book 進階來源測試。"""

from __future__ import annotations

from dataclasses import dataclass

from ai_quant_trading.data_collection.binance import BinanceSpotClient
from ai_quant_trading.data_collection.macro import FredPublicClient
from ai_quant_trading.data_collection.sec_fundamentals import SecCompanyFactsClient
from tests.data_collection.test_binance_client import FakeSession


@dataclass
class FakeResponse:
    payload: object | None = None
    text: str = ""

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self.payload


class ResponseSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.headers: dict[str, str] = {}

    def get(self, *_args: object, **_kwargs: object) -> FakeResponse:
        return self.responses.pop(0)


def test_order_book_snapshot_builds_depth_imbalance() -> None:
    session = FakeSession(
        [
            {
                "lastUpdateId": 123,
                "bids": [["100", "2"], ["99", "1"]],
                "asks": [["101", "1"], ["102", "1"]],
            }
        ]
    )
    frame = BinanceSpotClient(session=session).fetch_order_book_snapshot(
        "BTC/USDT",
        limit=20,
    )

    assert frame.iloc[0]["spread_bps"] > 0
    assert frame.iloc[0]["imbalance_5"] > 0
    assert frame.iloc[0]["bid_depth_20"] == 299
    assert frame.iloc[0]["ask_depth_20"] == 203


def test_fred_uses_publication_lag_as_available_timestamp() -> None:
    session = ResponseSession(
        [
            FakeResponse(
                text="observation_date,VIXCLS\n2024-01-02,14.5\n2024-01-03,15.0\n"
            )
        ]
    )
    frame = FredPublicClient(session=session).fetch_series("VIXCLS")

    assert frame.iloc[0]["timestamp"].startswith("2024-01-03")
    assert frame.iloc[1]["change"] == 0.5
    assert frame.iloc[0]["publication_lag_days"] == 1


def test_sec_fundamentals_start_on_day_after_filing() -> None:
    facts = {
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {
                        "USD": [
                            {
                                "start": "2023-01-01",
                                "end": "2023-03-31",
                                "val": 100,
                                "filed": "2023-05-01",
                                "form": "10-Q",
                            }
                        ]
                    }
                },
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            {
                                "start": "2023-01-01",
                                "end": "2023-03-31",
                                "val": 20,
                                "filed": "2023-05-01",
                                "form": "10-Q",
                            }
                        ]
                    }
                },
            }
        }
    }
    session = ResponseSession([FakeResponse(payload=facts)])
    client = SecCompanyFactsClient(
        user_agent="Researcher test@example.com",
        session=session,
    )

    frame = client.fetch_company_fundamentals("TEST", cik="1234")

    assert frame.iloc[0]["timestamp"].startswith("2023-05-02")
    assert frame.iloc[0]["fundamental_profit_margin"] == 0.2
    assert frame.iloc[0]["cik"] == "0000001234"
