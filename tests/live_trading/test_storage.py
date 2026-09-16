"""Live CSV 欄位遷移與排他寫入測試。"""

from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from ai_quant_trading.live_trading.storage import append_csv_row


def test_append_migrates_old_csv_header_before_writing(tmp_path) -> None:
    path = tmp_path / "orders.csv"
    path.write_text("timestamp,symbol\nold,BTC/USDT\n", encoding="utf-8")

    append_csv_row(
        path,
        ["timestamp", "market_type", "symbol"],
        {"timestamp": "new", "market_type": "usd_m_futures", "symbol": "BTC/USDT"},
    )

    frame = pd.read_csv(path)
    assert list(frame.columns) == ["timestamp", "market_type", "symbol"]
    assert len(frame) == 2
    assert frame.iloc[1]["market_type"] == "usd_m_futures"


def test_parallel_append_does_not_lose_or_merge_rows(tmp_path) -> None:
    path = tmp_path / "cycles.csv"
    columns = ["timestamp", "status"]

    def write(index: int) -> None:
        append_csv_row(path, columns, {"timestamp": index, "status": "hold"})

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(write, range(40)))

    frame = pd.read_csv(path)
    assert len(frame) == 40
    assert set(frame["timestamp"]) == set(range(40))
