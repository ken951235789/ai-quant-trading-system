"""大型歷史下載的校驗、續傳與發布測試，不連接外部交易所。"""

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from ai_quant_trading.data_collection.history import (
    HistoryStaging, decode_monthly_klines, download_btc_history, fetch_monthly_archive,
    validate_history_chunk,
)
from ai_quant_trading.market_clock import BTC_MULTITIMEFRAME_INTERVALS


def test_history_default_is_nine_without_changing_trading_default() -> None:
    assert download_btc_history.__kwdefaults__["intervals"] == (
        "1m", "3m", "5m", "15m", "30m", "1h", "4h", "12h", "1d",
    )
    assert BTC_MULTITIMEFRAME_INTERVALS == ("5m", "15m", "1h", "4h", "1d")


def make_zip(*, header: bool = False, timestamp: int = 1631059200000) -> bytes:
    stream = BytesIO()
    text = "open_time,open,high,low,close,volume,close_time,quote_volume,count,buy_base,buy_quote,ignore\n" if header else ""
    text += f"{timestamp},10,12,9,11,5,{timestamp + 59999},50,3,2,20,0\n"
    with ZipFile(stream, "w") as archive:
        archive.writestr("BTCUSDT-1m-2021-09.csv", text)
    return stream.getvalue()


@pytest.mark.parametrize("header", [False, True])
def test_archive_header_and_market_identity(header: bool) -> None:
    frame = decode_monthly_klines(make_zip(header=header), "1m")
    assert frame.iloc[0]["exchange"] == "binance_futures"
    assert frame.iloc[0]["timestamp"] == "2021-09-08T00:00:00+00:00"
    assert frame.iloc[0]["taker_buy_base_volume"] == 2


def test_archive_rejects_spot_microseconds() -> None:
    with pytest.raises(ValueError, match="毫秒"):
        decode_monthly_klines(make_zip(timestamp=1631059200000000), "1m")


def test_archive_rejects_unclosed_bar() -> None:
    future = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000) // 60000 * 60000 + 60000
    with pytest.raises(ValueError, match="尚未收盤"):
        decode_monthly_klines(make_zip(timestamp=future), "1m")


def test_empty_staging_does_not_replace_file(tmp_path: Path) -> None:
    staging = HistoryStaging(tmp_path / "stage.sqlite", "1m")
    target = tmp_path / "latest.csv"
    with pytest.raises(ValueError, match="不發布空"):
        staging.publish(target)
    assert not target.exists()
    staging.close()


@pytest.mark.parametrize("column,value", [("close_time_ms", 1), ("close", float("inf")), ("exchange", "binance")])
def test_invalid_market_rows_rejected(column: str, value: object) -> None:
    frame = decode_monthly_klines(make_zip(), "1m")
    frame[column] = value
    with pytest.raises(ValueError):
        validate_history_chunk(frame, "1m")


def test_staging_resume_gaps_and_latest_wins(tmp_path: Path) -> None:
    first = 1631059200000
    frame = decode_monthly_klines(make_zip(), "1m")
    path = tmp_path / "staging.sqlite"
    staging = HistoryStaging(path, "1m")
    staging.upsert(frame)
    assert staging.missing_ranges(first, first + 180000) == [(first + 60000, first + 180000)]
    staging.close()
    staging = HistoryStaging(path, "1m")
    corrected = frame.copy()
    corrected["close"] = 10.5
    staging.upsert(corrected)
    staging.upsert(decode_monthly_klines(make_zip(timestamp=first + 120000), "1m"))
    assert staging.missing_ranges(first, first + 180000) == [(first + 60000, first + 120000)]
    assert staging.connection.execute("SELECT COUNT(*) FROM bars").fetchone()[0] == 2
    assert staging.connection.execute("SELECT close FROM bars WHERE open_time_ms=?", (first,)).fetchone()[0] == 10.5
    staging.close()


def test_publish_keeps_external_context_and_new_bar(tmp_path: Path) -> None:
    first = 1631059200000
    staging = HistoryStaging(tmp_path / "stage.sqlite", "1m")
    frame = decode_monthly_klines(make_zip(), "1m")
    staging.upsert(frame)
    original = pd.concat([frame, decode_monthly_klines(make_zip(timestamp=first + 60000), "1m")])
    original["funding_rate"] = 0.0001
    original["timestamp"] = pd.to_datetime(original["timestamp"], utc=True).map(lambda value: value.isoformat())
    target = tmp_path / "latest.csv"
    original.to_csv(target, index=False)
    staging.publish(target)
    result = pd.read_csv(target)
    assert len(result) == 2
    assert result["funding_rate"].eq(0.0001).all()
    assert not target.with_suffix(".csv.lock").exists()
    assert not list(tmp_path.glob("*.tmp"))
    staging.close()


def test_publish_refreshes_lock_during_large_file_write(tmp_path: Path, monkeypatch) -> None:
    import ai_quant_trading.data_collection.history as history

    called = []
    original = history.os.utime

    def capture(path, times):
        called.append(Path(path))
        return original(path, times)

    monkeypatch.setattr(history.os, "utime", capture)
    staging = HistoryStaging(tmp_path / "stage.sqlite", "1m")
    staging.upsert(decode_monthly_klines(make_zip(), "1m"))
    target = tmp_path / "latest.csv"
    staging.publish(target)
    assert target.with_suffix(".csv.lock") in called
    staging.close()


def test_mixed_old_and_rest_timestamp_formats(tmp_path: Path) -> None:
    first = 1631059200000
    frame = pd.concat([decode_monthly_klines(make_zip(), "1m"),
                       decode_monthly_klines(make_zip(timestamp=first + 60000), "1m")], ignore_index=True)
    frame.loc[1, "timestamp"] = "2021-09-08T00:01:00.000Z"
    validate_history_chunk(frame, "1m")
    staging = HistoryStaging(tmp_path / "stage.sqlite", "1m")
    staging.upsert(frame)
    staging.publish(tmp_path / "latest.csv")
    result = pd.read_csv(tmp_path / "latest.csv")
    assert result["timestamp"].str.endswith("+00:00").all()
    staging.close()


def test_checksum_mismatch_rejects_archive() -> None:
    class Response:
        status_code = 200
        content = make_zip()
        text = "0" * 64 + "  BTCUSDT-1m-2021-09.zip"

        def raise_for_status(self):
            pass

    class Session:
        def get(self, *args, **kwargs):
            return Response()

    with pytest.raises(ValueError, match="SHA-256"):
        fetch_monthly_archive(Session(), "1m", pd.Timestamp("2021-09-01", tz="UTC"))
