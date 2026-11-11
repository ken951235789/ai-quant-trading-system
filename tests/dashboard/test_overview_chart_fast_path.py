"""總覽圖表專用計算的公開合成資料測試，不需要私人行情。"""

import numpy as np
import pandas as pd
import pytest

from ai_quant_trading.dashboard.overview_page import load_overview_chart


def test_chart_matches_full_history_without_training_pipeline(tmp_path, monkeypatch):
    from ai_quant_trading.dashboard import services

    def forbidden(*args, **kwargs):
        raise AssertionError("總覽不可執行訓練特徵工程")

    monkeypatch.setattr(services, "build_feature_dataset", forbidden)
    close = pd.Series(np.random.default_rng(42).uniform(100, 200, 5000))
    frame = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=len(close), freq="5min", tz="UTC"),
                          "open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1})
    file = tmp_path / "market.csv"
    frame.to_csv(file, index=False)
    result = load_overview_chart(str(file), file.stat().st_mtime_ns)
    assert len(result) == 1000
    assert not any(name.startswith("target") for name in result.columns)
    for window in (20, 60):
        np.testing.assert_allclose(result[f"sma_{window}"], close.rolling(window, min_periods=window).mean().tail(1000))


def test_short_history_cache_and_future_isolation(tmp_path):
    file = tmp_path / "market.csv"
    frame = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=70, freq="h", tz="UTC"),
                          "close": range(100, 170)})
    frame.to_csv(file, index=False)
    before = load_overview_chart(str(file), 1)
    assert before["sma_60"].iloc[:59].isna().all()
    frame.loc[69, "close"] = 1000
    frame.to_csv(file, index=False)
    after = load_overview_chart(str(file), 2)
    pd.testing.assert_frame_equal(before.iloc[:69], after.iloc[:69])
    assert after.iloc[-1]["sma_60"] != before.iloc[-1]["sma_60"]


def test_empty_chart_rejected(tmp_path):
    file = tmp_path / "empty.csv"
    file.write_text("timestamp,close\n")
    with pytest.raises(ValueError, match="為空"):
        load_overview_chart(str(file), 1)
