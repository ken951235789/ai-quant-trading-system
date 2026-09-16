"""Dashboard Plotly 圖表測試。"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ai_quant_trading.dashboard.charts import (
    make_candlestick_figure,
    make_equity_figure,
    make_indicator_figure,
    make_model_trading_figure,
)
from ai_quant_trading.features.indicators import build_feature_dataset


def make_feature_frame(rows: int = 230) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    close = 100 + index * 0.1 + np.sin(index / 5)
    source = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="D", tz="UTC"),
            "symbol": "TEST",
            "exchange": "binance",
            "interval": "1d",
            "open": close - 0.3,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 2_000 + index,
        }
    )
    return build_feature_dataset(source, annualization_periods=365, drop_na=False)


class DashboardChartsTest(unittest.TestCase):
    def test_candlestick_contains_price_overlay_and_volume(self):
        frame = make_feature_frame().tail(100)
        figure = make_candlestick_figure(frame, overlays=["sma_20"], show_volume=True)

        trace_types = [trace.type for trace in figure.data]
        self.assertEqual(trace_types, ["candlestick", "scatter", "bar"])

    def test_all_indicator_modes_build_a_figure(self):
        frame = make_feature_frame().tail(100)

        for mode in ["RSI", "MACD", "波動率", "成交量"]:
            with self.subTest(mode=mode):
                figure = make_indicator_figure(frame, mode)
                self.assertGreater(len(figure.data), 0)

    def test_dark_theme_updates_chart_background_and_text(self):
        frame = make_feature_frame().tail(100)

        figure = make_candlestick_figure(frame, dark_mode=True)

        self.assertEqual(figure.layout.paper_bgcolor, "#1b2022")
        self.assertEqual(figure.layout.plot_bgcolor, "#1b2022")
        self.assertEqual(figure.layout.font.color, "#e6edf3")
        self.assertEqual(figure.layout.legend.font.color, "#e6edf3")
        self.assertEqual(figure.layout.xaxis.tickfont.color, "#e6edf3")
        self.assertEqual(figure.layout.yaxis.title.font.color, "#e6edf3")
        self.assertEqual(figure.layout.hoverlabel.font.color, "#e6edf3")

    def test_theme_aware_supporting_charts_build(self):
        performance = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=2, tz="UTC"),
                "equity": [1000.0, 1010.0],
            }
        )

        equity_figure = make_equity_figure(performance, dark_mode=True)

        self.assertEqual(equity_figure.data[0].name, "總資產")

    def test_model_trading_chart_contains_orders_targets_and_forecast(self):
        frame = make_feature_frame().tail(60).copy()
        frame["_is_live"] = False
        frame.loc[frame.index[-1], "_is_live"] = True
        signal_time = frame.iloc[-10]["timestamp"]
        prediction_time = frame.iloc[-2]["timestamp"]
        orders = pd.DataFrame(
            {
                "timestamp": [signal_time],
                "fill_price": [float(frame.iloc[-10]["close"])],
                "notional": [100.0],
                "quantity": [1.0],
                "target_fraction": [0.25],
                "reason": ["rl_increase_long"],
                "side": ["BUY"],
            }
        )
        predictions = pd.DataFrame(
            {
                "timestamp": [prediction_time],
                "close": [float(frame.iloc[-2]["close"])],
                "model_target_fraction": [0.30],
                "approved_target_fraction": [0.25],
                "transformer_return_1": [0.001],
                "transformer_return_5": [0.003],
                "transformer_return_20": [-0.002],
            }
        )

        figure = make_model_trading_figure(
            frame,
            orders=orders,
            predictions=predictions,
            position={
                "quantity": 1.0,
                "entry_time": signal_time,
                "entry_price": float(frame.iloc[-10]["close"]),
                "stop_loss": float(frame.iloc[-10]["close"]) * 0.97,
                "take_profit": float(frame.iloc[-10]["close"]) * 1.06,
                "risk_budget": 10.0,
                "account_equity": 1000.0,
                "leverage": 1.0,
                "currency": "USDT",
            },
            interval="5m",
            dark_mode=True,
        )
        names = {trace.name for trace in figure.data}
        annotations = [str(annotation.text) for annotation in figure.layout.annotations]

        self.assertIn("即時未收盤 K 線", names)
        self.assertIn("RL 多單進場／加碼", names)
        self.assertIn("Transformer 預測端點", names)
        self.assertIn("RL 原始目標", names)
        self.assertIn("風控後最終目標", names)
        self.assertGreaterEqual(len(figure.layout.shapes), 5)
        self.assertTrue(any("停利" in text for text in annotations))
        self.assertTrue(any("停損" in text for text in annotations))
        self.assertTrue(any("R/R 2.00" in text for text in annotations))
        self.assertTrue(any("本金" in text and "部位" in text for text in annotations))
        self.assertTrue(
            any(
                "帳戶曝險" in text and "槓桿 1.00x" in text
                for text in annotations
            )
        )
        self.assertLess(
            float(figure.layout.yaxis.range[1]),
            float(frame.iloc[-10]["close"]) * 1.06,
        )

        full_range = make_model_trading_figure(
            frame,
            position={
                "quantity": 1.0,
                "entry_time": signal_time,
                "entry_price": float(frame.iloc[-10]["close"]),
                "stop_loss": float(frame.iloc[-10]["close"]) * 0.97,
                "take_profit": float(frame.iloc[-10]["close"]) * 1.06,
            },
            interval="5m",
            fit_position_levels=True,
        )
        self.assertGreater(
            float(full_range.layout.yaxis.range[1]),
            float(frame.iloc[-10]["close"]) * 1.06,
        )

    def test_model_trading_chart_supports_short_position_box(self):
        frame = make_feature_frame().tail(30).copy()
        entry = float(frame.iloc[-5]["close"])

        figure = make_model_trading_figure(
            frame,
            position={
                "quantity": -0.5,
                "entry_time": frame.iloc[-5]["timestamp"],
                "entry_price": entry,
                "stop_loss": entry * 1.02,
                "take_profit": entry * 0.96,
                "risk_budget": 5.0,
            },
            interval="1d",
        )
        annotations = [str(annotation.text) for annotation in figure.layout.annotations]

        self.assertTrue(any("空單進場" in text for text in annotations))
        self.assertTrue(any("R/R 2.00" in text for text in annotations))


if __name__ == "__main__":
    unittest.main()
