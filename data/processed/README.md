# data/processed

這裡存放清洗後資料與 AI 模型可用的特徵資料集。Step 3 產生的檔名格式為：

```text
features_{asset_class}_{exchange}_{symbol}_{interval}_{start}_{end}_h{target_horizon}.csv
```

例如 `features_crypto_binance_BTC-USDT_1d_20240101_20241231_h1.csv`，或 `features_us_equity_yahoo_finance_AAPL_1d_20240101_20241231_h5.csv`。

範例欄位：

- SMA
- EMA
- RSI
- MACD
- ATR
- Bollinger Band
- Volume Change
- OBV
- VWAP
- sentiment_score
- target

`future_return` 代表未來 N 期報酬，`target` 代表是否高於設定門檻。兩者都是訓練答案，不能洩漏到任何 feature 中。

Step 4 的回測結果位於：

```text
data/processed/backtests/{run_id}/
```

每次執行包含績效摘要 `summary.json`、逐期資金曲線 `equity_curve.csv` 與完整交易紀錄 `trades.csv`。

Step 5 的模型成品位於：

```text
data/processed/models/{run_id}/
```

其中 `test_predictions.csv` 只包含模型未見過的測試區段，應優先拿來做 Step 4／Step 6 回測；`all_predictions.csv` 含訓練與驗證區段，只供診斷，不可當成樣本外績效。

Step 9 的強化學習環境成品位於：

```text
data/processed/rl/environments/{run_id}/
```

每次包含 `environment.json`、`train.csv`、`validation.csv`、`test.csv` 與 `diagnostic.csv`。三段資料依時間順序切分，標準化平均值與標準差只從訓練段估計。

PPO／SAC 訓練結果位於環境資料夾內的 `training/{training_run_id}/`，包含最佳與最終模型、Checkpoint、訓練日誌，以及驗證／測試逐根評估。測試集只在選定最佳驗證模型後執行，不能反覆拿來調參。
