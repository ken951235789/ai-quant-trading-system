# tests

這個資料夾放單元測試與整合測試。

初期測試重點：

- 資料欄位是否正確
- 時間序列是否排序
- 特徵是否沒有使用未來資料
- 回測績效計算是否正確
- 風控是否能阻擋超出限制的交易
- 強化學習資料是否依時間切分且只用訓練段標準化
- Gymnasium 環境的觀測、動作、Reward、成本與終止條件是否正確
- PPO／SAC 是否能完成短訓練、保存模型、Checkpoint 與樣本外評估
- CUDA 不可用時是否拒絕 GPU 設定，而不是靜默退回 CPU

測試會隨各階段逐步補上。

## 執行測試

從專案根目錄執行：

```powershell
$env:PYTHONPATH="D:\AIQuantTradingSystem\src"
python -m unittest discover -s tests -p "test_*.py"
```

目前 Step 2 測試包含：

- Binance 回應解析
- OHLCV 資料驗證
- Market Data 資料驗證
- CSV 儲存路徑與內容檢查
- Yahoo Finance 美股 OHLCV 解析
