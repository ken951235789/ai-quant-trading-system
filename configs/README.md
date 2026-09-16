# 設定檔說明

本資料夾只保留 BTC/USDT USD-M 永續合約 15 分鐘系統會使用的設定範本。

- `default.yaml`：交易所、五週期資料、逐倉槓桿、交易成本與風控預設值。
- `btc_15m_transformer.example.json`：Transformer 架構與正式訓練參數範本。
- `rl_training.example.json`：SAC 正式訓練參數範本；一次設定只代表一個 seed。
- `startup_refresh.example.json`：桌面 App 啟動時的市場、新聞、FinBERT 與特徵更新設定。

正式實驗請另存一份設定並固定資料版本、時間切分與 seed，才能重現結果。API Key 不可
寫入這些檔案；目前 Binance Futures 真實下單仍被安全封鎖，未完成 Testnet Gateway
與驗收前不要設定實盤密鑰。
