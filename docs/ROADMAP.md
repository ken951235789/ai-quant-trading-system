# BTC 15m 開發 Roadmap

## 已完成

- BTC/USDT Binance USD-M Futures REST 五週期資料下載。
- Futures WebSocket K 線與 BookTicker 即時傳輸。
- Funding、OI、Basis、多空比、Taker Ratio、Spread 與 Order Book 快照。
- 已收盤 K 線過濾、固定 CSV 合併去重與舊資料覆寫。
- 五週期因果特徵、60/20/20 時序切分與 train-only scaler。
- Temporal Transformer 訓練、進度、checkpoint、推論與 latent 融合。
- SAC 連續多空目標曝險、PPO 離散比較動作。
- 永續帳務、2 倍預設／3 倍上限、fee、spread、slippage、funding 與 liquidation。
- 單筆 0.25%、單日 1%、最大回撤 10%、保證金 20%、名目曝險 50% 風控。
- Transformer/SAC 歷史回測與持久化模擬交易。
- 深淺色桌面 UI、原生 WebView2 EXE 與啟動自動更新。
- Transformer V3 與 SAC smoke 訓練、模擬倉流程及完整自動測試。
- 訓練前資料完整性檢查、中度 Drift 自動降倉、回撤曲線縮倉與入出金修正回撤。

## 目前模型狀態

- 舊版 Transformer V2、SAC smoke 成品與相依回測已退出主線並刪除。
- 目前只接受 Transformer V3 checkpoint；尚未包含可進入 Testnet 的正式候選模型。
- Smoke 的目的只限驗證端到端流程、風控與儲存，不代表預測優勢或實盤資格。

## 正式研究下一步

1. 準備涵蓋牛、熊、盤整與高波動的多年 5m/15m/1h/4h/1d Futures 資料。
2. 建立資料版本與 walk-forward folds，另保留從未調參的最終 holdout。
3. 用多個 seed 訓練 Transformer，先驗證校準、MAE、方向與 regime 穩定性。
4. 用至少 5 個 seed 訓練 SAC，比較無 Transformer、Transformer 與 FinBERT ablation。
5. 以 PF、Expectancy、Calmar、回撤、turnover、費用、funding 與零強平篩選候選。
6. 通過後連續執行 3 至 6 個月持久化模擬倉。

## 實盤前仍需完成或取得證據

- 綁定 Binance Futures Testnet，核對帳戶、One-way、逐倉、槓桿與 API 權限。
- User Data Stream 已成為主要訂單／持倉事件來源；仍需真實 Testnet 長時間驗證 listen key、
  斷線補帳、限流與維護期間的 gap 行為。
- 離線斷線重連、停損停利與 Kill Switch 演練已通過；仍需真實 Testnet 實機驗收。
- 完成時鐘同步、API 限流、通知送達、密鑰輪替與長時間故障演練。
- Testnet 穩定運行證據與人工審核；完成前維持 `trading_enabled: false`。
