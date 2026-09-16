# BTC 15 分鐘系統架構

## 系統邊界

目前唯一研究市場是 Binance `BTC/USDT` USD-M 永續合約，決策週期固定為 15 分鐘。
系統不是高頻撮合引擎；Futures Testnet／Live Gateway 已完成，但 Live 仍由品質、Champion、
資料時效、User Stream、對帳、緊急停機與人工確認共同鎖住。

```text
Binance Futures REST / WebSocket
        |
        v
統一 MarketEvent（確定性 ID／去重／排序／缺 K／checkpoint）
        |
        v
已收盤 5m / 15m / 1h / 4h / 1d K 線
Funding / OI / Basis / Spread / Order Book
        |
        v
因果特徵工程與 60/20/20 時序切分
        |
        v
資料品質守門（缺 K / 過期 / 未收盤 / Spread）
        |
        v
Temporal Transformer
報酬 + 波動 + Regime + Latent
        |
        v
SAC 連續目標曝險（多 / 空 / 平）
        |
        v
Regime Gate + 動態資金配置 + 固定風控
        |
        v
歷史回測 -> 持久化模擬倉 -> Futures Testnet -> 受控 Live
```

## 模組責任

### `data_collection`

下載 Futures K 線、24 小時統計、Funding、Open Interest、基差、多空比、主動買賣比、
Spread 與 Order Book。REST 固定檔只保留已收盤 K 線；WebSocket 提供未收盤即時快照。

### `features`

建立趨勢、動能、波動、成交量、市場結構、永續合約、微結構與時間特徵，並用 K 線
收盤時間做因果多週期對齊。較高週期尚未收盤時不可提前使用。

### `transformer`

讀取一段五週期特徵序列，輸出 1、5、20 根報酬、波動、regime 機率、不確定度與
latent。Scaler 只使用訓練區段配適。

### `reinforcement_learning`

SAC 是主要模型，輸出 -1 到 1 的連續目標曝險；PPO 使用 Hold、Long、Short、Close
作比較。Observation 同時包含市場、Transformer 與帳戶風險狀態。

### `trading`

定義 Transformer、SAC、市場狀態與資金管理之間的資料合約。Data Guard 會拒絕缺漏、
過期、未收盤、跳價或 Spread 過大的行情；Model Guard 使用訓練期 mean/std 偵測漂移；
Orchestrator 依序執行 Regime Detector 與 Capital Manager，且不會阻擋任何降風險交易。
`MarketEventBus` 讓歷史 CSV 與 WebSocket 使用相同事件 ID、排序、去重及 consumer 介面。

### `research`

把歷史 OHLCV 轉成正式 `MarketEvent`，以虛擬時間重播並在成功投遞後原子更新 checkpoint。
故障注入涵蓋重複、斷線、consumer 崩潰、亂序、缺 K 與損壞 OHLCV，並保存資料 SHA、
Git commit、套件版本、參數、結果摘要與可重現命令。

### `backtesting`

以本期收盤決策、下一根開盤成交，計入 fee、spread、slippage、funding、停損與強平，
統計 PF、Expectancy、Sharpe、Sortino、Calmar、回撤、turnover 與 liquidation。

### `risk`

限制逐倉槓桿、保證金、名目曝險、單筆風險、ATR 停損、單日虧損、連虧與最大回撤。
異常 spread、滑價、ATR、模型漂移與事件封鎖時禁止新倉。

### `paper_trading`

以永續合約帳務模擬做多與做空，持久化保存帳戶、訂單、成交、資產曲線與模型決策。
反向動作會先平倉，再於後續決策開立相反部位。

### `live_trading`

Futures Gateway 支援多空目標、ReduceOnly、唯一訂單 ID、未知結果查單、交易所停損停利、
User Data Stream、REST 對帳與 Kill Switch。正式環境只接受 Registry Champion；私有事件
心跳或對帳過期會自動啟用緊急停機，恢復後仍需人工解除。

### `dashboard`

提供資料更新、特徵建構、Transformer/SAC 訓練、AI 回測、模擬倉與機器人監控。桌面
版使用本機 Streamlit 服務加 WebView2，不會把資料送往外部 Dashboard 服務。

## 不偏差原則

- 只使用決策時刻以前已發布且可取得的資料。
- 依時間切分 60% 訓練、20% 驗證、20% 測試，不隨機打散。
- Scaler、缺值統計與特徵選擇只能從訓練區段估計。
- 測試集不可用來反覆挑參數；正式候選另保留最終 holdout。
- 模型成品必須帶資料來源、特徵欄位、風控、成本、seed 與版本 metadata。
