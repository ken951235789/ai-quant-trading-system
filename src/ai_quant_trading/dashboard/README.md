# dashboard

Streamlit 研究操作介面，直接串接 Step 2 到 Step 9。

## 第一版功能

- 從 Binance 下載加密貨幣 OHLCV 與 24 小時市場快照
- 直接搜尋 Binance 目前全部可交易 USDT 現貨，不必手動輸入交易對
- 從 Nasdaq 官方目錄取得 Nasdaq、NYSE 等美國掛牌證券，再由 Yahoo Finance 下載 OHLCV
- 選擇 K 線週期、日期範圍與下載筆數
- 選擇標的後自動下載足夠 K 線並建立完整特徵資料集，不必自己指定 CSV
- 調整圖表顯示 K 線數與 SMA / EMA 疊加線
- 查看 RSI、MACD、ATR、歷史波動率、成交量變化與 OBV
- 調整 FinBERT 推論、Transformer 架構與 PPO 多模態融合設定
- 顯示 FinBERT 情緒與 Transformer 多週期上下文
- 重跑 PPO 的訓練、驗證與保留測試歷史資料
- 重新推論 Transformer 歷史序列並建立可調整的多空目標部位
- 調整初始資金、手續費、滑價、空單成本與部位上限
- 顯示模型與 BTC 基準資金曲線、曝險、回撤、預測品質及交易紀錄
- 每種模型只覆寫保存最近一次回測，避免輸出無限累積
- 建立或續跑持久化模擬帳戶，自動下載最新資料並顯示模擬訂單、持倉與資產曲線
- 模擬帳戶中斷後會逐根補跑所有新 K 線，不會直接跳到最新價格
- PPO 成品選單顯示環境、演算法與資料截止日，未通過模型治理門檻時禁止真實送單
- 以中文／英文資產名稱快速選擇虛擬貨幣與美股，CSV 仍使用穩定市場代碼
- 查詢 Binance USD-M Testnet、Demo 或 Live 帳戶，驗證模型調倉與緊急 reduce-only 全平
- 讓已訓練模型經過 Step 6 風控、交易所規則、白名單與金額上限後執行一次實盤輪次
- 從模擬或 Binance 頁啟動獨立背景輪詢，顯示心跳、錯誤熔斷與安全停止
- 顯示 Binance Futures 停損停利條件單與緊急停機狀態
- 建立 Step 9 Gymnasium 投資組合環境，使用 CPU／CUDA 訓練 PPO／SAC
- 將短線回檔市場特徵、未實現報酬與持有時間加入 RL observation
- 調整完整 PPO／SAC 超參數、神經網路、Checkpoint、評估頻率與續跑模型
- 顯示驗證／測試資產曲線、訓練曲線、最佳模型與完整訓練設定
- 從側邊欄切換深色／淺色背景，交易圖表會同步套用目前主題
- 將功能整理為交易中心、模型中心與資料中心，中心內只載入目前選取的功能
- 在「交易中心 → 機器人監控」集中查看今日、本週、累計損益與資產曲線
- 每 5 秒更新機器人心跳、持倉、勝率、最大回撤、交易紀錄與風控停機狀態
- 在即時 15m K 線疊加 RL 多空進出場、原始／最終目標部位與 Transformer 預測端點
- 以紅綠風險報酬框顯示目前進場、停損、停利、數量、浮動損益、風險金額與 R/R
- 價格範圍可切換「K 線細節」與「完整停損停利」，避免遠距停利把短線蠟燭壓扁
- 依實際持倉顯示多空方向、使用本金、目前名目部位、帳戶曝險與槓桿倍數
- 模擬機器人運行時直接顯示 WebSocket 尚未收盤 K 線，但模型仍只使用已收盤資料
- 每次啟動桌面 App 時在背景增量更新既有行情、新聞、FinBERT 與特徵資料
- 左側只顯示自動更新進度，不需要額外按啟動按鈕
- PPO 與 Transformer 共用後端工作鎖；切換頁面也不能重複送出 GPU 工作
- 模擬機器人使用持續運行開關；開啟一次後，重開桌面 App 會自動恢復

## 機器人監控

開啟 App 後，在左側選擇「交易中心」，再選擇「機器人監控」。

1. 「模擬交易」會讀取 `data/paper_trading/{account_id}/` 的帳戶、績效與交易紀錄。
2. 「實盤／測試網」會讀取 `data/live_trading/{environment}/` 的帳戶快照、訂單與自動化心跳。
3. 今日與本週損益以台北時區計算；累計損益以帳戶初始資金為基準。
4. 「即時模型交易圖」顯示 RL 實際進出場、目標部位與 Transformer 的 1／5／20 根預測端點。
5. 「資產績效」可切換資產曲線、累計損益與回撤，並可查看 7 天、30 天、90 天或全部紀錄。
6. 「機器人狀況」會顯示執行中、未啟動、資料延遲、風控停機或執行失敗等狀態。
7. 模擬與 Live 都以 BTC USD-M 永續契約計算帶方向曝險；Live 固定逐倉並允許 1 至 3 倍，
   介面會分開顯示帳戶權益、名目部位與槓桿。

監控頁只呈現帳戶已保存的真實紀錄，不會製造預估獲利。建立模擬帳戶並至少執行一次交易輪次後，頁面才會開始累積曲線。

## 安裝

```powershell
cd D:\AIQuantTradingSystem
python -m pip install -e ".[dashboard,ai,rl]"
```

## 啟動

```powershell
python -m streamlit run scripts\run_dashboard.py
```

預設網址為 `http://localhost:8501`。

直接使用 `streamlit run` 時不會啟動桌面 Launcher 的外部更新工作；雙擊 Windows App 時會在介面服務就緒後自動執行，不需要另外按按鈕。

## Windows App

已建置的桌面版位於：

```text
D:\AIQuantTradingSystem\dist\AIQuantTradingSystem\AIQuantTradingSystem.exe
```

雙擊後會直接開啟 Windows WebView2 桌面視窗，不會另外開啟瀏覽器。背景服務會在本機尋找 `8501` 起第一個可用連接埠；所有 CSV 與回測輸出都寫在 EXE 同層的 `data`。AI 回測固定保存於 `data/processed/model_backtests/latest_ppo` 與 `latest_transformer`。按視窗右上角 X 時 Dashboard 服務會停止；已開啟「機器人持續運行」的模擬工作仍會繼續，需回到模擬機器人頁關閉切換。

FinBERT 情緒保存於 `data/processed/sentiment/`，AI 管線設定保存於 `data/config/ai_pipeline.json`。模擬帳戶保存於 `data/paper_trading/{account_id}/`。Step 9 環境保存在 `data/processed/rl/environments/{run_id}/`，PPO／SAC 成品保存在該環境的 `training/{training_run_id}/`。

實盤頁預設使用 Binance USD-M Testnet 且只驗證。API Key 請從 `.env.example` 建立 `.env`，
不要在 Dashboard 輸入或保存密鑰。Testnet、Demo、Live 的訂單與本地持倉保存在
`data/live_trading/{environment}/`；Futures 部位另用 `usd_m_futures:` 命名空間隔離。
模型多空調倉後會建立交易所端停損停利；手動區只提供緊急 reduce-only 全平。
