# BTC 短線模型特徵組合

## 目標與時間框架

- 交易標的：`BTC/USDT` USDⓈ-M 永續合約。
- 決策週期：`15m` 已收盤 K 線，先控制過度換手與執行噪音。
- 背景週期：`5m`、`1h`、`4h`、`1d`。
- Transformer：讀取多時間框架序列，輸出報酬分布、波動、市場狀態與不確定度。
- SAC／PPO：根據 Transformer 輸出、即時市場特徵及帳戶風險狀態，決定目標多空曝險。
- 槓桿：模型只能提出目標曝險，最終部位由獨立風控裁切；Futures Gateway 固定逐倉，
  僅允許 1 至 3 倍且預設 2 倍。

## 第一版採用組合

### 1. 價格、結構與趨勢

- 報酬：1、3、12 根 K 線對數報酬。
- K 線形態：實體、上下影線、振幅，全部除以收盤價正規化。
- EMA：20、50、200 的價格距離、斜率及 20/50、50/200 相對位置。
- 趨勢狀態：ADX 14、Efficiency Ratio 20、Choppiness Index 14。
- 關鍵價位：當日高低點、前日高低點、區間中點及其價格距離。
- 突破結構：Donchian 20 上下緣、Swing High/Low、BOS、CHoCH。

### 2. 動能與波動

- 動能：RSI 14、MACD Histogram / Close、ROC 12。
- 波動：ATR 14 / Close、Bollinger Band Width、Bollinger Z-score。
- 實現波動：5、20、60 根 K 線；另加入 Parkinson Volatility 20。
- 波動狀態：短期／中期實現波動比，用來分辨擴張與收縮。

### 3. 成交量與訂單流

- Volume Z-score 20、Relative Volume 20、Dollar Volume Z-score 20。
- 每日重設的 Session VWAP 距離；不能使用從資料第一天一路累積的 VWAP。
- Taker Buy Ratio、單根 Delta、Cumulative Volume Delta 斜率。
- CMF 20；OBV 只保留斜率，不把無界累積值直接交給模型。
- Trade Intensity、大額成交多空差、主動成交加速度。

### 4. 市場微結構

- Bid-Ask Spread（bps）。
- Order Book Imbalance：前 5、10、20 檔。
- Microprice Deviation、買賣深度、Depth Slope。
- 預估滑價、資料年齡、WebSocket 延遲與資料是否失效。

### 5. BTC 永續合約

- Funding Rate、Funding Percentile、距離下次資金費率結算時間。
- Open Interest Change、OI Z-score。
- Basis、Premium、Mark Price / Index Price 價差。
- Global Long/Short Ratio、Taker Buy/Sell Ratio。
- 多空 Liquidation Volume 與清算不平衡。

### 6. 時間與帳戶狀態

- 小時與星期使用 `sin/cos` 編碼，另標示亞洲、歐洲、美國時段及週末。
- 當日已實現波動、當日區間位置、重大事件禁止交易旗標。
- PPO 帳戶狀態：目前方向、曝險、進場價距離、未實現損益、持倉時間、回撤、當日損益、連續虧損、保證金安全度。

## Transformer 輸出給 PPO 的資料

- 未來 1、3、12 根 K 線的報酬分位數，不只預測單一漲跌方向。
- Bull、Neutral、Bear 機率。
- 預期波動與市場狀態：趨勢、震盪、高波動。
- 模型不確定度；不確定度過高時 PPO 應降低曝險或不交易。
- Latent Context 僅在時序外推驗證有效後才加入 PPO。

## 目前已經具備

- OHLCV、成交量、成交筆數、主動買方成交量。
- SMA/EMA 5、20、60、200，RSI、MACD、ROC、ATR、Bollinger Bands。
- 歷史波動、Volume Change、OBV、VWAP、Higher High/Low、20 根突破。
- 多時間框架合併機制，且會避免使用尚未收盤的高週期 K 線。
- Funding、OI 變化、Spread、Long/Short Ratio、Taker Ratio、Basis。
- 訂單簿 5/10/20 檔不平衡、Microprice、FinBERT 與 Transformer 欄位。
- PPO 已能表達多空目標曝險，但目前最大曝險仍限制在 1 倍。

## 必須補齊或修正

- 本機目前只有 `1d`、`4h` K 線，缺少訓練所需的 `1m`、`5m`、`15m`、`1h`。
- EMA 使用 60 而非 50；缺少 ADX、Donchian 下緣、Choppiness、Parkinson Volatility。
- Higher High/Low 目前只比較前一根，不是真正 Swing、BOS 或 CHoCH。
- VWAP 目前從整份資料起點累積，日內模型必須改為每日重設。
- 目前訂單簿只有少量快照，無法訓練；要用 WebSocket 持續保存深度與成交事件。
- 衍生品歷史資料量很短，Liquidation、Funding Percentile、OI Z-score 尚未加入。
- PPO 尚未模擬永續合約保證金、Funding、強平價格與 1～3 倍槓桿。
- 新增欄位後，舊 Transformer、PPO 與 Scaler 皆不可沿用，必須用同一份特徵定義重訓。

## 第一版暫緩

- SMA 與 EMA 全套同時使用、Ichimoku、Supertrend、Parabolic SAR、Aroon。
- Stochastic、Stochastic RSI、CCI、Williams %R、TSI 全部一起加入。
- Volume Profile、Footprint、Absorption、Spoofing 判斷。
- DXY、VIX、美股、ETF Flow 等跨市場特徵。

這些不是永遠不用，而是先避免高度共線、資料不足或時間對齊不可靠。第一版通過 Walk-forward、成本壓力測試與特徵消融後，再逐組加入比較。

## 驗證門檻

- 所有特徵只能使用當下已完成的 K 線與已收到的事件，禁止回看未來資料。
- 訓練、驗證、測試按時間切割，並使用 Walk-forward，不隨機打散。
- 回測必須計入 Maker/Taker Fee、Spread、Slippage、Funding 與延遲。
- 每組新增特徵都做 Ablation Test；若多個 Seed 的測試結果沒有一致改善，就移除。
- 先通過離線回測，再跑 Testnet／模擬倉，最後才考慮小額實盤。
