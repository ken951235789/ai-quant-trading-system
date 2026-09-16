# BTC 15m SAC + Transformer 落地說明

## 1. 系統定位

本系統只研究 `BTC/USDT` Binance USD-M 永續合約。15 分鐘是決策週期；5 分鐘提供
較細的執行脈絡，1 小時、4 小時與 1 日提供趨勢與市場狀態。所有輸入必須在決策時刻
已經收盤，訂單以下一根 15 分鐘 K 線開盤價模擬，避免 Look Ahead Bias。

SAC 是主要模型，輸出 `-1 ~ 1` 的連續目標曝險：負值做空、正值做多、接近零減倉或
平倉。PPO 只作離散比較，動作為 Hold、Long、Short、Close。Transformer 不直接下單，
它把多週期金融指標轉成市場 latent、未來 1/5/20 根報酬、波動與 regime 預測，供 SAC
與風控閘門使用。

## 2. 已納入資料

- K 線：OHLCV、quote volume、成交筆數、taker buy volume。
- 趨勢：EMA 20/50/200、SMA 200、ADX、DI、斜率與 ATR 標準化距離。
- 動能：RSI、MACD、ROC、短週期 RSI 2。
- 波動：ATR、Bollinger width、歷史波動、實現波動與波動 regime。
- 成交量：Relative Volume、VWAP、OBV、CMF 與 taker buy ratio。
- 結構：前高低、Swing、突破、BOS/CHoCH 類結構欄位與 K 線型態。
- 永續合約：Funding、Open Interest、basis、多空比、mark/index price。
- 微結構：Bid/Ask spread、microprice、5/10/20 檔深度與 imbalance。
- 時間：UTC 小時、星期、亞洲／歐洲／美國時段、週末與 funding 時點。
- 帳戶狀態：目前曝險、現金比例、平均進場價距離、回撤、持倉時間、已實現／
  未實現損益，以及距停損、停利與強平價的距離。

新聞 FinBERT 是可選資訊。沒有具時間戳且可重現的新聞資料時，
`finbert_available=0`，不可把補零誤當中性觀點。正式 SAC 默認不啟用 FinBERT；
只有因果對齊後的歷史覆蓋率至少 30% 時才允許納入訓練。

## 3. 資料與切分

目前資料檔：

```text
data/processed/features_mtf_crypto_binance_futures_BTC-USDT_15m_5m-15m-1h-4h-1d_h1.csv
```

建立資料時會按時間順序切成 60% 訓練、20% 驗證、20% 測試。缺值填補與標準化只從
訓練區段估計，再套到後面區段。不要隨機 shuffle，也不要用測試績效反覆挑參數。

## 4. 風控與執行

- 逐倉預設 2 倍，絕對上限 3 倍。
- 每筆最大風險 0.25%，單日虧損達 1% 停止新倉。
- 帳戶最大回撤 10%，連續虧損 3 筆停止交易。
- 最大保證金使用 20%，最大名目曝險 50%。
- ATR 停損限制在價格的 0.4% 到 1.5%。
- Spread 或預估滑價超過 5 bps、事件封鎖、資料缺口或異常波動時不開新倉。
- 新 SAC 的動作絕對值 `0～0.03` 是等待／續抱區，`0.03～0.10` 是主動平倉區，
  超過 `0.10` 才提出多空目標；舊模型仍使用原本的空手中性區契約。
- 固定保護性停利為 1.5%；模型仍可在到價前自行減倉或平倉。
- 毛利成本倍數與淨風報比不再宣告單一最佳值，必須用預先定義的 validation 消融選擇；
  final holdout 不可拿來調整門檻。
- 回測計入 taker fee、spread、slippage、funding 與 liquidation fee。
- 訓練 Episode 的本金預設在正負 10% 內抽樣，滑價在正負 1 bp 內抽樣；
  驗證與測試自動關閉這兩項隨機化。

## 5. 本次可執行驗收

Transformer V3 與 SAC 的 Smoke Test 只用來證明資料、Scaler、模型、儲存、
推論與交易環境可以跑通，不代表有預測優勢或實盤資格。舊模型成品已退出主線，
新的模擬與實盤環境只能引用 Transformer V3 Checkpoint。

回測結果保存在 `data/processed/model_backtests/latest_sac/`，再次執行會覆寫，不會
累積大量同類輸出。

## 6. 正式訓練方式

1. 準備至少涵蓋多種牛、熊、盤整與高波動狀態的歷史資料；目前約一個多月資料不足。
2. 先依 `configs/btc_15m_transformer.example.json` 訓練 Transformer。
3. 檢查測試區間 MAE、方向準確率、regime confusion matrix 與預測校準。
4. 建立新的 SAC 環境，選取通過檢查的 Transformer checkpoint。
5. 依 `configs/rl_training.example.json` 分別使用至少 5 個 seed 訓練。
6. 用 walk-forward 和完全保留的最終 holdout 比較 PF、expectancy、Sharpe、Sortino、
   Calmar、最大回撤、turnover、費用、滑價、funding 與強平次數。
7. 通過後持續跑 3 到 6 個月模擬倉，再做 Binance Futures Testnet。

最低研究門檻可先設為：測試 PF 大於 1.2、expectancy 大於 0、最大回撤小於 10%、零強平，
而且多個 seed 與多個 walk-forward 區間結果一致。這些只是篩選門檻，不是實盤保證。

## 7. Live 落地狀態

已完成 Binance USD-M Futures Testnet gateway、逐倉與 1 至 3 倍槓桿驗證、`reduceOnly`、
client order id 去重、多空條件保護單、持倉 reconciliation 與 kill switch。研究、回測、
模擬與下單均固定為 BTCUSDT USD-M 永續；Spot 模型不能送到 Futures。

Futures User Data Stream 主動成交回報、斷線重連與 REST 補帳已完成；離線安全演練也已驗證
保護單契約與 Kill Switch。尚未完成的是真實 API Key Testnet 長時間整合驗收與足夠營運
證據，因此正式資金仍保持鎖定。
