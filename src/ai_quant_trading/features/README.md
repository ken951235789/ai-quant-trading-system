# features

## 比例化與市場情境

`market_context.py` 提供訓練／執行共用的價格比例、跨週期共識、方向熵、
趨勢與反轉衝突及資料覆蓋率。原始 OHLCV 留作圖表、標籤與撮合，不進入新版模型輸入。
已完成因果對齊的資料才可用於情境計算；沒有來源時以 coverage=0 標記。
詳細說明見 `docs/compact_ai_inputs_20260915.md`。

## 多週期資料集

「特徵資料」頁可把 BTC/USDT 的 1m、3m、5m、15m、30m、1h、4h、12h、1d
依已收盤時間融合成 `features_mtf_*.csv`。決策週期保留 OHLCV，每個週期另外輸出
趨勢、動能、波動、成交量、市場結構、可用性與新鮮度，供 Transformer 與 SAC
共用。完整操作請參考
`docs/BTC_15m_SAC_Transformer_落地說明.md`。

多週期 schema v4 加入可量化的因果 SMC：確認式 Swing、BOS、CHoCH、流動性掃描與
三根 K 線 FVG。Pivot 必須等右側 K 線收盤，並從下一根才可使用，避免重繪與未來資訊。
主觀 Order Block、手動畫區與「機構一定在此下單」的假設不納入模型。

## 進階特徵融合

`advanced.py` 負責市場廣度與外部資料的時間對齊。所有合併都採
`backward-asof`，只能讓 K 線讀到當時已公開的最近資料。Dashboard 的
「進階資料 → 特徵融合」會直接更新固定的 `features_*.csv`，供 Transformer
與 SAC 使用。

`intraday.py` 額外提供 BTC 日內交易的因果式特徵，包括 Supertrend、Donchian、
Stoch RSI、MFI、Parkinson 波動、CMF、主動成交量差、CVD、Funding／OI 狀態及
UTC 交易時段。缺少微結構或合約資料時會輸出中性值，並用 available 欄位標記。

最新已收盤 K 線即使尚未產生未來 target，也會保留在特徵檔中，讓正式推論可以
輸出最新訊號；訓練器仍會自行排除沒有未來標籤的尾端樣本。

負責把 Step 2 的 OHLCV CSV 轉換成 AI 模型可使用的特徵資料集。

## 已完成特徵

- Trend：單期報酬、SMA 5/20/60/200、EMA 5/20/60/200
- Momentum：RSI 14、MACD、MACD Signal、MACD Histogram、ROC 12
- Volatility：ATR 14、Bollinger Band 20、Band Width、Historical Volatility 20
- Volume：Volume Change、OBV、累積 VWAP
- Market Structure：Higher High、Higher Low、突破前 20 期高點
- Short-term Reversal：RSI(2)、價格相對 EMA5／EMA200 距離、20 期布林 Z-score、
  回檔進場事件、修復退出事件與規則參考持倉

完整模型特徵欄位定義在 `indicators.py` 的 `FEATURE_COLUMNS`。

## Target 定義

- `future_return`：`未來第 N 期 close / 本期 close - 1`
- `target`：當 `future_return > target_threshold` 時為 1，否則為 0
- 最後 N 筆資料沒有未來價格，因此 target 會是缺值

`future_return` 和 `target` 只可作為訓練答案，不可放入模型輸入欄位。

## 使用方式

先從專案根目錄下載至少 300 筆資料，讓 MA200 有足夠的暖機資料：

```powershell
python scripts\download_market_data.py --exchange binance --symbols BTC/USDT --interval 1d --limit 300
```

再將一或多份 OHLCV 檔案批次轉成特徵資料集：

```powershell
python scripts\build_features.py --input data\raw\crypto\binance\ohlcv\ohlcv_crypto_binance_BTC-USDT_1d_latest.csv data\raw\crypto\binance\ohlcv\ohlcv_crypto_binance_ETH-USDT_1d_latest.csv --target-horizon 1
```

美股使用相同流程，只要把 `--input` 換成 Yahoo Finance OHLCV：

```powershell
python scripts\build_features.py --input data\raw\us_equity\yahoo_finance\ohlcv\你的檔名.csv --target-horizon 5
```

輸出會放在 `data/processed/`。預設會刪除 MA200 暖機期和最後 N 期的缺值列；使用 `--keep-na` 可保留這些列供人工檢查。

重要原則：任何 feature 只能使用當下與過去資料，不得使用未來資料。
短線回檔欄位只提供市場狀態與規則先驗；五年候選排名、保留測試報酬及勝率不會放入
特徵，避免把樣本外答案洩漏給模型。
Dashboard 可一次選擇最多 20 個標的、K 線週期與筆數，按下「自動下載並建立
特徵資料集」；系統會逐一下載並保存獨立 CSV，不需要先找檔案。快速清單會自動
帶入長期或短期標的，短期加密預設 `4h`、短期美股預設 `1h`。建立時會排除尚未
收盤的最後一根 K 線，並保存 `target_timestamp`，讓模型切分時清除跨越訓練／
驗證／測試邊界的標籤。
