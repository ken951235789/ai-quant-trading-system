# FinBERT 新聞情緒

此模組把外部新聞文字轉換為 SAC／PPO 可以使用的數值特徵。

## 自動來源

- 加密貨幣：CoinDesk 官方 RSS，作為加密市場整體新聞。
- 美股：Yahoo Finance 搜尋新聞，依本機已追蹤的股票代碼收集。
- 自訂新聞：可在「資料與 AI → AI 管線 → FinBERT 新聞」匯入 CSV。

處理流程：

1. 新聞資料至少包含 `published_at`、`collected_at` 與 `title` 或 `text`；可信資料商可用
   `available_at` 取代 `collected_at`。
2. `FinBERTAnalyzer` 輸出正面、負面、中性機率、信心與連續情緒分數。
3. `aggregate_finbert_features` 使用 `max(published_at, collected_at/available_at)`，只讓模型
   看到當時已實際取得的新聞。
4. 舊新聞依半衰期降低權重，超過聚合視窗便不再使用。

`finbert_available=0` 表示該根 K 線沒有可用新聞，不可把它解讀為真正的中性看法。

第一次執行 FinBERT 時會下載 Hugging Face 模型至 `data/models/finbert/`，之後會使用本機快取。

自動收集的原文與評分結果使用固定檔名並去重：

```text
data/raw/news/news_latest.csv
data/processed/sentiment/finbert_news_latest.csv
```

每次開啟桌面 App 時，系統會在背景更新已存在的市場、新聞與特徵資料。左側可查看進度或按「立即更新」；設定位於「資料與 AI → 市場資料 → 啟動自動更新」。

FinBERT 不是交易訊號。SAC／PPO 取得的欄位包含情緒、正負機率、信心、新聞數量、
情緒變化與新聞新鮮度；`finbert_available=0` 時必須視為沒有新聞觀點。RSS 只能從啟用後
逐步累積，不能自動補齊過去五年的完整新聞歷史。

正式訓練預設要求具時間戳的歷史新聞覆蓋率至少 30%。覆蓋不足時仍可收集與檢視情緒，
但訓練前檢查與 Live 品質閘門會拒絕把 FinBERT 當成模型輸入，避免大量缺值被誤認為
中性市場。這個門檻可在 AI 管線設定調高，不建議降低。
