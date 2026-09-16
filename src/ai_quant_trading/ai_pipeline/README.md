# AI 管線

此模組管理 PPO、FinBERT 與時序 Transformer 的共同設定。

`pipeline.json` 只保存模型名稱、架構、訓練超參數與功能開關，不保存 API Key。

資料流向：

1. FinBERT 將已發布新聞轉成情緒、信心、新聞數量與新鮮度。
2. Transformer 將一段市場序列轉成多週期報酬、波動與行情狀態。
3. PPO 同時讀取技術特徵、風控狀態與兩組 AI 上下文，輸出目標持倉比例。

FinBERT 或 Transformer 尚未產生正式輸出時，對應 `available` 欄位為 `0`，PPO 不應把零值誤認為有效看法。
