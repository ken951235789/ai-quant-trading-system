# 專案分享素材

以下文字可依社群規則調整。發文重點是研究方法與失敗結果，不宣稱能帶來投資報酬。

## 中文短文

我公開了一個 BTC 永續合約 AI 量化交易研究專案。它把多時間週期因果特徵、Transformer
V3、SAC 連續曝險、獨立風控、含成本回測、事件重播與 Testnet 串成同一套可測試流程。

目前 SAC 樣本外結果仍為負，因此 Live 預設鎖定；我也保留了失敗結果、資料契約與 final
holdout 規則。倉庫提供無 API Key 的五分鐘離線 Demo，歡迎針對資料洩漏、成本模型與
RL 評估方法提出建議。

## English Post

I published a research-first BTC perpetual-futures platform connecting causal
multi-timeframe features, Transformer V3 forecasts, SAC target exposure, independent
risk controls, cost-aware evaluation, event replay and a locked live gateway.

The current SAC candidate remains negative out of sample, so it was not promoted. The
repository keeps that result visible and includes a no-key, deterministic five-minute
demo. Feedback on leakage prevention, transaction-cost modelling and RL evaluation is
especially welcome.

## 發布檢查

- 確認貼文沒有 API Key、帳號、絕對本機路徑或未公開結果。
- 附上研究問題或具體技術問題，不跨社群重複洗版。
- 遵守各社群自我宣傳規則，回覆技術問題並修正文檔。
- 分享 GitHub Release 或 README，不上傳真實權重與私人資料。

適合的內容形式包括 Kaggle 可重現實驗、LinkedIn 工程摘要、學校專題文章，以及允許專案
分享的量化／機器學習社群。各平台規則會變動，發布前應重新確認。
