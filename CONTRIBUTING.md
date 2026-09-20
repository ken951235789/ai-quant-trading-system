# 貢獻指南

感謝你對這個研究專案有興趣。貢獻內容應優先改善可重現性、資料安全、測試覆蓋、風控或
研究方法，而不是只提高單次回測數字。

## 開發流程

1. 先建立 Issue，說明問題、資料期間、預期行為與可能風險。
2. 從 `main` 建立小範圍分支。
3. 不提交 API Key、`.env`、真實交易紀錄、模型權重或市場資料。
4. 新行為必須加入測試，中文註解只放在無法由程式本身清楚表達的地方。
5. 提交前執行下列驗證。

```powershell
python -m ruff check src tests scripts
python -m pytest -q
ai-quant-demo --output outputs/public_demo_smoke --bars 360 --seed 11
```

## 研究要求

- 時間序列不可隨機打散後宣稱為樣本外結果。
- Scaler、補值、特徵選擇與門檻只能使用訓練區段估計。
- 所有績效須揭露資料期間、seed、費用、滑價、Spread、Funding 與切分方式。
- final holdout 不可用於反覆挑參數。
- 不得使用「保證獲利」、「穩賺」或把訓練 reward 當作真實報酬的描述。

## Pull Request 範圍

每個 PR 應只處理一個清楚主題，並在說明中列出：問題、方法、測試、結果、限制與是否改變
任何資料／模型契約。安全弱點不要公開建立 Issue，請依 `SECURITY.md` 私下回報。
