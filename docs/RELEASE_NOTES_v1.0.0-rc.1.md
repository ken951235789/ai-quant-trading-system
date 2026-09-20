# v1.0.0-rc.1 Research Preview

這是第一個適合公開技術審查的研究候選版本，不是實盤交易產品。

## 包含內容

- BTC/USDT USD-M Futures 多時間週期資料與因果特徵流程。
- Transformer V3 多任務市場分析架構與 cross-fit OOS 預測。
- SAC 連續曝險、多空、續抱、平倉共用動作契約。
- 成本、Funding、保證金、強平與獨立固定風控。
- Walk-forward、消融、final holdout 與 Champion 升級流程。
- 模擬倉、Testnet gateway、事件重播、故障注入、Watchdog 與 Kill Switch。
- 不需要 API Key 或權重的五分鐘公開 Demo。

## 已知限制

- 5 根短線 Transformer 預測尚未穩定超越基準。
- 已評估的 SAC 正式候選在四次樣本外測試皆為負。
- 尚無模型通過 Champion 與正式資金品質閘門。
- 公開版不包含資料、權重、交易紀錄、憑證或可執行檔。

## 驗證

```powershell
python -m pip install -e ".[dev]"
python -m ruff check src tests scripts
python -m pytest -q
ai-quant-demo --open
```

完整數字請閱讀 `docs/RESEARCH_RESULTS.md`，公開範圍請閱讀
`docs/PUBLIC_RELEASE_SCOPE.md`。
