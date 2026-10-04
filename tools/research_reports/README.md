# 研究報告標準工具

這裡公開通用的報告骨架、驗證器與 JSON Schema，不依賴本機私有 Skills 或帳戶設定。

```bash
python tools/research_reports/new_report.py --type model_research --project AIQuantTradingSystem --scope "研究驗證" --output outputs/report.json
python tools/research_reports/validate_report.py outputs/report.json
```

研究入口 `scripts/run_probability_sweep.py` 會自動使用這些工具。
驗證器檢查必要欄位、狀態、證據及高風險發現；外部工具可再使用 `report.schema.json` 做完整 schema 驗證。
報告 PASS 不代表策略可以獲利，不能解除實盤交易品質閘門。
