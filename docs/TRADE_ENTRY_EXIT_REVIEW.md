# 逐筆進出場與盈虧比

此研究工具不更換模型、不調整門檻，也不啟動交易服務。
使用已保存的機率門檻研究成交與相同的 `frozen_fold_3.csv`，產生離線 HTML 及桌面研究包。
不是任意回測檔案匯入器。公開版本不附私人行情、模型、逐筆成交及圖像。

## 重現

輸出目錄必須尚未存在，以保留旧研究；以下參數為使用者自己的研究產物。

```powershell
python scripts/render_trade_review.py --sweep <門檻研究目錄> --study <候選研究目錄> --output <新圖表目錄>
```

開啟輸出目錄的 `trades.html`。圖表與 Plotly 內嵌，不需連網。
可選候選、無 AI／Transformer、特徵、seed、機率門檻、成本與區間。
交易依時間排列，不特別挑獲利範例；用箭頭、選單或成交列切換。
`desktop_bundle` 可透過回測中心研究檢視器讀取，先驗證 manifest 再呈現。

## 定義

- 預定報酬／風險：停利距離 ÷ 停損距離；這批固定契約為 2:1。
- 實際淨盈虧比：平均正淨報酬 ÷ 平均負淨報酬絕對值，缺贏單或輸單則不計算。
- 本筆實現 R：淨報酬 ÷ 初始停損距離比例，不是整體盈虧比。
- 成交價已含滑價和價差，再扣費用及 funding 準備金，不重複扣成本。
- 停損／停利只有出場棒，沒有精確棒內時間；狀態或到期退出採棒開盤。
- 不同成本情境重新撮合，成交路徑可能不同；後續 K 線是歷史，不是預測。
- 零成交不產生虛假盈虧比，圖表不等於實盤資格。

```powershell
python -m pytest tests/research/test_trade_review.py tests/dashboard/test_trade_review_page.py -q
python tools/research_reports/validate_report.py <新圖表目錄>/report.json
```

保存 `report.md`、標準 `report.json`、`verification.json`、`artifact_manifest.json`。
新增成交的日誌與訓練邊界另見[交易圖文與研究交接](JOURNAL_RESEARCH_20261008.md)。
