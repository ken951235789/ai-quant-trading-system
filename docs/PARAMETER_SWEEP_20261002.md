# BTC 大型參數比較：操作與限制

這是研究用 CLI，不會送出訂單、不會選取 Live 模型，也不修改 EXE。
使用既有約五年 BTCUSD-M 永續 15m 資料，完整遍歷預先固定的 11,664 組離散組合。
「完整」僅指本次有限網格，不代表市場上所有策略或任意連續參數。

## 先看成果

研究資料夾：`artifacts/reports/20261002/parameter_sweep/`。

- 雙擊 `charts.html`：離線開啟五張圖表及候選參數表，不需要網路或啟動交易程式。
- `report.md`：中文說明、時間切分、交易規則、成本、限制及下一步。
- `report.json`：符合專案 Skill 契約的機器報告。
- `parameters.csv`：11,664 組完整設定。
- `results.csv.gz`：174,960 個期間與成本情境，用 pandas `read_csv` 可直接讀取。
- `pooled_forward.csv`：全部參數在三段 Forward 合併後的結果。
- `candidate_table.csv`：原始基準與驗證排名前五；不是獲利保證或部署清單。
- `trades/`：上述候選的逐筆成交、成本、R 倍數及結算權益。
- `groups/`：72 個可續跑區塊，每塊包含 CSV、月收益 NPZ 和雜湊收據。
- `audit.json`、`pytest_results.xml`、`artifact_manifest.json`：驗證紀錄。

圖表區分「每筆名目本金報酬」與「帳戶的簡化結算曲線」。
曲線採初始 1000 USDT、每筆只配置當時結算權益 10%、無槓桿。
不包含未平倉浮虧、清算、單日虧損停機、稅負與真實交易容量。
因此不可把它當作任何真實帳戶的可實現報酬。

## 重新執行

在專案根目錄 PowerShell 使用已安裝依賴的 Python：

```powershell
# 在自己的專案根目錄執行。
$env:PYTHONPATH = 'src'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
python -X utf8 scripts/run_parameter_sweep.py `
  --source outputs/kaggle_strategy_event_20260924/staging/data/btc_15m.csv `
  --output artifacts/reports/NEW_RUN/parameter_sweep
```

`NEW_RUN` 要換成新的研究識別碼，不覆蓋先前結果。
腳本會拒絕已存在的輸出資料夾；中途停止時，以原指令加 `--resume` 繼續。
續跑前會比對行情、程式、網格設定及完成區塊雜湊；任一不同就拒絕混合結果。
`progress.json` 可查看目前進度。完整執行後才會產生 `research.json`。

```powershell
python -X utf8 scripts/audit_parameter_sweep.py artifacts/reports/NEW_RUN/parameter_sweep
python -X utf8 scripts/plot_parameter_sweep.py artifacts/reports/NEW_RUN/parameter_sweep
```

依賴為既有 NumPy、pandas，加上 matplotlib 繪圖；不需要 GPU、Numba 或新模型訓練。
繪圖器在 Windows 自動載入微軟正黑體；移到其他系統時需自行提供支援中文字體。

完整標準報告另外需要測試 XML 與報告骨架：

```powershell
python -X utf8 -m pytest -q --junitxml=artifacts/reports/NEW_RUN/parameter_sweep/pytest_results.xml
python .agents/skills/ai-quant-engineering/scripts/new_report.py `
  --type model_research --project AIQuantTradingSystem `
  --scope 'BTC 有限網格研究' --output artifacts/reports/NEW_RUN/parameter_sweep/report.json
python -X utf8 scripts/report_parameter_sweep.py artifacts/reports/NEW_RUN/parameter_sweep
python .agents/skills/ai-quant-engineering/scripts/validate_report.py artifacts/reports/NEW_RUN/parameter_sweep/report.json
```

## 如何理解比較

1. Search 選各進場家族/週期前五，Validation 再選；後續 Forward 不參與挑選。
2. 全部 Forward 結果仍公開，避免只呈現最好的一組，但不可從中事後升級模型。
3. 放寬 ADX、縮短冷卻等設定可能增加交易，也會改變持倉時間與後續進場集合。
4. 零成本為正但基本成本為負，表示訊號優勢未抵過交易摩擦，不應直接拿零成本當策略。
5. 熱圖取其他參數的中位數，不取最好值；用來看結果是否只集中在少數尖峰。
6. 大量搜尋容易偶然找到漂亮曲線；月份區塊最大統計量只能作敏感度診斷，不能消除所有研究偏誤。
7. 歷史已被反覆研究，不是新的 final holdout。合格候選仍需要未見行情或前向模擬。

## 程式責任

- `grid_execution.py`：同一事件成交契約的批次計算、時間框訊號、冷卻及參考引擎核對。
- `parameter_sweep.py`：固定範圍、資料切分、分批續跑、搜尋/驗證隔離、統計與候選匯出。
- `audit_parameter_sweep.py`：檢查雜湊、完整情境數、月收益、選擇重現及交易算式。
- `plot_parameter_sweep.py`：只讀結果製圖，不挑選新策略。
- `report_parameter_sweep.py`：整理標準化中文報告，不更改品質閘門。

新候選通過研究後，仍須建立與 Transformer 標籤一致的版本契約；
先驗證 Transformer 過濾是否改善成本後收益，再比較固定部位、波動率部位與 SAC 部位管理。
