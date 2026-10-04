# Transformer 機率門檻研究操作

## 用途

對18個固定候選模型的既有預測，測試0%至100%、每次1%的機率門檻。
不訓練、不送單、不改部署設定。策略、成本、持有上限、冷卻與2bps淨收益閘門保持一致。
0%仍有收益閘門，不是無AI；另行產生無AI基準。每個成本情境重新撮合，不是單純減成本常數。

## 前置條件

1. 依專案安裝研究依賴與 `matplotlib`，準備相同版本的候選研究成品和原始白名單封包。
2. 成品應包含 `study.json`、`plan.json`、來源快照、18個模型、預測、成交、完整性清單。
3. 公開儲存庫沒有提供行情、模型權重或逐筆交易；只有下列程式及去識別化研究摘要。
4. 所有檔案必須可信。SHA256檢查失敗應停止，不能為了跑通而跳過資料或程式一致性驗證。

```bash
python scripts/run_probability_sweep.py --study outputs/candidate_feature_results --bundle-root outputs/staging --output outputs/probability_sweep
```

資料夾必須是新路徑，禁止覆蓋舊研究。入口將數值執行緒限制為2。
公開版透過 `tools/research_reports/` 建立及驗證報告，不需要 `.agents` 私人目錄。
原研究程式若與當前 checkout 不同，稽核會拒絕；應使用與成品配對的研究版本，不可混用結果。

## 輸出與限制

- `all_thresholds.csv`：逐模型、逐門檻、逐成本和區間的明細；包含機率/收益閘門通過數。
- `seed_summary.csv`：seed分布，收益平均只計有成交模型；成交數平均包含全部seeds。
- `baselines.csv`：無AI比較；`schedules/`：逐筆可追溯成交，不建議公開。
- 三張圖：淨期望值、成交數、固定10%配置的結算收益。
- `plan.json`、`report.md`、`report.json`、`artifact_manifest.json`：契約與證據。

零成交不是零風險。少於8筆不計算區塊CI；8筆不代表樣本充分，CI也沒有多重比較校正。
結算收益未包含持倉浮虧、清算、保證金及日損停機。Funding是正準備金，稅務未驗證。
這是已看歷史的敏感度，不可挑Test最高數字就升級Champion。
本次比較見 [結果與圖表](PROBABILITY_THRESHOLD_RESULTS_20261005.md)。
