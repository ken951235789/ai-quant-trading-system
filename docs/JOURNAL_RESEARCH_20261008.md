# 交易圖文、帳務恢復與研究交接

此更新是研究與工程能力，不是獲利或實盤認證。公開版本不附原始行情、模型、帳戶、
逐筆交易及圖片。部署模型未替換；正式訓練與 SAC 升級仍等待研究證據。

## 功能與使用位置

| 功能 | 位置 | 已知邊界 |
|---|---|---|
| 歷史進出場 K 線 | 回測中心的研究成交檢視 | 只讀已核對的研究包，不重新選最佳策略 |
| 逐筆交易圖片與紀錄 | 機器人監控 → 最近交易 → 圖文日誌 | 進場、加碼、部分平倉及平倉；風險未知時不捏造 R |
| 模擬帳務恢復 | RL 模擬交易週期自動使用 | SQLite 提交、CSV 與帳戶中斷可續作；外部改檔衝突會停止 |
| 確認成交圖文 | Testnet／實盤監控 → 訂單 → 交易所成交圖文 | 從既有耐久事件投影，不以送单成功冒充成交 |
| 圖文封存 | 圖文日誌 → 日誌維護 | 每批最多 100 筆，校驗 ZIP 後才移除可重建散檔 |
| 候選訓練交接 | `export_candidate_handoff.py` | 全候選而非只學成交樣本；標籤與輸入分開 |
| 模型退出提案 | `audit_exit_proposals.py` | 不同 R 各自訓練、合法 OOS、品質閘門；未接實盤 |
| 持倉浮虧核對 | `audit_trade_account.py` | 固定成交路徑，非完整保證金與清算驗證 |

啟動介面前安裝 `python -m pip install -e ".[dashboard,ai,rl,database]"`。
可使用既有桌面建置或 `streamlit run src/ai_quant_trading/dashboard/app.py --server.address=127.0.0.1`。
不要為了查看報告填入正式交易密鑰或解除 Live 閘門。

## 交易紀錄如何用於研究

每筆圖文保存決策當下資訊和後續結果，兩者必須分開。平倉後的 K 線、MFE／MAE、
實現 R、退出原因只能作標籤或診斷，不能成為同一筆進場的模型輸入。

`export_candidate_handoff.py` 從既有 Transformer Dataset 重建正規化市場矩陣，
以候選索引引用歷史序列。保存所有候選、重疊權重、切分與 purge 標記，
避免只學到曾經成交的選擇偏誤；候選數不等於獨立交易數。
来源雜湊、完整交易契約、特徵順序和 scaler 必須與訓練一致。
舊日誌沒有來源證明時，不會自動冒充合法訓練資料。

以下 `<...>` 是使用者自己的檔案位置，公開 repo 不提供私人資料或模型。
輸出必須選新目錄，不能覆寫舊研究。

```powershell
python scripts/export_candidate_handoff.py --run <研究模型目錄> --source <原凍結CSV> --output <新交接目錄>
python scripts/export_trade_journal.py --journal <帳戶圖文日誌目錄> --output <新匯出目錄>
python scripts/audit_exit_proposals.py --runs <1R模型目錄> <3R模型目錄> --source <原凍結CSV> --output <新提案目錄>
python scripts/audit_trade_account.py --source <原凍結CSV> --training-json <training.json> --trades <不重疊成交CSV> --allocation 0.1 --output <新權益目錄>
```

`inputs.npz` 不含標籤，`labels.csv` 分離存放未來結果；`handoff.json` 和
`artifact_manifest.json` 保存契約與完整性。通過時間核對的 OOS 仍不是全新 final holdout。
單純存下交易圖片不會自動改善 Transformer 或 SAC。

## 本次驗證

- 20 根歷史 15m K 棒、腳本動作：13 次成交操作、8 筆平倉、13 張圖片，重跑不重複記帳。
- 候選交接 1,040 筆：Train 587、Calibration 90、Selection 90、Test 269、purged 4。
- 小型 V3 六次一輪訓練：3 種候選 × 1R／3R、seed 42、CPU threads 2，共約 125 秒（含前後處理）。
- 六組雙閘門 Test 均 0 成交；共同 269 候選沒有退出提案，不等於零風險或盈利。
- 既有 91 筆無 AI 固定路徑，研究初始資金 1,000 USDT、10% 名目部位，帳戶報酬 -0.7514%，
  收盤最大回撤 1.3490%，棒內不利界線 1.3535%。這不是新模型績效。

單輪模型明顯不足以作正式優劣比較；沒有因本次結果擴大模型、提高槓桿、改種子或降低門檻。
完整程式測試與發布檢查另外記錄於[安全檢查](SECURITY_AUDIT_20261008.md)。

## 下一輪最小研究

固定 2R、停損 2ATR、32 根持有期、55% 與淨收益 2bps，先比較
`compact_combined` 的 MSE／Smooth L1：3 個候選 × 2 種損失 × 3 seeds × 3 向前区間，54 次。
V3 固定 d_model=64、2 層、96 根序列、最多 20 epochs、patience=5。
這個縮小矩陣不取代完整的新舊特徵消融。

```powershell
python scripts/run_candidate_training_research.py --source <完整15m研究CSV> --output <新計畫目錄> --feature-study --variants F_compact_combined --paired-loss --seeds 42 137 2026
```

預設只產生計畫。`--smoke` 為小型測試；正式長時間研究要明確使用 `--run-research`。
所有模式固定原生 CPU 執行緒 2，正式配置應先對實際硬體做完整 fold 基準測試。
單輪六模型的耗時不能線性保證五年、54 次完整研究的時間。

停止條件：資料或成本核對失敗立即停；校準、收益排序、跨期與成本壓力沒有可信改善，
就不進 SAC。只有合法且可信的相同 OOS 訊號，才比較固定部位、波動部位與 SAC。
不能以 Test 最好看的 R 選動作，也不能用降門檻增加交易來取代證據。

## 尚未驗證

BUY／SELL 不一定是開多／開空。交易所圖文目前只確認 fill，不虛構完整往返淨收益。
未知初始部位、佣金幣別換算、實際歷史 funding、保護單連結與風險基準，
都需要另外對帳；本次沒有連 Testnet／實盤驗證。
保證金階梯、mark price 清算、稅後收益、未來封存期間亦未驗證。

封存需手動執行；PNG 本身已壓縮，不保證 ZIP 大幅縮小容量。
檔案校驗與恢復不能取代異機備份，也不保證斷電或磁碟損壞零資料遺失。
