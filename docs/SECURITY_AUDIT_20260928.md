# 公開研究版本發布檢查

- 日期：2026-09-28
- Report ID：SECURITY_REVIEW-20260928-7D25F499
- 範圍：本次公開原始碼、暫存區快照、既有可達 Git 歷史及固定依賴清單。
- 結論：WARN。未發現可辨識的真實憑證；仍有靜態分析提醒，不能宣稱完全安全。
- 機器可讀紀錄：[JSON](security/public_release_20260928.json)。

## 這次公開什麼

同步成本後 Transformer 評估、Calibration／Selection 隔離、SAC 輸入消融、
策略事件研究、Kaggle 安全封包與監控修正，以及去識別化的負面研究結果。
保留公開版 Demo、安全設定與研究定位，不匯入私人 Git 歷史或升學資料。
事件研究 checkpoint 不可直接接入舊 SAC、模擬或實盤推論。

## 驗證證據

| 檢查 | 結果 |
|---|---|
| Gitleaks 8.30.1 可達 Git 歷史 | 發布前 12 個 commits，0 筆秘密字串命中 |
| Gitleaks 暫存快照 | 約 3.1 MB 文字及既有素材，0 筆未處理命中 |
| 公開檔案守門 | index blob 檢查通過；禁止憑證、資料、模型、資料庫、輸出與未審查二進位檔 |
| `.env.example` | 交易所及通知憑證為空，資料庫密碼只有明確占位值 |
| Ruff、compileall、git diff --check | 通過 |
| 完整 Pytest | 564 passed、4 subtests passed，94.64 秒 |
| 公開 Demo | 360 根合成 K 線，正常生成 HTML／JSON，不下載行情或送單 |
| Kaggle builder 帳號參數 | 自己的帳號及另指定 Dataset owner 兩種路徑通過離線測試 |
| pip-audit 2.10.1 | 固定清單 123 個套件，0 已知漏洞、0 跳過 |
| Bandit 1.9.4 | 0 high、9 medium、51 low；不是零警告 |

Gitleaks 使用官方 Windows 發行檔，執行前已比對官方 SHA-256。
掃描報告以 redaction 輸出並保留在私人工作目錄，不公開疑似秘密原文。
初次快照掃描將兩個相鄰空白 API 欄位誤判為跨行秘密；已確認值皆為空，
改成帶說明的空字串，dotenv 解析測試通過，未以整個 `.env.example` 白名單跳過掃描。

依賴掃描使用 PyPI 漏洞資料及 `requirements/locks/windows-py313-cpu.txt` 的固定版本，
不代表任意新安裝環境。Windows TLS 使用系統信任庫，沒有停用憑證驗證。

## 剩餘風險

1. 九項 Bandit medium 皆為 Kaggle runner 的固定 `/tmp` 路徑（B108）。
   不應在多人共用、不受信任的主機直接執行；後續宜改成每次執行獨立的安全暫存目錄。
   本次沒有為了消除數字而全面抑制這些提醒。
2. low 提醒主要涉及 subprocess、匯入及例外處理，不能用其嚴重度推定不可能被利用。
   SQL 識別字驗證及既有局部 nosec 註記保留；資料值仍走參數化查詢。
3. 舊公開提交含非秘密的 Kaggle 帳號／私人 Dataset 識別字。本次已移除目前程式與
   文件中的固定值，但沒有重寫歷史；識別字仍可能在舊提交看到。沒有因此公開認證金鑰。
4. 本次沒有滲透測試、真實下單、重新訓練 GPU 模型或重新打包 EXE。
   也沒有重新驗證 GitHub 帳戶後台的所有安全設定或外部 Fork／快取。
5. 模型、原始行情、逐筆預測、交易紀錄、個人帳戶設定及日誌沒有上傳。
   GitHub 公開版不是私人完整備份，仍需另外保存研究產物。

## 後續發布

使用 `python scripts/check_public_repository.py --staged` 檢查即將提交的版本，
再對匯出的 index 與 Git 歷史執行 Gitleaks，最後核對遠端 commit 與 CI。
每次資料、模型、依賴或外部服務改版都需要重新驗證，不沿用本報告作永久安全保證。
