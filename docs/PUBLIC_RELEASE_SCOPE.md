# 公開版本安全與範圍說明

## 收錄內容

- Python 原始碼、單元測試與整合測試
- 不含真值的設定範例
- 資料庫 migration、Docker Compose 與監控設定
- 系統架構、研究流程及彙總後的實驗結果

## 排除內容

- `.env`、API Key、Secret、Token、Email 應用程式密碼與私鑰
- 原始及處理後行情、新聞資料與 Order Book 快照
- Transformer／SAC checkpoint、Replay Buffer 與 Scaler
- PostgreSQL、SQLite、DuckDB 備份及快取
- 模擬倉／Testnet／Live 的帳戶、訂單、成交與通知紀錄
- EXE、可攜包、建置目錄、日誌、Kaggle 輸出與暫存檔
- 私有操作報告、舊 Git 歷史與本機秘密目錄

研究文件中的 `outputs/`、`artifacts/` 及行情路徑表示自行重跑後的本機產物，
不是公開儲存庫附帶的檔案。公開報告只保留彙總、契約與重現指令，
移除個人本金、稅務身分及私人雲端工作連結。

## 發布前檢查

首次公開時採白名單建立獨立的 Git 歷史；後續更新延續公開歷史，不匯入私人工作版
的 Git 物件或分支，也不以重寫歷史假裝敏感資料從未公開。發布前執行：

1. Gitleaks 秘密字串掃描。
2. Email、本機使用者路徑、私人 IP 與 GitHub 帳號識別掃描。
3. 副檔名、檔案大小及敏感檔名 allowlist 檢查。
4. Git staged snapshot 與完整新 Git 歷史再次掃描。
5. 原始碼靜態檢查及測試。

`python scripts/check_public_repository.py --staged` 直接檢查 index blob，阻擋私人產物、
敏感檔名、非範例 Email、本機使用者路徑及未審查二進位檔。此檢查也在 CI 執行，
但不取代 Gitleaks、人工 diff 審查或外部私人備份。

自動檢查只能降低風險，不能證明軟體絕對不存在漏洞或未知形式的敏感資訊。
