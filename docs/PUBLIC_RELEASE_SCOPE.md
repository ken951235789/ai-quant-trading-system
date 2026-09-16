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

## 發布前檢查

公開快照採白名單複製並建立全新的 Git 歷史，避免把原始工作目錄中的忽略檔或舊提交帶入。發布前執行：

1. Gitleaks 秘密字串掃描。
2. Email、本機使用者路徑、私人 IP 與 GitHub 帳號識別掃描。
3. 副檔名、檔案大小及敏感檔名 allowlist 檢查。
4. Git staged snapshot 與完整新 Git 歷史再次掃描。
5. 原始碼靜態檢查及測試。

自動檢查只能降低風險，不能證明軟體絕對不存在漏洞或未知形式的敏感資訊。
