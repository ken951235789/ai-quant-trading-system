# 公開研究版本安全稽核

檢查日期：2026-09-20

## 檢查範圍

本次檢查涵蓋公開儲存庫目前的主分支、遠端分支、版本標籤、Release 對應原始碼、所有
可達 Git commit、追蹤檔案名稱與內容。私人工作目錄、未追蹤行情、模型、資料庫、交易紀錄
及本機 `.env` 不在公開儲存庫中。

## 結果摘要

| 檢查 | 結果 |
|---|---|
| GitHub Secret Scanning | 已啟用，0 筆 open alert |
| GitHub Push Protection | 已啟用 |
| 私鑰、GitHub／AWS／Google／Slack／Stripe Token、JWT | 未偵測到 |
| 含帳密 URL 與一般祕密賦值 | 僅命中 `change_me`、空白欄位與安全測試資料 |
| `.env`、Key、模型、資料庫、行情、壓縮檔 | 未被 Git 追蹤 |
| Email、Webhook、電話與本機使用者路徑 | 僅命中 `example.com`、假 Webhook、測試時間字串與路徑占位符 |
| Release 自行上傳資產 | 0；只有 GitHub 自動產生的標籤原始碼封存 |
| `pip-audit` 專案依賴 | 未發現已知漏洞 |
| Bandit | 0 high、7 medium、35 low |
| Ruff | 通過 |
| Pytest | 510 passed、4 subtests passed |

## 已人工確認的命中

- `.env.example` 的交易所、Email 與 Webhook 憑證欄位皆為空白。
- PostgreSQL 與 Grafana 的 `change_me` 是明確占位值，不是部署密碼。
- 測試信箱使用 `example.com` 保留網域，Webhook 使用不存在的測試路徑。
- 測試中的 `user:password` 連線字串只驗證敏感內容遮罩，不是實際資料庫憑證。
- `C:\Users\你的帳號` 是操作說明占位符，不是公開者的 Windows 使用者名稱。
- Git commit 作者 Email 使用 GitHub 提供的 `users.noreply.github.com` 位址。

## 靜態分析說明

Bandit 沒有發現 high severity 問題。7 個 medium severity 提醒都位於 Kaggle 執行腳本的
固定 `/tmp` 暫存路徑，與憑證或個資外洩無關；正式交易與 Dashboard 原始碼沒有 medium
或 high severity 結果。Low severity 主要是參數陣列形式的 `subprocess`、非密碼用途的重連
jitter，以及關閉連線時的 best-effort 例外處理。

## 公開邊界

- `.gitignore` 排除 `.env`、祕密、模型權重、CSV／Parquet、資料庫、交易紀錄、日誌、
  壓縮檔與建置輸出。
- Docker 服務只映射到 `127.0.0.1`，密碼由 `.env` 注入，不寫入 Compose。
- GitHub Actions 權限為 `contents: read`，工作流程沒有讀取或輸出 Repository Secrets。
- Windows 打包器只複製 `.env.example`，並在成品含 `.env` 時直接拒絕發布。

## 限制

本次沒有發現真實憑證、私人資料或已知依賴漏洞。這代表在上述分支、版本、掃描規則與
檢查日期內沒有可辨識的洩漏，不等同於數學上證明永遠不存在未知漏洞。若任何真實金鑰曾
被貼入 Git、終端輸出或 Issue，即使後來刪除，也應立即在服務端撤銷並重新簽發。
