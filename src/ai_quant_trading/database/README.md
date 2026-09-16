# database

本模組提供 PostgreSQL 交易狀態中心。歷史 K 線與模型特徵不放進這個 OLTP 資料庫，
它們後續改存 partitioned Parquet，再由 DuckDB 查詢。

## 負責範圍

- `config.py`：讀取後端、連線池與逾時設定，輸出 log 時遮蔽密碼。
- `engine.py`：建立 SQLAlchemy Engine、連線池與健康檢查。
- `models.py`：事件、訂單、成交、部位、風控與對帳資料表。
- `repository.py`：唯一可由交易流程使用的 transaction 入口。
- `migrations/`：Alembic schema 版本，不可直接修改已套用的 migration。

## 安全原則

1. 預設 `AI_QUANT_STORAGE_BACKEND=file`，不會因為安裝 SQL 套件就自動切換。
2. 切到 `postgres` 後若 DB 斷線會直接失敗，不會偷偷寫回 CSV 造成雙重真相。
3. `client_order_id` 是冪等鍵；相同事件由 SHA-256 `record_key` 去重。
4. 部位更新在單一 transaction 內加 row lock，但 Binance 網路呼叫不放進 DB transaction。
5. CSV/JSON 只作遷移來源與人工匯出，PostgreSQL 啟用後才是交易狀態來源。

## 本機啟動

先安裝 Docker Desktop，複製 `.env.example` 為 `.env` 並更換 PostgreSQL 密碼：

```powershell
python -m pip install -e ".[database]"
docker compose up -d postgres
python -m alembic upgrade head
python scripts\database_admin.py health
```

把既有 Testnet 紀錄安全匯入；重跑時相同事件會略過：

```powershell
python scripts\database_admin.py import-live `
  --root data\live_trading `
  --environment testnet
```

確認筆數與 Dashboard 後才把 `.env` 改成：

```dotenv
AI_QUANT_STORAGE_BACKEND=postgres
```

停止 container 不會刪資料：

```powershell
docker compose stop postgres
```

不要對正式資料執行 `docker compose down -v`，`-v` 會刪除 named volume。

## 備份與還原

先停止所有實盤／Testnet 背景工作，再建立 PostgreSQL 自訂格式備份：

```powershell
New-Item data\database\backups -ItemType Directory -Force
docker compose exec -T postgres pg_dump `
  -U ai_quant_app -d ai_quant -Fc -f /tmp/ai_quant.dump
docker compose cp postgres:/tmp/ai_quant.dump `
  data\database\backups\ai_quant.dump
```

還原會覆蓋資料，必須在交易程序完全停止且已另存現況後執行：

```powershell
docker compose cp data\database\backups\ai_quant.dump `
  postgres:/tmp/ai_quant.dump
docker compose exec -T postgres pg_restore `
  -U ai_quant_app -d ai_quant --clean --if-exists /tmp/ai_quant.dump
```
