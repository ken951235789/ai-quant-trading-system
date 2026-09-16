# PostgreSQL Schema Migration

本目錄由 Alembic 管理交易狀態資料表。先在專案根目錄設定 `.env`，再執行：

```powershell
docker compose up -d postgres
python -m alembic upgrade head
```

正式環境不可用 `Base.metadata.create_all()` 取代 migration，也不可修改已經套用的版本檔；
需要變更 schema 時建立新的 revision，並先在 Testnet 資料庫驗證升級與回復流程。
