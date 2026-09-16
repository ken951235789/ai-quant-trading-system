# data/database

這裡只放本機分析資料、DuckDB 暫存檔或 PostgreSQL 備份，不放 container 的正式資料目錄。

- PostgreSQL 使用 Docker named volume `postgres_data` 保存交易狀態。
- Parquet 歷史市場資料放在 `data/raw` 或後續的 data lake 分區。
- DuckDB 直接查 Parquet；需要建立暫存資料庫時才放在本目錄。
- PostgreSQL 跨電腦備份使用 `pg_dump`，不能只複製 Docker volume。

`*.dump`、`*.sql.gz` 與本目錄內容已忽略 Git，避免交易資料與帳戶資訊進入版本庫。
