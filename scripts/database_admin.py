"""PostgreSQL 健康檢查、建表與既有實盤檔案匯入工具。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Sequence

import pandas as pd

from ai_quant_trading.database.config import DatabaseSettings
from ai_quant_trading.database.engine import check_database_health, create_database_engine
from ai_quant_trading.database.repository import PostgresTradingRepository


LIVE_RECORD_FILES = {
    "orders.csv": "orders",
    "cycles.csv": "cycles",
    "account_snapshots.csv": "account_snapshots",
    "rl_context.csv": "rl_context",
    "protections.csv": "protections",
}


def _postgres_settings(env_path: str | Path | None) -> DatabaseSettings:
    loaded = DatabaseSettings.from_environment(env_path)
    if not loaded.database_url:
        raise ValueError("請在 .env 設定 AI_QUANT_DATABASE_URL")
    return DatabaseSettings(
        backend="postgres",
        database_url=loaded.database_url,
        pool_size=loaded.pool_size,
        max_overflow=loaded.max_overflow,
        connect_timeout_seconds=loaded.connect_timeout_seconds,
    )


def _repository(settings: DatabaseSettings) -> PostgresTradingRepository:
    return PostgresTradingRepository(create_database_engine(settings))


def _compose_command(project_root: Path, *arguments: str) -> list[str]:
    return [
        "docker",
        "compose",
        "--project-directory",
        str(project_root),
        "-f",
        str(project_root / "compose.yaml"),
        *arguments,
    ]


def _database_identity() -> tuple[str, str]:
    database = os.getenv("POSTGRES_DB", "ai_quant").strip()
    username = os.getenv("POSTGRES_USER", "ai_quant_app").strip()
    if not database.replace("_", "").isalnum() or not username.replace("_", "").isalnum():
        raise ValueError("PostgreSQL 資料庫或使用者名稱含不允許字元")
    return database, username


def _backup_database(project_root: Path, destination: Path) -> Path:
    """透過容器內 pg_dump 產生 custom-format 備份並驗證目錄。"""
    database, username = _database_identity()
    destination.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = destination / f"{database}_{timestamp}.dump"
    command = _compose_command(
        project_root,
        "exec",
        "-T",
        "postgres",
        "pg_dump",
        "-U",
        username,
        "-d",
        database,
        "-Fc",
        "--no-owner",
        "--no-acl",
    )
    with backup.open("wb") as stream:
        subprocess.run(command, check=True, stdout=stream)
    with backup.open("rb") as stream:
        subprocess.run(
            _compose_command(project_root, "exec", "-T", "postgres", "pg_restore", "-l"),
            check=True,
            stdin=stream,
            stdout=subprocess.DEVNULL,
        )
    digest_state = hashlib.sha256()
    with backup.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest_state.update(chunk)
    digest = digest_state.hexdigest()
    backup.with_suffix(".dump.sha256").write_text(f"{digest}  {backup.name}\n", encoding="ascii")
    return backup


def _restore_drill(project_root: Path, backup: Path) -> None:
    """還原到隔離暫存資料庫、執行查詢後刪除，不觸碰正式資料庫。"""
    _database, username = _database_identity()
    drill_database = "ai_quant_restore_drill"
    base = _compose_command(project_root, "exec", "-T", "postgres")
    subprocess.run([*base, "dropdb", "-U", username, "--if-exists", drill_database], check=True)
    try:
        subprocess.run([*base, "createdb", "-U", username, drill_database], check=True)
        with backup.open("rb") as stream:
            subprocess.run(
                [*base, "pg_restore", "-U", username, "-d", drill_database, "--no-owner", "--no-acl"],
                check=True,
                stdin=stream,
            )
        subprocess.run(
            [
                *base,
                "psql",
                "-U",
                username,
                "-d",
                drill_database,
                "-v",
                "ON_ERROR_STOP=1",
                "-c",
                "SELECT count(*) FROM exchange_events; SELECT version_num FROM alembic_version;",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    finally:
        subprocess.run([*base, "dropdb", "-U", username, "--if-exists", drill_database], check=True)


def _import_live_files(
    repository: PostgresTradingRepository,
    root: Path,
    environment: str,
) -> tuple[int, int]:
    environment_dir = root.resolve() / environment
    inserted = 0
    duplicates = 0
    for filename, record_type in LIVE_RECORD_FILES.items():
        path = environment_dir / filename
        if not path.exists() or path.stat().st_size == 0:
            continue
        for row in pd.read_csv(path).to_dict(orient="records"):
            if repository.append_record(record_type, environment, row):
                inserted += 1
            else:
                duplicates += 1

    positions_path = environment_dir / "positions.json"
    if positions_path.exists():
        payload = json.loads(positions_path.read_text(encoding="utf-8"))
        repository.save_positions(environment, payload)
    return inserted, duplicates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Quant PostgreSQL 管理工具")
    parser.add_argument("--env-file", default=".env", help="環境變數檔案")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("health", help="檢查 PostgreSQL 連線")
    subparsers.add_parser("create-schema", help="僅供本機開發快速建表")
    importer = subparsers.add_parser("import-live", help="匯入既有 CSV/JSON 實盤紀錄")
    importer.add_argument("--root", default="data/live_trading")
    importer.add_argument(
        "--environment", choices=["testnet", "demo", "live"], default="testnet"
    )
    backup = subparsers.add_parser("backup", help="建立並驗證 PostgreSQL 備份")
    backup.add_argument("--destination")
    drill = subparsers.add_parser("restore-drill", help="把備份還原到隔離資料庫驗收")
    drill.add_argument("--backup", required=True)
    subparsers.add_parser("restore-latest", help="還原演練最近一份主要備份")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = _postgres_settings(args.env_file)
    engine = create_database_engine(settings)
    if args.command == "health":
        healthy, message = check_database_health(engine)
        print(message)
        return 0 if healthy else 1
    if args.command == "create-schema":
        from ai_quant_trading.database.models import Base

        Base.metadata.create_all(engine)
        print("本機開發 schema 已建立；正式環境請使用 alembic upgrade head")
        return 0
    if args.command == "import-live":
        inserted, duplicates = _import_live_files(
            PostgresTradingRepository(engine),
            Path(args.root),
            args.environment,
        )
        print(f"匯入完成：新增 {inserted} 筆、略過重複 {duplicates} 筆")
        return 0
    if args.command == "backup":
        project_root = Path(__file__).resolve().parents[1]
        destination = Path(
            args.destination
            or os.getenv(
                "AI_QUANT_BACKUP_DIR",
                str(project_root / "data" / "database" / "backups"),
            )
        ).resolve()
        created = _backup_database(project_root, destination)
        secondary_text = os.getenv("AI_QUANT_BACKUP_SECONDARY_DIR", "").strip()
        if secondary_text:
            secondary = Path(secondary_text).resolve()
            secondary.mkdir(parents=True, exist_ok=True)
            shutil.copy2(created, secondary / created.name)
            shutil.copy2(
                created.with_suffix(".dump.sha256"),
                secondary / created.with_suffix(".dump.sha256").name,
            )
        print(f"備份完成：{created}")
        return 0
    if args.command == "restore-drill":
        project_root = Path(__file__).resolve().parents[1]
        _restore_drill(project_root, Path(args.backup).resolve())
        print("隔離還原演練通過")
        return 0
    if args.command == "restore-latest":
        project_root = Path(__file__).resolve().parents[1]
        backup_dir = Path(
            os.getenv(
                "AI_QUANT_BACKUP_DIR",
                str(project_root / "data" / "database" / "backups"),
            )
        ).resolve()
        backups = sorted(backup_dir.glob("*.dump"), key=lambda path: path.stat().st_mtime)
        if not backups:
            raise FileNotFoundError(f"沒有可供演練的備份：{backup_dir}")
        _restore_drill(project_root, backups[-1])
        print(f"最近備份還原演練通過：{backups[-1].name}")
        return 0
    raise AssertionError(f"未處理的命令：{args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
