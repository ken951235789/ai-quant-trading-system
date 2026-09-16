"""Windows 桌面版打包期間的使用者資料保護測試。"""

from __future__ import annotations

import os

import pytest

from scripts import build_windows_exe as builder


def test_runtime_backup_restores_newer_desktop_data(tmp_path, monkeypatch) -> None:
    dist = tmp_path / "dist" / "AIQuantTradingSystem"
    backup = tmp_path / "runtime-backup"
    account = dist / "data" / "paper_trading" / "demo" / "account.json"
    account.parent.mkdir(parents=True)
    account.write_text("desktop account", encoding="utf-8")
    os.utime(account, (200, 200))
    monkeypatch.setattr(builder, "DIST_DIR", dist)
    monkeypatch.setattr(builder, "RUNTIME_BACKUP_DIR", backup)

    builder._backup_runtime_data()
    source_account = dist / "data" / "paper_trading" / "demo" / "account.json"
    source_account.parent.mkdir(parents=True)
    source_account.write_text("source account", encoding="utf-8")
    os.utime(source_account, (100, 100))
    builder._restore_runtime_data()

    assert source_account.read_text(encoding="utf-8") == "desktop account"
    assert not backup.exists()


def test_runtime_backup_refuses_to_overwrite_unrestored_backup(
    tmp_path,
    monkeypatch,
) -> None:
    dist = tmp_path / "dist" / "AIQuantTradingSystem"
    backup = tmp_path / "runtime-backup"
    (dist / "data").mkdir(parents=True)
    backup.mkdir()
    monkeypatch.setattr(builder, "DIST_DIR", dist)
    monkeypatch.setattr(builder, "RUNTIME_BACKUP_DIR", backup)

    with pytest.raises(RuntimeError, match="尚未還原"):
        builder._backup_runtime_data()


def test_project_runtime_merge_adds_new_files_and_keeps_newer_desktop_files(
    tmp_path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    dist = tmp_path / "dist" / "AIQuantTradingSystem"
    source_new = project / "data" / "processed" / "features_mtf.csv"
    source_shared = project / "data" / "config" / "settings.json"
    dist_shared = dist / "data" / "config" / "settings.json"
    source_new.parent.mkdir(parents=True)
    source_shared.parent.mkdir(parents=True)
    dist_shared.parent.mkdir(parents=True)
    source_new.write_text("nine timeframes", encoding="utf-8")
    source_shared.write_text("project", encoding="utf-8")
    dist_shared.write_text("desktop", encoding="utf-8")
    os.utime(source_new, (200, 200))
    os.utime(source_shared, (100, 100))
    os.utime(dist_shared, (300, 300))
    monkeypatch.setattr(builder, "PROJECT_ROOT", project)
    monkeypatch.setattr(builder, "DIST_DIR", dist)

    builder._merge_project_runtime_directories()

    assert (dist / "data" / "processed" / "features_mtf.csv").read_text(
        encoding="utf-8"
    ) == "nine timeframes"
    assert dist_shared.read_text(encoding="utf-8") == "desktop"


def test_project_runtime_merge_skips_rebuildable_model_cache(
    tmp_path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    dist = tmp_path / "dist" / "AIQuantTradingSystem"
    model = project / "data" / "models" / "finbert" / "model.safetensors"
    validation = project / "data" / "paper_trading_validation" / "result.csv"
    feature = project / "data" / "processed" / "features_mtf.csv"
    for path in [model, validation, feature]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("data", encoding="utf-8")
    monkeypatch.setattr(builder, "PROJECT_ROOT", project)
    monkeypatch.setattr(builder, "DIST_DIR", dist)

    builder._merge_project_runtime_directories()

    assert not (dist / "data" / "models").exists()
    assert not (dist / "data" / "paper_trading_validation").exists()
    assert (dist / "data" / "processed" / "features_mtf.csv").exists()


def test_packaging_cleanup_removes_only_known_dist_data(
    tmp_path,
    monkeypatch,
) -> None:
    dist = tmp_path / "dist" / "AIQuantTradingSystem"
    model = dist / "data" / "models" / "finbert.bin"
    feature = dist / "data" / "processed" / "features_mtf.csv"
    for path in [model, feature]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("data", encoding="utf-8")
    monkeypatch.setattr(builder, "DIST_DIR", dist)

    builder._remove_rebuildable_packaging_data()

    assert not (dist / "data" / "models").exists()
    assert feature.exists()


def test_packaging_removes_transient_locks_but_keeps_account_data(
    tmp_path,
    monkeypatch,
) -> None:
    dist = tmp_path / "dist" / "AIQuantTradingSystem"
    lock = dist / "data" / "config" / "startup_refresh.lock"
    runner_lock = dist / "data" / "paper_trading" / "demo" / "runner.lock"
    account = dist / "data" / "paper_trading" / "demo" / "account.json"
    for path in [lock, runner_lock, account]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("data", encoding="utf-8")
    monkeypatch.setattr(builder, "DIST_DIR", dist)

    builder._remove_transient_runtime_files()

    assert not lock.exists()
    assert not runner_lock.exists()
    assert account.exists()


def test_legacy_dist_secret_is_moved_outside_app_directory(
    tmp_path,
    monkeypatch,
) -> None:
    dist = tmp_path / "dist" / "AIQuantTradingSystem"
    user_env = tmp_path / "user-config" / ".env"
    legacy = dist / ".env"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("BINANCE_TESTNET_API_KEY=secret\n", encoding="utf-8")
    monkeypatch.setattr(builder, "DIST_DIR", dist)
    monkeypatch.setattr(builder, "USER_ENV_PATH", user_env)
    monkeypatch.setattr(builder, "_harden_secret_acl", lambda _path: None)

    migrated = builder._migrate_legacy_runtime_secret()

    assert migrated == user_env
    assert user_env.read_text(encoding="utf-8") == "BINANCE_TESTNET_API_KEY=secret\n"
    assert not legacy.exists()
    assert ".env" not in builder.PRESERVED_RUNTIME_PATHS


def test_packaging_validation_rejects_secret_inside_dist(
    tmp_path,
    monkeypatch,
) -> None:
    dist = tmp_path / "dist" / "AIQuantTradingSystem"
    for path in [
        dist / "data" / "raw" / "README.md",
        dist / "data" / "processed" / "features.csv",
        dist / "configs" / "profile.json",
        dist / "docs" / "README.md",
        dist / "使用說明.txt",
        dist / ".env",
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test", encoding="utf-8")
    monkeypatch.setattr(builder, "DIST_DIR", dist)

    with pytest.raises(RuntimeError, match="不可包含 .env"):
        builder._validate_runtime_data()
