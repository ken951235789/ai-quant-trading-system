"""可攜模型包匯出、驗證與匯入測試。"""

from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pytest

from ai_quant_trading.operations.integrity import verify_artifact_manifest
from ai_quant_trading.training_portable.model_package import (
    export_rl_model_package,
    export_transformer_model_package,
    import_rl_model_package,
    verify_model_package,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_transformer_package_can_be_verified(tmp_path: Path) -> None:
    run = tmp_path / "models" / "transformer_run"
    _write_json(run / "training.json", {"status": "complete"})
    (run / "best_model.pt").write_bytes(b"trusted-transformer")
    (run / "history.csv").write_text("epoch,loss\n1,0.2\n", encoding="utf-8")

    result = export_transformer_model_package(run, tmp_path / "exports")
    manifest = verify_model_package(result.archive_path)

    assert result.package_type == "transformer"
    assert manifest["package_type"] == "transformer"
    assert any(item["path"] == "best_model.pt" for item in manifest["files"])


def test_rl_package_round_trip_preserves_portable_layout(tmp_path: Path) -> None:
    environment = tmp_path / "environments" / "btc_15m"
    run = environment / "training" / "sac_run"
    _write_json(
        environment / "environment.json",
        {
            "source": {
                "exchange": "binance_futures",
                "symbol": "BTC/USDT",
                "interval": "15m",
            },
            "feature_columns": ["return_1"],
        },
    )
    (environment / "ai").mkdir(parents=True)
    (environment / "ai" / "transformer_model.pt").write_bytes(b"transformer")
    _write_json(
        run / "training.json",
        {
            "status": "complete",
            "training_config": {"algorithm": "sac"},
        },
    )
    (run / "best_model").mkdir(parents=True)
    (run / "best_model" / "best_model.zip").write_bytes(b"stable-baselines-model")
    (environment / "test.csv").write_text("timestamp,close\n0,1\n", encoding="utf-8")

    result = export_rl_model_package(run, tmp_path / "exports")
    manifest = verify_model_package(result.archive_path)
    imported = import_rl_model_package(
        result.archive_path,
        tmp_path / "imported",
        trusted_local_artifact=True,
    )

    assert result.package_type == "rl"
    assert manifest["entry_training_dir"] == "environment/training/sac_run"
    assert (imported / "training.json").is_file()
    assert (imported / "best_model" / "best_model.zip").is_file()
    assert verify_artifact_manifest(imported)
    assert (imported.parent.parent / "environment.json").is_file()
    assert (imported.parent.parent / "ai" / "transformer_model.pt").is_file()


def test_rl_package_import_requires_explicit_trust(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="只可明確信任"):
        import_rl_model_package(tmp_path / "unknown.zip", tmp_path / "imported")


def test_model_package_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "traversal.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("package/manifest.json", "{}")
        bundle.writestr("../outside.txt", "attack")

    with pytest.raises(ValueError, match="不安全路徑"):
        verify_model_package(archive)


def test_model_package_rejects_compression_bomb(tmp_path: Path) -> None:
    archive = tmp_path / "bomb.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("package/manifest.json", "{}")
        bundle.writestr("package/huge.txt", b"0" * (2 * 1024 * 1024))

    with pytest.raises(ValueError, match="壓縮比例異常"):
        verify_model_package(archive)


def test_model_package_rejects_unlisted_extra_file(tmp_path: Path) -> None:
    run = tmp_path / "models" / "transformer_run"
    _write_json(run / "training.json", {"status": "complete"})
    (run / "best_model.pt").write_bytes(b"trusted-transformer")
    result = export_transformer_model_package(run, tmp_path / "exports")
    rewritten = tmp_path / "with-extra.zip"
    with (
        zipfile.ZipFile(result.archive_path) as source,
        zipfile.ZipFile(rewritten, "w", compression=zipfile.ZIP_DEFLATED) as target,
    ):
        for entry in source.infolist():
            target.writestr(entry, source.read(entry))
        target.writestr(f"{result.package_id}/unexpected.py", "print('unexpected')")

    with pytest.raises(ValueError, match="未列出的額外檔案"):
        verify_model_package(rewritten)
