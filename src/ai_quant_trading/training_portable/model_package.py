"""匯出與驗證可攜式 Transformer／SAC／PPO 模型包。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from pathlib import PurePosixPath
import re
import shutil
import stat
import tempfile
from typing import Any
from uuid import uuid4
import zipfile

from ai_quant_trading.operations.integrity import (
    build_artifact_manifest,
    verify_artifact_manifest,
)


MODEL_PACKAGE_SCHEMA_VERSION = 1
MAX_ARCHIVE_FILES = 4096
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 8 * 1024**3
MAX_ARCHIVE_MEMBER_BYTES = 4 * 1024**3
MAX_ARCHIVE_COMPRESSION_RATIO = 500
PACKAGE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")


@dataclass(frozen=True, slots=True)
class ModelPackageResult:
    """模型包輸出位置與摘要。"""

    archive_path: Path
    package_type: str
    package_id: str
    files: int
    bytes: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest(root: Path, package_type: str, package_id: str) -> dict[str, Any]:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "manifest.json":
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return {
        "schema_version": MODEL_PACKAGE_SCHEMA_VERSION,
        "package_type": package_type,
        "package_id": package_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "trusted_local_artifacts_only": True,
        "files": files,
    }


def _write_archive(staging: Path, output_dir: Path, package_id: str) -> ModelPackageResult:
    _validated_package_id(package_id)
    manifest = _read_json(staging / "manifest.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f"{package_id}.zip"
    temporary = archive.with_suffix(f".zip.{uuid4().hex}.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(item for item in staging.rglob("*") if item.is_file()):
                bundle.write(path, Path(package_id) / path.relative_to(staging))
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)
    entries = list(manifest["files"])
    return ModelPackageResult(
        archive_path=archive,
        package_type=str(manifest["package_type"]),
        package_id=package_id,
        files=len(entries),
        bytes=sum(int(item["size"]) for item in entries),
    )


def export_transformer_model_package(
    training_dir: str | Path,
    output_dir: str | Path,
) -> ModelPackageResult:
    """封裝 Transformer checkpoint、特徵契約、訓練摘要與測試預測。"""
    source = Path(training_dir).resolve()
    metadata_path = source / "training.json"
    checkpoint = source / "best_model.pt"
    if not metadata_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError("Transformer 訓練資料夾缺少 training.json 或 best_model.pt")
    metadata = _read_json(metadata_path)
    if metadata.get("status") != "complete":
        raise ValueError("只允許匯出已完成的 Transformer 訓練")
    package_id = f"transformer_{source.name}"
    with tempfile.TemporaryDirectory(prefix="aiquant-transformer-") as temporary:
        staging = Path(temporary)
        for name in ["best_model.pt", "training.json", "history.csv", "test_predictions.csv"]:
            candidate = source / name
            if candidate.is_file():
                shutil.copy2(candidate, staging / name)
        payload = _manifest(staging, "transformer", package_id)
        (staging / "manifest.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return _write_archive(staging, Path(output_dir).resolve(), package_id)


def export_rl_model_package(
    training_dir: str | Path,
    output_dir: str | Path,
    *,
    include_evaluation_data: bool = True,
) -> ModelPackageResult:
    """封裝 SAC/PPO 模型及其完整環境、Transformer 與標準化契約。"""
    run = Path(training_dir).resolve()
    metadata_path = run / "training.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"找不到 RL training.json：{metadata_path}")
    metadata = _read_json(metadata_path)
    if metadata.get("status") != "complete":
        raise ValueError("只允許匯出已完成的 SAC/PPO 訓練")
    environment = run.parent.parent
    if not (environment / "environment.json").is_file():
        raise FileNotFoundError("RL 訓練資料夾不在有效 environment/training 結構內")
    selected = run / "best_model" / "best_model.zip"
    if not selected.is_file():
        selected = run / "final_model.zip"
    if not selected.is_file():
        raise FileNotFoundError("RL 訓練沒有 best_model.zip 或 final_model.zip")

    algorithm = str(dict(metadata.get("training_config", {})).get("algorithm", "rl"))
    package_id = f"{algorithm}_{environment.name}_{run.name}"
    with tempfile.TemporaryDirectory(prefix="aiquant-rl-") as temporary:
        staging = Path(temporary)
        target_environment = staging / "environment"
        target_run = target_environment / "training" / run.name
        target_run.mkdir(parents=True)
        shutil.copy2(environment / "environment.json", target_environment / "environment.json")
        if (environment / "ai").is_dir():
            shutil.copytree(environment / "ai", target_environment / "ai")
        for name in [
            "training.json",
            "progress.json",
            "validation_evaluation.csv",
            "test_evaluation.csv",
        ]:
            candidate = run / name
            if candidate.is_file():
                shutil.copy2(candidate, target_run / name)
        (target_run / "best_model").mkdir()
        shutil.copy2(selected, target_run / "best_model" / "best_model.zip")
        # 可攜包只帶入選定模型，因此必須依封裝後的實際內容重建清單。
        build_artifact_manifest(target_run)
        if include_evaluation_data:
            for name in ["validation.csv", "test.csv", "diagnostic.csv"]:
                candidate = environment / name
                if candidate.is_file():
                    shutil.copy2(candidate, target_environment / name)
        payload = _manifest(staging, "rl", package_id)
        payload["entry_training_dir"] = f"environment/training/{run.name}"
        (staging / "manifest.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return _write_archive(staging, Path(output_dir).resolve(), package_id)


def _validated_members(bundle: zipfile.ZipFile) -> tuple[list[zipfile.ZipInfo], str]:
    """驗證 ZIP 路徑、類型與解壓資源上限，避免路徑穿越與壓縮炸彈。"""
    members = bundle.infolist()
    if not members or len(members) > MAX_ARCHIVE_FILES:
        raise ValueError("模型 ZIP 的檔案數量不合理")
    roots: set[str] = set()
    normalized_names: set[str] = set()
    total_size = 0
    for member in members:
        raw_name = member.filename
        relative = PurePosixPath(raw_name)
        if (
            not raw_name
            or raw_name.startswith(("/", "\\"))
            or "\\" in raw_name
            or relative.is_absolute()
            or any(part in {"", ".", ".."} for part in relative.parts)
            or ":" in relative.parts[0]
        ):
            raise ValueError("模型 ZIP 包含不安全路徑")
        normalized = relative.as_posix().casefold()
        if normalized in normalized_names:
            raise ValueError("模型 ZIP 包含重複或大小寫衝突的路徑")
        normalized_names.add(normalized)
        roots.add(relative.parts[0])
        if member.flag_bits & 0x1:
            raise ValueError("模型 ZIP 不接受加密項目")
        unix_type = (member.external_attr >> 16) & 0o170000
        if unix_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
            raise ValueError("模型 ZIP 不接受符號連結或特殊檔案")
        if member.file_size > MAX_ARCHIVE_MEMBER_BYTES:
            raise ValueError("模型 ZIP 單一檔案超過安全上限")
        total_size += member.file_size
        if total_size > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
            raise ValueError("模型 ZIP 解壓總量超過安全上限")
        if (
            member.file_size >= 1024 * 1024
            and member.compress_size > 0
            and member.file_size / member.compress_size > MAX_ARCHIVE_COMPRESSION_RATIO
        ):
            raise ValueError("模型 ZIP 壓縮比例異常，可能是壓縮炸彈")
    if len(roots) != 1:
        raise ValueError("模型 ZIP 必須只有一個頂層資料夾")
    return members, roots.pop()


def _safe_extract(archive: Path, destination: Path) -> Path:
    """通過完整 ZIP 安全驗證後解壓縮，回傳模型包根目錄。"""
    with zipfile.ZipFile(archive) as bundle:
        members, root_name = _validated_members(bundle)
        destination_root = destination.resolve()
        for member in members:
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination_root):
                raise ValueError("模型 ZIP 包含不安全路徑")
        bundle.extractall(destination)
    return destination / root_name


def _validated_package_id(value: object) -> str:
    package_id = str(value or "")
    if not PACKAGE_ID_PATTERN.fullmatch(package_id) or package_id in {".", ".."}:
        raise ValueError("模型包 package_id 格式不安全")
    return package_id


def verify_model_package(archive_path: str | Path) -> dict[str, Any]:
    """驗證模型包結構、檔案大小與 SHA-256。"""
    archive = Path(archive_path).resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"找不到模型包：{archive}")
    with tempfile.TemporaryDirectory(prefix="aiquant-verify-") as temporary:
        root = _safe_extract(archive, Path(temporary))
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("模型包缺少 manifest.json")
        manifest = _read_json(manifest_path)
        if int(manifest.get("schema_version", 0)) != MODEL_PACKAGE_SCHEMA_VERSION:
            raise ValueError("不支援的模型包版本")
        _validated_package_id(manifest.get("package_id"))
        entries = manifest.get("files", [])
        if not isinstance(entries, list):
            raise ValueError("模型包 manifest 的 files 格式錯誤")
        listed_paths: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("模型包 manifest 包含無效項目")
            relative = PurePosixPath(str(entry.get("path", "")))
            if (
                relative.is_absolute()
                or not relative.parts
                or any(part in {"", ".", ".."} for part in relative.parts)
            ):
                raise ValueError("模型包 manifest 包含不安全路徑")
            normalized = relative.as_posix()
            if normalized in listed_paths:
                raise ValueError("模型包 manifest 包含重複檔案")
            listed_paths.add(normalized)
            path = root.joinpath(*relative.parts)
            if not path.is_file():
                raise ValueError(f"模型包缺少檔案：{normalized}")
            if path.stat().st_size != int(entry["size"]):
                raise ValueError(f"模型檔案大小不符：{normalized}")
            if _sha256(path) != str(entry["sha256"]):
                raise ValueError(f"模型檔案雜湊不符：{normalized}")
        actual_paths = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path != manifest_path
        }
        if listed_paths != actual_paths:
            raise ValueError("模型 ZIP 含有 manifest 未列出的額外檔案")
        return manifest


def import_rl_model_package(
    archive_path: str | Path,
    environments_dir: str | Path,
    *,
    trusted_local_artifact: bool = False,
) -> Path:
    """驗證並匯入 RL 模型包，回傳可交給交易程式的 training 目錄。"""
    if not trusted_local_artifact:
        raise PermissionError("SAC/PPO 模型可能含可執行的序列化內容；只可明確信任自己產生的模型包")
    manifest = verify_model_package(archive_path)
    if manifest.get("package_type") != "rl":
        raise ValueError("這不是 SAC/PPO 模型包")
    package_id = _validated_package_id(manifest.get("package_id"))
    target_root = Path(environments_dir).resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="aiquant-import-") as temporary:
        package_root = _safe_extract(Path(archive_path).resolve(), Path(temporary))
        source_environment = package_root / "environment"
        if not (source_environment / "environment.json").is_file():
            raise ValueError("RL 模型包缺少 environment.json")
        destination = (target_root / f"imported_{package_id}").resolve()
        if not destination.is_relative_to(target_root):
            raise ValueError("模型包目的地超出 RL 環境目錄")
        if destination.exists():
            raise FileExistsError(f"模型已匯入：{destination}")
        shutil.copytree(source_environment, destination)
    relative = PurePosixPath(str(manifest.get("entry_training_dir", "")))
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.parts[:2] != ("environment", "training")
        or len(relative.parts) != 3
    ):
        raise ValueError("模型包 entry_training_dir 格式不安全")
    run_name = relative.name
    training_dir = destination / "training" / run_name
    if not (training_dir / "training.json").is_file():
        raise ValueError("匯入後找不到 RL training.json")
    verify_artifact_manifest(training_dir, required=True)
    return training_dir
