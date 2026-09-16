"""模型與設定成品的 SHA-256 完整性清單。"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable


DEFAULT_SUFFIXES = {".json", ".yaml", ".yml", ".zip", ".pt", ".safetensors"}
MANIFEST_NAME = "artifact_manifest.json"


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """串流計算檔案雜湊，避免大型模型一次載入記憶體。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_artifact_manifest(
    root: str | Path,
    files: Iterable[str | Path] | None = None,
) -> Path:
    """建立可攜式完整性清單；清單本身不納入雜湊。"""
    base = Path(root).resolve()
    if files is None:
        candidates = [
            path
            for path in base.rglob("*")
            if path.is_file() and path.suffix.lower() in DEFAULT_SUFFIXES
        ]
    else:
        candidates = [Path(path).resolve() for path in files]
    artifacts: dict[str, dict[str, int | str]] = {}
    for path in sorted(set(candidates)):
        if path.name == MANIFEST_NAME:
            continue
        try:
            relative = path.relative_to(base)
        except ValueError as exc:
            raise ValueError(f"成品不可位於清單根目錄之外：{path}") from exc
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"成品必須是根目錄內的一般檔案：{path}")
        artifacts[relative.as_posix()] = {
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
        }
    manifest = base / MANIFEST_NAME
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hash_algorithm": "sha256",
        "artifacts": artifacts,
    }
    manifest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def verify_artifact_manifest(root: str | Path, *, required: bool = True) -> list[str]:
    """驗證清單內每個檔案；任何缺檔、大小或雜湊差異都 fail closed。"""
    base = Path(root).resolve()
    manifest = base / MANIFEST_NAME
    if not manifest.is_file():
        if required:
            raise FileNotFoundError(f"缺少模型完整性清單：{manifest}")
        return []
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("hash_algorithm") != "sha256":
        raise ValueError("模型完整性清單版本或演算法不支援")
    verified: list[str] = []
    for relative, expected in dict(payload.get("artifacts", {})).items():
        path = (base / relative).resolve()
        try:
            path.relative_to(base)
        except ValueError as exc:
            raise ValueError("完整性清單包含目錄穿越路徑") from exc
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"模型成品遺失或不是一般檔案：{relative}")
        if path.stat().st_size != int(expected["size"]):
            raise ValueError(f"模型成品大小不符：{relative}")
        if sha256_file(path) != str(expected["sha256"]):
            raise ValueError(f"模型成品 SHA-256 不符：{relative}")
        verified.append(relative)
    return verified
