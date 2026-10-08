"""圖文衍生檔封存；資料庫、原始成交與模型完全不刪除。"""

import hashlib
import json
from pathlib import Path
import re
import os
import tempfile
import zipfile

from filelock import FileLock

from ai_quant_trading.persistence import write_json_atomic


def _path(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if path.parent != root or not re.fullmatch(r"[a-f0-9]{64}\.(png|json|zip)", name):
        raise ValueError("封存只允許本日誌內的雜湊命名檔")
    return path


def read_asset(root, record_id, suffix="png"):
    """封存檔不解壓到任意路徑，逐檔校驗後讀取。"""
    root = Path(root).resolve()
    loose = _path(root, f"{record_id}.{suffix}")
    if loose.is_file():
        if loose.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("圖文衍生檔超過安全讀取上限")
        return loose.read_bytes()
    if (root / "archives").resolve().parent != root:
        raise ValueError("封存資料夾超出日誌範圍")
    manifest = root / "archives" / f"{record_id}.json"
    if not manifest.is_file():
        return None
    receipt = json.loads(manifest.read_text(encoding="utf-8"))
    archive = _path(root / "archives", f"{record_id}.zip")
    name = f"{record_id}.{suffix}"
    try:
        with zipfile.ZipFile(archive) as bundle:
            info = bundle.getinfo(name)
            if info.file_size > 64 * 1024 * 1024:
                raise ValueError("圖文衍生檔超過安全讀取上限")
            data = bundle.read(name)
    except zipfile.BadZipFile as error:
        raise ValueError("封存檔損壞") from error
    if hashlib.sha256(data).hexdigest() != receipt["sha256"][name]:
        raise ValueError("封存圖文校驗失敗")
    return data


def archive_journal(journal, *, keep_recent=200, limit=50, prune=False):
    """可重入封存；只有明確 prune 且 ZIP 與收據皆驗證成功才移除衍生副本。"""
    if keep_recent < 0 or not 1 <= limit <= 1000:
        raise ValueError("封存保留數與批次大小不合法")
    root = journal.root.resolve()
    directory = (root / "archives").resolve()
    if directory.parent != root:
        raise ValueError("封存資料夾超出日誌範圍")
    directory.mkdir(parents=True, exist_ok=True)
    count = 0
    with FileLock(str(root / "archive.lock"), timeout=10):
        with journal.connect() as db:
            items = db.execute("SELECT id FROM records WHERE status='complete' ORDER BY timestamp DESC,id LIMIT -1 OFFSET ?", (keep_recent,)).fetchall()
        for (record_id,) in reversed(items):
            files = [_path(root, f"{record_id}.{suffix}") for suffix in ("json", "png")]
            if not any(p.exists() for p in files):
                continue
            archive = _path(directory, f"{record_id}.zip")
            receipt = _path(directory, f"{record_id}.json")
            if not archive.exists():
                if not all(p.is_file() for p in files):
                    continue
                data = {p.name: p.read_bytes() for p in files}
                descriptor, name = tempfile.mkstemp(dir=directory, suffix=".zip.tmp")
                os.close(descriptor)
                temporary = Path(name)
                try:
                    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                        for name, content in data.items():
                            bundle.writestr(name, content)
                    with zipfile.ZipFile(temporary) as bundle:
                        if any(bundle.read(name) != content for name, content in data.items()):
                            raise ValueError("封存寫入校驗失敗")
                    temporary.replace(archive)
                finally:
                    temporary.unlink(missing_ok=True)
            # 中斷於 ZIP 完成但收據尚未落盤時，可由 ZIP 與仍存在的原檔重建。
            if not receipt.exists():
                with zipfile.ZipFile(archive) as bundle:
                    hashes = {}
                    for file in files:
                        content = bundle.read(file.name)
                        if not file.exists() or content != file.read_bytes():
                            raise ValueError("缺少可驗證的封存原檔")
                        hashes[file.name] = hashlib.sha256(content).hexdigest()
                write_json_atomic(receipt, {"schema": "journal_archive_v1", "sha256": hashes})
            expected = json.loads(receipt.read_text(encoding="utf-8"))["sha256"]
            with zipfile.ZipFile(archive) as bundle:
                for file in files:
                    digest = hashlib.sha256(bundle.read(file.name)).hexdigest()
                    if digest != expected[file.name] or (file.exists() and hashlib.sha256(file.read_bytes()).hexdigest() != digest):
                        raise ValueError("封存與衍生檔不同，禁止刪除")
            if prune:
                for file in files:
                    file.unlink(missing_ok=True)
            count += 1
            if count >= limit:
                break
    return {"archived": count, "pruned": prune, "keep_recent": keep_recent}
