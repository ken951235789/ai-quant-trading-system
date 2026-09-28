"""檢查公開檔案邊界；只輸出規則與位置，不印出疑似私人內容。

這不是完整的秘密掃描器，發布時仍須另外執行 Gitleaks。
--staged 直接檢查 Git index，避免工作檔安全但暫存版本仍含敏感資料。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import subprocess


PRIVATE_ROOTS = {"outputs", "artifacts", "dist", "build", "logs", ".agents"}
PRIVATE_SUFFIXES = {
    ".pt", ".pth", ".ckpt", ".onnx", ".safetensors", ".pkl", ".pickle", ".joblib",
    ".csv", ".tsv", ".parquet", ".feather", ".arrow", ".db", ".duckdb", ".sqlite",
    ".sqlite3", ".zip", ".7z", ".tar", ".gz", ".exe", ".dll", ".key", ".pem",
    ".p12", ".pfx", ".log", ".dump", ".pyc",
}
EMAIL = re.compile(r"[\w.+-]+@([\w.-]+\.[A-Za-z]{2,})")
WINDOWS_USER = re.compile(r'''[A-Za-z]:[\\/]+Users[\\/]+([^\\/\s`"']+)''')
PLACEHOLDER_USERS = {"你的帳號", "<username>", "YOUR_USERNAME", "example", "test", "your-user"}


def inspect_file(name: str, data: bytes, mode: str = "100644") -> list[dict]:
    """檢查單一快照檔案；測試只使用合成內容。"""
    path = PurePosixPath(name)
    findings: list[dict] = []

    def flag(rule: str, line: int = 0) -> None:
        findings.append({"file": name, "line": line, "rule": rule})

    if mode not in {"100644", "100755"}:
        flag("non_regular_file")
    if path.parts[0] in PRIVATE_ROOTS or path.suffix.lower() in PRIVATE_SUFFIXES:
        flag("private_artifact")
    if (path.name.startswith(".env") and path.name != ".env.example") or path.name in {
        "kaggle.json", "secrets.toml", "id_rsa", "id_ed25519",
    }:
        flag("credential_filename")
    if "secrets" in path.parts or (path.parts[0] == "data" and path.name != "README.md"):
        flag("private_data_directory")
    if len(data) > 2_000_000:
        flag("oversized_file")
    if name == "docs/assets/social-preview.png" and data.startswith(b"\x89PNG\r\n\x1a\n"):
        return findings
    try:
        content = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        flag("unreviewed_binary")
        return findings
    for number, line in enumerate(content.splitlines(), 1):
        for match in EMAIL.finditer(line):
            domain = match.group(1).lower()
            if domain not in {"example.com", "example.org", "example.net", "users.noreply.github.com"} and not domain.endswith((".test", ".invalid", ".example")):
                flag("non_example_email", number)
        for match in WINDOWS_USER.finditer(line):
            if match.group(1) not in PLACEHOLDER_USERS:
                flag("personal_windows_path", number)
    return findings


def index_files(root: Path) -> list[tuple[str, bytes, str]]:
    """以 batch 讀取暫存區 blob，不使用工作目錄中的替代內容。"""
    entries = subprocess.check_output(["git", "ls-files", "--stage", "-z"], cwd=root)
    items = []
    for entry in entries.split(b"\0"):
        if not entry:
            continue
        metadata, name = entry.split(b"\t", 1)
        mode, oid, stage = metadata.decode("ascii").split()
        if stage != "0":
            raise ValueError("暫存區仍有未解決衝突，禁止發布")
        items.append((name.decode("utf-8"), mode, oid))
    process = subprocess.run(
        ["git", "cat-file", "--batch"], cwd=root,
        input="".join(f"{item[2]}\n" for item in items).encode("ascii"),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )
    offset = 0
    result = []
    for name, mode, oid in items:
        end = process.stdout.index(b"\n", offset)
        actual_oid, kind, size = process.stdout[offset:end].split()
        if actual_oid.decode("ascii") != oid or kind != b"blob":
            raise ValueError("暫存區包含非 blob 物件，禁止發布")
        offset = end + 1
        length = int(size)
        result.append((name, process.stdout[offset:offset + length], mode))
        offset += length + 1
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.staged:
        files = index_files(root)
    else:
        names = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root,
        ).decode("utf-8").split("\0")
        files = [(name, (root / name).read_bytes(), "120000" if (root / name).is_symlink() else "100644")
                 for name in sorted(set(names)) if name and (root / name).exists()]
    findings = [finding for name, data, mode in files for finding in inspect_file(name, data, mode)]
    print(json.dumps({"files": len(files), "findings": findings}, ensure_ascii=False, indent=2))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
