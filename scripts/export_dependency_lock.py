"""從目前已驗證環境輸出專案依賴閉包鎖檔。"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
import platform
import sys
import tomllib

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def _project_requirements(project_root: Path) -> list[Requirement]:
    payload = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
    project = dict(payload["project"])
    values = list(project.get("dependencies", []))
    for requirements in dict(project.get("optional-dependencies", {})).values():
        values.extend(requirements)
    return [Requirement(str(value)) for value in values]


def _marker_applies(requirement: Requirement, active_extras: set[str]) -> bool:
    if requirement.marker is None:
        return True
    environment = default_environment()
    return any(
        requirement.marker.evaluate({**environment, "extra": extra})
        for extra in ("", *sorted(active_extras))
    )


def resolve_installed_dependency_closure(project_root: Path) -> dict[str, str]:
    """解析直接與傳遞依賴；缺少套件時拒絕產生不完整 lock。"""
    roots = _project_requirements(project_root)
    queue: deque[tuple[str, set[str]]] = deque()
    for requirement in roots:
        if _marker_applies(requirement, set(requirement.extras)):
            queue.append((canonicalize_name(requirement.name), set(requirement.extras)))

    versions: dict[str, str] = {}
    known_extras: dict[str, set[str]] = {}
    while queue:
        name, requested_extras = queue.popleft()
        previous_extras = known_extras.setdefault(name, set())
        extras_changed = not requested_extras.issubset(previous_extras)
        previous_extras.update(requested_extras)
        if name in versions and not extras_changed:
            continue
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"目前環境缺少依賴：{name}") from exc
        versions[name] = distribution.version
        for raw_requirement in distribution.requires or ():
            requirement = Requirement(raw_requirement)
            if not _marker_applies(requirement, previous_extras):
                continue
            queue.append(
                (
                    canonicalize_name(requirement.name),
                    set(requirement.extras),
                )
            )
    return versions


def export_lock(project_root: Path, output: Path) -> Path:
    versions = resolve_installed_dependency_closure(project_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    lines = [
        "# AIQuantTradingSystem 可重現依賴約束",
        f"# generated_at_utc={generated_at}",
        f"# python={platform.python_version()}",
        f"# platform={platform.platform()}",
        "# 此檔鎖定目前已通過測試的完整依賴閉包；CUDA 環境需在目標 GPU 主機另行產生。",
        "",
        *(f"{name}=={versions[name]}" for name in sorted(versions)),
        "",
    ]
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(output)
    return output


def check_lock(project_root: Path, lock_path: Path) -> None:
    """確認鎖檔與目前環境的完整依賴閉包完全一致。"""
    expected: dict[str, str] = {}
    for raw_line in lock_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            raise ValueError(f"鎖檔只接受精確版本：{line}")
        name, version = line.split("==", maxsplit=1)
        expected[canonicalize_name(name)] = version
    actual = resolve_installed_dependency_closure(project_root)
    missing = sorted(set(actual) - set(expected))
    extra = sorted(set(expected) - set(actual))
    changed = sorted(name for name in set(actual) & set(expected) if actual[name] != expected[name])
    if missing or extra or changed:
        details = []
        if missing:
            details.append("鎖檔缺少：" + ", ".join(missing))
        if extra:
            details.append("鎖檔多出：" + ", ".join(extra))
        if changed:
            details.append(
                "版本不同："
                + ", ".join(f"{name}({expected[name]} != {actual[name]})" for name in changed)
            )
        raise RuntimeError("；".join(details))


def main() -> int:
    parser = argparse.ArgumentParser(description="輸出目前 Python 環境的依賴閉包鎖檔")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--check",
        action="store_true",
        help="只驗證目前環境是否符合既有鎖檔，不覆寫檔案",
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output = args.output if args.output.is_absolute() else project_root / args.output
    path = output.resolve()
    if args.check:
        check_lock(project_root, path)
        print(f"鎖檔驗證通過：{path}")
    else:
        path = export_lock(project_root, path)
        print(f"已輸出 {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
