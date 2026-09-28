"""事件研究上傳包的白名單、完整性與安全解壓測試。"""

import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile

import pytest


def load_script(name):
    path = Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_script("kaggle_strategy_event_runner")
builder = load_script("build_kaggle_strategy_event")


def make_package(root):
    root.mkdir()
    source = root / "training_config.json"
    source.write_text("{}", encoding="utf-8")
    manifest = {"study": "btc_strategy_event_v1", "research_only": True,
                "files": {source.name: {"sha256": runner.sha256(source), "size": 2}}}
    (root / "strategy_event_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


@pytest.mark.parametrize("compressed", [False, True])
def test_bundle_integrity_in_expanded_and_zip_modes(tmp_path, compressed):
    package = make_package(tmp_path / "package")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    if compressed:
        with ZipFile(inputs / "strategy_event_bundle.zip", "w") as archive:
            for path in package.iterdir():
                archive.write(path, path.name)
    else:
        inputs = package
    root, manifest = runner.prepare_bundle(inputs, tmp_path / "expanded")
    assert (root / "training_config.json").is_file()
    assert manifest["research_only"] is True


def test_modified_or_extra_files_rejected(tmp_path):
    package = make_package(tmp_path / "package")
    (package / ".env").write_text("fixture=true", encoding="utf-8")
    with pytest.raises(ValueError, match="額外檔案"):
        runner.prepare_bundle(package, tmp_path / "expanded")
    (package / ".env").unlink()
    (package / "training_config.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="完整性"):
        runner.prepare_bundle(package, tmp_path / "expanded")


@pytest.mark.parametrize("name", ["../outside.txt", "/absolute.txt", "a\\..\\outside.txt"])
def test_unsafe_zip_path_rejected(tmp_path, name):
    with ZipFile(tmp_path / "strategy_event_bundle.zip", "w") as archive:
        archive.writestr(name, "fixture")
    with pytest.raises(ValueError, match="不安全"):
        runner.prepare_bundle(tmp_path, tmp_path / "expanded")


def test_dependency_closure_excludes_unrelated_private_files(tmp_path):
    package = tmp_path / "ai_quant_trading"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "a.py").write_text("from .b import value\n", encoding="utf-8")
    (package / "b.py").write_text("value = 1\n", encoding="utf-8")
    (package / "unrelated.py").write_text("private_value = 2\n", encoding="utf-8")
    (package / ".env").write_text("fixture=true\n", encoding="utf-8")
    paths = builder.local_dependencies(tmp_path, "ai_quant_trading.a")
    assert {path.name for path in paths} == {"__init__.py", "a.py", "b.py"}


def test_credential_scan_does_not_print_secret_value(tmp_path):
    path = tmp_path / "unsafe.py"
    marker = "ghp_" + "x" * 30
    path.write_text(f"value = {marker!r}", encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        builder.check_source_text(path)
    assert marker not in str(caught.value)
