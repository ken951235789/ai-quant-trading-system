"""驗證 Kaggle 自動解壓縮與 ZIP 掛載兩種輸入，不需要連線或訓練。"""

import importlib.util
from pathlib import Path
from zipfile import ZipFile

import pytest

spec = importlib.util.spec_from_file_location(
    "v34_sac_runner",
    Path(__file__).resolve().parents[2] / "scripts/kaggle_sac_v34_ablation_runner.py",
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

FILES = {
    "formal_manifest.json": "{}",
    "data/transformer_v3_formal.csv": "timestamp,close\n",
    "src/ai_quant_trading/__init__.py": "",
}


@pytest.mark.parametrize("compressed", [False, True])
def test_locates_expanded_and_compressed_bundle(tmp_path, compressed):
    inputs = tmp_path / "input"
    inputs.mkdir()
    if compressed:
        with ZipFile(inputs / "transformer_v3_formal_bundle.zip", "w") as archive:
            for name, content in FILES.items():
                archive.writestr(name, content)
    else:
        for name, content in FILES.items():
            path = inputs / "dataset" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    root = runner.prepare_bundle_root(inputs, tmp_path / "extracted")
    assert (root / "data/transformer_v3_formal.csv").is_file()


def test_missing_input_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match="訓練包"):
        runner.prepare_bundle_root(tmp_path, tmp_path / "extracted")


def test_archive_cannot_escape_destination(tmp_path):
    with ZipFile(tmp_path / "transformer_v3_formal_bundle.zip", "w") as archive:
        archive.writestr("../outside.txt", "invalid")
    with pytest.raises(ValueError, match="超出"):
        runner.prepare_bundle_root(tmp_path, tmp_path / "extracted")
    assert not (tmp_path / "outside.txt").exists()
