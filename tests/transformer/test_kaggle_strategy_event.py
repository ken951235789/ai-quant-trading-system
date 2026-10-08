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


def test_candidate_bundle_keeps_flow_but_excludes_private_columns(tmp_path):
    import numpy as np
    import pandas as pd

    root = Path(__file__).parents[2]
    n = 100
    frame = pd.DataFrame({"timestamp": pd.date_range("2023-01-01", periods=n, freq="15min", tz="UTC"),
        "symbol": "BTC/USDT", "exchange": "binance_futures", "interval": "15m",
        "open": 100., "high": 101., "low": 99., "close": 100., "volume": 10.,
        "quote_asset_volume": 1000., "number_of_trades": np.full(n, 5),
        "taker_buy_base_volume": 6., "taker_buy_quote_volume": 600., "private_account_note": "DO_NOT_UPLOAD"})
    source, flow, output = tmp_path / "source.csv", tmp_path / "flow.csv", tmp_path / "out"
    frame[list(builder.SAFE_COLUMNS)].to_csv(source, index=False)
    frame.to_csv(flow, index=False)
    result = builder.build(source, root / "configs/transformer_strategy_event_v2.example.json", output,
                           "fixture", candidate_features=True, flow_source=flow)
    assert result["kernel_id"] == "fixture/btc-transformer-candidate-features-v1"
    package, manifest = runner.prepare_bundle(output / "dataset", tmp_path / "expanded")
    assert manifest["study"] == "btc_candidate_feature_v1"
    data = pd.read_csv(package / "data/btc_15m.csv")
    assert len(data.columns) == 13 and "private_account_note" not in data
    assert "taker_buy_base_volume" in data
    assert (package / "src/ai_quant_trading/research/candidate_evaluation.py").exists()
    assert (package / "scripts/run_candidate_training_research.py").exists()
    plan = json.loads((package / "execution_plan.json").read_text(encoding="utf-8"))
    assert plan["planned_runs"] == 18 and plan["seeds"] == [42, 137, 2026]
    assert plan["variants"] == ["F_existing", "F_compact_combined"]
    metadata = json.loads((output / "kernel/kernel-metadata.json").read_text(encoding="utf-8"))
    assert metadata["is_private"] and not metadata["enable_internet"]


def test_unapproved_config_fields_fail_before_packaging(tmp_path):
    config = tmp_path / "bad.json"
    config.write_text(json.dumps({"model": {}, "training": {}, "account": "private"}))
    with pytest.raises(ValueError, match="私人設定"):
        builder.build(tmp_path / "not_read.csv", config, tmp_path / "out", "fixture")
