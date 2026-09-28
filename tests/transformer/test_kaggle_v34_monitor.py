"""監控必須等待完成，而且第二階段只允許提交一次。"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location(
    "v34_monitor", Path(__file__).resolve().parents[2] / "scripts/monitor_kaggle_v34.py"
)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


class FakeAPI:
    def __init__(self):
        self.status = "running"
        self.pushes = 0

    def kernels_status(self, _kernel):
        return SimpleNamespace(status=self.status)

    def kernels_output(self, kernel, path, **_kwargs):
        name = (
            "five_seed_summary.json" if kernel.endswith("transformer") else "sac_v34_summary.json"
        )
        (Path(path) / name).write_text(
            json.dumps({"status": "complete", "quality_gate": {"passed": False}}), encoding="utf-8"
        )
        return [], None

    def kernels_push(self, _path, **_kwargs):
        self.pushes += 1
        self.status = "running"
        return SimpleNamespace(error=None)


def test_monitor_transitions_and_never_resubmits(tmp_path):
    for folder, name in (("kernel", "transformer"), ("kernel_sac", "sac")):
        directory = tmp_path / folder
        directory.mkdir()
        (directory / "kernel-metadata.json").write_text(
            json.dumps({"id": "test/" + name}), encoding="utf-8"
        )
    api = FakeAPI()
    assert monitor.advance(api, tmp_path)["stage"] == "transformer"
    assert api.pushes == 0
    api.status = "complete"
    assert monitor.advance(api, tmp_path)["stage"] == "sac"
    monitor.advance(api, tmp_path)
    assert api.pushes == 1
    api.status = "complete"
    assert monitor.advance(api, tmp_path)["status"] == "complete"
    monitor.advance(api, tmp_path)
    assert api.pushes == 1


def test_identical_summary_copies_are_accepted(tmp_path):
    nested = tmp_path / "selected"
    nested.mkdir()
    summary = {"status": "complete", "selected_seed": 11}
    for folder in (tmp_path, nested):
        (folder / "five_seed_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    path, loaded = monitor._load_summary(tmp_path, "five_seed_summary.json")
    assert path == tmp_path / "five_seed_summary.json"
    assert loaded == summary


def test_conflicting_or_missing_summaries_are_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="缺少"):
        monitor._load_summary(tmp_path, "five_seed_summary.json")
    nested = tmp_path / "selected"
    nested.mkdir()
    for index, folder in enumerate((tmp_path, nested)):
        (folder / "five_seed_summary.json").write_text(
            json.dumps({"status": "complete", "selected_seed": index}), encoding="utf-8"
        )
    with pytest.raises(RuntimeError, match="不一致"):
        monitor._load_summary(tmp_path, "five_seed_summary.json")


def test_uncertain_submission_is_not_automatically_retried(tmp_path):
    for folder, name in (("kernel", "transformer"), ("kernel_sac", "sac")):
        directory = tmp_path / folder
        directory.mkdir()
        (directory / "kernel-metadata.json").write_text(
            json.dumps({"id": "test/" + name}), encoding="utf-8"
        )

    class UncertainAPI(FakeAPI):
        def kernels_push(self, *_args, **_kwargs):
            self.pushes += 1
            self.status = "running"
            raise TimeoutError("伺服器可能已接受提交")

    api = UncertainAPI()
    api.status = "complete"
    with pytest.raises(TimeoutError):
        monitor.advance(api, tmp_path)
    state = json.loads((tmp_path / "monitor_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "submission_pending"
    assert state["stage"] == "sac"
    monitor.advance(api, tmp_path)
    assert api.pushes == 1
    api.status = "complete"
    assert monitor.advance(api, tmp_path)["status"] == "complete"
    monitor.advance(api, tmp_path)
    assert api.pushes == 1


def test_submission_uses_server_canonical_slug(tmp_path):
    for folder, name in (("kernel", "transformer"), ("kernel_sac", "sac")):
        directory = tmp_path / folder
        directory.mkdir()
        (directory / "kernel-metadata.json").write_text(
            json.dumps({"id": "test/" + name}), encoding="utf-8"
        )

    class CanonicalAPI(FakeAPI):
        def kernels_push(self, *args, **kwargs):
            super().kernels_push(*args, **kwargs)
            return SimpleNamespace(error=None, url="https://www.kaggle.com/code/test/actual-sac")

        def kernels_status(self, kernel):
            if self.pushes:
                assert kernel == "test/actual-sac"
            return super().kernels_status(kernel)

    api = CanonicalAPI()
    api.status = "complete"
    state = monitor.advance(api, tmp_path)
    assert state["sac_kernel"] == "test/actual-sac"
    monitor.advance(api, tmp_path)
    assert api.pushes == 1
