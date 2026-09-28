"""單次檢查 V3.4 訓練，完成後下載證據並且只提交一次後續 SAC 研究。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.parse import urlparse


def _save(path: Path, state: dict) -> None:
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _download(api, kernel: str, destination: Path, pattern: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    cursor = None
    while True:
        _, cursor = api.kernels_output(
            kernel,
            str(destination),
            file_pattern=pattern,
            force=False,
            quiet=True,
            page_token=cursor,
        )
        if not cursor:
            break


def _load_summary(results: Path, name: str) -> tuple[Path, dict]:
    candidates = sorted(results.rglob(name), key=lambda path: (len(path.parts), str(path)))
    if not candidates:
        raise RuntimeError(f"完成後缺少 {name}，保留狀態供下次重新下載")
    summary = json.loads(candidates[0].read_text(encoding="utf-8"))
    # 候選模型目錄也會附摘要；完全相同的副本可接受，不一致時禁止接續提交。
    for candidate in candidates[1:]:
        if json.loads(candidate.read_text(encoding="utf-8")) != summary:
            raise RuntimeError(f"發現內容不一致的 {name}，必須先核對研究版本")
    return candidates[0], summary


def advance(api, root: Path) -> dict:
    state_path = root / "monitor_state.json"
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else {
            "stage": "transformer",
            "status": "monitoring",
            "sac_submitted": False,
        }
    )
    if state.get("status") in {"complete", "failed"}:
        return state
    transformer = json.loads((root / "kernel/kernel-metadata.json").read_text(encoding="utf-8"))
    sac = json.loads((root / "kernel_sac/kernel-metadata.json").read_text(encoding="utf-8"))
    stage = str(state["stage"])
    kernel = transformer["id"] if stage == "transformer" else state.get("sac_kernel", sac["id"])
    status = str(api.kernels_status(kernel).status).rsplit(".", 1)[-1].lower()
    state.update({"kernel": kernel, "remote_status": status})
    if status not in {"complete", "error", "cancelled", "canceled"}:
        _save(state_path, state)
        return state
    results = root / f"{stage}_results"
    pattern = r".*(?:five_seed_summary\.json|five_seed_progress\.json|candidate\.zip|sac_v34_summary\.json|sac_v34_research_artifacts\.zip|\.log)$"
    _download(api, kernel, results, pattern)
    if status != "complete":
        state.update({"status": "failed", "reason": f"{stage}: {status}", "results": str(results)})
        _save(state_path, state)
        return state
    name = "five_seed_summary.json" if stage == "transformer" else "sac_v34_summary.json"
    summary_path, summary = _load_summary(results, name)
    if summary.get("status") != "complete":
        state.update({"status": "failed", "reason": f"{stage} 摘要未完成", "results": str(results)})
    elif stage == "transformer":
        state["transformer_summary"] = str(summary_path)
        state["transformer_quality_passed"] = bool(
            summary.get("quality_gate", {}).get("passed", False)
        )
        state["transformer_validation_passed"] = bool(
            summary.get("validation_gate", {}).get("passed", False)
        )
        # 不論分析師是否過研究門檻，三組輸入消融都能檢查 SAC 是否需要其資訊。
        # 先持久化提交意圖；連線中斷時下次只查同一 kernel，避免重複消耗額度。
        state.update({"stage": "sac", "status": "submission_pending", "sac_submitted": True})
        _save(state_path, state)
        response = api.kernels_push(str(root / "kernel_sac"), acc="NvidiaTeslaT4")
        error = getattr(response, "error", None)
        if error:
            raise RuntimeError(f"SAC 提交失敗：{error}")
        # Kaggle 可能依標題產生不同 slug，後續以提交回應的正式網址查詢。
        url = urlparse(str(getattr(response, "url", "")))
        parts = url.path.strip("/").split("/")
        canonical = sac["id"]
        if (
            url.hostname in {"kaggle.com", "www.kaggle.com"}
            and len(parts) == 3
            and parts[0] == "code"
        ):
            canonical = "/".join(parts[1:])
        state.update(
            {
                "status": "monitoring",
                "kernel": canonical,
                "sac_kernel": canonical,
                "remote_status": "submitted",
            }
        )
    else:
        state.update(
            {"status": "complete", "sac_summary": str(summary_path), "deployment_allowed": False}
        )
    _save(state_path, state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs/kaggle_v34_economic_20260923",
    )
    args = parser.parse_args()
    import truststore

    truststore.inject_into_ssl()
    from filelock import FileLock
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    with FileLock(str(args.root / "monitor.lock"), timeout=1):
        state = advance(api, args.root)
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
