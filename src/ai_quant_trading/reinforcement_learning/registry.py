"""RL Champion／Challenger 模型登錄、升級與回滾。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ai_quant_trading.operations.integrity import (
    MANIFEST_NAME,
    sha256_file,
    verify_artifact_manifest,
)
from ai_quant_trading.paper_trading.storage import (
    list_paper_accounts,
    load_account_state,
    read_account_csv,
)
from ai_quant_trading.reinforcement_learning.quality import assess_rl_training_quality


@dataclass(frozen=True, slots=True)
class PaperValidationEvidence:
    """模型是否累積足夠的真實時間模擬交易證據。"""

    eligible: bool
    accounts: int
    cycles: int
    observed_days: float
    total_return: float | None
    max_drawdown: float | None
    reasons: tuple[str, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _training_payload(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir).resolve() / "training.json"
    if not path.exists():
        raise FileNotFoundError(f"找不到 RL 訓練中繼資料：{path}")
    payload = _read_json(path)
    if payload.get("status") != "complete":
        raise ValueError("只有完成的 RL 訓練可以登錄")
    return payload


def _expert_kind(payload: dict[str, Any]) -> str:
    kind = str(dict(payload.get("environment_expert", {})).get("kind", "general"))
    return kind if kind in {"long_term", "short_term", "general"} else "general"


def load_model_registry(path: str | Path) -> dict[str, Any]:
    """讀取 registry；首次使用時回傳空白結構。"""
    source = Path(path)
    if not source.exists():
        return {"version": 1, "updated_at": None, "experts": {}}
    payload = _read_json(source)
    payload.setdefault("version", 1)
    payload.setdefault("updated_at", None)
    payload.setdefault("experts", {})
    return payload


def _find_research_summary(run_dir: Path) -> Path | None:
    for parent in run_dir.parents:
        candidate = parent / "summary.json"
        if candidate.exists():
            return candidate
    return None


def _summary_candidate_path(summary_path: Path, summary: dict[str, Any]) -> Path:
    relative = str(summary.get("candidate_run_relative") or "").strip()
    if not relative:
        raise ValueError("Protocol v2 研究摘要缺少 candidate_run_relative")
    return _resolve_experiment_path(summary_path, relative, label="候選模型")


def _resolve_experiment_path(summary_path: Path, relative: str, *, label: str) -> Path:
    """只允許研究證據指向同一實驗目錄內的相對路徑。"""
    raw_path = Path(relative)
    if raw_path.is_absolute():
        raise ValueError(f"{label}必須使用實驗內相對路徑")
    experiment_dir = summary_path.parent.resolve()
    resolved = (experiment_dir / raw_path).resolve()
    try:
        resolved.relative_to(experiment_dir)
    except ValueError as exc:
        raise ValueError(f"{label}路徑超出研究實驗目錄") from exc
    return resolved


def _validate_final_holdout_evidence(
    run: Path,
    summary_path: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    """驗證候選、封存資料與模型成品的完整證據鏈。"""
    if int(summary.get("research_protocol_version") or 0) < 2:
        raise ValueError("研究摘要缺少 final holdout protocol v2")
    if not summary.get("selection_eligible"):
        raise ValueError("模型選擇階段未通過穩健性門檻")
    if _summary_candidate_path(summary_path, summary) != run:
        raise ValueError("只有研究摘要選出的 candidate_run 可以升級")
    final_holdout = dict(summary.get("final_holdout", {}))
    if final_holdout.get("status") != "complete" or not final_holdout.get("eligible"):
        reasons = "；".join(str(item) for item in final_holdout.get("reasons", []))
        suffix = f"：{reasons}" if reasons else ""
        raise ValueError("Final holdout 尚未通過品質門檻" + suffix)
    if bool(final_holdout.get("selection_use", True)):
        raise ValueError("Final holdout 被標記為曾參與模型選擇")
    evidence_relative = str(final_holdout.get("candidate_run_relative") or "").strip()
    if not evidence_relative:
        raise ValueError("Final holdout 證據缺少 candidate_run_relative")
    evidence_candidate = _resolve_experiment_path(
        summary_path,
        evidence_relative,
        label="Final holdout 候選模型",
    )
    if evidence_candidate != run:
        raise ValueError("Final holdout 評估的候選模型與目前模型不一致")

    verify_artifact_manifest(run, required=True)
    candidate_manifest = run / MANIFEST_NAME
    if sha256_file(candidate_manifest) != str(final_holdout.get("candidate_manifest_sha256", "")):
        raise ValueError("候選模型完整性清單與 final holdout 證據不一致")

    holdout_relative = str(final_holdout.get("holdout_dir_relative") or "final_holdout")
    holdout_dir = _resolve_experiment_path(
        summary_path,
        holdout_relative,
        label="Final holdout 資料",
    )
    verify_artifact_manifest(holdout_dir, required=True)
    if sha256_file(holdout_dir / MANIFEST_NAME) != str(
        final_holdout.get("holdout_manifest_sha256", "")
    ):
        raise ValueError("Final holdout 資料完整性清單與研究摘要不一致")
    evaluation_relative = str(
        final_holdout.get("evaluation_csv_relative") or "final_holdout_evaluation.csv"
    )
    evaluation_path = _resolve_experiment_path(
        summary_path,
        evaluation_relative,
        label="Final holdout 評估明細",
    )
    if not evaluation_path.is_file():
        raise FileNotFoundError("Final holdout 評估明細遺失")
    if sha256_file(evaluation_path) != str(final_holdout.get("evaluation_csv_sha256", "")):
        raise ValueError("Final holdout 評估明細與研究摘要不一致")
    return final_holdout


def register_model_challenger(
    registry_path: str | Path,
    run_dir: str | Path,
    *,
    research_summary: str | Path | None = None,
) -> dict[str, Any]:
    """把完成模型加入候選區；登錄本身不會改變交易使用中的模型。"""
    run = Path(run_dir).resolve()
    training = _training_payload(run)
    kind = _expert_kind(training)
    summary_path = (
        Path(research_summary).resolve() if research_summary else _find_research_summary(run)
    )
    quality = assess_rl_training_quality(training)
    final_holdout_status = "missing"
    if summary_path and summary_path.is_file():
        final_holdout_status = str(
            dict(_read_json(summary_path).get("final_holdout", {})).get("status", "missing")
        )
    entry: dict[str, object] = {
        "run_dir": str(run),
        "registered_at": _utc_now(),
        "status": "challenger",
        "training_quality_eligible": quality.eligible,
        "training_quality_reasons": list(quality.reasons),
        "research_summary": str(summary_path) if summary_path else None,
        "final_holdout_status": final_holdout_status,
    }
    registry = load_model_registry(registry_path)
    experts = dict(registry["experts"])
    expert = dict(experts.get(kind, {"champion": None, "challengers": [], "history": []}))
    challengers = [
        item for item in list(expert.get("challengers", [])) if item.get("run_dir") != str(run)
    ]
    challengers.insert(0, entry)
    expert["challengers"] = challengers
    experts[kind] = expert
    registry.update({"updated_at": _utc_now(), "experts": experts})
    _write_json(Path(registry_path), registry)
    return entry


def assess_paper_model_evidence(
    paper_root: str | Path,
    run_dir: str | Path,
    *,
    expert_kind: str,
) -> PaperValidationEvidence:
    """依長短期週期檢查模擬倉的天數、輪次、回撤與期末報酬。"""
    run_name = Path(run_dir).name
    if expert_kind == "long_term":
        minimum_days, minimum_cycles, maximum_drawdown = 180, 60, 0.12
    elif expert_kind == "short_term":
        minimum_days, minimum_cycles, maximum_drawdown = 56, 200, 0.06
    else:
        minimum_days, minimum_cycles, maximum_drawdown = 30, 100, 0.15
    frames: list[pd.DataFrame] = []
    account_count = 0
    halted = False
    for paths in list_paper_accounts(paper_root):
        try:
            state = load_account_state(paths)
        except (OSError, ValueError, TypeError):
            continue
        if state.model_kind != "rl" or state.model_dir != run_name:
            continue
        performance = read_account_csv(paths.performance_csv)
        if performance.empty:
            continue
        account_count += 1
        halted = halted or state.risk_halted
        frame = performance.copy()
        frame["account_id"] = state.account_id
        frames.append(frame)
    reasons: list[str] = []
    if not frames:
        reasons.append("找不到此模型的模擬交易紀錄")
        return PaperValidationEvidence(False, 0, 0, 0.0, None, None, tuple(reasons))
    combined = pd.concat(frames, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True, errors="coerce")
    combined["total_return"] = pd.to_numeric(combined["total_return"], errors="coerce")
    combined["drawdown"] = pd.to_numeric(combined["drawdown"], errors="coerce")
    combined = combined.dropna(subset=["timestamp"]).sort_values(["account_id", "timestamp"])
    account_spans = combined.groupby("account_id")["timestamp"].agg(["min", "max"])
    span_days = (account_spans["max"] - account_spans["min"]).dt.total_seconds() / 86_400
    # 必須至少有一個帳戶獨立走完觀察期間，不能把互不連續的帳戶拼成長期證據。
    observed_days = float(span_days.max()) if not span_days.empty else 0.0
    cycles = len(combined)
    terminal_returns = (
        combined.dropna(subset=["total_return"])
        .groupby("account_id", sort=False)
        .tail(1)["total_return"]
    )
    total_return = float(terminal_returns.median()) if not terminal_returns.empty else None
    drawdowns = combined["drawdown"].dropna()
    max_drawdown = abs(float(drawdowns.min())) if not drawdowns.empty else None
    if observed_days < minimum_days:
        reasons.append(f"模擬觀察 {observed_days:.1f} 天，至少需要 {minimum_days} 天")
    if cycles < minimum_cycles:
        reasons.append(f"模擬輪次 {cycles}，至少需要 {minimum_cycles} 次")
    if total_return is None or total_return <= 0:
        reasons.append("模擬帳戶期末報酬中位數必須大於 0")
    if max_drawdown is None or max_drawdown > maximum_drawdown:
        text = "N/A" if max_drawdown is None else f"{max_drawdown:.2%}"
        reasons.append(f"模擬最大回撤 {text}，上限為 {maximum_drawdown:.0%}")
    if halted:
        reasons.append("至少一個模擬帳戶仍處於風控停機")
    return PaperValidationEvidence(
        not reasons,
        account_count,
        cycles,
        observed_days,
        total_return,
        max_drawdown,
        tuple(reasons),
    )


def promote_model_challenger(
    registry_path: str | Path,
    run_dir: str | Path,
    *,
    paper_root: str | Path,
) -> dict[str, Any]:
    """通過訓練、研究與模擬三道門檻後，才可升級為 Champion。"""
    run = Path(run_dir).resolve()
    training = _training_payload(run)
    kind = _expert_kind(training)
    quality = assess_rl_training_quality(training)
    if not quality.eligible:
        raise ValueError("模型未通過單次樣本外品質門檻：" + "；".join(quality.reasons))
    registry = load_model_registry(registry_path)
    experts = dict(registry["experts"])
    expert = dict(experts.get(kind, {"champion": None, "challengers": [], "history": []}))
    registered_runs = {str(item.get("run_dir", "")) for item in expert.get("challengers", [])}
    if str(run) not in registered_runs:
        raise ValueError("候選模型尚未登錄為 Challenger，禁止直接升級")
    summary_path = _find_research_summary(run)
    if summary_path is None:
        raise ValueError("模型不屬於多 seed Walk-forward 研究實驗")
    summary = _read_json(summary_path)
    if not summary.get("eligible"):
        raise ValueError("多 seed Walk-forward 摘要尚未通過穩健性門檻")
    final_holdout = _validate_final_holdout_evidence(run, summary_path, summary)
    evidence = assess_paper_model_evidence(paper_root, run, expert_kind=kind)
    if not evidence.eligible:
        raise ValueError("模擬倉證據不足：" + "；".join(evidence.reasons))

    previous = expert.get("champion")
    history = list(expert.get("history", []))
    if previous:
        history.append(
            {**dict(previous), "retired_at": _utc_now(), "reason": "promoted_new_champion"}
        )
    champion = {
        "run_dir": str(run),
        "promoted_at": _utc_now(),
        "status": "champion",
        "promotion_evidence_version": 2,
        "research_summary": str(summary_path),
        "research_summary_sha256": sha256_file(summary_path),
        "candidate_manifest_sha256": sha256_file(run / MANIFEST_NAME),
        "final_holdout_manifest_sha256": str(final_holdout["holdout_manifest_sha256"]),
        "final_holdout_evaluated_at": final_holdout.get("evaluated_at"),
        "final_holdout_metrics": dict(final_holdout.get("metrics", {})),
        "paper_evidence": asdict(evidence),
    }
    expert["champion"] = champion
    expert["history"] = history
    expert["challengers"] = [
        item for item in list(expert.get("challengers", [])) if item.get("run_dir") != str(run)
    ]
    experts[kind] = expert
    registry.update({"updated_at": _utc_now(), "experts": experts})
    _write_json(Path(registry_path), registry)
    return champion


def rollback_model_champion(registry_path: str | Path, expert_kind: str) -> dict[str, Any]:
    """把上一個 Champion 還原；只修改登錄指標，不直接送出任何交易。"""
    registry = load_model_registry(registry_path)
    experts = dict(registry["experts"])
    expert = dict(experts.get(expert_kind, {}))
    history = list(expert.get("history", []))
    if not history:
        raise ValueError("沒有可回滾的 Champion")
    restored = dict(history.pop())
    restored.pop("retired_at", None)
    restored.pop("reason", None)
    restored.update({"status": "champion", "restored_at": _utc_now()})
    current = expert.get("champion")
    if current:
        challengers = list(expert.get("challengers", []))
        challengers.insert(
            0, {**dict(current), "status": "challenger", "rolled_back_at": _utc_now()}
        )
        expert["challengers"] = challengers
    expert["champion"] = restored
    expert["history"] = history
    experts[expert_kind] = expert
    registry.update({"updated_at": _utc_now(), "experts": experts})
    _write_json(Path(registry_path), registry)
    return restored


def registered_champion(registry_path: str | Path, expert_kind: str) -> Path | None:
    """取得指定專家的 Champion 路徑。"""
    registry = load_model_registry(registry_path)
    champion = dict(dict(registry.get("experts", {})).get(expert_kind, {})).get("champion")
    if not champion:
        return None
    path = Path(str(dict(champion).get("run_dir", "")))
    if (path / "training.json").exists():
        return path
    registry_root = Path(registry_path).resolve().parent
    matches = list(registry_root.rglob(f"{path.name}/training.json"))
    return matches[0].parent if len(matches) == 1 else None


def ensure_registered_champion(
    registry_path: str | Path,
    run_dir: str | Path,
    expert_kind: str,
) -> None:
    """正式環境只允許 Registry 已核准的 Champion，避免 CLI 繞過升級流程。"""
    registry = load_model_registry(registry_path)
    champion_record = dict(
        dict(registry.get("experts", {})).get(expert_kind, {}).get("champion") or {}
    )
    champion = registered_champion(registry_path, expert_kind)
    if champion is None or not champion_record:
        raise ValueError(f"{expert_kind} 尚未設定可供實盤的 Champion 模型")
    if champion.resolve() != Path(run_dir).resolve():
        raise ValueError("目前模型不是 Registry 核准的 Champion，禁止正式下單")
    summary_path = Path(str(champion_record.get("research_summary", ""))).resolve()
    if not summary_path.is_file():
        raise FileNotFoundError("Champion 的研究摘要遺失")
    if sha256_file(summary_path) != str(champion_record.get("research_summary_sha256", "")):
        raise ValueError("Champion 研究摘要已變更，禁止正式下單")
    if sha256_file(champion / MANIFEST_NAME) != str(
        champion_record.get("candidate_manifest_sha256", "")
    ):
        raise ValueError("Champion 模型清單已變更，禁止正式下單")
    summary = _read_json(summary_path)
    _validate_final_holdout_evidence(champion.resolve(), summary_path, summary)
