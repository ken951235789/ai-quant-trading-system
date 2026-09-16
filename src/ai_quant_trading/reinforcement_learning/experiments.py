"""多 seed、Walk-forward 與穩健性彙整研究。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
from typing import Any, Callable

import pandas as pd

from ai_quant_trading.operations.integrity import (
    MANIFEST_NAME,
    build_artifact_manifest,
    sha256_file,
    verify_artifact_manifest,
)
from ai_quant_trading.reinforcement_learning.policy import load_rl_policy
from ai_quant_trading.reinforcement_learning.quality import (
    assess_rl_evaluation_quality,
    assess_rl_training_quality,
)
from ai_quant_trading.reinforcement_learning.training import (
    RLTrainingResult,
    evaluate_rl_markets,
    train_rl_agent,
)
from ai_quant_trading.reinforcement_learning.training_config import RLTrainingConfig


@dataclass(frozen=True, slots=True)
class RLResearchConfig:
    """一次穩健性實驗使用的 seeds 與時間視窗數。"""

    seeds: tuple[int, ...] = (11, 23, 42, 67, 101)
    walk_forward_folds: int = 3
    min_rows_per_split: int = 20
    final_holdout_fraction: float = 0.10
    min_final_holdout_rows: int = 100

    def __post_init__(self) -> None:
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seed 不可重複")
        if not self.seeds or any(seed < 0 for seed in self.seeds):
            raise ValueError("至少需要一個非負整數 seed")
        if not 0 <= self.walk_forward_folds <= 8:
            raise ValueError("walk_forward_folds 必須介於 0 與 8")
        if self.min_rows_per_split <= 0:
            raise ValueError("min_rows_per_split 必須大於 0")
        if not 0.05 <= self.final_holdout_fraction <= 0.30:
            raise ValueError("final_holdout_fraction 必須介於 0.05 與 0.30")
        if self.min_final_holdout_rows <= 0:
            raise ValueError("min_final_holdout_rows 必須大於 0")


@dataclass(frozen=True, slots=True)
class RLResearchResult:
    """一次研究實驗的資料夾、摘要與候選模型。"""

    experiment_dir: Path
    summary_json: Path
    runs_csv: Path
    run_dirs: tuple[Path, ...]
    eligible: bool
    candidate_run: Path | None


ResearchProgressCallback = Callable[[dict[str, object]], None]
Trainer = Callable[..., RLTrainingResult]
HoldoutEvaluator = Callable[..., dict[str, object]]


def parse_seed_list(value: str) -> tuple[int, ...]:
    """解析逗號分隔 seed，保留輸入順序並拒絕重複值。"""
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("Seeds 請使用逗號分隔的非負整數") from exc
    config = RLResearchConfig(seeds=seeds, walk_forward_folds=0)
    return config.seeds


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _fold_boundaries(
    rows: int,
    original_train_rows: int,
    folds: int,
    minimum: int,
) -> list[tuple[int, int, int]]:
    """建立擴展訓練窗；每一 fold 的測試區段互不重疊。"""
    remaining = rows - original_train_rows
    block = remaining // (folds + 1)
    if original_train_rows < minimum or block < minimum:
        raise ValueError(f"Walk-forward 資料不足：訓練 {original_train_rows} 筆、每窗 {block} 筆")
    boundaries: list[tuple[int, int, int]] = []
    for index in range(folds):
        train_end = original_train_rows + block * index
        validation_end = train_end + block
        test_end = rows if index == folds - 1 else validation_end + block
        if test_end - validation_end < minimum:
            raise ValueError("Walk-forward 最後測試窗資料不足")
        boundaries.append((train_end, validation_end, test_end))
    return boundaries


def _split_final_holdout(
    frame: pd.DataFrame,
    *,
    fraction: float,
    minimum_rows: int,
    original_train_rows: int,
    folds: int,
    min_rows_per_split: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """從資料尾端封存 final holdout，且不讓選模 folds 接觸。"""
    holdout_rows = max(minimum_rows, math.ceil(len(frame) * fraction))
    maximum_holdout = len(frame) - original_train_rows - (folds + 1) * min_rows_per_split
    if maximum_holdout < minimum_rows or holdout_rows > maximum_holdout:
        raise ValueError(
            "資料不足以同時建立 Walk-forward 與 final holdout："
            f"總計 {len(frame)} 筆、初始訓練 {original_train_rows} 筆、"
            f"holdout 至少 {minimum_rows} 筆"
        )
    selection = frame.iloc[:-holdout_rows].reset_index(drop=True)
    holdout = frame.iloc[-holdout_rows:].reset_index(drop=True)
    if "timestamp" in frame and selection["timestamp"].max() >= holdout["timestamp"].min():
        raise ValueError("Final holdout 時間邊界與模型選擇資料重疊")
    return selection, holdout


def _load_market_frame(environment_dir: Path, files: dict[str, str]) -> pd.DataFrame:
    frames = [
        pd.read_csv(environment_dir / files[name]) for name in ("train", "validation", "test")
    ]
    frame = pd.concat(frames, ignore_index=True)
    if "timestamp" in frame:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
        frame = frame.drop_duplicates("timestamp", keep="last")
    return frame.reset_index(drop=True)


def _write_fold_frames(
    frame: pd.DataFrame,
    fold_dir: Path,
    files: dict[str, str],
    boundaries: tuple[int, int, int],
) -> dict[str, int]:
    train_end, validation_end, test_end = boundaries
    slices = {
        "train": frame.iloc[:train_end],
        "validation": frame.iloc[train_end:validation_end],
        "test": frame.iloc[validation_end:test_end],
    }
    sizes: dict[str, int] = {}
    for name, split in slices.items():
        target = fold_dir / files[name]
        target.parent.mkdir(parents=True, exist_ok=True)
        split.to_csv(target, index=False, encoding="utf-8")
        sizes[name] = len(split)
    return sizes


def create_walk_forward_environments(
    environment_dir: str | Path,
    output_dir: str | Path,
    *,
    folds: int = 3,
    min_rows_per_split: int = 20,
    final_holdout_fraction: float = 0.10,
    min_final_holdout_rows: int = 100,
) -> tuple[Path, ...]:
    """建立模型選擇 folds，並將資料尾端封存為獨立 final holdout。"""
    source_dir = Path(environment_dir).resolve()
    metadata = _read_json(source_dir / "environment.json")
    if folds <= 0:
        return (source_dir,)
    if not 0.05 <= final_holdout_fraction <= 0.30:
        raise ValueError("final_holdout_fraction 必須介於 0.05 與 0.30")
    if min_final_holdout_rows <= 0:
        raise ValueError("min_final_holdout_rows 必須大於 0")
    target_root = Path(output_dir).resolve()
    target_root.mkdir(parents=True, exist_ok=False)
    final_holdout_root = target_root.parent / "final_holdout"
    final_holdout_root.mkdir(parents=True, exist_ok=False)
    universal = metadata.get("environment_kind") == "universal"
    fold_dirs = tuple(target_root / f"fold_{index:02d}" for index in range(1, folds + 1))
    for fold_dir in fold_dirs:
        fold_dir.mkdir(parents=True)
        source_ai = source_dir / "ai"
        if source_ai.is_dir():
            shutil.copytree(source_ai, fold_dir / "ai")

    if universal:
        markets = dict(metadata.get("markets", {}))
        if not markets:
            raise ValueError("通用環境缺少 markets 中繼資料")
        fold_markets = [dict() for _ in fold_dirs]
        fold_totals = [{"train": 0, "validation": 0, "test": 0} for _ in fold_dirs]
        holdout_markets: dict[str, object] = {}
        holdout_total = 0
        holdout_starts: list[pd.Timestamp] = []
        holdout_ends: list[pd.Timestamp] = []
        for market_name, raw_market in markets.items():
            market = dict(raw_market)
            files = {key: str(value) for key, value in dict(market["files"]).items()}
            frame = _load_market_frame(source_dir, files)
            original_train = int(dict(market["split_rows"])["train"])
            selection_frame, holdout_frame = _split_final_holdout(
                frame,
                fraction=final_holdout_fraction,
                minimum_rows=min_final_holdout_rows,
                original_train_rows=original_train,
                folds=folds,
                min_rows_per_split=min_rows_per_split,
            )
            boundaries = _fold_boundaries(
                len(selection_frame), original_train, folds, min_rows_per_split
            )
            for index, (fold_dir, boundary) in enumerate(zip(fold_dirs, boundaries, strict=True)):
                sizes = _write_fold_frames(selection_frame, fold_dir, files, boundary)
                payload = {**market, "split_rows": sizes}
                payload["start"] = str(selection_frame.iloc[0]["timestamp"])
                payload["end"] = str(selection_frame.iloc[boundary[2] - 1]["timestamp"])
                fold_markets[index][market_name] = payload
                for split_name, size in sizes.items():
                    fold_totals[index][split_name] += size
            holdout_path = final_holdout_root / files["test"]
            holdout_path.parent.mkdir(parents=True, exist_ok=True)
            holdout_frame.to_csv(holdout_path, index=False, encoding="utf-8")
            holdout_markets[market_name] = {
                **market,
                "files": {"test": files["test"]},
                "split_rows": {"final_holdout": len(holdout_frame)},
                "start": str(holdout_frame.iloc[0]["timestamp"]),
                "end": str(holdout_frame.iloc[-1]["timestamp"]),
            }
            holdout_total += len(holdout_frame)
            holdout_starts.append(pd.Timestamp(holdout_frame.iloc[0]["timestamp"]))
            holdout_ends.append(pd.Timestamp(holdout_frame.iloc[-1]["timestamp"]))
        for index, fold_dir in enumerate(fold_dirs):
            payload = {
                **metadata,
                "status": "environment_ready",
                "training_started": False,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "markets": fold_markets[index],
                "split_rows": fold_totals[index],
                "walk_forward": {
                    "parent_environment": str(source_dir),
                    "fold": index + 1,
                    "folds": folds,
                    "method": "expanding_train_non_overlapping_test",
                    "normalization": "沿用最初訓練段參數，沒有使用未來資料",
                    "final_holdout_excluded": True,
                },
            }
            _write_json(fold_dir / "environment.json", payload)
        _write_json(
            final_holdout_root / "environment.json",
            {
                **metadata,
                "status": "final_holdout_sealed",
                "training_started": False,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "markets": holdout_markets,
                "split_rows": {"final_holdout": holdout_total},
                "final_holdout": {
                    "sealed": True,
                    "fraction": final_holdout_fraction,
                    "minimum_rows_per_market": min_final_holdout_rows,
                    "selection_use": False,
                    "start": str(min(holdout_starts)),
                    "end": str(max(holdout_ends)),
                },
            },
        )
    else:
        files = {name: f"{name}.csv" for name in ("train", "validation", "test")}
        frame = _load_market_frame(source_dir, files)
        original_train = int(dict(metadata["split_rows"])["train"])
        selection_frame, holdout_frame = _split_final_holdout(
            frame,
            fraction=final_holdout_fraction,
            minimum_rows=min_final_holdout_rows,
            original_train_rows=original_train,
            folds=folds,
            min_rows_per_split=min_rows_per_split,
        )
        boundaries = _fold_boundaries(
            len(selection_frame), original_train, folds, min_rows_per_split
        )
        for index, (fold_dir, boundary) in enumerate(zip(fold_dirs, boundaries, strict=True)):
            sizes = _write_fold_frames(selection_frame, fold_dir, files, boundary)
            payload = {
                **metadata,
                "status": "environment_ready",
                "training_started": False,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "split_rows": sizes,
                "walk_forward": {
                    "parent_environment": str(source_dir),
                    "fold": index + 1,
                    "folds": folds,
                    "method": "expanding_train_non_overlapping_test",
                    "normalization": "沿用最初訓練段參數，沒有使用未來資料",
                    "final_holdout_excluded": True,
                },
            }
            _write_json(fold_dir / "environment.json", payload)
        holdout_frame.to_csv(
            final_holdout_root / "test.csv",
            index=False,
            encoding="utf-8",
        )
        _write_json(
            final_holdout_root / "environment.json",
            {
                **metadata,
                "status": "final_holdout_sealed",
                "training_started": False,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "split_rows": {"final_holdout": len(holdout_frame)},
                "final_holdout": {
                    "sealed": True,
                    "fraction": final_holdout_fraction,
                    "minimum_rows": min_final_holdout_rows,
                    "selection_use": False,
                    "start": str(holdout_frame.iloc[0]["timestamp"]),
                    "end": str(holdout_frame.iloc[-1]["timestamp"]),
                },
            },
        )
    build_artifact_manifest(
        final_holdout_root,
        files=[path for path in final_holdout_root.rglob("*") if path.is_file()],
    )
    return fold_dirs


def _load_final_holdout(
    holdout_dir: Path,
) -> tuple[pd.DataFrame | dict[str, pd.DataFrame], dict[str, Any]]:
    verify_artifact_manifest(holdout_dir, required=True)
    metadata = _read_json(holdout_dir / "environment.json")
    if metadata.get("status") != "final_holdout_sealed":
        raise ValueError("Final holdout 尚未封存")
    if metadata.get("environment_kind") == "universal":
        frames: dict[str, pd.DataFrame] = {}
        for market_name, raw_market in dict(metadata.get("markets", {})).items():
            relative = str(dict(raw_market).get("files", {}).get("test", ""))
            path = holdout_dir / relative
            if not relative or not path.is_file():
                raise FileNotFoundError(f"Final holdout 缺少 {market_name} 測試資料：{path}")
            frames[str(market_name)] = pd.read_csv(path)
        if not frames:
            raise ValueError("Final holdout 沒有市場資料")
        return frames, metadata
    path = holdout_dir / "test.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Final holdout 缺少測試資料：{path}")
    return pd.read_csv(path), metadata


def evaluate_research_final_holdout(
    candidate_run: str | Path,
    holdout_dir: str | Path,
    output_csv: str | Path,
    *,
    device: str = "cpu",
) -> dict[str, object]:
    """候選確定後只評估封存資料，輸出不可用於再次挑選模型。"""
    candidate = Path(candidate_run).resolve()
    holdout = Path(holdout_dir).resolve()
    output = Path(output_csv).resolve()
    if output.exists():
        raise FileExistsError("Final holdout 已評估，禁止覆寫後重新挑選")
    policy = load_rl_policy(candidate, device=device)
    frames, metadata = _load_final_holdout(holdout)
    rows, metrics, market_metrics = evaluate_rl_markets(
        policy.model,
        frames,
        policy.feature_columns,
        policy.env_config,
        deterministic=True,
        seed=int(dict(policy.training_metadata.get("training_config", {})).get("seed", 0)),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output, index=False, encoding="utf-8")
    expert_kind = str(
        dict(policy.training_metadata.get("environment_expert", {})).get("kind", "general")
    )
    quality = assess_rl_evaluation_quality(metrics, expert_kind=expert_kind)
    manifest = candidate / MANIFEST_NAME
    holdout_manifest = holdout / MANIFEST_NAME
    holdout_detail = dict(metadata.get("final_holdout", {}))
    return {
        "status": "complete",
        "eligible": quality.eligible,
        "reasons": list(quality.reasons),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_protocol": "deterministic_once_after_selection",
        "selection_use": False,
        "candidate_run": str(candidate),
        "candidate_manifest_sha256": sha256_file(manifest),
        "holdout_manifest_sha256": sha256_file(holdout_manifest),
        "holdout_start": holdout_detail.get("start"),
        "holdout_end": holdout_detail.get("end"),
        "evaluation_csv": str(output),
        "evaluation_csv_sha256": sha256_file(output),
        "metrics": metrics,
        "market_metrics": market_metrics,
        "quality": asdict(quality),
    }


def _finite(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def aggregate_research_runs(run_dirs: list[Path]) -> tuple[pd.DataFrame, dict[str, object]]:
    """彙整 seed 與 fold，並以中位數、最差值判斷是否值得進模擬倉。"""
    records: list[dict[str, object]] = []
    for run_dir in run_dirs:
        payload = _read_json(run_dir / "training.json")
        config = dict(payload.get("training_config", {}))
        test = dict(dict(payload.get("metrics", {})).get("test", {}))
        validation = dict(dict(payload.get("metrics", {})).get("validation", {}))
        environment_path = Path(str(payload.get("environment_dir", "")))
        try:
            environment = _read_json(environment_path / "environment.json")
        except (OSError, ValueError):
            environment = {}
        quality = assess_rl_training_quality(payload)
        records.append(
            {
                "seed": int(config.get("seed", 0)),
                "fold": int(dict(environment.get("walk_forward", {})).get("fold", 1)),
                "expert_kind": str(
                    dict(payload.get("environment_expert", {})).get("kind", "general")
                ),
                "run_dir": str(run_dir.resolve()),
                "quality_eligible": quality.eligible,
                "validation_return": _finite(validation.get("total_return")),
                "test_return": _finite(test.get("total_return")),
                "test_sharpe": _finite(test.get("sharpe_ratio")),
                "max_drawdown": _finite(test.get("max_drawdown"), 1.0),
                "cost_to_capital": _finite(test.get("fee_to_initial_capital"), 1.0),
                "turnover_to_capital": _finite(test.get("turnover_to_initial_capital")),
                "positive_market_ratio": _finite(test.get("positive_market_ratio")),
            }
        )
    frame = pd.DataFrame(records).sort_values(["seed", "fold"]).reset_index(drop=True)
    if frame.empty:
        raise ValueError("沒有可彙整的訓練紀錄")
    seed_summary = frame.groupby("seed", as_index=False).agg(
        median_test_return=("test_return", "median"),
        worst_test_return=("test_return", "min"),
        median_sharpe=("test_sharpe", "median"),
        worst_drawdown=("max_drawdown", "max"),
        median_cost=("cost_to_capital", "median"),
        eligible_ratio=("quality_eligible", "mean"),
    )
    seed_summary["robust_score"] = (
        seed_summary["median_test_return"]
        + seed_summary["median_sharpe"] * 0.02
        - seed_summary["worst_drawdown"]
        - seed_summary["median_cost"]
    )
    best_seed = int(seed_summary.sort_values("robust_score", ascending=False).iloc[0]["seed"])
    candidate = frame.loc[frame["seed"].eq(best_seed)].sort_values("fold").iloc[-1]
    seed_count = int(frame["seed"].nunique())
    fold_count = int(frame["fold"].nunique())
    positive_ratio = float((frame["test_return"] > 0).mean())
    median_return = float(frame["test_return"].median())
    median_sharpe = float(frame["test_sharpe"].median())
    worst_drawdown = float(frame["max_drawdown"].max())
    median_cost = float(frame["cost_to_capital"].median())
    quality_eligible_ratio = float(frame["quality_eligible"].mean())
    expert_kinds = set(frame["expert_kind"].astype(str))
    drawdown_limit = min(
        {"long_term": 0.12, "short_term": 0.06}.get(kind, 0.15) for kind in expert_kinds
    )
    reasons: list[str] = []
    if seed_count < 3:
        reasons.append("至少需要 3 個不同 seed")
    if fold_count < 2:
        reasons.append("至少需要 2 個 Walk-forward 測試窗")
    if median_return <= 0:
        reasons.append("所有 run 的測試報酬中位數必須大於 0")
    if positive_ratio < 0.60:
        reasons.append("正報酬 run 比例必須至少 60%")
    if median_sharpe <= 0:
        reasons.append("測試 Sharpe 中位數必須大於 0")
    if worst_drawdown > drawdown_limit:
        reasons.append(f"最差測試回撤不可超過 {drawdown_limit:.0%}")
    if median_cost > 0.02:
        reasons.append("交易成本占本金中位數不可超過 2%")
    if quality_eligible_ratio < 0.60:
        reasons.append("通過單次樣本外品質門檻的 run 比例必須至少 60%")
    if not bool(candidate["quality_eligible"]):
        reasons.append("最新資料訓練出的候選 run 未通過單次樣本外品質門檻")
    summary: dict[str, object] = {
        "eligible": not reasons,
        "reasons": reasons,
        "seed_count": seed_count,
        "fold_count": fold_count,
        "run_count": len(frame),
        "positive_run_ratio": positive_ratio,
        "median_test_return": median_return,
        "median_test_sharpe": median_sharpe,
        "worst_max_drawdown": worst_drawdown,
        "max_drawdown_limit": drawdown_limit,
        "median_cost_to_capital": median_cost,
        "quality_eligible_ratio": quality_eligible_ratio,
        "candidate_seed": best_seed,
        "candidate_run": str(candidate["run_dir"]),
        "seed_summary": seed_summary.to_dict(orient="records"),
    }
    return frame, summary


def run_rl_research_experiment(
    environment_dir: str | Path,
    training_config: RLTrainingConfig,
    research_config: RLResearchConfig | None = None,
    *,
    progress_callback: ResearchProgressCallback | None = None,
    backend_status: Any | None = None,
    trainer: Trainer = train_rl_agent,
    holdout_evaluator: HoldoutEvaluator = evaluate_research_final_holdout,
) -> RLResearchResult:
    """依 fold 與 seed 串行訓練，降低同一張 GPU 同時搶記憶體的風險。"""
    research = research_config or RLResearchConfig()
    source_dir = Path(environment_dir).resolve()
    experiment_dir = source_dir / "research" / _utc_stamp()
    experiment_dir.mkdir(parents=True, exist_ok=False)
    final_holdout_dir: Path | None = None
    if research.walk_forward_folds:
        environments = create_walk_forward_environments(
            source_dir,
            experiment_dir / "environments",
            folds=research.walk_forward_folds,
            min_rows_per_split=research.min_rows_per_split,
            final_holdout_fraction=research.final_holdout_fraction,
            min_final_holdout_rows=research.min_final_holdout_rows,
        )
        final_holdout_dir = experiment_dir / "final_holdout"
    else:
        environments = (source_dir,)
    total_runs = len(environments) * len(research.seeds)
    run_dirs: list[Path] = []
    run_number = 0
    for seed in research.seeds:
        for fold, fold_environment in enumerate(environments, start=1):
            current_index = run_number

            def report(payload: dict[str, object]) -> None:
                if progress_callback is None:
                    return
                local_progress = min(max(_finite(payload.get("progress")), 0.0), 1.0)
                progress_callback(
                    {
                        **payload,
                        "progress": (current_index + local_progress) / total_runs,
                        "experiment_run": current_index + 1,
                        "experiment_runs": total_runs,
                        "experiment_seed": seed,
                        "experiment_fold": fold,
                    }
                )

            result = trainer(
                fold_environment,
                replace(training_config, seed=seed),
                progress_callback=report,
                backend_status=backend_status,
            )
            run_dirs.append(result.paths.run_dir)
            run_number += 1
    runs, summary = aggregate_research_runs(run_dirs)
    candidate_path = Path(str(summary["candidate_run"]))
    selection_eligible = bool(summary["eligible"])
    selection_reasons = list(summary["reasons"])
    final_holdout: dict[str, object]
    if final_holdout_dir is not None and selection_eligible:
        if progress_callback is not None:
            progress_callback(
                {
                    "status": "final_holdout",
                    "progress": 1.0,
                    "experiment_run": total_runs,
                    "experiment_runs": total_runs,
                }
            )
        final_holdout = holdout_evaluator(
            candidate_path,
            final_holdout_dir,
            experiment_dir / "final_holdout_evaluation.csv",
            device=training_config.device,
        )
        final_holdout["holdout_dir_relative"] = "final_holdout"
        final_holdout["evaluation_csv_relative"] = "final_holdout_evaluation.csv"
    elif final_holdout_dir is not None:
        final_holdout = {
            "status": "sealed_not_evaluated",
            "eligible": False,
            "reasons": ["模型選擇門檻未通過，因此未開啟 final holdout"],
            "selection_use": False,
        }
    else:
        final_holdout = {
            "status": "not_created",
            "eligible": False,
            "reasons": ["未啟用 Walk-forward，因此沒有建立 final holdout"],
            "selection_use": False,
        }
    holdout_eligible = bool(final_holdout.get("eligible", False))
    final_reasons = [*selection_reasons, *list(final_holdout.get("reasons", []))]
    summary.update(
        {
            "research_protocol_version": 2,
            "selection_eligible": selection_eligible,
            "selection_reasons": selection_reasons,
            "final_holdout": final_holdout,
            "eligible": selection_eligible and holdout_eligible,
            "reasons": final_reasons,
        }
    )
    try:
        candidate_relative = candidate_path.resolve().relative_to(experiment_dir.resolve())
    except ValueError:
        candidate_relative = None
    summary["candidate_run_relative"] = (
        str(candidate_relative).replace("\\", "/") if candidate_relative else None
    )
    if candidate_relative and final_holdout.get("status") == "complete":
        final_holdout["candidate_run_relative"] = str(candidate_relative).replace("\\", "/")
    summary.update(
        {
            "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_environment": str(source_dir),
            "training_config": training_config.to_dict(),
            "research_config": {
                "seeds": list(research.seeds),
                "walk_forward_folds": research.walk_forward_folds,
                "min_rows_per_split": research.min_rows_per_split,
                "final_holdout_fraction": research.final_holdout_fraction,
                "min_final_holdout_rows": research.min_final_holdout_rows,
            },
        }
    )
    runs_csv = experiment_dir / "runs.csv"
    summary_json = experiment_dir / "summary.json"
    runs.to_csv(runs_csv, index=False, encoding="utf-8")
    _write_json(summary_json, summary)
    candidate = candidate_path if summary.get("candidate_run") else None
    return RLResearchResult(
        experiment_dir,
        summary_json,
        runs_csv,
        tuple(run_dirs),
        bool(summary["eligible"]),
        candidate,
    )


def list_rl_research_experiments(output_dir: str | Path) -> list[Path]:
    """列出已有 summary.json 的穩健性研究，最近完成者優先。"""
    root = Path(output_dir)
    if not root.exists():
        return []
    paths = [path.parent for path in root.rglob("summary.json")]
    return sorted(paths, key=lambda path: path.stat().st_mtime, reverse=True)
