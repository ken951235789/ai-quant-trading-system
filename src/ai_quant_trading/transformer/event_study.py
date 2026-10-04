"""事件模型的配對消融與 expanding walk-forward；不接交易或部署入口。"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
from time import monotonic

import numpy as np
import pandas as pd

from ai_quant_trading.operations.integrity import build_artifact_manifest, sha256_file
from ai_quant_trading.persistence import write_json_atomic
from ai_quant_trading.transformer.config import TemporalTransformerConfig, TransformerTrainingConfig
from ai_quant_trading.transformer.economics import block_confidence_interval
from ai_quant_trading.transformer.strategy_events import StrategyEventConfig, select_event_trades, validate_event_bars
from ai_quant_trading.transformer.training import train_temporal_transformer


VARIANTS = {
    "A_legacy_smooth": ("legacy_v1", "smooth_l1"),
    "B_legacy_mse": ("legacy_v1", "mse"),
    "C_context_smooth": ("context_v2", "smooth_l1"),
    "D_context_mse": ("context_v2", "mse"),
}


def study_windows(rows: int) -> list[dict[str, int | str]]:
    """前 60% 消融；60-90% 分三段向前驗證；最後 10% 本次不讀入模型。"""
    if rows < 20_000:
        raise ValueError("正式事件研究至少需要 20,000 棒，不能用小樣本冒充五年比較")
    development = int(rows * 0.60)
    windows = [{"stage": "ablation", "fold": 0, "source_end": development,
                "train_end": int(development * 0.60), "validation_end": int(development * 0.80)}]
    for fold, (start, end) in enumerate(((0.60, 0.70), (0.70, 0.80), (0.80, 0.90)), 1):
        cutoff = int(rows * start)
        windows.append({"stage": "walk_forward", "fold": fold, "source_end": int(rows * end),
                        "train_end": int(cutoff * 0.75), "validation_end": cutoff})
    return windows


def window_config(base: TransformerTrainingConfig, window: dict,
                  variant: str, seed: int) -> TransformerTrainingConfig:
    """用半棒餘裕避開浮點截斷，最後仍驗證實際整數切點。"""
    rows = window["source_end"]
    train = (window["train_end"] + 0.25) / rows
    validation = (window["validation_end"] + 0.25) / rows - train
    if int(rows * train) != window["train_end"] or int(rows * (train + validation)) != window["validation_end"]:
        raise ValueError("浮點資料切分未符合預先登記邊界")
    profile, loss = VARIANTS[variant]
    return replace(base, seed=seed, train_fraction=train, validation_fraction=validation,
                   return_loss_kind=loss, max_rows_per_source=None,
                   checkpoint_metric="event_prediction_skill_score",
                   strategy_event_config={**base.strategy_event_config, "feature_profile": profile})


def choose_challenger(records: list[dict]) -> str:
    """只看消融 Selection，不讀 Test 指標；全部為負也只是研究挑戰者。"""
    scores = {}
    for variant in list(VARIANTS)[1:]:
        values = [item["selection"]["metrics"]["event_prediction_skill_score"]
                  for item in records if item["stage"] == "ablation" and item["variant"] == variant]
        if not values or not np.isfinite(values).all():
            raise ValueError("挑戰組的 Selection 證據不足")
        scores[variant] = float(np.mean(values))
    return max(scores, key=lambda key: (scores[key], -list(VARIANTS).index(key)))


def factorial_effects(records: list[dict], split: str, metric: str) -> list[dict]:
    """配對種子下，分離特徵主效應、損失主效應與交互作用。"""
    result = []
    ablation = [item for item in records if item["stage"] == "ablation"]
    for seed in sorted({item["seed"] for item in ablation}):
        values = {item["variant"]: item[split]["metrics"][metric]
                  for item in ablation if item["seed"] == seed}
        if set(values) != set(VARIANTS) or not np.isfinite(list(values.values())).all():
            raise ValueError("消融必須有相同種子的四組完整結果")
        a, b, c, d = (values[name] for name in VARIANTS)
        result.append({"seed": seed, "feature_effect": ((c - a) + (d - b)) / 2,
                       "loss_effect": ((b - a) + (d - c)) / 2,
                       "interaction": d - c - b + a})
    return result


def _distribution(values: list[float]) -> dict:
    data = np.asarray(values, dtype=float)
    return {"n": len(data), "mean": float(data.mean()), "std": float(data.std(ddof=1)) if len(data) > 1 else 0.0,
            "min": float(data.min()), "max": float(data.max())}


def summarize_study(records: list[dict]) -> dict:
    metrics = ("event_return_skill", "event_probability_skill", "event_prediction_skill_score")
    comparisons = []
    for stage in ("ablation", "walk_forward"):
        for variant in VARIANTS:
            subset = [item for item in records if item["stage"] == stage and item["variant"] == variant]
            if not subset:
                continue
            for split in ("selection", "test"):
                comparisons.append({"stage": stage, "variant": variant, "split": split,
                    "metrics": {metric: _distribution([item[split]["metrics"][metric] for item in subset])
                                for metric in metrics},
                    "filtered_trades_per_run": [item[split]["policies"]["filtered"]["trades"] for item in subset],
                    "passed_runs": sum(item[split]["research_gate"]["passed"] for item in subset),
                    "runs": len(subset)})
    effects = {split: {metric: {effect: _distribution([row[effect] for row in factorial_effects(records, split, metric)])
                               for effect in ("feature_effect", "loss_effect", "interaction")}
                      for metric in metrics} for split in ("selection", "test")}
    return {"comparisons": comparisons, "factorial_effects": effects,
            "interpretation": "正 skill 效應表示比對照好；不代表淨收益為正。種子與時間窗不是互相獨立樣本。"}


def _pooled_forward(records: list[dict], root: Path, config: TransformerTrainingConfig) -> list[dict]:
    """各 seed 獨立串接不重疊 Test 交易，禁止把三個 seed 的同一筆行情當三筆樣本。"""
    output = []
    horizon = int(config.strategy_event_config["max_holding_bars"])
    for variant in VARIANTS:
        subset = [item for item in records if item["stage"] == "walk_forward" and item["variant"] == variant]
        for seed in sorted({item["seed"] for item in subset}):
            frames = []
            for item in sorted((r for r in subset if r["seed"] == seed), key=lambda r: r["fold"]):
                frame = pd.read_csv(root / item["run_dir"] / "test_predictions.csv")
                frame["walk_forward_fold"] = item["fold"]
                frames.append(frame)
            predictions = pd.concat(frames, ignore_index=True)
            if predictions.timestamp_ns.duplicated().any():
                raise ValueError("向前驗證 Test 出現重複決策時間")
            for filtered in (False, True):
                trades, _ = select_event_trades(predictions, horizon,
                    StrategyEventConfig(**config.strategy_event_config), config.economic_minimum_edge_bps,
                    filtered=filtered)
                values = trades[f"actual_return_{horizon}"].to_numpy(dtype=float)
                low, high = block_confidence_interval(values)
                name = f"wf_{variant}_seed{seed}_{'filtered' if filtered else 'baseline'}_trades.csv"
                trades.to_csv(root / name, index=False)
                output.append({"variant": variant, "seed": seed, "policy": "filtered" if filtered else "baseline",
                    "trades": len(values), "mean_net_return": float(values.mean()) if len(values) else None,
                    "std_net_return": float(values.std(ddof=1)) if len(values) > 1 else None,
                    "worst_trade": float(values.min()) if len(values) else None,
                    "ci_low": low if len(values) > 1 else None, "ci_high": high if len(values) > 1 else None,
                    "win_rate": float((values > 0).mean()) if len(values) else None,
                    "extra_6bps_net_mean": float(values.mean() - 0.0006) if len(values) else None,
                    "path": name})
    return output


def run_event_study(source: Path, config_path: Path, output: Path,
                    seeds: tuple[int, ...] = (42, 137, 2026)) -> Path:
    """先凍結計畫再執行；每個工作獨立成品，失敗不冒充完成或自動部署。"""
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or any(seed < 0 or seed >= 2**32 for seed in seeds):
        raise ValueError("至少需要兩個不重複、合法的配對種子")
    if output.exists():
        raise FileExistsError("研究目錄已存在；禁止覆蓋舊實驗，請指定新目錄")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    model = TemporalTransformerConfig(**payload["model"])
    base = TransformerTrainingConfig(**payload["training"])
    if base.trading_target_mode != "strategy_event" or base.calibration_fraction <= 0:
        raise ValueError("消融研究需要事件模式與獨立機率校準")
    # 此研究固定歷史標籤、交易規則與 CPU 預算，只變動已登記的兩個因素。
    base = replace(base, device="cpu", cpu_threads=2, num_workers=0, mixed_precision=False,
                   recency_half_life_days=None)
    source_hash = sha256_file(source)
    frame = pd.read_csv(source)
    validate_event_bars(frame)
    if sha256_file(source) != source_hash:
        raise ValueError("讀取過程來源變動，必須改用不可變快照")
    windows = study_windows(len(frame))
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    plan = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "hypotheses": {"features": "情境特徵提高樣本外收益與機率 skill", "loss": "MSE 改善平均淨收益估計"},
        "source": str(source.resolve()), "source_sha256": source_hash, "source_rows": len(frame),
        "config_sha256": sha256_file(config_path), "model": model.to_dict(), "training": base.to_dict(),
        "seeds": list(seeds), "variants": VARIANTS, "windows": windows,
        "window_times": [{**window, **{f"{name}_last_open_utc": str(frame.iloc[int(window[name]) - 1].timestamp)
                                     for name in ("source_end", "train_end", "validation_end")}}
                         for window in windows],
        "selection_rule": "最高平均 ablation Selection skill 的非 A 組；平手按 B/C/D 排序；不使用 Test；負分也不取得部署資格",
        "max_runs": 10 * len(seeds), "untouched_this_study_start": int(len(frame) * 0.90),
        "pristine_holdout": False, "live_eligible": False,
        "code_sha256": {str(path.relative_to(root)): sha256_file(path)
                        for path in sorted((root / "src/ai_quant_trading").rglob("*.py"))},
        "environment": {"python": platform.python_version(),
                        **{name: importlib.metadata.version(name) for name in ("torch", "numpy", "pandas")}},
        "limitations": ["同一歷史資料已用於過往研究，沒有全新 final holdout", "seed 配對仍有小樣本不確定性",
                        "新舊特徵使輸入層參數數量不同，主體寬深與訓練預算固定", "不是投資組合／SAC／稅後績效"]}
    write_json_atomic(output / "plan.json", plan)
    started = monotonic()
    records = []

    def execute(source_path: Path, window: dict, variant: str, seed: int) -> None:
        job = f"{window['stage']}_f{window['fold']}_{variant}_s{seed}"
        config = window_config(base, window, variant, seed)
        last_epoch = [-1]

        def progress(event: dict) -> None:
            status = {"status": "running", "job": job, "completed": len(records), "total": plan["max_runs"],
                      "elapsed_seconds": monotonic() - started, "training": event}
            write_json_atomic(output / "progress.json", status)
            epoch = int(event.get("epoch", 0))
            if epoch != last_epoch[0] or event.get("status") == "complete":
                print(f"[{len(records) + 1}/{plan['max_runs']}] {job} {event.get('status')} epoch={epoch}", flush=True)
                last_epoch[0] = epoch

        result = train_temporal_transformer([source_path], model, config, output / "runs", run_name=job,
                                            progress_callback=progress)
        summary = json.loads(result.summary_json.read_text(encoding="utf-8"))
        predictions = pd.read_csv(result.run_dir / "test_predictions.csv")
        if (predictions.endpoint < window["validation_end"] + config.embargo_bars).any():
            raise ValueError("Test 預測早於預先登記的 OOS 邊界")
        if (predictions.event_exit_endpoint >= window["source_end"]).any():
            raise ValueError("Test 成交標籤超出本 fold 結束")
        row = {"stage": window["stage"], "fold": window["fold"], "variant": variant, "seed": seed,
               "run_dir": result.run_dir.relative_to(output).as_posix(),
               "source_slice_sha256": sha256_file(source_path), "duration_seconds": summary["duration_seconds"],
               "best_epoch": summary["best_epoch"], "trainable_parameters": summary["trainable_parameters"],
               "features": len(summary["feature_columns"]), "sample_counts": summary["sample_counts"],
               "calibration_protocol": summary["calibration_protocol"]}
        for split, file_split in (("selection", "validation"), ("test", "test")):
            row[split] = json.loads((result.run_dir / f"{file_split}_event_diagnostics.json").read_text(encoding="utf-8"))
        records.append(row)
        write_json_atomic(output / "records.json", records)

    try:
        with TemporaryDirectory(prefix="aiquant_event_study_") as temporary:
            sliced = Path(temporary) / "source.csv"
            first = windows[0]
            frame.iloc[:first["source_end"]].to_csv(sliced, index=False)
            for variant in VARIANTS:
                for seed in seeds:
                    execute(sliced, first, variant, seed)
            challenger = choose_challenger(records)
            write_json_atomic(output / "selection.json", {"challenger": challenger,
                "selection_only": True, "ablation_record_sha256": sha256_file(output / "records.json"),
                "research_only": True, "live_eligible": False})
            print(f"向前驗證挑戰组：{challenger}（僅依 Selection；不是部署升級）", flush=True)
            for window in windows[1:]:
                frame.iloc[:window["source_end"]].to_csv(sliced, index=False)
                for variant in ("A_legacy_smooth", challenger):
                    for seed in seeds:
                        execute(sliced, window, variant, seed)
        if sha256_file(source) != source_hash:
            raise ValueError("研究期間來源變動，結果不可認定有效")
        summary = {**summarize_study(records), "status": "complete", "technical_success": True,
            "live_eligible": False, "challenger": challenger, "completed_runs": len(records),
            "elapsed_seconds": monotonic() - started, "source_unchanged": True,
            "walk_forward_pooled": _pooled_forward(records, output, base)}
        write_json_atomic(output / "study.json", summary)
        write_json_atomic(output / "progress.json", {"status": "complete", "completed": len(records),
                          "total": plan["max_runs"], "elapsed_seconds": monotonic() - started})
        build_artifact_manifest(output)
    except BaseException as error:
        write_json_atomic(output / "progress.json", {"status": "failed", "completed": len(records),
                          "error_type": type(error).__name__, "elapsed_seconds": monotonic() - started})
        raise
    return output / "study.json"
