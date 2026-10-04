"""依契約先驗證 Transformer，再開放固定訊號的 SAC 資金配置研究。"""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

import numpy as np
import pandas as pd

from ai_quant_trading.operations.integrity import build_artifact_manifest, sha256_file
from ai_quant_trading.persistence import write_json_atomic
from ai_quant_trading.research.sizing import SizingConfig, run_sizing_comparison
from ai_quant_trading.transformer.config import TemporalTransformerConfig, TransformerTrainingConfig
from ai_quant_trading.transformer.event_diagnostics import diagnose_events, training_reference
from ai_quant_trading.transformer.event_study import window_config
from ai_quant_trading.transformer.strategy_events import StrategyEventConfig, prepare_event_frame, select_event_trades, validate_event_bars
from ai_quant_trading.transformer.training import train_temporal_transformer


def digest_contract(contract: dict) -> str:
    return hashlib.sha256(json.dumps(contract, sort_keys=True, ensure_ascii=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def build_contract(model: TemporalTransformerConfig, training: TransformerTrainingConfig,
                   sizing: SizingConfig) -> dict:
    if training.trading_target_mode != "strategy_event" or model.architecture_version != 3:
        raise ValueError("固定訊號研究只接受明確的 V3 事件模型")
    event = StrategyEventConfig(**training.strategy_event_config)
    if model.return_horizons != (event.max_holding_bars,) or training.calibration_fraction <= 0:
        raise ValueError("horizon、事件出場或校準契約不一致")
    if training.direction_threshold_bps != training.economic_minimum_edge_bps:
        raise ValueError("成功機率標籤與淨收益接受門檻必須相同")
    return {"schema_version": 1, "name": "btc15m_event_entry_sizing_v1", "research_only": True,
        "exchange": "binance_usdm", "symbol": "BTCUSDT", "interval": "15m", "timezone": "UTC",
        "transformer_architecture": 3, "target_mode": "strategy_event",
        "horizon_bars": event.max_holding_bars,
        "return_units": "fractional_net_return_on_entry_notional",
        "expected_return_is_gross": False,
        "probability_event": "net_return_exceeds_economic_minimum_edge_bps",
        "candidate_rule": "closed_1h_trend_confirmed_15m_breakout",
        "entry": "next_bar_open", "exit": "shared_strategy_event_replay_stop_first",
        "event": asdict(event), "fee_bps_per_side": training.fee_bps_per_side,
        "slippage_bps_per_side": training.slippage_bps_per_side,
        "minimum_edge_bps": training.economic_minimum_edge_bps,
        "funding": "conservative_reserve_not_historical_settlement", "tax_verified": False,
        "sac_action": "entry_allocation_only_0_to_1", "sizing": asdict(sizing),
        "same_schedule_when_skipped": True, "free_exit_enabled": False,
        "compatible_with_existing_live_models": False, "live_eligible": False}


def chronological_windows(rows: int, folds: int = 3) -> list[dict]:
    if rows < 20_000 or folds < 2:
        raise ValueError("長區間研究需要至少 20,000 棒與兩個時間折")
    edges = np.linspace(int(rows * .40), int(rows * .90), folds + 1, dtype=int)
    return [{"stage": "crossfit", "fold": i + 1, "source_end": int(end),
             "train_end": int(start * .75), "validation_end": int(start)}
            for i, (start, end) in enumerate(zip(edges[:-1], edges[1:]))]


def split_oos(frame: pd.DataFrame, start: int, end: int, embargo: int) -> dict[str, pd.DataFrame]:
    """按預登記 K 棒邊界切分，不因可交易事件稀疏而移動切點。"""
    a, b = start + int((end - start) * .60), start + int((end - start) * .80)
    windows = {"train": (start, a), "validation": (a + embargo, b), "test": (b + embargo, end)}
    return {name: frame.loc[frame.endpoint.ge(low) & frame.endpoint.lt(high)
                            & frame.event_exit_endpoint.lt(high)].copy().reset_index(drop=True)
            for name, (low, high) in windows.items()}


def assess_transformer(splits: dict[str, pd.DataFrame], horizon: int, event: StrategyEventConfig,
                       training: TransformerTrainingConfig, minimum_trades: int) -> dict:
    """只讀 OOS Train 與 Validation，SAC Test 不參與 gate 或排名。"""
    train, validation = splits["train"], splits["validation"]
    if len(train) < 2 or len(validation) < 2:
        return {"passed": False, "reason": "insufficient_oos_samples"}
    values = train[f"actual_return_{horizon}"]
    reference = training_reference(float(values.mean()), float(train[f"actual_tradeability_{horizon}"].sum()), len(train))
    diagnosis = diagnose_events(validation, horizon, event, training.economic_minimum_edge_bps,
        reference, slippage_bps_per_side=training.slippage_bps_per_side, minimum_trades=minimum_trades)
    diagnosis["reference_source"] = "earlier_crossfit_oos_train_only"
    # 排序門檻僅由更早的 OOS Train 決定，不能在 Validation/Test 重切分位挑好交易。
    threshold = float(train[f"predicted_return_{horizon}"].quantile(.80))
    high = validation.loc[validation[f"predicted_return_{horizon}"].ge(threshold), f"actual_return_{horizon}"]
    low = validation.loc[validation[f"predicted_return_{horizon}"].lt(threshold), f"actual_return_{horizon}"]
    rank_check = len(high) >= 10 and len(low) >= 10 and high.mean() > low.mean()
    checks = dict(diagnosis["research_gate"]["checks"])
    checks["higher_scores_have_higher_realized_net_return"] = bool(rank_check)
    filtered = diagnosis["policies"]["filtered"]
    checks["positive_after_additional_6bps"] = filtered.get("extra_6bps_net_mean", -1) > 0
    return {"passed": all(checks.values()), "checks": checks, "diagnostics": diagnosis,
            "score_ranking": {"threshold_fit_on": "oos_train", "threshold": threshold,
                "high_n": len(high), "low_n": len(low),
                "high_mean": float(high.mean()) if len(high) else None,
                "low_mean": float(low.mean()) if len(low) else None}}


def filtered_splits(splits: dict[str, pd.DataFrame], horizon: int, event: StrategyEventConfig,
                    minimum_edge_bps: float) -> dict[str, pd.DataFrame]:
    return {name: select_event_trades(frame, horizon, event, minimum_edge_bps, filtered=True)[0]
            for name, frame in splits.items()}


def require_sizing_gate(gates: dict[str, dict], selected_seed: int) -> None:
    # 保守要求五個預登記 seed 都通過，不能只拿偶然最高分那個作為穩定性證明。
    if len(gates) < 5 or not all(value.get("passed") is True for value in gates.values()):
        raise ValueError("Transformer 多種子樣本外品質未通過，禁止正式 SAC sizing 訓練")
    if str(selected_seed) not in gates:
        raise ValueError("選中的 Transformer 不在預登記結果內")


def run_staged_training(source: Path, config_path: Path, output: Path) -> dict:
    source, config_path, output = source.resolve(), config_path.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("研究目錄已存在，禁止覆寫")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    transformer_path = (config_path.parent / payload["transformer_config"]).resolve()
    template = json.loads(transformer_path.read_text(encoding="utf-8"))
    model = TemporalTransformerConfig(**template["model"])
    base = replace(TransformerTrainingConfig(**template["training"]), device="cpu", cpu_threads=2,
                   num_workers=0, mixed_precision=False, recency_half_life_days=None)
    sizing = SizingConfig(**{**payload["sizing"], "seeds": tuple(payload["seeds"])})
    seeds = tuple(payload["seeds"])
    if len(seeds) < 5 or len(set(seeds)) != len(seeds) or any(not isinstance(s, int) or not 0 <= s < 2**32 for s in seeds):
        raise ValueError("正式分階段研究需要至少五個合法且不重複的 seeds")
    contract = build_contract(model, base, sizing)
    contract_hash = digest_contract(contract)
    source_hash = sha256_file(source)
    bars = pd.read_csv(source)
    validate_event_bars(bars)
    if sha256_file(source) != source_hash:
        raise ValueError("來源讀取期間已變更")
    windows = chronological_windows(len(bars), int(payload["folds"]))
    event = StrategyEventConfig(**base.strategy_event_config)
    # 只準備本研究允許的前 90%；最後 10% 不建特徵、不計績效。
    prepared = prepare_event_frame(bars.iloc[:windows[-1]["source_end"]], event)
    output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[3]
    code_paths = sorted((root / "src/ai_quant_trading/transformer").glob("*.py")) + [Path(__file__), Path(__file__).with_name("sizing.py")]
    plan = {"created_at": datetime.now(timezone.utc).isoformat(), "contract": contract,
        "contract_sha256": contract_hash, "source": str(source.resolve()), "source_sha256": source_hash,
        "source_rows": len(bars), "source_start": str(bars.timestamp.iloc[0]), "source_end": str(bars.timestamp.iloc[-1]),
        "model": model.to_dict(), "training": base.to_dict(), "seeds": seeds, "windows": windows,
        "configuration_sha256": sha256_file(config_path), "transformer_configuration_sha256": sha256_file(transformer_path),
        "code_sha256": {str(p.relative_to(root)): sha256_file(p) for p in code_paths},
        "gate": {"minimum_trades": int(payload["minimum_trades"]), "all_five_seeds": True,
                 "positive_net_ci_lower_bound": True, "positive_return_and_probability_skill": True,
                 "ranking_threshold_fit_on": "earlier_oos_train", "stress_extra_bps": 6},
        "sac_split": "OOS 前60% Train、中20% Validation、後20% Test；依K棒邊界、purge退出與embargo",
        "reserved_tail_start": windows[-1]["source_end"], "pristine_holdout": False,
        "limitations": ["這批歷史曾用於其他研究，尾端不是全新的holdout", "固定事件sizing只計結算權益，不代表棒內回撤或可部署",
                        "funding是準備金，費率為假設，稅務未驗證", "模型不可載入原有SAC/Paper/Live入口"],
        "live_eligible": False}
    write_json_atomic(output / "plan.json", plan)
    records, predictions = [], {seed: [] for seed in seeds}
    started = monotonic()
    total = len(windows) * len(seeds)
    try:
        with TemporaryDirectory(prefix="aiquant_staged_") as temp:
            sliced = Path(temp) / "source.csv"
            for window in windows:
                bars.iloc[:window["source_end"]].to_csv(sliced, index=False)
                for seed in seeds:
                    config = window_config(base, window, "D_context_mse", seed)
                    job = f"fold{window['fold']}_seed{seed}"
                    last_epoch = [-1]

                    def progress(info: dict) -> None:
                        write_json_atomic(output / "progress.json", {"status": "training_transformer", "job": job,
                            "completed": len(records), "total": total, "elapsed_seconds": monotonic() - started, "training": info})
                        epoch = int(info.get("epoch", 0))
                        if epoch != last_epoch[0] or info.get("status") == "complete":
                            print(f"[{len(records)+1}/{total}] {job} {info.get('status')} epoch={epoch}", flush=True)
                            last_epoch[0] = epoch

                    result = train_temporal_transformer([sliced], model, config, output / "transformer", run_name=job, progress_callback=progress)
                    frame = pd.read_csv(result.run_dir / "test_predictions.csv")
                    if frame.empty or frame.endpoint.lt(window["validation_end"] + config.embargo_bars).any() or frame.event_exit_endpoint.ge(window["source_end"]).any():
                        raise ValueError("cross-fit 預測不符合事前窗口")
                    causal = prepared.iloc[frame.endpoint.to_numpy(dtype=int)]
                    frame["causal_atr_fraction"] = (causal.event_atr / causal.close).to_numpy()
                    frame["crossfit_fold"] = window["fold"]
                    frame["fit_end_endpoint"] = window["validation_end"] - 1
                    frame["checkpoint_sha256"] = sha256_file(result.model_path)
                    frame["contract_sha256"] = contract_hash
                    predictions[seed].append(frame)
                    record = {"fold": window["fold"], "seed": seed, "checkpoint": str(result.model_path.relative_to(output)),
                              "checkpoint_sha256": sha256_file(result.model_path), "oos_events": len(frame),
                              "source_slice_sha256": sha256_file(sliced)}
                    records.append(record)
                    write_json_atomic(output / "records.json", records)
        gates, partitions = {}, {}
        for seed, parts in predictions.items():
            merged = pd.concat(parts, ignore_index=True).sort_values("endpoint").reset_index(drop=True)
            if merged.endpoint.duplicated().any() or not merged.endpoint.gt(merged.fit_end_endpoint).all():
                raise ValueError("cross-fit 資料重疊或不是樣本外")
            merged.to_csv(output / f"oos_seed{seed}.csv", index=False)
            partitions[seed] = split_oos(merged, windows[0]["validation_end"], windows[-1]["source_end"], event.max_holding_bars)
            gates[str(seed)] = assess_transformer(partitions[seed], event.max_holding_bars, event, base, int(payload["minimum_trades"]))
        write_json_atomic(output / "transformer_gate.json", gates)
        passed = all(g["passed"] for g in gates.values())
        summary = {"status": "complete", "contract_sha256": contract_hash, "transformer_runs": len(records),
            "oos_events_by_seed": {str(seed): sum(len(p) for p in parts) for seed, parts in predictions.items()},
            "oos_start": str(bars.timestamp.iloc[windows[0]["validation_end"]]),
            "oos_end": str(bars.timestamp.iloc[windows[-1]["source_end"] - 1]),
            "transformer_gate_passed": passed, "sac_stage": "blocked_by_transformer_gate", "stage4": "locked",
            "sac_test_evaluated": False, "tail_evaluated": False, "live_eligible": False}
        if passed:
            selected_seed = max(seeds, key=lambda seed: gates[str(seed)]["diagnostics"]["policies"]["filtered"]["net_expectancy_ci_low"])
            require_sizing_gate(gates, selected_seed)
            selected = filtered_splits(partitions[selected_seed], event.max_holding_bars, event, base.economic_minimum_edge_bps)
            if min(map(len, selected.values())) < int(payload["minimum_trades"]):
                summary["sac_stage"] = "blocked_by_event_sample_count"
            else:
                write_json_atomic(output / "sizing_selection.json", {"seed": selected_seed, "selection_source": "oos_validation_only", "gates_sha256": sha256_file(output / "transformer_gate.json")})
                results = run_sizing_comparison(selected, event.max_holding_bars, sizing, output / "sizing", contract_hash)
                summary.update(sac_stage="complete", sac_test_evaluated=True, stage4=results["stage4"])
        if sha256_file(source) != source_hash or sha256_file(config_path) != plan["configuration_sha256"] or sha256_file(transformer_path) != plan["transformer_configuration_sha256"]:
            raise ValueError("研究期間資料或設定有變更")
        for name, expected in plan["code_sha256"].items():
            if sha256_file(root / name) != expected:
                raise ValueError(f"研究程式已變更：{name}")
        summary["elapsed_seconds"] = monotonic() - started
        write_json_atomic(output / "result.json", summary)
        write_json_atomic(output / "progress.json", {**summary, "completed": len(records), "total": total})
        build_artifact_manifest(output)
        return summary
    except BaseException as error:
        write_json_atomic(output / "progress.json", {"status": "failed", "completed": len(records), "error_type": type(error).__name__, "elapsed_seconds": monotonic() - started})
        raise
