"""候選契約研究：預設只列計畫；煙霧與完整研究均須明確指定。"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from time import monotonic

# 在載入 NumPy / Torch 前限制原生數值執行緒，避免本機 CPU 爆滿。
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ai_quant_trading.operations.integrity import build_artifact_manifest  # noqa: E402
from ai_quant_trading.research.candidate_contract import (  # noqa: E402
    CandidateContract, FAMILIES, candidate_outcomes, prepare_candidate_frame,
)
from ai_quant_trading.research.candidate_evaluation import (  # noqa: E402
    compare_execution_policies, fit_statistical_filter, score_diagnostics,
)
from ai_quant_trading.research.candidate_features import (  # noqa: E402
    FEATURE_SETS, enrich_flow_source, validate_flow,
)
from ai_quant_trading.research.event_attribution import attribute_events  # noqa: E402
from ai_quant_trading.transformer.config import (  # noqa: E402
    TemporalTransformerConfig, TransformerTrainingConfig,
)
from ai_quant_trading.transformer.dataset import prepare_transformer_datasets  # noqa: E402
from ai_quant_trading.transformer.event_study import VARIANTS, study_windows  # noqa: E402
from ai_quant_trading.transformer.training import (  # noqa: E402
    _split_calibration, train_temporal_transformer,
)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def manifest_all(root):
    # 共用工具預設只追蹤模型與 JSON；研究必須另外包含預測、成交與快照 CSV。
    return build_artifact_manifest(root, [p for p in Path(root).rglob("*") if p.is_file()])


FEATURE_VARIANTS = {f"F_{name}": name for name in FEATURE_SETS}


def research_plan(seeds, feature_study=False):
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("種子必須為不重複的非負整數")
    value = {"version": "candidate_training_research_v1", "seeds": seeds,
        "families": list(FAMILIES), "variants": list(VARIANTS), "walk_forward_folds": 3,
        "feature_group_ablation": "D_context_mse_without_cost",
        "formal_runs": len(seeds) * 3 * len(FAMILIES) * (len(VARIANTS) + 1),
        "mechanisms": {"original": "原突破基準", "trend_pullback": "順勢回調",
                       "vwap_reversion": "低趨勢區間均值回歸"},
        "selection": "不依舊 Forward 排名挑選；共同 stop=2ATR,R=2,hold=32,cooldown=4,strict=false",
        "costs": "fee=5bp/side,slip=2bp/side,spread=1bp,funding_reserve=1bp/8h; stress slip+3bp/side",
        "success_probability": .55, "minimum_net_edge_bps": 2,
        "research_only": True, "live_eligible": False, "p2_sac": "BLOCKED_pending_evidence",
        "history_status": "repeated_research_not_pristine_holdout",
        "future_holdout": {"start": "after_all_project_observed_history_not_this_fold_end",
            "state": "BLOCKED_requires_observation_ledger_then_prospective_collection",
            "release": "需另行凍結模型、門檻、期間與交易樣本要求後一次評估；不得邊看邊調"},
        "stopping": ["資料/契約/因果/成本對帳失敗立即停止",
            "樣本不足保留失敗紀錄，不放寬規則或換 seed",
            "不勝過 Train 常數或統計基準則不進 SAC",
            "無跨期成本後優勢、壓力失敗或過度集中則不部署"],
        "account_scope": "名目交易報酬；沒有完整浮虧、保證金、清算或稅後資格",
        "statistics": "種子不增加獨立行情樣本；候選重疊權重只用 Train；block CI 只作描述"}
    if feature_study:
        value.update(version="candidate_feature_study_v1", variants=list(FEATURE_VARIANTS),
            feature_group_ablation="existing/compact/flow/setup/combined; all MSE",
            formal_runs=len(seeds) * 3 * len(FAMILIES) * len(FEATURE_VARIANTS),
            hypotheses={"compact": "減少重複價格表達，不預設改善",
                "flow": "已收盤主動成交方向與活動可增加條件資訊",
                "setup": "已知價格空間與訊號時效可區分候選品質"},
            unchanged="同來源/候選/標籤/成本/門檻/切分/小型V3；只變輸入欄位",
            feature_version="candidate_features_v1")
    return value


def configurations(payload, family, variant, seed, window, smoke=False, reward_r=2.0):
    paired_loss = variant.endswith("_smooth_l1") and variant.removesuffix("_smooth_l1") in FEATURE_VARIANTS
    base_variant = variant.removesuffix("_smooth_l1") if paired_loss else variant
    feature_ablation = "without_cost" if variant == "D_context_mse_without_cost" else "none"
    feature_set = FEATURE_VARIANTS.get(base_variant, "existing")
    profile, loss = ("context_v2", "smooth_l1" if paired_loss else "mse") if base_variant in FEATURE_VARIANTS else VARIANTS[
        "D_context_mse" if feature_ablation != "none" else variant]
    contract = CandidateContract(family=family, feature_profile=profile, feature_ablation=feature_ablation,
        feature_set=feature_set, reward_r=reward_r,
        regime_exit={"original": "loss", "trend_pullback": "opposite", "vwap_reversion": "disabled"}[family])
    model = TemporalTransformerConfig(**payload["model"])
    training = TransformerTrainingConfig(**payload["training"])
    rows = window["source_end"]
    train = (window["train_end"] + .25) / rows
    validation = (window["validation_end"] + .25) / rows - train
    training = replace(training, strategy_event_config=contract.event_config().to_dict(),
        candidate_contract=contract.to_dict(), return_loss_kind=loss, seed=seed,
        train_fraction=train, validation_fraction=validation, cpu_threads=2, num_workers=0,
        max_rows_per_source=None, epochs=1 if smoke else training.epochs,
        device="cpu" if smoke else training.device, mixed_precision=False if smoke else training.mixed_precision)
    if int(rows * train) != window["train_end"] or int(rows * (train + validation)) != window["validation_end"]:
        raise ValueError("資料切點浮點誤差")
    if (model.return_horizons != (contract.holding_bars,) or model.d_model != 64 or model.n_layers != 2
            or model.sequence_length != 96 or model.quantile_levels != (.1, .5, .9)):
        raise ValueError("此研究固定小型 V3：d_model=64,n_layers=2,sequence=96,horizon=32,quantiles=.1/.5/.9")
    return contract, model, training


def evaluate_run(result, source, contract, model, training):
    data = prepare_transformer_datasets([source], model, training)
    calibration, selection, protocol = _split_calibration(data, training)
    series = data.train.series[0]
    boundaries = [(data.train, int(len(series.features) * training.train_fraction)),
        (calibration, selection.references[0][1]),
        (selection, int(len(series.features) * (training.train_fraction + training.validation_fraction))),
        (data.test, len(series.features))]
    for subset, boundary in boundaries:
        ends = series.event_metadata["label_end_endpoint"][[p for _, p in subset.references]]
        if (ends >= boundary).any():
            raise ValueError("候選標籤跨越時間隔離邊界")
    frame = prepare_candidate_frame(pd.read_csv(source), contract)
    outcomes = candidate_outcomes(frame, contract)
    attributed = attribute_events(frame, outcomes, contract, sha(source))
    attributed.to_csv(result.run_dir / "candidate_attribution.csv", index=False)
    train_points = [point for _, point in data.train.references]
    fitted = fit_statistical_filter(attributed, train_points, training.direction_threshold_bps)
    write_json(result.run_dir / "statistical_filter.json", fitted)
    selection_predictions = pd.read_csv(result.run_dir / "validation_predictions.csv")
    h = contract.holding_bars
    edges = np.unique(np.quantile(selection_predictions[f"predicted_return_{h}"], [.2, .4, .6, .8])).tolist()
    comparisons = {}
    for split in ("validation", "test"):
        predictions = pd.read_csv(result.run_dir / f"{split}_predictions.csv")
        aligned = outcomes.set_index("endpoint").loc[predictions.endpoint]
        np.testing.assert_allclose(predictions[f"actual_return_{h}"], aligned.net_return, rtol=1e-5, atol=1e-8)
        comparisons[split] = {
            "policies": compare_execution_policies(frame, predictions, fitted, contract, sha(source),
                training.economic_minimum_edge_bps, result.run_dir / f"{split}_policies", gate_ablation=True),
            "prediction_diagnostics": score_diagnostics(predictions, h, edges)}
    value = {"contract": data.diagnostics["candidate_contract"], "calibration_protocol": protocol,
        "samples": {**data.sample_counts, "calibration": len(calibration), "selection": len(selection)},
        "comparisons": comparisons, "engineering_checks": {"labels_equal_replay": True,
            "label_boundaries": True, "cost_reconciliation": True},
        "research_quality": "INSUFFICIENT_EVIDENCE", "live_eligible": False,
        "limits": ["OHLC 退出棒路徑不明，MFE/MAE 是界線不是精確逐筆值",
            "候選歸因表可重疊，policies 才是不重疊成交；三種策略不是組合帳戶",
            "分數分組邊界來自 Selection；未依 Test 修改任何門檻",
            "本次沒有估計多重搜尋校正後實盤資格，亦不解鎖 SAC 或 Champion"]}
    write_json(result.run_dir / "candidate_evaluation.json", value)
    manifest_all(result.run_dir)
    return value


def paired_summary(records):
    """配對比較不把同一行情的種子視為獨立交易，也不挑 Test 贏家。"""
    grouped = {}
    for record in records:
        key = (record["family"], record["fold"], record["seed"])
        grouped.setdefault(key, {})[record["variant"]] = record
    effects, feature_effects, loss_effects = [], [], []
    for key, variants in grouped.items():
        for name in FEATURE_VARIANTS:
            if name in variants and name + "_smooth_l1" in variants:
                for metric in ("event_return_skill", "event_probability_skill", "event_prediction_skill_score"):
                    loss_effects.append({"family": key[0], "fold": key[1], "seed": key[2],
                        "feature_variant": name, "metric": metric,
                        "mse_minus_smooth_l1": variants[name]["test_prediction_metrics"][metric]
                            - variants[name + "_smooth_l1"]["test_prediction_metrics"][metric]})
        if set(FEATURE_VARIANTS).issubset(variants):
            for metric in ("event_return_skill", "event_probability_skill", "event_prediction_skill_score"):
                old, compact, flow, setup, combined = [variants[v]["test_prediction_metrics"][metric]
                                                      for v in FEATURE_VARIANTS]
                feature_effects.append({"family": key[0], "fold": key[1], "seed": key[2], "metric": metric,
                    "compact_minus_existing": compact - old, "flow_minus_compact": flow - compact,
                    "setup_minus_compact": setup - compact, "combined_minus_compact": combined - compact,
                    "flow_setup_interaction": combined - flow - setup + compact})
        if not set(VARIANTS).issubset(variants):
            continue
        for metric in ("event_return_skill", "event_probability_skill", "event_prediction_skill_score"):
            a, b, c, d = [variants[v]["test_prediction_metrics"][metric] for v in VARIANTS]
            effects.append({"family": key[0], "fold": key[1], "seed": key[2], "metric": metric,
                "context_effect": ((c - a) + (d - b)) / 2,
                "mse_effect": ((b - a) + (d - c)) / 2, "interaction": d - c - b + a})
    return {"factorial_effects": effects, "complete_pairs": len(effects) // 3,
            "paired_loss_effects": loss_effects,
            "feature_effects": feature_effects, "complete_feature_pairs": len(feature_effects) // 3,
            "selection": "no_test_winner_or_deployment", "scope": "seed/fold 非獨立；效果是預測 skill 差，不是獲利證據"}


def assess_sac_readiness(records, smoke):
    """只提供是否值得繼續的證據清單，不用單次正收益取代既有 OOS 資金配置閘門。"""
    rows = []
    for r in records:
        base = r["test_policies"]["transformer_base"]
        stress = r["test_policies"]["transformer_stress"]
        rows.append({"name": r["name"], "trades": base["trades"], "mean": base["mean"],
                     "stress_mean": stress["mean"], "skill": r["test_prediction_metrics"]["event_return_skill"]})
    reasons = ["尚未產生供 SAC Train/Validation/Test 使用且符合本候選契約的跨擬合 OOS 訊號",
               "完整帳戶浮虧/清算及新的封存行情未驗證；不可藉槓桿放大未證實優勢"]
    if smoke:
        reasons.insert(0, "本次只有單輪工程煙霧，沒有充分訓練或跨期證據")
    if not rows or all(r["trades"] == 0 for r in rows):
        reasons.append("所有本次 AI 篩選零成交，沒有可用事件供 SAC 公平比較")
    if rows and any(r["stress_mean"] is None or r["stress_mean"] <= 0 for r in rows):
        reasons.append("成本壓力下仍有負收益或無交易證據")
    return {"status": "BLOCKED", "recommendation": "先驗證 Transformer，不啟動 SAC 成效訓練",
        "smoke_only": smoke, "reasons": reasons, "observations": rows,
        "next_comparison": "通過證據與契約審查後，固定相同 OOS 事件比較固定/波動率/SAC 入場部位",
        "action_scope": "entry_allocation_only; no_leverage; no_free_exit",
        "not_a_replacement_for": "staged_training.require_sizing_gate",
        "live_eligible": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/transformer_strategy_event_v2.example.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 137, 2026])
    parser.add_argument("--smoke-rows", type=int, default=48000)
    parser.add_argument("--feature-study", action="store_true", help="五組新特徵研究，預設 MSE，可另加 paired-loss")
    parser.add_argument("--paired-loss", action="store_true", help="同一特徵、種子與時段額外配對 Smooth L1；不擴大模型")
    parser.add_argument("--reward-r", type=float, choices=[1.0, 2.0, 3.0], default=2.0,
                        help="獨立退出研究的固定停利 R；重新產生契約和標籤，不沿用舊分數")
    parser.add_argument("--flow-source", type=Path, help="本機完整成交欄位 CSV；逐根比對 OHLCV 後補入")
    parser.add_argument("--variants", nargs="+", choices=list(FEATURE_VARIANTS), help="事先限定新特徵比較組，不依結果追加")
    parser.add_argument("--folds", type=int, nargs="+", choices=[1, 2, 3], help="事先限定向前時段；未指定則三段")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--run-research", action="store_true", help="明確啟動完整矩陣，可能長時間運算")
    args = parser.parse_args(argv)
    if args.paired_loss and not args.feature_study:
        raise ValueError("paired-loss 只支援明確的新特徵配對研究")
    if args.reward_r != 2.0 and args.paired_loss:
        raise ValueError("退出研究與損失消融應分開，不同時變更")
    if args.variants and (not args.feature_study or len(set(args.variants)) != len(args.variants)):
        raise ValueError("variants 僅供新特徵研究且不得重複")
    if args.folds and len(set(args.folds)) != len(args.folds):
        raise ValueError("folds 不得重複")
    started = monotonic()
    plan = research_plan(args.seeds, args.feature_study)
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    source_sha = sha(args.source)
    raw = pd.read_csv(args.source)
    flow_sha = None
    if args.flow_source:
        flow_sha = sha(args.flow_source)
        raw = enrich_flow_source(raw, pd.read_csv(args.flow_source))
        plan["flow_source"] = {"path": str(args.flow_source.resolve()), "sha256": flow_sha,
                               "matched_ohlcv_rows": len(raw)}
    if args.feature_study:
        validate_flow(raw)
    observed_source_end = str(raw.timestamp.iloc[-1])
    if args.output.exists():
        raise ValueError("輸出目錄已存在，請使用新目錄；禁止覆寫研究")
    args.output.mkdir(parents=True)
    all_windows = [w for w in study_windows(len(raw)) if w["stage"] == "walk_forward"]
    if args.folds:
        all_windows = [w for w in all_windows if w["fold"] in args.folds]
    if args.variants or args.folds:
        plan["full_matrix_runs"] = plan["formal_runs"]
        if args.variants:
            plan["variants"] = args.variants
        plan["selected_folds"] = [w["fold"] for w in all_windows]
        plan["walk_forward_folds"] = len(all_windows)
        variant_count = len(args.variants or FEATURE_VARIANTS) if args.feature_study else len(VARIANTS) + 1
        plan["formal_runs"] = len(args.seeds) * len(FAMILIES) * variant_count * len(all_windows)
        plan["limited_matrix"] = "事前限制；單一時段不能冒充完整三段向前驗證"
    plan.update(gate_ablation=["none", "probability", "return", "both"], reward_r=args.reward_r,
                paired_loss=args.paired_loss, dynamic_exit_enabled=False)
    plan["selection"] = f"不依舊 Forward 排名挑選；共同 stop=2ATR,R={args.reward_r:g},hold=32,cooldown=4,strict=false"
    if args.paired_loss:
        plan["variants"] = [v for name in plan["variants"] for v in (name, name + "_smooth_l1")]
        plan["formal_runs"] *= 2
        plan["feature_group_ablation"] = "同一特徵群與 seed，配對 MSE / Smooth L1"
        plan["unchanged"] = "同來源/候選/標籤/成本/門檻/切分/小型V3；特徵與損失採配對比較"
    raw = raw.iloc[:int(len(raw) * .9)].copy()
    plan.update(source=str(args.source.resolve()), source_sha256=source_sha,
        observed_source_end=observed_source_end,
        frozen_source_end=str(raw.timestamp.iloc[-1]), mode="smoke" if args.smoke else "research" if args.run_research else "plan",
        model=payload["model"], config_sha256=sha(args.config), windows=all_windows,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        code_sha256={str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "src").rglob("*.py"))},
        runner_sha256=sha(__file__))
    write_json(args.output / "plan.json", plan)
    if not (args.smoke or args.run_research):
        print(json.dumps({"status": "plan_only", "formal_runs": plan["formal_runs"],
                          "live_eligible": False}, ensure_ascii=False), flush=True)
        return 0
    if args.smoke:
        raw = raw.tail(args.smoke_rows).reset_index(drop=True)
        windows = [{"fold": 0, "source_end": len(raw), "train_end": int(len(raw) * .6),
                    "validation_end": int(len(raw) * .8)}]
        variants = args.variants or (list(FEATURE_VARIANTS) if args.feature_study else ["D_context_mse"])
        seeds = [args.seeds[0]]
    else:
        windows = all_windows
        variants = args.variants or (list(FEATURE_VARIANTS) if args.feature_study else list(VARIANTS) + ["D_context_mse_without_cost"])
        seeds = args.seeds
    if args.paired_loss:
        variants = [v for name in variants for v in (name, name + "_smooth_l1")]
    plan["executed_windows"] = windows
    plan["executed_seeds"] = seeds
    plan["executed_variants"] = variants
    plan["executed_runs"] = len(windows) * len(seeds) * len(variants) * len(FAMILIES)
    write_json(args.output / "plan.json", plan)
    records, failures = [], []
    write_json(args.output / "progress.json", {"status": "running", "completed": 0})
    for window in windows:
        source = args.output / f"frozen_fold_{window['fold']}.csv"
        raw.iloc[:window["source_end"]].to_csv(source, index=False)
        for family in FAMILIES:
            for variant in variants:
                for seed in seeds:
                    name = f"f{window['fold']}_{family}_{variant}_s{seed}"
                    contract, model, training = configurations(payload, family, variant, seed, window, args.smoke, args.reward_r)
                    print(f"開始 {name}", flush=True)
                    try:
                        run_started = monotonic()
                        last = [-1]
                        def progress(event):
                            percent = int(float(event.get("progress", 0)) * 100)
                            if percent // 10 != last[0]:
                                print(f"{name}: {percent}% {event.get('status')}", flush=True)
                                last[0] = percent // 10
                        result = train_temporal_transformer([source], model, training, args.output / "runs",
                                                            run_name=name, progress_callback=progress)
                        training_seconds = monotonic() - run_started
                        evaluation = evaluate_run(result, source, contract, model, training)
                        training_summary = json.loads(result.summary_json.read_text(encoding="utf-8"))
                        records.append({"name": name, "family": family, "variant": variant, "seed": seed,
                            "fold": window["fold"], "run_dir": str(result.run_dir), "samples": evaluation["samples"],
                            "feature_count": len(evaluation["contract"]["feature_columns"]),
                            "training_seconds": training_seconds, "evaluation_seconds": monotonic() - run_started - training_seconds,
                            "test_prediction_metrics": {key: training_summary["test_metrics"][key] for key in (
                                "event_return_skill", "event_probability_skill", "event_prediction_skill_score")},
                            "test_policies": evaluation["comparisons"]["test"]["policies"]})
                    except ValueError as error:
                        # 樣本不足不放寬策略，也不能讓部分完成冒充整個矩陣通過。
                        failures.append({"name": name, "error": str(error)})
                        print(f"停止候選 {name}: {error}", flush=True)
                        write_json(args.output / "failures.json", failures)
                        if not any(token in str(error) for token in ("資料不足", "至少各需要", "沒有符合")):
                            write_json(args.output / "progress.json", {"status": "failed", "run": name, "error": str(error)})
                            raise
                    write_json(args.output / "progress.json", {"status": "running", "completed": len(records),
                                                              "failed": len(failures)})
    unchanged = sha(args.source) == source_sha and (not args.flow_source or sha(args.flow_source) == flow_sha)
    summary = {"engineering": "PASS" if records and not failures and unchanged else "FAIL",
        "research_quality": "INSUFFICIENT_EVIDENCE", "smoke_only": args.smoke,
        "live_eligible": False, "p2_sac": "BLOCKED", "source_unchanged": unchanged,
        "elapsed_seconds": monotonic() - started, "records": records, "failures": failures}
    summary["paired_comparison"] = paired_summary(records)
    summary["sac_assessment"] = assess_sac_readiness(records, args.smoke)
    write_json(args.output / "study.json", summary)
    write_json(args.output / "progress.json", {"status": "complete", "completed": len(records), "failed": len(failures)})
    manifest_all(args.output)
    print(json.dumps({k: v for k, v in summary.items() if k not in {"records", "failures"}}, ensure_ascii=False), flush=True)
    return 0 if summary["engineering"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
