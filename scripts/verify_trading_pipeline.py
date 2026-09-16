"""使用本機 RL 成品回放指定根數，驗證完整交易治理管線。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import pandas as pd

from ai_quant_trading.paper_trading import run_rl_paper_cycle
from ai_quant_trading.paper_trading.storage import read_account_csv
from ai_quant_trading.reinforcement_learning import (
    assess_rl_training_quality,
    load_rl_policy,
)
from ai_quant_trading.persistence import write_json_atomic


def _resolve_source(model_dir: Path, configured: str, supplied: str | None) -> Path:
    if supplied:
        path = Path(supplied)
    else:
        path = Path(configured)
        if not path.is_absolute():
            candidates = [Path.cwd() / path, model_dir.parent.parent / path]
            path = next((candidate for candidate in candidates if candidate.is_file()), path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到驗收市場資料：{path}")
    return path.resolve()


def verify_pipeline(
    model_dir: str | Path,
    *,
    input_path: str | None = None,
    bars: int = 20,
    output_root: str | Path | None = None,
    device: str = "cpu",
    end_at: str | None = None,
) -> dict[str, object]:
    """回放最後 N 根 K 線，並回傳可序列化的驗收摘要。"""
    if bars < 2:
        raise ValueError("bars 至少為 2")
    training_dir = Path(model_dir).resolve()
    policy = load_rl_policy(training_dir, device=device)
    source = _resolve_source(
        training_dir,
        str(policy.environment_metadata.get("source_path", "")),
        input_path,
    )
    market = pd.read_csv(source)
    if end_at:
        timestamps = pd.to_datetime(market["timestamp"], utc=True, errors="coerce")
        cutoff = pd.to_datetime(end_at, utc=True, errors="raise")
        market = market.loc[timestamps <= cutoff].reset_index(drop=True)
    minimum_history = max(250, bars)
    replay = market.tail(minimum_history + bars - 1).reset_index(drop=True)
    if len(replay) < bars:
        raise ValueError(f"市場資料只有 {len(replay)} 根，無法回放 {bars} 根")
    initial_end = len(replay) - bars + 1

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if output_root is None:
        temporary = tempfile.TemporaryDirectory(prefix="ai-quant-verify-")
        storage_root = Path(temporary.name)
    else:
        storage_root = Path(output_root).resolve()
    common = {
        "policy": policy,
        "model_dir": training_dir,
        "root_dir": storage_root,
        "account_id": (
            "pipeline_verify_" + pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S%f")
        ),
    }
    try:
        run_rl_paper_cycle(replay.iloc[:initial_end], **common)
        result = run_rl_paper_cycle(replay, **common)
        duplicate = run_rl_paper_cycle(replay, **common)
        predictions = read_account_csv(result.paths.predictions_csv).tail(bars)
        performance = read_account_csv(result.paths.performance_csv).tail(bars)
        orders = read_account_csv(result.paths.orders_csv)
        quality = assess_rl_training_quality(policy.training_metadata)
        ai_context = dict(policy.environment_metadata.get("ai_context", {}))

        def numeric_column_ready(frame: pd.DataFrame, column: str) -> bool:
            return column in frame and pd.to_numeric(frame[column], errors="coerce").notna().all()

        transformer_required = bool(ai_context.get("transformer_enabled", False))
        finbert_required = bool(ai_context.get("finbert_enabled", False))
        role_status: dict[str, dict[str, object]] = {
            "data_engineer": {
                "operational": len(performance) == bars,
                "evidence": f"processed={len(performance)}/{bars}",
            },
            "finbert_analyst": {
                "operational": (not finbert_required) or numeric_column_ready(
                    predictions, "finbert_available"
                ),
                "required": finbert_required,
                "evidence": {
                    "environment_coverage": dict(ai_context.get("coverage", {}))
                    .get("aggregate", {})
                    .get("finbert"),
                    "active_bars": int(
                        pd.to_numeric(
                            predictions.get("finbert_available", pd.Series(dtype=float)),
                            errors="coerce",
                        ).fillna(0.0).ge(0.5).sum()
                    ),
                },
            },
            "transformer_analyst": {
                "operational": (not transformer_required) or (
                    "market_regime" in predictions
                    and predictions["market_regime"].astype(str).str.len().gt(0).all()
                ),
                "required": transformer_required,
                "evidence": sorted(set(predictions.get("market_regime", pd.Series(dtype=str)).astype(str))),
            },
            "sac_trader": {
                "operational": numeric_column_ready(predictions, "model_target_fraction"),
                "evidence": "model_target_fraction",
            },
            "regime_analyst": {
                "operational": numeric_column_ready(predictions, "regime_risk_multiplier"),
                "evidence": "regime_risk_multiplier",
            },
            "capital_manager": {
                "operational": numeric_column_ready(predictions, "capital_multiplier"),
                "evidence": "capital_multiplier",
            },
            "model_monitor": {
                "operational": numeric_column_ready(predictions, "model_input_drift_score"),
                "evidence": "model_input_drift_score",
            },
            "fixed_risk_manager": {
                "operational": numeric_column_ready(predictions, "target_risk_multiplier"),
                "evidence": "target_risk_multiplier",
            },
            "portfolio_manager": {
                "operational": numeric_column_ready(predictions, "portfolio_gross_exposure"),
                "evidence": "portfolio_gross_exposure",
            },
            "execution_engine": {
                "operational": result.processed,
                "evidence": f"orders={len(orders)}",
            },
            "accountant": {
                "operational": numeric_column_ready(performance, "equity")
                and pd.to_numeric(performance["equity"], errors="coerce").gt(0).all(),
                "evidence": "positive_equity",
            },
            "idempotency_guard": {
                "operational": not duplicate.processed,
                "evidence": duplicate.message,
            },
        }
        for role in role_status.values():
            role["operational"] = bool(role["operational"])
        summary: dict[str, object] = {
            "status": "passed" if all(
                bool(item["operational"]) for item in role_status.values()
            ) else "failed",
            "model_dir": str(training_dir),
            "input_path": str(source),
            "algorithm": str(
                dict(policy.training_metadata.get("training_config", {})).get(
                    "algorithm", "unknown"
                )
            ),
            "bars_requested": bars,
            "replay_end_at": end_at,
            "bars_processed": len(performance),
            "predictions": len(predictions),
            "orders": len(orders),
            "final_equity": (
                float(performance.iloc[-1]["equity"]) if not performance.empty else None
            ),
            "last_model_target": (
                float(predictions.iloc[-1]["model_target_fraction"])
                if not predictions.empty
                else None
            ),
            "last_approved_target": (
                float(predictions.iloc[-1]["approved_target_fraction"])
                if not predictions.empty
                else None
            ),
            "regimes": (
                predictions["market_regime"].value_counts(dropna=False).to_dict()
                if "market_regime" in predictions
                else {}
            ),
            "duplicate_blocked": not duplicate.processed,
            "model_quality_eligible": quality.eligible,
            "model_quality_reasons": list(quality.reasons),
            "roles": role_status,
        }
        if output_root is not None:
            summary["account_dir"] = str(result.paths.account_dir)
        if not summary["duplicate_blocked"] or summary["bars_processed"] != bars:
            summary["status"] = "failed"
        if output_root is not None:
            report_path = result.paths.account_dir / "verification_summary.json"
            summary["report_path"] = str(report_path)
            write_json_atomic(report_path, summary)
        return summary
    finally:
        if temporary is not None:
            temporary.cleanup()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="回放 RL 模型並驗證交易治理管線")
    parser.add_argument("--model-dir", required=True, help="SAC／PPO training 資料夾")
    parser.add_argument("--input", help="選填；覆蓋 environment.json 的 source_path")
    parser.add_argument("--bars", type=int, default=20, help="回放 K 線數量，預設 20")
    parser.add_argument("--output-root", help="選填；保留驗收帳戶資料的資料夾")
    parser.add_argument("--device", default="cpu", help="cpu、cuda 或 auto")
    parser.add_argument("--end-at", help="選填；只回放此 UTC 時間以前的歷史資料")
    return parser


def main(arguments: list[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    summary = verify_pipeline(
        args.model_dir,
        input_path=args.input,
        bars=args.bars,
        output_root=args.output_root,
        device=args.device,
        end_at=args.end_at,
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
