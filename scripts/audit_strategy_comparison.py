"""核對多策略比較的相同樣本、成本、交易統計與來源完整性。"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai_quant_trading.operations.integrity import build_artifact_manifest, sha256_file, verify_artifact_manifest
from ai_quant_trading.persistence import write_json_atomic


def audit(root: Path) -> dict:
    verify_artifact_manifest(root)
    project = Path(__file__).resolve().parents[1]
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
    for name, digest in plan["source_sha256"].items():
        if sha256_file(Path(name)) != digest:
            raise ValueError("來源檔案已改變")
    for name, digest in plan["code_sha256"].items():
        if sha256_file(project / name) != digest:
            raise ValueError("比較程式已改變")
    configs = plan["environment_configs"]
    excluded = {"fee_rate", "slippage_rate", "spread_rate", "short_borrow_rate_annual", "liquidation_fee_rate"}
    risks = [{k: v for k, v in config.items() if k not in excluded} for config in configs.values()]
    if any(risk != risks[0] for risk in risks):
        raise ValueError("成本情境以外的風控設定不一致")
    keys = [(r["split"], r["scenario"], r["policy"]) for r in summary["results"]]
    expected = {(split, scenario, name) for split in ("validation", "test")
                for scenario in configs for name in plan["policy_names"]}
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("比較情境缺漏或重複")
    rows = list(summary["results"])
    supplemental = root / "final_checkpoint_results.json"
    supplemental_count = 0
    if supplemental.exists():
        final_plan = json.loads((root / "final_checkpoint_plan.json").read_text(encoding="utf-8"))
        if sha256_file(root / "plan.json") != final_plan["original_plan_sha256"]:
            raise ValueError("補充研究使用不同的原始計畫")
        if sha256_file(project / "scripts/compare_sac_final_checkpoint.py") != final_plan["code_sha256"]:
            raise ValueError("補充研究程式已改變")
        if sha256_file(Path(plan["training_dir"]) / "final_model.zip") != final_plan["final_model_sha256"]:
            raise ValueError("最終模型已改變")
        extra = json.loads(supplemental.read_text(encoding="utf-8"))["results"]
        extra_keys = [(r["split"], r["scenario"], r["policy"]) for r in extra]
        extra_expected = {(split, scenario, "sac_final") for split in ("validation", "test")
                          for scenario in configs}
        if len(extra_keys) != len(extra_expected) or set(extra_keys) != extra_expected:
            raise ValueError("最終模型補充情境缺漏或重複")
        if any(row["config"] != configs[row["scenario"]] for row in extra):
            raise ValueError("最終模型未沿用相同環境")
        rows.extend(extra)
        supplemental_count = len(extra)
    hashes, timestamp_hashes = {}, {}
    for row in rows:
        path = root / row["path"]
        frame = pd.read_csv(path)
        metrics = row["metrics"]
        window = plan["windows"][row["split"]]
        times = pd.to_datetime(frame.timestamp, utc=True)
        expected_times = pd.date_range(pd.Timestamp(window["start"]) + pd.Timedelta(minutes=15),
                                       pd.Timestamp(window["end"]), freq="15min")
        if not np.array_equal(times.array, expected_times.array):
            raise ValueError("比較期間不同或缺棒")
        initial = configs[row["scenario"]]["initial_capital"]
        np.testing.assert_allclose(frame.equity.iloc[-1] / initial - 1, metrics["total_return"], atol=1e-12)
        equity = np.r_[initial, frame.equity.to_numpy()]
        drawdown = equity / np.maximum.accumulate(equity) - 1
        np.testing.assert_allclose(abs(drawdown.min()), metrics["max_drawdown"], atol=1e-12)
        closed = frame.loc[frame.trade_closed.astype(bool), "closed_trade_pnl"]
        if len(closed) != metrics["closed_trades"]:
            raise ValueError("已平倉筆數不一致")
        if len(closed):
            np.testing.assert_allclose(closed.mean(), metrics["expectancy"], atol=1e-12)
            np.testing.assert_allclose((closed > 0).mean(), metrics["win_rate"], atol=1e-12)
        elif metrics["win_rate"] is not None or metrics["expectancy"] is not None:
            raise ValueError("空手沒有可估計的交易勝率或期望值")
        # 沒有期末未平倉時，完整往返損益應能對回帳戶權益變化。
        if abs(float(frame.quantity.iloc[-1])) < 1e-10:
            np.testing.assert_allclose(closed.sum(), frame.equity.iloc[-1] - initial, atol=1e-8)
        for column, key in (("fee", "total_fees"), ("slippage_cost", "total_slippage"),
                            ("funding_cost", "total_funding")):
            np.testing.assert_allclose(frame[column].sum(), metrics[key], atol=1e-10)
        if row["scenario"] == "zero":
            np.testing.assert_allclose(frame[["fee", "slippage_cost", "funding_cost"]], 0, atol=1e-12)
        if row["policy"] == "cash":
            np.testing.assert_allclose(frame.equity, initial, atol=1e-12)
        hashes[row["path"]] = sha256_file(path)
        timestamp_hashes.setdefault(row["split"], []).append(tuple(times.astype("int64")))
    for timelines in timestamp_hashes.values():
        if any(timeline != timelines[0] for timeline in timelines):
            raise ValueError("策略執行時間軸不同")
    return {"status": "PASS", "scenarios": len(rows), "main_scenarios": len(keys),
            "supplemental_scenarios": supplemental_count, "csv_sha256": hashes,
            "same_timeline": True, "same_risk_limits": True, "source_unchanged": True,
            "cost_totals_reconciled": True, "closed_pnl_reconciled_when_flat": True,
            "scope": "同期間、同環境的會計／契約核對，不等同長期獲利或真實撮合證明"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    args = parser.parse_args()
    result = audit(args.study)
    write_json_atomic(args.study / "audit.json", result)
    build_artifact_manifest(args.study, [p for p in args.study.iterdir() if p.is_file()])
    print(f"PASS：{result['scenarios']} 組同環境比較與損益核對")
