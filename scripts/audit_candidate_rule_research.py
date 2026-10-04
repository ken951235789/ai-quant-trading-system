"""核對固定規則研究的來源、逐筆成本、時間窗口及候選選擇。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai_quant_trading.operations.integrity import sha256_file, verify_artifact_manifest
from ai_quant_trading.persistence import write_json_atomic


def audit(root: Path, project: Path) -> dict:
    verified = verify_artifact_manifest(root)
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    research = json.loads((root / "research.json").read_text(encoding="utf-8"))
    if sha256_file(Path(plan["source"])) != plan["source_sha256"]:
        raise ValueError("來源資料雜湊不符")
    for name, expected in plan["code_sha256"].items():
        if sha256_file(project / name) != expected:
            raise ValueError(f"研究程式已改變：{name}")
    windows = {row["name"]: row for row in plan["windows"]}
    rules = {row["name"]: row for row in plan["rules"]}
    expected_keys = {(w, r, c) for w in windows for r in rules for c in plan["costs_bps"]}
    keys = [(row["window"], row["rule"], row["scenario"]) for row in research["results"]]
    if len(keys) != len(expected_keys) or set(keys) != expected_keys:
        raise ValueError("情境缺漏或重複")
    checked, grouped = {}, {}
    for row in research["results"]:
        path = root / row["path"]
        frame = pd.read_csv(path)
        window = windows[row["window"]]
        metrics = row["metrics"]
        if len(frame) != metrics["trades"] or frame.endpoint.duplicated().any():
            raise ValueError("交易筆數不符或重複")
        checked[row["path"]] = sha256_file(path)
        if row["window"] != "development":
            grouped.setdefault((row["rule"], row["scenario"]), []).append(frame)
        if frame.empty:
            if metrics["mean"] is not None:
                raise ValueError("空交易不得宣稱具有期望值")
            continue
        if not frame.endpoint.is_monotonic_increasing:
            raise ValueError("事件順序錯誤")
        if not ((frame.endpoint >= window["start"])
                & (frame.endpoint + plan["common_tail_purge_bars"] < window["stop"])
                & (frame.entry_endpoint == frame.endpoint + 1)
                & (frame.exit_endpoint >= frame.entry_endpoint)
                & (frame.exit_endpoint <= frame.endpoint + rules[row["rule"]]["holding_bars"] + 1)).all():
            raise ValueError("進出場窗口違反契約")
        if not (frame.endpoint.iloc[1:].to_numpy() >= frame.exit_endpoint.iloc[:-1].to_numpy() + 5).all():
            raise ValueError("持倉重疊或冷卻不足")
        if not frame.side.isin([-1, 1]).all():
            raise ValueError("方向錯誤")
        cost = plan["costs_bps"][row["scenario"]]
        ratio = frame.exit_price / frame.entry_price
        fees = cost["fee"] / 10_000 * (1 + ratio)
        np.testing.assert_allclose(frame.fee_return, fees, atol=1e-14, rtol=1e-12)
        np.testing.assert_allclose(frame.net_return, frame.side * (ratio - 1) - fees - frame.funding_return,
                                   atol=1e-14, rtol=1e-12)
        np.testing.assert_allclose(frame.net_return, frame.gross_same_fills - frame.fee_return
                                   - frame.funding_return - frame.slippage_spread_return, atol=1e-14)
        if len(frame):
            np.testing.assert_allclose(frame.net_return.mean(), metrics["mean"], atol=1e-14)
            np.testing.assert_allclose(frame.net_return.min(), metrics["worst"], atol=1e-14)
            np.testing.assert_allclose((frame.net_return > 0).mean(), metrics["win_rate"], atol=1e-14)
    pooled_keys = [(r["rule"], r["scenario"]) for r in research["pooled_forward"]]
    if len(pooled_keys) != len(grouped) or set(pooled_keys) != set(grouped):
        raise ValueError("合併情境缺漏或重複")
    for row in research["pooled_forward"]:
        frame = pd.read_csv(root / row["path"])
        expected = pd.concat(grouped[(row["rule"], row["scenario"])], ignore_index=True)
        pd.testing.assert_frame_equal(frame, expected, check_exact=False, atol=1e-14, rtol=1e-12)
        if len(frame) != row["metrics"]["trades"] or frame.endpoint.duplicated().any():
            raise ValueError("合併交易錯誤")
        if len(frame):
            np.testing.assert_allclose(frame.net_return.mean(), row["metrics"]["mean"], atol=1e-14)
        elif row["metrics"]["mean"] is not None:
            raise ValueError("空交易不得宣稱具有期望值")
        checked[row["path"]] = sha256_file(root / row["path"])
    eligible = []
    for rule in rules:
        subset = {r["scenario"]: r["metrics"] for r in research["results"]
                  if r["window"] == "development" and r["rule"] == rule}
        base, stress = subset["base"], subset["stress"]
        if base["trades"] >= 100 and base["ci_low"] > 0 and stress["trades"] > 0 and stress["mean"] > 0:
            eligible.append((rule, base["mean"]))
    expected = max(eligible, key=lambda pair: pair[1])[0] if eligible else None
    if research["selection"]["selected"] != expected or research["live_eligible"]:
        raise ValueError("候選選擇或研究資格不符")
    return {"status": "PASS", "verified_artifact_files": len(verified), "scenarios": len(keys),
            "trade_csv_count": len(checked), "trade_csv_sha256": checked, "selected": expected,
            "source_unchanged": True, "code_unchanged": True,
            "checks": ["逐筆淨收益與雙邊費用", "成本分解", "下一棒成交", "持倉及冷卻",
                       "時間窗口與尾端留白", "逐段合併無重複", "依開發集選擇", "研究不可部署"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    args = parser.parse_args()
    result = audit(args.study.resolve(), Path(__file__).resolve().parents[1])
    write_json_atomic(args.study / "audit.json", result)
    print(f"PASS：{result['scenarios']} 情境，{result['trade_csv_count']} 個逐筆交易 CSV")
