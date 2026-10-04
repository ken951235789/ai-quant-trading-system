"""獨立核對分階段研究的版本、樣本外邊界、資料契約與未通過時的鎖定狀態。"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai_quant_trading.operations.integrity import build_artifact_manifest, sha256_file, verify_artifact_manifest
from ai_quant_trading.persistence import write_json_atomic
from ai_quant_trading.research.staged_training import assess_transformer, digest_contract, split_oos
from ai_quant_trading.transformer.config import TransformerTrainingConfig
from ai_quant_trading.transformer.strategy_events import StrategyEventConfig, prepare_event_frame


def require(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit(root: Path) -> dict:
    verify_artifact_manifest(root)
    project = Path(__file__).resolve().parents[1]
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    records = json.loads((root / "records.json").read_text(encoding="utf-8"))
    gates = json.loads((root / "transformer_gate.json").read_text(encoding="utf-8"))
    contract = plan["contract"]
    require(digest_contract(contract) == plan["contract_sha256"] == result["contract_sha256"], "交易契約雜湊不一致")
    require(sha256_file(Path(plan["source"])) == plan["source_sha256"], "原始資料已改變")
    for name, expected in plan["code_sha256"].items():
        if sha256_file(project / name) != expected:
            raise ValueError(f"執行程式版本已改變：{name}")
    expected = {(window["fold"], seed) for window in plan["windows"] for seed in plan["seeds"]}
    require({(r["fold"], r["seed"]) for r in records} == expected and len(records) == len(expected), "訓練工作有缺漏或重複")
    bars = pd.read_csv(plan["source"])
    times = pd.to_datetime(bars.timestamp, utc=True).astype("int64").to_numpy()
    event = StrategyEventConfig(**contract["event"])
    training = TransformerTrainingConfig(**plan["training"])
    horizon = contract["horizon_bars"]
    prepared = prepare_event_frame(bars.iloc[:plan["reserved_tail_start"]], event)
    for record in records:
        checkpoint = root / record["checkpoint"]
        verify_artifact_manifest(checkpoint.parent)
        require(sha256_file(checkpoint) == record["checkpoint_sha256"], "模型雜湊不一致")
    checks = []
    for seed in plan["seeds"]:
        frame = pd.read_csv(root / f"oos_seed{seed}.csv")
        require(not frame.endpoint.duplicated().any(), "OOS 時間點重複")
        require(frame.contract_sha256.eq(plan["contract_sha256"]).all(), "OOS 契約不一致")
        require(frame.endpoint.gt(frame.fit_end_endpoint).all(), "預測早於 fit 截止時間")
        np.testing.assert_array_equal(frame.timestamp_ns, times[frame.endpoint.to_numpy(dtype=int)])
        causal = prepared.iloc[frame.endpoint.to_numpy(dtype=int)]
        np.testing.assert_allclose(frame.causal_atr_fraction, (causal.event_atr / causal.close).to_numpy(), atol=1e-12)
        for window in plan["windows"]:
            part = frame.loc[frame.crossfit_fold.eq(window["fold"])]
            require(len(part) > 0, "OOS 區塊為空")
            require(part.endpoint.ge(window["validation_end"] + training.embargo_bars).all(), "OOS 預測早於 embargo")
            require(part.event_exit_endpoint.lt(window["source_end"]).all(), "標籤跨越窗口")
            record = next(r for r in records if r["seed"] == seed and r["fold"] == window["fold"])
            require(part.checkpoint_sha256.eq(record["checkpoint_sha256"]).all(), "預測模型來源不一致")
            original = pd.read_csv((root / record["checkpoint"]).parent / "test_predictions.csv")
            require(len(part) == len(original), "合併預測筆數不同")
            for column in original:
                if pd.api.types.is_numeric_dtype(original[column]):
                    np.testing.assert_allclose(part[column], original[column], atol=1e-12)
                else:
                    require(part[column].reset_index(drop=True).equals(original[column]), "合併預測內容不同")
        splits = split_oos(frame, plan["windows"][0]["validation_end"], plan["reserved_tail_start"], horizon)
        reproduced = assess_transformer(splits, horizon, event, training, plan["gate"]["minimum_trades"])
        require(reproduced["passed"] == gates[str(seed)]["passed"], "品質結論不能重現")
        require(reproduced.get("checks") == gates[str(seed)].get("checks"), "品質條件不能重現")
        checks.append({"seed": seed, "oos_candidates": len(frame),
                       "split_candidates": {name: len(part) for name, part in splits.items()},
                       "gate_reproduced": True})
    passed = all(g["passed"] for g in gates.values())
    require(passed == result["transformer_gate_passed"], "摘要與品質結論不一致")
    if not passed:
        require(result["sac_stage"] == "blocked_by_transformer_gate", "未通過但未阻擋 SAC")
        require(result["stage4"] == "locked" and not result["sac_test_evaluated"], "未通過卻開放下一階段")
        require(not (root / "sizing").exists(), "未通過卻產生正式 SAC 工作")
    require(result["tail_evaluated"] is False and result["live_eligible"] is False, "超出授權研究範圍")
    return {"status": "PASS", "transformer_runs": len(records), "checks": checks,
            "contract_consistent": True, "source_unchanged": True, "oos_verified": True,
            "gate_reproduced": True, "scope": "研究證據核對，不是長期獲利或實盤認證"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    args = parser.parse_args()
    result = audit(args.study)
    write_json_atomic(args.study / "audit.json", result)
    # 研究證據不只包含權重，也封存預測 CSV、測試 XML 與人類報告。
    build_artifact_manifest(args.study, files=[path for path in args.study.rglob("*") if path.is_file()])
    print(json.dumps(result, ensure_ascii=False, indent=2))
