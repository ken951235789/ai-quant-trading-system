"""唯讀核對候選研究的來源、契約、成品與逐筆帳務，不改變模型或品質閘門。"""

import argparse
import json
from pathlib import Path, PurePosixPath
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_quant_trading.operations.integrity import verify_artifact_manifest, sha256_file  # noqa: E402
from ai_quant_trading.research.candidate_contract import CandidateContract, prepare_candidate_frame  # noqa: E402
from ai_quant_trading.transformer.inference import _load_checkpoint  # noqa: E402
from ai_quant_trading.transformer.strategy_events import replay_event  # noqa: E402


def verify_trade_table(trades, cooldown):
    """空 CSV 的 dtype 可能是 object；仍明確驗證欄位及有限值，不把零成交當獲利。"""
    fields = ["endpoint", "exit_endpoint", "gross_same_path_return", "spread_return",
              "slippage_return", "fee_return", "funding_return", "net_return"]
    values = trades[fields].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("交易帳務含非有限值")
    if len(trades) > 1 and not np.all(values[1:, 0] > values[:-1, 1] + cooldown):
        raise ValueError("成交有重疊或未遵守冷卻")
    net = values[:, 2] - values[:, 3:7].sum(axis=1)
    np.testing.assert_allclose(net, values[:, 7], atol=1e-12, rtol=1e-10)
    return len(trades)


def portable_path(saved, parent):
    """只映射已下載的研究檔名，不信任雲端絕對路徑，也不修改已簽名的原始 JSON。"""
    name = PurePosixPath(str(saved).replace("\\", "/")).name
    if not name or name in {".", ".."}:
        raise ValueError("研究檔名不合法")
    path = (Path(parent) / name).resolve()
    path.relative_to(Path(parent).resolve())
    return path


def audit(root, bundle_root=None):
    root = Path(root).resolve()
    bundle_root = Path(bundle_root).resolve() if bundle_root else None
    verified = verify_artifact_manifest(root)
    study = json.loads((root / "study.json").read_text(encoding="utf-8"))
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    if study["engineering"] != "PASS" or study["live_eligible"] or study["p2_sac"] != "BLOCKED":
        raise ValueError("研究執行狀態或部署隔離不符")
    source_path = bundle_root / "data/btc_15m.csv" if bundle_root else Path(plan["source"])
    if sha256_file(source_path) != plan["source_sha256"]:
        raise ValueError("原始來源已變動")
    if "flow_source" in plan and sha256_file(Path(plan["flow_source"]["path"])) != plan["flow_source"]["sha256"]:
        raise ValueError("補充成交來源已變動")
    for path, digest in plan["code_sha256"].items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError(f"研究計算版本不同：{path}")
        if bundle_root and sha256_file(bundle_root / path) != digest:
            raise ValueError(f"雲端封包版本不同：{path}")
    runner_root = bundle_root or ROOT
    if sha256_file(runner_root / "scripts/run_candidate_training_research.py") != plan["runner_sha256"]:
        raise ValueError("研究執行入口版本不同")
    counts = {"manifest_files": len(verified), "models": 0, "prediction_rows": 0,
              "nonoverlap_trade_rows": 0, "slow_reference_probes": 0}
    paired_labels = {}
    for record in study["records"]:
        run = portable_path(record["run_dir"], root / "runs") if bundle_root else Path(record["run_dir"]).resolve()
        run.relative_to(root)
        evaluation = json.loads((run / "candidate_evaluation.json").read_text(encoding="utf-8"))
        payload = torch.load(run / "best_model.pt", map_location="cpu", weights_only=True)
        contract = CandidateContract(**payload["training_config"]["candidate_contract"])
        if contract.digest != evaluation["contract"]["contract_sha256"]:
            raise ValueError("模型與研究契約不一致")
        if payload["data_diagnostics"]["candidate_contract"] != evaluation["contract"]:
            raise ValueError("checkpoint 與歸因資料來源不一致")
        source = portable_path(payload["sources"][0]["path"], root) if bundle_root else Path(payload["sources"][0]["path"]).resolve()
        source.relative_to(root)
        if sha256_file(source) != evaluation["contract"]["source_sha256"]:
            raise ValueError("訓練快照已變更")
        try:
            _load_checkpoint(run / "best_model.pt", torch.device("cpu"))
        except ValueError as exc:
            if "研究模型" not in str(exc):
                raise
        else:
            raise ValueError("研究模型竟能进入部署推論")
        frame = prepare_candidate_frame(pd.read_csv(source), contract)
        attribution = pd.read_csv(run / "candidate_attribution.csv").set_index("endpoint", drop=False)
        if attribution.event_id.duplicated().any() or attribution.source_sha256.nunique() != 1:
            raise ValueError("候選身分重複或混資料")
        for split in ("validation", "test"):
            predictions = pd.read_csv(run / f"{split}_predictions.csv")
            aligned = attribution.loc[predictions.endpoint]
            np.testing.assert_allclose(predictions[f"actual_return_{contract.holding_bars}"], aligned.net_return,
                                       rtol=1e-5, atol=1e-8)
            np.testing.assert_array_equal(predictions.event_exit_endpoint, aligned.exit_endpoint)
            key = (record["family"], record["fold"], record["seed"], split)
            labels = predictions[["endpoint", "event_exit_endpoint", f"actual_return_{contract.holding_bars}"]].to_numpy()
            if key in paired_labels:
                np.testing.assert_allclose(labels, paired_labels[key], atol=1e-8, rtol=1e-5)
            paired_labels[key] = labels
            counts["prediction_rows"] += len(predictions)
        for row in attribution.iloc[np.linspace(0, len(attribution) - 1, min(12, len(attribution)), dtype=int)].itertuples():
            reference = replay_event(frame, row.endpoint, contract.event_config(),
                fee_bps_per_side=contract.fee_bps, slippage_bps_per_side=contract.slippage_bps,
                regime_exit=contract.regime_exit)
            for key, expected in reference.items():
                if isinstance(expected, str):
                    if getattr(row, key) != expected:
                        raise ValueError("參考退出原因不一致")
                else:
                    np.testing.assert_allclose(getattr(row, key), expected, atol=1e-12, rtol=1e-10)
            counts["slow_reference_probes"] += 1
        for path in run.glob("*_policies/*_trades.csv"):
            trades = pd.read_csv(path)
            counts["nonoverlap_trade_rows"] += verify_trade_table(trades, contract.cooldown_bars)
        counts["models"] += 1
    return {"status": "PASS", **counts, "live_eligible": False,
            "scope": "只驗證列明工程證據，不授予策略或完整帳戶績效資格"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--bundle-root", type=Path, help="下載 Kaggle 成品時，提供原本白名單 staging 封包")
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.bundle_root), ensure_ascii=False, indent=2))
