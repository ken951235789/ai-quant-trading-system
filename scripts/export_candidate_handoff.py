"""從已完成研究重建 Transformer 序列交接包，不啟動訓練。"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_quant_trading.operations.integrity import sha256_file, verify_artifact_manifest  # noqa: E402
from ai_quant_trading.research.candidate_contract import CandidateContract, candidate_outcomes, prepare_candidate_frame  # noqa: E402
from ai_quant_trading.research.candidate_handoff import export_candidate_handoff  # noqa: E402
from ai_quant_trading.transformer.config import TemporalTransformerConfig, TransformerTrainingConfig  # noqa: E402
from ai_quant_trading.transformer.dataset import prepare_transformer_datasets  # noqa: E402
from ai_quant_trading.transformer.training import _split_calibration  # noqa: E402


def main():
    import pandas as pd
    import torch
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    verify_artifact_manifest(args.run, required=True)
    summary = json.loads((args.run / "training.json").read_text(encoding="utf-8"))
    model = TemporalTransformerConfig(**summary["model_config"])
    training = TransformerTrainingConfig(**summary["training_config"])
    contract = CandidateContract(**training.candidate_contract)
    source_sha = sha256_file(args.source)
    original_sha = summary["data_diagnostics"].get("candidate_contract", {}).get("source_sha256")
    if not original_sha:
        evaluation = json.loads((args.run / "candidate_evaluation.json").read_text(encoding="utf-8"))
        original_sha = evaluation["contract"].get("source_sha256")
    # 舊模型沒有可驗證來源欄位時拒絕自動替換來源，不能只看檔名相同。
    if not original_sha or original_sha != source_sha:
        raise ValueError("來源雜湊缺失或不符，不能證明為原訓練資料")
    # checkpoint 保存的是剪枝後維度；重建時先給已登記事件特徵足夠容量，再核對結果。
    diagnostics = prepare_transformer_datasets([args.source], replace(model, input_features=max(128, model.input_features), feature_group_ids=()), training)
    if diagnostics.resolved_model_config.to_dict() != model.to_dict():
        raise ValueError("重建模型輸入契約與 checkpoint 不同")
    if diagnostics.scaler.to_dict() != summary["scaler"]:
        raise ValueError("重建正規化器與 checkpoint 不同，拒絕交接")
    if original_sha and original_sha != source_sha:
        raise ValueError("來源雜湊不符")
    source_contract = diagnostics.diagnostics.get("candidate_contract", {})
    if source_contract != summary["data_diagnostics"].get("candidate_contract", {}):
        raise ValueError("重建來源契約與原研究不一致")
    calibration, selection, _ = _split_calibration(diagnostics, training)
    frame = prepare_candidate_frame(pd.read_csv(args.source), contract)
    print(json.dumps(export_candidate_handoff(diagnostics, calibration, selection,
        candidate_outcomes(frame, contract), contract, args.output, source_sha256=source_sha,
        checkpoint_sha256=sha256_file(args.run / summary["artifacts"]["model"])), ensure_ascii=False))


if __name__ == "__main__":
    main()
