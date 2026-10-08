"""將既有事件 Dataset 交接為可追溯研究資料，不從成交圖像反推輸入。"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai_quant_trading.operations.integrity import build_artifact_manifest, verify_artifact_manifest
from ai_quant_trading.persistence import write_json_atomic


def export_candidate_handoff(data, calibration, selection, outcomes, contract, output,
                             *, source_sha256, checkpoint_sha256):
    """保存一份時序矩陣與所有候選索引，不展開重複的 96 根視窗。"""
    output = Path(output)
    if len(data.train.series) != 1:
        raise ValueError("此候選交接目前要求單一 BTC 來源")
    if any(len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
           for digest in (source_sha256, checkpoint_sha256)):
        raise ValueError("交接需要來源與實際 checkpoint 的 SHA256")
    output.mkdir(parents=True, exist_ok=False)
    series = data.train.series[0]
    splits = {"train": data.train, "calibration": calibration, "selection": selection, "test": data.test}
    membership, upper = {}, {}
    ordered = list(splits)
    for index, (name, subset) in enumerate(splits.items()):
        if not subset.references:
            raise ValueError("交接區間不能為空")
        for source_index, endpoint in subset.references:
            if source_index != 0 or endpoint in membership:
                raise ValueError("候選切分重複或跨來源")
            membership[endpoint] = name
        upper[name] = min(p for _, p in splits[ordered[index + 1]].references) if index < 3 else len(series.features)
    last_fit = max(p for _, p in selection.references)
    end_values = series.event_metadata["label_end_endpoint"]
    # Selection 影響 checkpoint/閘門，合法最終預測必須晚於這些標籤完成時間。
    model_known_endpoint = max(int(end_values[p]) for _, p in selection.references)
    items = []
    for row in outcomes.sort_values("endpoint").itertuples():
        point, end = int(row.endpoint), int(row.label_end_endpoint)
        if not 0 <= point < end < len(series.timestamps_ns):
            raise ValueError("候選標籤時間超出來源")
        split = membership.get(point, "purged_or_warmup")
        if split in splits and end >= upper[split]:
            raise ValueError("交接拒絕跨切分邊界的標籤")
        sample_id = hashlib.sha256(f"{source_sha256}|{contract.digest}|{point}".encode()).hexdigest()
        items.append({"id": sample_id, "endpoint": point, "label_end_endpoint": end,
            "split": split, "known_at": pd.Timestamp(series.timestamps_ns[point], tz="UTC") + pd.Timedelta(minutes=15),
            "label_available_at": pd.Timestamp(series.timestamps_ns[end], tz="UTC"),
            "train_weight": float(series.sample_weights[point]) if split == "train" else None,
            "prediction_oos_allowed": split == "test" and point > max(last_fit, model_known_endpoint),
            "selected_for_trade": None, "all_candidates_not_only_fills": True})
    references = pd.DataFrame(items)
    np.savez_compressed(output / "inputs.npz", features=series.features,
                        feature_mask=series.feature_mask, timestamps_ns=series.timestamps_ns)
    references.to_csv(output / "candidates.csv", index=False)
    outcomes.merge(references[["endpoint", "id", "split"]], on="endpoint", validate="one_to_one").to_csv(output / "labels.csv", index=False)
    write_json_atomic(output / "handoff.json", {"schema": "candidate_handoff_v1", "contract": contract.metadata(),
        "source_sha256": source_sha256, "checkpoint_sha256": checkpoint_sha256,
        "scaler": data.scaler.to_dict(), "sequence_length": data.train.sequence_length,
        "rows": len(references), "splits": references.split.value_counts().to_dict(),
        "model_information_end_endpoint": model_known_endpoint,
        "all_outcomes_are_labels_not_inputs": True, "live_eligible": False,
        "scope": "完整候選研究；重複研究的 Test 不是全新 holdout；OOS 時間合法不代表品質合格"})
    build_artifact_manifest(output)
    return {"rows": len(references), "splits": references.split.value_counts().to_dict()}


def load_candidate_sequence(root, candidate_id, *, require_oos=False):
    """只回傳當時的輸入，不把 labels.csv 或出場圖片混入特徵。"""
    root = Path(root)
    verify_artifact_manifest(root, required=True)
    meta = json.loads((root / "handoff.json").read_text(encoding="utf-8"))
    rows = pd.read_csv(root / "candidates.csv")
    matched = rows.loc[rows.id.eq(candidate_id)]
    if len(matched) != 1:
        raise ValueError("候選識別不存在或不唯一")
    row = matched.iloc[0]
    if row.split == "purged_or_warmup" or (require_oos and not bool(row.prediction_oos_allowed)):
        raise ValueError("候選不符合允許的訓練或 OOS 區間")
    point, length = int(row.endpoint), meta["sequence_length"]
    if point + 1 < length:
        raise ValueError("候選歷史序列不足")
    with np.load(root / "inputs.npz", allow_pickle=False) as values:
        return {"features": values["features"][point-length+1:point+1].copy(),
                "feature_mask": values["feature_mask"][point-length+1:point+1].copy(),
                "known_at": row.known_at, "contract_sha256": meta["contract"]["contract_sha256"]}


def join_trade_journal(journal, handoff, output):
    """實際成交僅連結回已保存候選；缺少來源證明時不冒充可訓練樣本。"""
    handoff, output = Path(handoff), Path(output)
    verify_artifact_manifest(handoff, required=True)
    meta = json.loads((handoff / "handoff.json").read_text(encoding="utf-8"))
    candidates = pd.read_csv(handoff / "candidates.csv")
    lookup = {pd.Timestamp(row.known_at): row for row in candidates.itertuples()}
    joined, excluded = [], []
    for item in journal.records(limit=1_000_000):
        entry = item["record"]["entry"]
        decision = entry.get("decision") or {}
        provenance = decision.get("candidate_handoff") or {}
        row = lookup.get(pd.Timestamp(decision["known_at"])) if decision.get("known_at") else None
        valid = row is not None and provenance.get("source_sha256") == meta["source_sha256"] and provenance.get("contract_sha256") == meta["contract"]["contract_sha256"]
        if not valid:
            excluded.append({"record_id": item["id"], "reason": "missing_exact_candidate_source_or_contract_proof"})
            continue
        joined.append({"record_id": item["id"], "candidate_id": row.id, "split": row.split,
                       "oos_allowed": bool(row.prediction_oos_allowed), "automatic_training": False})
    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output / "journal_links.json", {"joined": joined, "excluded": excluded,
        "live_eligible": False, "selection_bias": "保留所有候選；成交子集不可取代整體候選"})
    build_artifact_manifest(output)
    return {"joined": len(joined), "excluded": len(excluded)}
