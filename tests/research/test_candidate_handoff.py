"""訓練交接包含全候選，標籤與觀察分離且拒絕跨界。"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ai_quant_trading.research.candidate_contract import CandidateContract
from ai_quant_trading.research.candidate_handoff import export_candidate_handoff, load_candidate_sequence


def fixture():
    features = np.arange(60, dtype=np.float32).reshape(30, 2)
    series = SimpleNamespace(features=features, feature_mask=np.ones_like(features),
        timestamps_ns=pd.date_range("2026-01-01", periods=30, freq="15min", tz="UTC").as_unit("ns").asi8,
        sample_weights=np.ones(30), event_metadata={"label_end_endpoint": np.arange(30)+2})
    subsets = [SimpleNamespace(references=((0,p),), sequence_length=3, series=[series]) for p in (3,9,15,21)]
    data = SimpleNamespace(train=subsets[0], test=subsets[3], scaler=SimpleNamespace(to_dict=lambda: {}))
    outcomes = pd.DataFrame({"endpoint": [3,6,9,15,21], "label_end_endpoint": [5,8,11,17,23], "net_return": [.1]*5})
    return data, subsets[1], subsets[2], outcomes


def test_handoff_all_candidates_and_no_labels_in_inputs(tmp_path):
    data, calibration, selection, outcomes = fixture()
    output = tmp_path / "handoff"
    result = export_candidate_handoff(data, calibration, selection, outcomes, CandidateContract(), output,
        source_sha256="a"*64, checkpoint_sha256="b"*64)
    assert result["rows"] == 5 and result["splits"]["purged_or_warmup"] == 1
    rows = pd.read_csv(output / "candidates.csv")
    chosen = rows.loc[rows.split.eq("test")].iloc[0]
    result = load_candidate_sequence(output, chosen.id, require_oos=True)
    np.testing.assert_array_equal(result["features"], data.train.series[0].features[19:22])
    with np.load(output / "inputs.npz", allow_pickle=False) as content:
        assert set(content.files) == {"features", "feature_mask", "timestamps_ns"}
    with pytest.raises(ValueError):
        load_candidate_sequence(output, rows.loc[rows.split.eq("train"), "id"].iloc[0], require_oos=True)


def test_handoff_cross_boundary_label_rejected(tmp_path):
    data, calibration, selection, outcomes = fixture()
    outcomes.loc[0, "label_end_endpoint"] = 9
    with pytest.raises(ValueError, match="邊界"):
        export_candidate_handoff(data, calibration, selection, outcomes, CandidateContract(), tmp_path / "bad",
            source_sha256="a"*64, checkpoint_sha256="b"*64)
