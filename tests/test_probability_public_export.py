"""公開彙總匯出只接受固定分類與數值，不能夾帶私人欄位。"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("probability_export", Path(__file__).resolve().parents[1] / "scripts/export_probability_research.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    row = {name: 0. for name in module.SUMMARY_COLUMNS}
    row.update(family="original", variant="F_existing", split="test", scenario="base")
    return pd.DataFrame([row])


def test_unlisted_private_columns_removed():
    frame = fixture()
    frame["private_extra"] = "DO_NOT_EXPORT"
    result = module.sanitize_summary(frame)
    assert "DO_NOT_EXPORT" not in str(result)
    assert len(result["columns"]) == len(result["rows"][0])


@pytest.mark.parametrize("column", ["family", "mean_net_mean"])
def test_free_text_rejected(column):
    frame = fixture().astype(object)
    frame.loc[0, column] = "unapproved_text"
    with pytest.raises(ValueError):
        module.sanitize_summary(frame)
