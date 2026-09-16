"""CPU 預算與檔案快取的回歸測試。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from ai_quant_trading.performance import (
    apply_thread_environment,
    clamp_cpu_threads,
)
from ai_quant_trading.persistence import (
    clear_csv_snapshot_cache,
    read_csv_snapshot,
)


def test_thread_environment_applies_cpu_budget() -> None:
    environment = {"OMP_NUM_THREADS": "99", "EXISTING": "kept"}

    apply_thread_environment(environment, 2, overwrite=True)

    assert environment["OMP_NUM_THREADS"] == "2"
    assert environment["MKL_NUM_THREADS"] == "2"
    assert environment["OPENBLAS_NUM_THREADS"] == "2"
    assert environment["NUMEXPR_NUM_THREADS"] == "2"
    assert environment["EXISTING"] == "kept"


def test_invalid_cpu_budget_is_rejected() -> None:
    with pytest.raises(ValueError, match="cpu_threads"):
        clamp_cpu_threads(0)


def test_csv_snapshot_reuses_unchanged_file_and_invalidates_on_write(
    tmp_path: Path,
) -> None:
    source = tmp_path / "account.csv"
    source.write_text("value\n1\n", encoding="utf-8")
    clear_csv_snapshot_cache()

    with patch(
        "ai_quant_trading.persistence.pd.read_csv",
        wraps=pd.read_csv,
    ) as reader:
        first = read_csv_snapshot(source)
        second = read_csv_snapshot(source)
        source.write_text("value\n22\n", encoding="utf-8")
        third = read_csv_snapshot(source)

    assert reader.call_count == 2
    assert first.iloc[0]["value"] == 1
    assert second.iloc[0]["value"] == 1
    assert third.iloc[0]["value"] == 22
