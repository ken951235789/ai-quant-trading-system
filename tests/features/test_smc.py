"""因果 SMC 特徵測試。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from ai_quant_trading.features.smc import (
    CAUSAL_SMC_FEATURE_COLUMNS,
    build_causal_smc_features,
)


def _market(rows: int = 120) -> pd.DataFrame:
    index = np.arange(rows, dtype="float64")
    close = 100 + np.sin(index / 5) * 4 + index * 0.03
    return pd.DataFrame(
        {
            "open": close - np.sin(index) * 0.2,
            "high": close + 0.8,
            "low": close - 0.8,
            "close": close,
        }
    )


def test_smc_columns_are_fixed_and_numeric() -> None:
    result = build_causal_smc_features(_market())

    assert tuple(result.columns) == CAUSAL_SMC_FEATURE_COLUMNS
    assert all(pd.api.types.is_numeric_dtype(result[column]) for column in result)


def test_future_bars_do_not_repaint_existing_smc_features() -> None:
    full = _market(140)
    prefix = full.iloc[:100].copy()

    before = build_causal_smc_features(prefix)
    after = build_causal_smc_features(full).iloc[:100]

    assert_frame_equal(before, after, check_exact=False, atol=1e-12, rtol=1e-12)


def test_bos_and_choch_are_mutually_exclusive() -> None:
    result = build_causal_smc_features(_market())

    assert not ((result["smc_bos_up"] > 0) & (result["smc_choch_up"] > 0)).any()
    assert not ((result["smc_bos_down"] > 0) & (result["smc_choch_down"] > 0)).any()
