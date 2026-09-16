"""Transformer 訓練與推論共用的特徵完整性合約。"""

from __future__ import annotations

import re
from typing import Sequence

from ai_quant_trading.features.market_context import is_price_level_feature


OPTIONAL_FEATURE_PREFIXES = (
    "finbert_",
    "sentiment_",
    "news_",
    "macro_",
    "breadth_",
    "fundamental_",
)


def build_transformer_feature_contract(columns: Sequence[str]) -> dict[str, object]:
    """建立核心特徵與必要多週期群組，供執行期採 fail-closed 判斷。"""
    feature_columns = tuple(str(column) for column in columns)
    core = tuple(
        column
        for column in feature_columns
        if not column.lower().startswith(OPTIONAL_FEATURE_PREFIXES)
    )
    if not core:
        core = feature_columns
    timeframes = sorted(
        {
            match.group(1)
            for column in core
            if (match := re.match(r"^mtf_(\d+[mhd])_", column.lower())) is not None
        }
    )
    return {
        "schema_version": 2,
        "selection_version": "timeframe_indicator_balanced_v2",
        # 舊 checkpoint 的實際輸入不可被新版中繼資料誤標成比例化輸入。
        "raw_price_levels_allowed": any(is_price_level_feature(column) for column in feature_columns),
        "core_feature_columns": list(core),
        "optional_feature_columns": [
            column for column in feature_columns if column not in core
        ],
        "required_timeframes": timeframes,
        "minimum_core_coverage": 0.25,
    }
