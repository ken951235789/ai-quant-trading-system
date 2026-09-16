"""建立不洩漏未來資訊的強化學習時序資料。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ai_quant_trading.features.contracts import ensure_safe_model_features
from ai_quant_trading.features.indicators import FEATURE_COLUMNS
from ai_quant_trading.reinforcement_learning.config import RLSplitConfig
from ai_quant_trading.reinforcement_learning.feature_contract import attach_expected_return, resolve_expected_return_contract


MARKET_COLUMNS = [
    "timestamp",
    "symbol",
    "exchange",
    "interval",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

# 這些欄位供撮合與硬風控使用，會原樣保存但不會混進標準化後的模型特徵。
EXECUTION_CONTEXT_COLUMNS = [
    "funding_rate",
    "spread_bps",
    "expected_return",
    "event_blackout",
    "transformer_available",
    "transformer_oos",
    "transformer_oos_fold",
    "finbert_available",
]


@dataclass(slots=True)
class PreparedRLDataset:
    """已標準化的完整資料與三個時序區段。"""

    frame: pd.DataFrame
    feature_columns: list[str]
    feature_mean: pd.Series
    feature_std: pd.Series
    train_end: int
    validation_end: int
    split_config: RLSplitConfig
    availability_coverage: dict[str, float]
    expected_return_contract: dict[str, object] = field(default_factory=dict)
    feature_contract: dict[str, object] = field(default_factory=dict)

    @property
    def train(self) -> pd.DataFrame:
        return self.frame.iloc[: self.train_end].reset_index(drop=True)

    @property
    def validation(self) -> pd.DataFrame:
        return self.frame.iloc[self.train_end : self.validation_end].reset_index(drop=True)

    @property
    def test(self) -> pd.DataFrame:
        return self.frame.iloc[self.validation_end :].reset_index(drop=True)

    @property
    def observation_size(self) -> int:
        """市場特徵加上五個帳戶與持倉狀態。"""
        return len(self.feature_columns) + 5


def prepare_rl_dataset(
    frame: pd.DataFrame,
    feature_columns: list[str] | None = None,
    split_config: RLSplitConfig | None = None,
) -> PreparedRLDataset:
    """清理、時序切分並只用訓練區段估計特徵標準化參數。"""
    selected_features = list(feature_columns or FEATURE_COLUMNS)
    expected_contract = frame.attrs.get("expected_return_contract")
    expected_contract = (
        resolve_expected_return_contract({"expected_return_contract": expected_contract})
        if expected_contract else {}
    )
    if expected_contract:
        # 即使呼叫端誤填 expected_return，也重新依同一份契約取得未標準化報酬。
        frame = attach_expected_return(frame, expected_contract)
    feature_contract = dict(frame.attrs.get("rl_feature_contract", {}))
    if feature_contract:
        feature_contract["feature_columns"] = selected_features
    split = split_config or RLSplitConfig()
    if not selected_features:
        raise ValueError("至少需要一個強化學習市場特徵")
    ensure_safe_model_features(selected_features, model_name="強化學習")
    required = [*MARKET_COLUMNS, *selected_features]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"強化學習資料缺少必要欄位：{missing}")

    execution_context = [
        column
        for column in EXECUTION_CONTEXT_COLUMNS
        if column in frame.columns and column not in required
    ]
    if expected_contract:
        source_column = str(expected_contract["source_column"])
        if source_column not in required and source_column not in execution_context:
            execution_context.append(source_column)
    result = frame[[*required, *execution_context]].copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
    numeric = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        *selected_features,
        *execution_context,
    ]
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="coerce")
    result[numeric] = result[numeric].replace([np.inf, -np.inf], np.nan)
    result = result.dropna(
        subset=["timestamp", "open", "high", "low", "close", "volume", *selected_features]
    ).sort_values("timestamp")
    result = result.reset_index(drop=True)
    if result.empty:
        raise ValueError("清理缺值後沒有可用的強化學習資料")
    if result["timestamp"].duplicated().any():
        raise ValueError("強化學習資料的 timestamp 不可重複")
    if (result[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("強化學習 OHLC 價格必須大於 0")

    availability_coverage = {
        column: float(
            pd.to_numeric(result[column], errors="coerce").fillna(0.0).clip(0.0, 1.0).mean()
        )
        for column in ("finbert_available", "transformer_available")
        if column in result
    }

    rows = len(result)
    train_end = int(rows * split.train_fraction)
    validation_end = train_end + int(rows * split.validation_fraction)
    sizes = [train_end, validation_end - train_end, rows - validation_end]
    if min(sizes) < split.min_rows_per_split:
        raise ValueError(
            "資料量不足：強化學習訓練、驗證、測試各至少需要 "
            f"{split.min_rows_per_split} 筆，目前分別為 {sizes}"
        )

    train_features = result.iloc[:train_end][selected_features]
    feature_mean = train_features.mean()
    feature_std = train_features.std(ddof=0)
    # 幾乎常數的欄位只剩浮點誤差，若直接相除會把無意義雜訊放大。
    feature_std = feature_std.mask(feature_std.abs() < 1e-12, 1.0).fillna(1.0)
    result[selected_features] = ((result[selected_features] - feature_mean) / feature_std).clip(
        -10.0, 10.0
    )
    # 多週期資料會一次加入數百個欄位；先整理連續記憶體，避免後續新增切分欄位時碎片化。
    result = result.copy()
    result["dataset_split"] = "test"
    result.loc[: train_end - 1, "dataset_split"] = "train"
    result.loc[train_end : validation_end - 1, "dataset_split"] = "validation"
    return PreparedRLDataset(
        result,
        selected_features,
        feature_mean,
        feature_std,
        train_end,
        validation_end,
        split,
        availability_coverage,
        expected_contract,
        feature_contract,
    )
