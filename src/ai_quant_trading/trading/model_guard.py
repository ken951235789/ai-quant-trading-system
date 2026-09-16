"""比較即時模型輸入與訓練期標準化分布，偵測資料漂移。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class ModelInputHealth:
    """模型輸入完整性與漂移結果；嚴重漂移時只允許降風險。"""

    ready: bool
    drifted: bool
    severe_drift: bool
    checked_rows: int
    missing_columns: tuple[str, ...]
    latest_missing_columns: tuple[str, ...]
    mean_absolute_zscore: float
    maximum_absolute_zscore: float
    extreme_value_fraction: float
    drift_score: float
    risk_multiplier: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "drifted": self.drifted,
            "severe_drift": self.severe_drift,
            "checked_rows": self.checked_rows,
            "missing_columns": list(self.missing_columns),
            "latest_missing_columns": list(self.latest_missing_columns),
            "mean_absolute_zscore": self.mean_absolute_zscore,
            "maximum_absolute_zscore": self.maximum_absolute_zscore,
            "extreme_value_fraction": self.extreme_value_fraction,
            "drift_score": self.drift_score,
            "risk_multiplier": self.risk_multiplier,
            "reasons": list(self.reasons),
        }


def assess_model_input_health(
    frame: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
    means: pd.Series | dict[str, float],
    stds: pd.Series | dict[str, float],
    *,
    lookback_rows: int = 20,
    extreme_zscore: float = 5.0,
    drift_fraction: float = 0.20,
    severe_fraction: float = 0.50,
    severe_zscore: float = 50.0,
    minimum_drift_risk_multiplier: float = 0.25,
) -> ModelInputHealth:
    """以訓練期 mean/std 評估最近輸入，不用未來資料重新估計基準。"""
    if not 0 < drift_fraction < severe_fraction <= 1:
        raise ValueError("漂移比例門檻必須符合 0 < drift < severe <= 1")
    if extreme_zscore <= 0 or severe_zscore <= extreme_zscore:
        raise ValueError("Z-score 門檻必須符合 0 < extreme < severe")
    if not 0 <= minimum_drift_risk_multiplier <= 1:
        raise ValueError("最低漂移風險倍率必須介於 0 與 1")
    columns = list(feature_columns)
    missing = tuple(column for column in columns if column not in frame.columns)
    reasons: list[str] = []
    if missing or frame.empty:
        if missing:
            reasons.append(f"缺少 {len(missing)} 個模型特徵")
        if frame.empty:
            reasons.append("沒有模型輸入資料")
        return ModelInputHealth(
            False,
            False,
            False,
            0,
            missing,
            (),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            tuple(reasons),
        )
    values = frame[columns].tail(max(int(lookback_rows), 1)).apply(
        pd.to_numeric, errors="coerce"
    )
    latest_missing = tuple(column for column in columns if pd.isna(values.iloc[-1][column]))
    if latest_missing:
        reasons.append(f"最新 K 線有 {len(latest_missing)} 個模型特徵無效")
    mean_values = pd.Series(means, dtype=float).reindex(columns)
    std_values = pd.Series(stds, dtype=float).reindex(columns).replace(0.0, 1.0)
    normalization_missing = mean_values.isna() | std_values.isna()
    normalization_columns = tuple(mean_values.index[normalization_missing])
    if normalization_columns:
        reasons.append(f"缺少 {len(normalization_columns)} 個訓練期標準化參數")
    zscores = (values - mean_values) / std_values
    finite = np.isfinite(zscores.to_numpy(dtype=float))
    finite_values = np.abs(zscores.to_numpy(dtype=float)[finite])
    mean_abs = float(finite_values.mean()) if finite_values.size else 0.0
    max_abs = float(finite_values.max()) if finite_values.size else 0.0
    extreme_fraction = (
        float((finite_values > extreme_zscore).mean()) if finite_values.size else 0.0
    )
    drifted = extreme_fraction >= drift_fraction
    severe = extreme_fraction >= severe_fraction or max_abs >= severe_zscore
    drift_score = float(
        np.clip(
            (extreme_fraction - drift_fraction) / (severe_fraction - drift_fraction),
            0.0,
            1.0,
        )
    )
    if severe:
        risk_multiplier = 0.0
    elif drifted:
        risk_multiplier = float(
            1.0 - drift_score * (1.0 - minimum_drift_risk_multiplier)
        )
    else:
        risk_multiplier = 1.0
    if drifted:
        reasons.append("模型輸入與訓練期分布有明顯偏移")
    if severe:
        reasons.append("模型輸入發生嚴重漂移，只允許降低風險")
    ready = not latest_missing and not normalization_columns
    return ModelInputHealth(
        ready,
        drifted,
        severe,
        len(values),
        missing,
        latest_missing,
        mean_abs,
        max_abs,
        extreme_fraction,
        drift_score,
        risk_multiplier,
        tuple(reasons),
    )
