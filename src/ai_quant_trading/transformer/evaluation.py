"""以持久化真實標籤核對預測，禁止用另一組門檻重新創造測試答案。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calibration_diagnostics(frame: pd.DataFrame, horizons: tuple[int, ...]) -> dict[str, float]:
    """校準前後同樣本 Brier／NLL，包含類別比例及預測 HOLD 比例。"""
    result = {}
    for horizon in horizons:
        actual = frame[f"actual_direction_{horizon}"].to_numpy(dtype=int)
        one_hot = np.eye(3)[actual]
        for prefix, label in (("raw_", "raw"), ("", "calibrated")):
            probabilities = frame[[f"{prefix}{side}_probability_{horizon}"
                                   for side in ("down", "neutral", "up")]].to_numpy()
            result[f"{label}_direction_brier_{horizon}"] = float(
                np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))
            )
            result[f"{label}_direction_nll_{horizon}"] = float(-np.log(np.clip(
                probabilities[np.arange(len(actual)), actual], 1e-12, 1,
            )).mean())
            result[f"{label}_predicted_hold_fraction_{horizon}"] = float(
                (probabilities.argmax(axis=1) == 1).mean()
            )
            trade_column = f"{prefix}tradeability_probability_{horizon}"
            if trade_column in frame:
                truth = frame[f"actual_tradeability_{horizon}"].to_numpy()
                result[f"{label}_tradeability_brier_{horizon}"] = float(
                    np.mean((frame[trade_column].to_numpy() - truth) ** 2)
                )
        result[f"actual_hold_fraction_{horizon}"] = float((actual == 1).mean())
    return result


def validate_ensemble_labels(frames: list[pd.DataFrame], horizons: tuple[int, ...]) -> None:
    """逐列核對時間、市場、索引與實際標籤；舊檔缺標籤時明確拒絕。"""
    if not frames or any(frame.empty for frame in frames):
        raise ValueError("集成需要非空預測資料")
    identity = ["series_index", "source_key", "endpoint", "timestamp_ns", "label_contract"]
    labels = [f"actual_direction_{horizon}" for horizon in horizons]
    required = identity + labels
    for frame in frames:
        if set(required).difference(frame):
            raise ValueError("預測缺少身分或 actual_direction 真實標籤；請重新推論，不能重建答案")
        if frame[required].isna().any().any() or frame.duplicated(identity).any():
            raise ValueError("預測身分／標籤包含缺值或重複")
        if not frame[labels].isin([0, 1, 2]).all().all():
            raise ValueError("方向標籤必須為 0、1、2")
        if not frame[required].equals(frames[0][required]):
            raise ValueError("不同 seed 的測試時間／市場／標籤沒有完全對齊")
        for horizon in horizons:
            probabilities = frame[[f"{side}_probability_{horizon}"
                                   for side in ("down", "neutral", "up")]].to_numpy()
            if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
                    or (probabilities > 1).any()
                    or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5, rtol=0)):
                raise ValueError("集成機率不合法")
