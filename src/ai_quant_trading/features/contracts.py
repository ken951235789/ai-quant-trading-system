"""模型特徵名稱的共用安全契約。"""

from __future__ import annotations

import re
from collections.abc import Iterable


_FORBIDDEN_EXACT_NAMES = frozenset(
    {
        "dataset_split",
        "label",
        "split",
        "target",
        "target_horizon",
        "target_timestamp",
        "y",
    }
)
_FORBIDDEN_TOKENS = frozenset({"future", "hindsight", "label", "target"})
_FORBIDDEN_PHRASES = (
    "max_adverse_excursion",
    "max_favorable_excursion",
    "stop_loss_hit",
    "take_profit_hit",
)


def forbidden_model_feature_reason(
    column: str,
    *,
    forbid_transformer_outputs: bool = False,
) -> str | None:
    """回傳欄位不可進入模型的原因；合法欄位回傳 ``None``。"""
    normalized = column.strip().lower()
    if not normalized:
        return "特徵名稱不可為空"
    if normalized in _FORBIDDEN_EXACT_NAMES:
        return "欄位是資料切分、標籤或預測目標"
    tokens = {token for token in re.split(r"[^a-z0-9]+", normalized) if token}
    matched = sorted(tokens.intersection(_FORBIDDEN_TOKENS))
    if matched:
        return f"欄位名稱包含事後資訊識別字：{', '.join(matched)}"
    if any(phrase in normalized for phrase in _FORBIDDEN_PHRASES):
        return "欄位是交易後才能確認的結果"
    if forbid_transformer_outputs and (
        normalized.startswith(("transformer_", "u_transformer_")) or normalized == "expected_return"
    ):
        return "Transformer 不可把既有 Transformer 輸出當成自身輸入"
    return None


def ensure_safe_model_features(
    columns: Iterable[str],
    *,
    model_name: str,
    forbid_transformer_outputs: bool = False,
) -> None:
    """在模型資料邊界拒絕標籤、未來資訊與其他事後欄位。"""
    rejected = {
        str(column): reason
        for column in columns
        if (
            reason := forbidden_model_feature_reason(
                str(column),
                forbid_transformer_outputs=forbid_transformer_outputs,
            )
        )
        is not None
    }
    if rejected:
        details = "；".join(f"{column}（{reason}）" for column, reason in rejected.items())
        raise ValueError(f"{model_name} 特徵包含禁止欄位：{details}")
