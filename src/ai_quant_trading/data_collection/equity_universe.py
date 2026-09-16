"""Dashboard 與 CLI 可共用的美股研究快速清單。"""

from __future__ import annotations

from ai_quant_trading.data_collection.assets import US_EQUITY_PRESETS


def preset_symbols_text(preset: str) -> str:
    """將快速清單轉成 Dashboard 文字輸入的格式。"""
    if preset not in US_EQUITY_PRESETS:
        raise ValueError(f"未知的美股快速清單：{preset}")
    return ", ".join(US_EQUITY_PRESETS[preset])
