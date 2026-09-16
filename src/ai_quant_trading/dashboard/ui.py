"""Dashboard 共用的介面狀態工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from ai_quant_trading.persistence import write_json_atomic


THEME_PREFERENCE_FILE = Path("data/config/dashboard_preferences.json")


def theme_preference_path(project_root: str | Path) -> Path:
    """回傳可隨整套程式搬移的介面偏好設定位置。"""
    return Path(project_root).resolve() / THEME_PREFERENCE_FILE


def load_dark_mode_preference(project_root: str | Path) -> bool:
    """讀取深色模式偏好；損壞或舊版設定一律安全回到淺色。"""
    path = theme_preference_path(project_root)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return False
    return bool(payload.get("dark_mode", False)) if isinstance(payload, dict) else False


def initialize_theme_state(project_root: str | Path) -> None:
    """每個 Streamlit 工作階段只在第一次由磁碟載入主題。"""
    if "dark_mode" not in st.session_state:
        st.session_state["dark_mode"] = load_dark_mode_preference(project_root)


def save_theme_preference(project_root: str | Path) -> Path:
    """保存目前主題，供 App 關閉或電腦重開後繼續使用。"""
    return write_json_atomic(
        theme_preference_path(project_root),
        {
            "schema_version": 1,
            "dark_mode": bool(st.session_state.get("dark_mode", False)),
        },
    )


def themed_widget_key(name: str) -> str:
    """讓 Canvas 類元件在切換主題時重新掛載並重繪。"""
    theme = "dark" if st.session_state.get("dark_mode", False) else "light"
    return f"{name}__{theme}"


def themed_dataframe_data(frame: pd.DataFrame) -> pd.DataFrame | pd.io.formats.style.Styler:
    """深色模式為 Canvas 表格提供明確的儲存格前景與背景色。"""
    if not st.session_state.get("dark_mode", False):
        return frame
    return frame.style.set_properties(
        **{
            "background-color": "#1b2022",
            "color": "#eef2f1",
            "border-color": "#3a4240",
        }
    )


def themed_dataframe(frame: pd.DataFrame, **kwargs: Any) -> Any:
    """以目前主題呈現可互動資料表。"""
    return st.dataframe(themed_dataframe_data(frame), **kwargs)
