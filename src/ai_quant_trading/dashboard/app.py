"""AI Quant Trading System 的 Streamlit 操作介面。"""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import date, timedelta
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from ai_quant_trading.dashboard.ai_pipeline_page import render_ai_pipeline_page
from ai_quant_trading.dashboard.charts import (
    make_candlestick_figure,
    make_equity_figure,
    make_indicator_figure,
)
from ai_quant_trading.dashboard.services import (
    ensure_chart_features,
    list_ohlcv_files,
    list_processed_files,
    load_market_frame,
    relative_file_label,
    summarize_market,
)
from ai_quant_trading.dashboard.runtime import get_runtime_root
from ai_quant_trading.dashboard.live_page import render_live_trading_page
from ai_quant_trading.dashboard.overview_page import render_trading_overview_page
from ai_quant_trading.dashboard.realtime_status import render_realtime_status
from ai_quant_trading.dashboard.ui import (
    initialize_theme_state,
    save_theme_preference,
    themed_dataframe,
    themed_widget_key,
)
from ai_quant_trading.data_collection.binance import BinanceApiError, INTERVAL_TO_MS
from ai_quant_trading.data_collection.collectors import (
    collect_binance_futures_data,
    collect_binance_futures_timeframe_bundle,
)
from ai_quant_trading.features.builder import (
    build_features_from_csv_batch,
    infer_annualization_periods,
)
from ai_quant_trading.features.multitimeframe import (
    BTC_MULTITIMEFRAME_INTERVALS,
    build_multitimeframe_feature_datasets,
    multitimeframe_numeric_columns,
    multitimeframe_source_intervals,
)
from ai_quant_trading.market_clock import interval_duration
from ai_quant_trading.risk import DynamicLeverageConfig, RiskConfig


PROJECT_ROOT = get_runtime_root()
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_BACKTEST_DIR = PROCESSED_DIR / "model_backtests"
RL_DIR = PROCESSED_DIR / "rl" / "environments"
TRANSFORMER_DIR = PROCESSED_DIR / "transformer"
PAPER_DIR = PROJECT_ROOT / "data" / "paper_trading"
LIVE_DIR = PROJECT_ROOT / "data" / "live_trading"
PLOTLY_CONFIG = {
    "displaylogo": False,
    "scrollZoom": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def _apply_style(dark_mode: bool) -> None:
    """依使用者選擇套用完整的深色或淺色工作介面。"""
    palette = (
        {
            "ink": "#eef2f1",
            "muted": "#aab3b0",
            "line": "#3a4240",
            "surface": "#1b2022",
            "canvas": "#121516",
            "sidebar": "#171a1c",
            "accent": "#36aa8c",
            "on_accent": "#071c16",
            "hover": "#2a3230",
            "scheme": "dark",
        }
        if dark_mode
        else {
            "ink": "#263442",
            "muted": "#6b7885",
            "line": "#dce3e8",
            "surface": "#ffffff",
            "canvas": "#f5f7f9",
            "sidebar": "#edf1f4",
            "accent": "#2f6fad",
            "on_accent": "#ffffff",
            "hover": "#e5ebf0",
            "scheme": "light",
        }
    )
    st.markdown(
        """
        <style>
        :root {
            color-scheme: %(scheme)s;
            --ink: %(ink)s;
            --muted: %(muted)s;
            --line: %(line)s;
            --surface: %(surface)s;
            --canvas: %(canvas)s;
            --sidebar: %(sidebar)s;
            --accent: %(accent)s;
            --on-accent: %(on_accent)s;
            --hover: %(hover)s;
            --text-color: %(ink)s;
            --background-color: %(canvas)s;
            --secondary-background-color: %(surface)s;
            --primary-color: %(accent)s;
            --border-color: %(line)s;
        }
        .stApp, [data-testid="stAppViewContainer"] {
            background: var(--canvas);
            color: var(--ink);
        }
        [data-testid="stHeader"] {
            background: var(--canvas);
        }
        [data-testid="stSidebar"] {
            background: var(--sidebar);
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span {
            color: var(--ink);
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: var(--muted);
        }
        .block-container {
            max-width: 1440px;
            padding-top: 4.75rem;
            padding-bottom: 3rem;
        }
        .st-key-center_navigation {
            background: var(--canvas);
            border-bottom: 1px solid var(--line);
            padding: 0.45rem 0 0.65rem;
            margin-bottom: 0.35rem;
        }
        .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp label,
        .stApp [data-testid="stWidgetLabel"] {
            color: var(--ink) !important;
            letter-spacing: 0;
        }
        .stApp [data-testid="stMarkdownContainer"],
        .stApp [data-testid="stMarkdownContainer"] li,
        .stApp [data-testid="stMarkdownContainer"] strong,
        .stApp [data-testid="stMarkdownContainer"] em,
        .stApp [data-testid="stMarkdownContainer"] blockquote,
        .stApp [data-testid="stMarkdownContainer"] table,
        .stApp [data-testid="stMarkdownContainer"] th,
        .stApp [data-testid="stMarkdownContainer"] td,
        .stApp [data-testid="stText"],
        .stApp [data-testid="stText"] * {
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink) !important;
        }
        .stApp a,
        .stApp [data-testid="stMarkdownContainer"] a,
        [data-baseweb="popover"] a,
        [role="dialog"] a {
            color: var(--accent) !important;
            -webkit-text-fill-color: var(--accent) !important;
        }
        .stApp [data-testid="stMarkdownContainer"] blockquote {
            border-left-color: var(--accent) !important;
        }
        .stApp [data-testid="stMarkdownContainer"] table,
        .stApp [data-testid="stMarkdownContainer"] th,
        .stApp [data-testid="stMarkdownContainer"] td {
            border-color: var(--line) !important;
            background: var(--surface) !important;
        }
        [data-testid="stCaptionContainer"] p {
            color: var(--muted) !important;
        }
        .stApp [data-testid="stCode"],
        .stApp [data-testid="stCodeBlock"],
        .stApp [data-testid="stCode"] > div,
        .stApp [data-testid="stCodeBlock"] > div,
        .stApp pre {
            background: %(surface)s !important;
            background-color: %(surface)s !important;
            border-color: %(line)s !important;
            color: %(ink)s !important;
        }
        .stApp [data-testid="stCode"] *,
        .stApp [data-testid="stCodeBlock"] *,
        .stApp pre code,
        .stApp code {
            color: %(ink)s !important;
            -webkit-text-fill-color: %(ink)s !important;
        }
        h1 {
            font-size: 1.75rem !important;
            line-height: 1.25 !important;
        }
        h2 {
            font-size: 1.3rem !important;
        }
        div[data-testid="stMetric"] {
            background: var(--surface);
            border-left: 3px solid var(--accent);
            padding: 0.8rem 1rem;
        }
        div[data-testid="stMetricLabel"],
        div[data-testid="stMetricLabel"] * {
            color: var(--muted) !important;
            -webkit-text-fill-color: var(--muted) !important;
        }
        div[data-testid="stMetricValue"],
        div[data-testid="stMetricValue"] * {
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink) !important;
            font-size: 1.35rem;
            line-height: 1.2;
            white-space: normal;
            overflow-wrap: anywhere;
        }
        div.stButton > button, div.stDownloadButton > button,
        button[data-testid^="stBaseButton-"] {
            background: var(--surface);
            border-color: var(--line);
            color: var(--ink) !important;
            border-radius: 4px;
            min-height: 2.55rem;
            -webkit-text-fill-color: currentColor !important;
        }
        div.stButton > button:hover, div.stDownloadButton > button:hover,
        button[data-testid^="stBaseButton-"]:hover {
            background: var(--hover);
            border-color: var(--accent);
            color: var(--ink);
        }
        .stApp button,
        .stApp [role="button"] {
            color: var(--ink) !important;
            -webkit-text-fill-color: currentColor !important;
        }
        .stApp button p,
        .stApp button span,
        .stApp button div,
        .stApp button svg,
        .stApp [role="button"] p,
        .stApp [role="button"] span {
            color: inherit !important;
            -webkit-text-fill-color: currentColor !important;
        }
        .stApp button svg {
            fill: currentColor;
            stroke: currentColor;
        }
        .stApp button:disabled,
        .stApp button:disabled p,
        .stApp button:disabled span {
            color: var(--muted) !important;
            -webkit-text-fill-color: var(--muted) !important;
            opacity: 0.6;
        }
        div.stButton > button[kind="primary"],
        button[data-testid="stBaseButton-primary"] {
            background: var(--accent) !important;
            border-color: var(--accent) !important;
            color: var(--on-accent) !important;
            -webkit-text-fill-color: var(--on-accent) !important;
        }
        div.stButton > button[kind="primary"] *,
        button[data-testid="stBaseButton-primary"] * {
            color: var(--on-accent) !important;
            -webkit-text-fill-color: var(--on-accent) !important;
        }
        div.stButton > button[kind="primary"]:hover,
        button[data-testid="stBaseButton-primary"]:hover {
            filter: brightness(0.94);
            color: var(--on-accent) !important;
        }
        button[data-variant="segmented_control"] {
            background: var(--surface) !important;
            border-color: var(--line) !important;
            color: var(--ink) !important;
        }
        [data-testid="stPills"] button,
        [data-testid="stSegmentedControl"] button,
        [data-baseweb="button-group"] button {
            background: var(--surface) !important;
            border-color: var(--line) !important;
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink) !important;
        }
        [data-testid="stPills"] button[aria-pressed="true"],
        [data-testid="stSegmentedControl"] button[aria-pressed="true"] {
            background: var(--accent) !important;
            color: var(--on-accent) !important;
            -webkit-text-fill-color: var(--on-accent) !important;
        }
        button[data-variant="segmented_control"]:hover {
            background: var(--hover) !important;
        }
        button[data-variant="segmented_control"][data-selected="true"] {
            background: var(--accent) !important;
            color: var(--on-accent) !important;
        }
        button[data-variant="segmented_control"][aria-pressed="true"],
        [data-baseweb="button-group"] button[aria-pressed="true"] {
            background: var(--accent) !important;
            color: var(--on-accent) !important;
            -webkit-text-fill-color: var(--on-accent) !important;
        }
        button[data-variant="segmented_control"][data-selected="true"] *,
        button[data-variant="segmented_control"][aria-pressed="true"] *,
        [data-baseweb="button-group"] button[aria-pressed="true"] * {
            color: var(--on-accent) !important;
            -webkit-text-fill-color: var(--on-accent) !important;
        }
        button[aria-label="Open"],
        button[aria-label="Clear all"],
        button[kind="elementToolbar"] {
            background: transparent !important;
            color: var(--ink) !important;
        }
        button[aria-label="Open"]:hover,
        button[aria-label="Clear all"]:hover,
        button[kind="elementToolbar"]:hover {
            background: var(--hover) !important;
        }
        button[data-testid="stNumberInputStepDown"],
        button[data-testid="stNumberInputStepUp"] {
            background: var(--hover) !important;
            border-color: var(--line) !important;
            color: var(--ink) !important;
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="base-input"],
        [data-testid="stSelectbox"] .react-aria-ComboBox > div,
        [data-testid="stMultiSelect"] .react-aria-ComboBox > div,
        [data-testid="stTextInputRootElement"],
        [data-testid="stTextAreaRootElement"],
        [data-testid="stDateInputRootElement"],
        [data-testid="stTimeInputRootElement"],
        textarea {
            background: var(--surface) !important;
            border-color: var(--line) !important;
            color: var(--ink) !important;
            border-radius: 4px;
        }
        div[data-baseweb="select"] span,
        div[data-baseweb="input"] input,
        div[data-baseweb="base-input"] input,
        [data-testid="stNumberInputContainer"] input,
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea,
        [data-testid="stDateInput"] input,
        [data-testid="stTimeInput"] input,
        [data-testid="stSelectbox"] [role="combobox"],
        [data-testid="stMultiSelect"] [role="combobox"],
        textarea {
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink) !important;
            caret-color: var(--accent) !important;
        }
        [data-testid="stNumberInputContainer"],
        [data-testid="stTextInput"] > div,
        [data-testid="stTextArea"] > div,
        [data-testid="stTextInputRootElement"],
        [data-testid="stTextAreaRootElement"],
        [data-testid="stDateInputRootElement"],
        [data-testid="stTimeInputRootElement"] {
            background: var(--surface) !important;
            border-color: var(--line) !important;
        }
        .stApp input::placeholder,
        .stApp textarea::placeholder {
            color: var(--muted) !important;
            -webkit-text-fill-color: var(--muted) !important;
            opacity: 1;
        }
        [data-baseweb="popover"], [role="listbox"] {
            background: var(--surface);
            color: var(--ink);
        }
        [data-baseweb="popover"] *,
        [role="listbox"] * {
            color: var(--ink);
            -webkit-text-fill-color: var(--ink);
        }
        [role="option"] {
            background: var(--surface);
            color: var(--ink);
        }
        [role="option"]:hover {
            background: var(--hover);
        }
        [role="option"][aria-selected="true"],
        [data-baseweb="menu"] [aria-selected="true"] {
            background: var(--hover) !important;
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink) !important;
        }
        [data-baseweb="tag"] {
            background: var(--hover) !important;
            border-color: var(--line) !important;
            color: var(--ink) !important;
        }
        [data-baseweb="tag"] * {
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink) !important;
        }
        button[data-baseweb="tab"] {
            color: var(--muted) !important;
            -webkit-text-fill-color: var(--muted) !important;
        }
        button[data-baseweb="tab"] * {
            color: var(--muted) !important;
            -webkit-text-fill-color: var(--muted) !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink) !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] * {
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink) !important;
        }
        [data-testid="stToggleSwitch"] span,
        [data-testid="stRadio"] label,
        [data-testid="stCheckbox"] label {
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink) !important;
        }
        [data-testid="stFileUploaderDropzone"],
        [data-testid="stFileUploaderDropzone"] section,
        [data-testid="stFileUploaderDropzone"] button,
        [data-testid="stFileUploader"] section {
            background: var(--surface) !important;
            border-color: var(--line) !important;
            color: var(--ink) !important;
        }
        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] small *,
        [data-testid="stFileUploaderDropzoneInstructions"],
        [data-testid="stFileUploaderDropzoneInstructions"] * {
            color: var(--muted) !important;
            -webkit-text-fill-color: var(--muted) !important;
        }
        [data-testid="stExpander"] details,
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary * {
            background: var(--surface) !important;
            border-color: var(--line) !important;
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink) !important;
        }
        [data-testid="stForm"],
        [data-testid="stStatusWidget"],
        [data-testid="stStatusWidget"] details,
        [data-testid="stStatusWidget"] summary {
            background: var(--surface) !important;
            border-color: var(--line) !important;
            color: var(--ink) !important;
        }
        [data-testid="stStatusWidget"] *,
        [data-testid="stProgress"] *,
        [data-testid="stSpinner"] * {
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink) !important;
        }
        .stApp [data-testid="stSliderThumbValue"],
        .stApp [data-testid="stSliderThumbValue"] * {
            color: var(--on-accent) !important;
            -webkit-text-fill-color: var(--on-accent) !important;
        }
        [data-testid="stExpander"], [data-testid="stDataFrame"] {
            background: var(--surface);
            border-color: var(--line);
        }
        [data-testid="stAlert"] p,
        [data-testid="stAlert"] span,
        [data-testid="stTooltipContent"] p,
        [data-testid="stTooltipContent"] span {
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink) !important;
        }
        [data-testid="stTooltipIcon"] button,
        [data-testid="stTooltipHoverTarget"] button {
            background: transparent !important;
            border-color: transparent !important;
            color: var(--accent) !important;
            -webkit-text-fill-color: var(--accent) !important;
        }
        [data-testid="stTooltipIcon"] button svg,
        [data-testid="stTooltipHoverTarget"] button svg {
            color: var(--accent) !important;
            fill: none !important;
            stroke: var(--accent) !important;
        }
        [data-testid="stAlert"],
        [data-testid="stToast"],
        [data-testid="stNotification"],
        [data-testid="stTooltipContent"],
        [role="tooltip"],
        [role="dialog"],
        [data-baseweb="modal"] {
            background: var(--surface) !important;
            border-color: var(--line) !important;
            color: var(--ink) !important;
        }
        [role="tooltip"] [data-testid="stTooltipContent"],
        [role="tooltip"] [data-testid="stMarkdownContainer"] {
            background: var(--surface) !important;
            border-color: var(--line) !important;
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink) !important;
            opacity: 1 !important;
            visibility: visible !important;
        }
        [data-testid="stToast"] *,
        [data-testid="stNotification"] *,
        [data-testid="stTooltipContent"] *,
        [role="tooltip"] *,
        [role="dialog"] * {
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink) !important;
        }
        [role="tooltip"] p,
        [role="tooltip"] li,
        [role="tooltip"] strong,
        [role="tooltip"] em,
        [role="tooltip"] code,
        [data-testid="stTooltipContent"] p,
        [data-testid="stTooltipContent"] li,
        [data-testid="stTooltipContent"] code {
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink) !important;
            opacity: 1 !important;
        }
        [role="tooltip"] code,
        [data-testid="stTooltipContent"] code {
            background: var(--hover) !important;
        }
        [data-testid="stToolbar"],
        [data-testid="stToolbar"] button,
        [data-testid="stMainMenu"] {
            color: var(--ink) !important;
        }
        [data-testid="stElementToolbarButtonContainer"] {
            background: var(--surface) !important;
            color: var(--ink) !important;
        }
        [data-testid="stDataFrame"] .stDataFrameGlideDataEditor {
            --gdg-text-dark: var(--ink) !important;
            --gdg-text-medium: var(--ink) !important;
            --gdg-text-light: var(--muted) !important;
            --gdg-text-bubble: var(--ink) !important;
            --gdg-bg-icon-header: var(--muted) !important;
            --gdg-text-header: var(--muted) !important;
            --gdg-text-group-header: var(--muted) !important;
            --gdg-bg-group-header: var(--sidebar) !important;
            --gdg-bg-cell: var(--surface) !important;
            --gdg-bg-cell-medium: var(--surface) !important;
            --gdg-bg-header: var(--sidebar) !important;
            --gdg-bg-bubble: var(--hover) !important;
            --gdg-border-color: var(--line) !important;
            --gdg-horizontal-border-color: var(--line) !important;
            --gdg-link-color: var(--accent) !important;
        }
        hr {
            border-color: var(--line);
        }
        [data-testid="stPlotlyChart"] {
            background: var(--surface);
            border: 1px solid var(--line);
        }
        @media (max-width: 720px) {
            .block-container {
                padding-top: 4.25rem;
                padding-left: 0.75rem;
                padding-right: 0.75rem;
            }
            .st-key-center_navigation {
                padding-top: 0.35rem;
            }
            h1 {
                font-size: 1.45rem !important;
            }
        }
        </style>
        """
        % palette,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def _cached_load_market_frame(path: str, modified_at: float) -> pd.DataFrame:
    """依檔案修改時間快取 CSV，避免每次操作都重新讀取。"""
    del modified_at
    return load_market_frame(path)


def _load_path(path: Path) -> pd.DataFrame:
    return _cached_load_market_frame(str(path), path.stat().st_mtime)


def _format_path(path: Path) -> str:
    return relative_file_label(path, PROJECT_ROOT)


def _render_header(page: str) -> None:
    st.title(page)
    st.caption("研究模式 · 實盤下單預設關閉")


@st.fragment(run_every=5)
def _render_startup_refresh_sidebar() -> None:
    """顯示 App 自動更新工作的狀態與進度。"""
    from ai_quant_trading.startup_refresh import load_startup_refresh_status

    status = load_startup_refresh_status(PROJECT_ROOT)
    state = str(status.get("status", "idle"))
    message = str(status.get("message", "尚未執行資料更新"))
    label = {
        "running": "資料更新中",
        "completed": "資料已更新",
        "partial": "部分資料未更新",
        "failed": "資料更新失敗",
        "disabled": "自動更新已停用",
        "idle": "等待資料更新",
    }.get(state, "資料更新狀態")
    st.sidebar.caption(f"{label} · {message}")
    if state == "running":
        completed = int(status.get("completed", 0) or 0)
        total = int(status.get("total", 0) or 0)
        fraction = completed / total if total > 0 else 0.0
        st.sidebar.progress(min(max(fraction, 0.0), 1.0))


def _render_sidebar() -> str:
    st.sidebar.subheader("AI Quant")
    st.sidebar.toggle(
        "深色模式",
        key="dark_mode",
        on_change=save_theme_preference,
        args=(PROJECT_ROOT,),
    )
    st.sidebar.divider()
    center = st.sidebar.radio(
        "主功能",
        ["交易中心", "模型中心", "資料中心"],
    )
    st.sidebar.divider()
    from ai_quant_trading.reinforcement_learning import list_rl_training_runs

    st.sidebar.caption(
        f"Market {len(list_ohlcv_files(RAW_DIR))} · "
        f"Features {len(list_processed_files(PROCESSED_DIR))} · "
        f"RL {len(list_rl_training_runs(RL_DIR))}"
    )
    _render_startup_refresh_sidebar()
    return center


def _is_dark_mode() -> bool:
    """回傳目前工作階段選擇的介面主題。"""
    return bool(st.session_state.get("dark_mode", False))


def _render_universal_rl_builder(market_files: list[Path]) -> None:
    """由 Raw OHLCV 或 Step 3 成品建立不跨市場跳接價格的專家環境。"""
    from ai_quant_trading.features import build_feature_dataset
    from ai_quant_trading.ai_pipeline import load_ai_pipeline_config
    from ai_quant_trading.sentiment import aggregate_finbert_features, load_scored_news
    from ai_quant_trading.transformer import (
        infer_transformer_context_frame,
        transformer_oos_provenance,
    )
    from ai_quant_trading.reinforcement_learning import (
        PortfolioEnvConfig,
        RLSplitConfig,
        UniversalPortfolioTradingEnv,
        build_expert_profile,
        prepare_universal_rl_dataset,
        run_environment_diagnostic,
        save_universal_rl_environment,
        validate_expert_interval,
    )

    expert_label = st.segmented_control(
        "交易專家",
        ["長期交易專家", "短期交易專家", "一般研究模型"],
        default="長期交易專家",
        width="stretch",
        key="universal_expert_kind",
    )
    expert_kind = {
        "長期交易專家": "long_term",
        "短期交易專家": "short_term",
        "一般研究模型": "general",
    }[str(expert_label)]
    profile = build_expert_profile(expert_kind)
    st.caption(
        f"{profile.label} · 建議 {profile.recommended_algorithm.upper()} · "
        f"適用週期 {'、'.join(profile.allowed_intervals)}"
    )

    default_paths: list[Path] = []
    default_markets: set[tuple[str, str, str]] = set()
    default_interval: str | None = None
    for path in market_files:
        try:
            first = pd.read_csv(path, nrows=1).iloc[0]
            market_key = (
                str(first.get("exchange", "")),
                str(first.get("symbol", path.stem)),
                str(first.get("interval", "")),
            )
        except (OSError, ValueError, IndexError):
            continue
        if market_key in default_markets:
            continue
        if expert_kind != "general" and market_key[2].lower() not in profile.allowed_intervals:
            continue
        if default_interval is None:
            default_interval = market_key[2].lower()
        if market_key[2].lower() != default_interval:
            continue
        default_markets.add(market_key)
        default_paths.append(path)
        if len(default_paths) >= 7:
            break

    selected_paths = st.multiselect(
        "多市場 OHLCV／特徵資料",
        market_files,
        default=default_paths,
        format_func=_format_path,
        max_selections=20,
        key=f"universal_rl_files_{expert_kind}",
    )
    ai_config = load_ai_pipeline_config(PROJECT_ROOT / "data/config/ai_pipeline.json")
    transformer_models = sorted(
        TRANSFORMER_DIR.glob("models/*/best_model.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    selected_intervals: set[str] = set()
    selected_mtf_flags: list[bool] = []
    selected_mtf_contexts: set[str] = set()
    for path in selected_paths:
        try:
            first = pd.read_csv(path, nrows=1).iloc[0]
            selected_intervals.add(str(first.get("interval", "")).lower())
            uses_mtf = "mtf_source_intervals" in first.index
            selected_mtf_flags.append(uses_mtf)
            if uses_mtf:
                selected_mtf_contexts.add(str(first.get("mtf_source_intervals", "")))
        except (OSError, ValueError, IndexError):
            continue
    selected_uses_mtf = bool(selected_mtf_flags) and all(selected_mtf_flags)
    if any(selected_mtf_flags) and not selected_uses_mtf:
        st.warning("通用 PPO 不可混用單週期與多週期市場資料。")
    elif selected_uses_mtf:
        st.success(
            "多週期 PPO：決策週期 "
            f"{'、'.join(sorted(selected_intervals))} · 觀察 "
            f"{'；'.join(sorted(selected_mtf_contexts))}"
        )

    def compatible_transformer(checkpoint: Path) -> bool:
        """只顯示使用相同 K 線週期訓練的 Transformer。"""
        try:
            summary = json.loads((checkpoint.parent / "training.json").read_text(encoding="utf-8"))
            trained_intervals = {
                str(source.get("interval", "")).lower()
                for source in summary.get("sources", [])
                if source.get("interval")
            }
            model_uses_mtf = any(
                str(column).startswith("mtf_") for column in summary.get("feature_columns", [])
            )
            train_samples = int(dict(summary.get("sample_counts", {})).get("train", 0))
            return bool(
                summary.get("status") == "complete"
                and train_samples >= 5_000
                and selected_intervals
                and selected_intervals.issubset(trained_intervals)
                and model_uses_mtf == selected_uses_mtf
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False

    transformer_models = [
        checkpoint for checkpoint in transformer_models if compatible_transformer(checkpoint)
    ]
    selected_transformer: Path | None = None
    if ai_config.ppo_use_transformer:
        selected_transformer = (
            st.selectbox(
                "Transformer 模型",
                transformer_models,
                format_func=lambda path: path.parent.name,
                key=f"universal_transformer_{expert_kind}",
                help="建立環境時會重新推論所選市場，並把模型封裝到 RL 環境內。",
            )
            if transformer_models
            else None
        )
        if selected_transformer is None:
            st.warning("AI 管線已啟用 Transformer，請先在 Transformer 訓練頁建立模型。")
    finbert_news_path = PROCESSED_DIR / "sentiment" / "finbert_news_latest.csv"
    st.caption(
        f"AI 特徵：FinBERT {'啟用' if ai_config.ppo_use_finbert else '停用'}｜"
        f"Transformer {'啟用' if ai_config.ppo_use_transformer else '停用'}"
    )
    settings = st.columns(4)
    with settings[0]:
        history_years = st.number_input(
            "最近資料年數",
            1,
            20,
            10 if expert_kind == "long_term" else 5,
            1,
            key=f"universal_years_{expert_kind}",
        )
    with settings[1]:
        train_percent = st.number_input("訓練集 %", 40, 80, 60, 5, key="universal_train")
    with settings[2]:
        validation_percent = st.number_input("驗證集 %", 10, 30, 20, 5, key="universal_validation")
    with settings[3]:
        initial_capital = st.number_input(
            "初始資金", 100.0, 100_000_000.0, 50_000.0, 1_000.0, key="universal_capital"
        )
    risk = st.columns(4)
    with risk[0]:
        max_position = st.number_input(
            "最大持倉 %",
            1,
            100,
            int(profile.environment.max_position_fraction * 100),
            1,
            key=f"universal_position_{expert_kind}",
        )
    with risk[1]:
        max_drawdown = st.number_input(
            "最大回撤 %",
            2,
            80,
            int(profile.environment.max_drawdown_limit * 100),
            1,
            key=f"universal_drawdown_{expert_kind}",
        )
    with risk[2]:
        fee_percent = st.number_input("手續費 %", 0.0, 5.0, 0.1, 0.01, key="universal_fee")
    with risk[3]:
        slippage_percent = st.number_input("滑價 %", 0.0, 5.0, 0.05, 0.01, key="universal_slippage")
    advanced = st.columns(4)
    with advanced[0]:
        episode_length = st.number_input(
            "Episode K 線",
            32,
            10_000,
            int(profile.environment.episode_length or 256),
            32,
            key=f"universal_episode_{expert_kind}",
        )
    with advanced[1]:
        holding_reference = st.number_input(
            "持有期參考 K 線",
            1,
            100,
            profile.environment.holding_period_reference,
            1,
            key=f"universal_holding_{expert_kind}",
        )
    with advanced[2]:
        drawdown_penalty = st.number_input(
            "回撤懲罰",
            0.0,
            10.0,
            profile.environment.drawdown_penalty,
            0.1,
            key=f"universal_dd_penalty_{expert_kind}",
        )
    with advanced[3]:
        turnover_penalty = st.number_input(
            "換手懲罰",
            0.0,
            1.0,
            profile.environment.turnover_penalty,
            0.001,
            key=f"universal_turnover_{expert_kind}",
        )
    with st.expander("專家風控與 Reward"):
        governor = st.columns(6)
        with governor[0]:
            deadband_percent = st.number_input(
                "調倉死區 %",
                0.0,
                50.0,
                profile.environment.rebalance_deadband * 100,
                0.5,
                key=f"universal_deadband_{expert_kind}",
            )
        with governor[1]:
            minimum_holding = st.number_input(
                "最短持有 K 線",
                0,
                500,
                profile.environment.minimum_holding_bars,
                1,
                key=f"universal_min_hold_{expert_kind}",
            )
        with governor[2]:
            soft_drawdown_percent = st.number_input(
                "軟性回撤 %",
                0.1,
                79.0,
                float(profile.environment.soft_drawdown_limit or 0.08) * 100,
                0.5,
                key=f"universal_soft_dd_{expert_kind}",
            )
        with governor[3]:
            soft_multiplier_percent = st.number_input(
                "回撤後曝險 %",
                1.0,
                100.0,
                profile.environment.soft_drawdown_multiplier * 100,
                5.0,
                key=f"universal_soft_mult_{expert_kind}",
            )
        with governor[4]:
            downside_penalty = st.number_input(
                "下行懲罰",
                0.0,
                10.0,
                profile.environment.downside_penalty,
                0.05,
                key=f"universal_downside_{expert_kind}",
            )
        with governor[5]:
            concentration_penalty = st.number_input(
                "集中度懲罰",
                0.0,
                1.0,
                profile.environment.concentration_penalty,
                0.001,
                key=f"universal_concentration_{expert_kind}",
            )
        shorting = st.columns(3)
        with shorting[0]:
            allow_short = st.toggle(
                "允許策略作空",
                value=profile.environment.allow_short,
                disabled=expert_kind == "long_term",
                key=f"universal_allow_short_{expert_kind}",
            )
        if expert_kind == "long_term":
            allow_short = False
        with shorting[1]:
            max_short_percent = st.number_input(
                "最大空頭持倉 %",
                1.0,
                100.0,
                max(profile.environment.max_short_fraction * 100, 1.0),
                1.0,
                disabled=not allow_short,
                key=f"universal_max_short_{expert_kind}",
            )
        with shorting[2]:
            short_borrow_rate_percent = st.number_input(
                "空頭年化持有成本 %",
                0.0,
                200.0,
                profile.environment.short_borrow_rate_annual * 100,
                0.5,
                disabled=not allow_short,
                key=f"universal_short_rate_{expert_kind}",
            )
        st.caption("正目標持倉代表做多、負目標持倉代表做空；空頭成本會逐根 K 線扣除。")
    test_percent = 100 - int(train_percent) - int(validation_percent)
    st.caption(
        f"每個市場各自切分 {int(train_percent)}%／{int(validation_percent)}%／"
        f"{test_percent}%；共同 mean/std 只由所有訓練段估計"
    )
    if st.button(
        "建立通用多市場環境",
        type="primary",
        icon=":material/hub:",
        width="stretch",
        key="build_universal_rl",
    ):
        try:
            if len(selected_paths) < 2:
                raise ValueError("通用模型至少要選兩個不同市場")
            if test_percent < 10:
                raise ValueError("測試集比例至少需要 10%")
            if ai_config.ppo_use_transformer and selected_transformer is None:
                raise ValueError("請先選擇可用的 Transformer 模型")
            scored_news = (
                load_scored_news(finbert_news_path) if ai_config.ppo_use_finbert else pd.DataFrame()
            )
            if ai_config.ppo_use_finbert and not finbert_news_path.exists():
                raise FileNotFoundError("找不到 FinBERT 新聞分數，請先執行 AI 管線")
            frames: dict[str, pd.DataFrame] = {}
            source_paths: dict[str, str] = {}
            intervals: set[str] = set()
            transformer_provenance: dict[str, object] = {}
            for path in selected_paths:
                frame = _load_path(path)
                if "sma_200" not in frame.columns or "short_rsi_2" not in frame.columns:
                    frame = build_feature_dataset(
                        frame,
                        target_horizon=1,
                        annualization_periods=infer_annualization_periods(frame),
                        drop_na=True,
                    )
                timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
                latest = timestamps.max()
                if pd.isna(latest):
                    raise ValueError(f"{path.name} 沒有有效 timestamp")
                frame = frame.loc[
                    timestamps >= latest - pd.DateOffset(years=int(history_years))
                ].reset_index(drop=True)
                if ai_config.ppo_use_finbert:
                    frame = aggregate_finbert_features(frame, scored_news, ai_config.finbert)
                if ai_config.ppo_use_transformer:
                    if selected_transformer is None:
                        raise RuntimeError("Transformer 已啟用但模型選擇狀態遺失")
                    frame = infer_transformer_context_frame(
                        selected_transformer,
                        frame,
                        device="auto",
                    )
                    provenance = transformer_oos_provenance(selected_transformer, frame)
                    safe_start = pd.Timestamp(str(provenance["safe_rl_start_exclusive"]))
                    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
                    frame = frame.loc[
                        (frame["timestamp"] > safe_start)
                        & (
                            pd.to_numeric(frame["transformer_available"], errors="coerce")
                            >= 0.5
                        )
                    ].reset_index(drop=True)
                    transformer_provenance[str(path)] = provenance
                first = frame.iloc[0]
                exchange = str(first.get("exchange", "unknown"))
                symbol = str(first.get("symbol", path.stem))
                interval = str(first.get("interval", "unknown"))
                market = f"{exchange}:{symbol}:{interval}"
                if market in frames:
                    raise ValueError(f"重複選到相同市場：{market}")
                frames[market] = frame
                source_paths[market] = str(path)
                intervals.add(interval)
            if len(intervals) != 1:
                raise ValueError("通用環境必須使用相同 K 線週期，避免報酬期間定義不一致")
            validate_expert_interval(expert_kind, next(iter(intervals)))
            split = RLSplitConfig(
                train_fraction=float(train_percent) / 100,
                validation_fraction=float(validation_percent) / 100,
                min_rows_per_split=20,
            )
            env_config = PortfolioEnvConfig(
                initial_capital=float(initial_capital),
                fee_rate=float(fee_percent) / 100,
                slippage_rate=float(slippage_percent) / 100,
                max_position_fraction=float(max_position) / 100,
                allow_short=bool(allow_short),
                max_short_fraction=(float(max_short_percent) / 100 if allow_short else 0.0),
                short_borrow_rate_annual=(
                    float(short_borrow_rate_percent) / 100 if allow_short else 0.0
                ),
                drawdown_penalty=float(drawdown_penalty),
                turnover_penalty=float(turnover_penalty),
                max_drawdown_limit=float(max_drawdown) / 100,
                episode_length=int(episode_length),
                random_start=True,
                holding_period_reference=int(holding_reference),
                expert_kind=expert_kind,
                rebalance_deadband=float(deadband_percent) / 100,
                minimum_holding_bars=int(minimum_holding),
                soft_drawdown_limit=float(soft_drawdown_percent) / 100,
                soft_drawdown_multiplier=float(soft_multiplier_percent) / 100,
                downside_penalty=float(downside_penalty),
                concentration_penalty=float(concentration_penalty),
            )
            with st.spinner("正在逐市場切分、共同標準化並診斷環境..."):
                dataset = prepare_universal_rl_dataset(
                    frames,
                    split,
                    expert_kind=expert_kind,
                    use_finbert=ai_config.ppo_use_finbert,
                    use_transformer=ai_config.ppo_use_transformer,
                )
                environment = UniversalPortfolioTradingEnv(
                    {name: market.train for name, market in dataset.markets.items()},
                    dataset.feature_columns,
                    env_config,
                    selection_mode="cycle",
                )
                diagnostic = run_environment_diagnostic(environment)
                paths = save_universal_rl_environment(
                    dataset,
                    env_config,
                    RL_DIR,
                    source_paths=source_paths,
                    diagnostic=diagnostic,
                    transformer_checkpoint=selected_transformer,
                    transformer_provenance=transformer_provenance,
                    finbert_scored_news_path=finbert_news_path.relative_to(PROJECT_ROOT),
                    finbert_config=asdict(ai_config.finbert),
                )
            st.session_state["universal_rl_result"] = {
                "run_dir": str(paths.run_dir),
                "markets": len(dataset.markets),
                "features": len(dataset.feature_columns),
                "expert": profile.label,
                "train_rows": sum(len(item.train) for item in dataset.markets.values()),
                "action_min": -env_config.max_short_fraction,
                "action_max": env_config.max_position_fraction,
                "diagnostic": diagnostic,
            }
            st.success("通用環境建立完成，可切換到「模型訓練」使用 PPO 或 SAC。")
        except (ValueError, OSError, ImportError, RuntimeError) as exc:
            st.error(str(exc))
    result = st.session_state.get("universal_rl_result")
    if isinstance(result, dict):
        metrics = st.columns(4)
        metrics[0].metric("市場數", result.get("markets", "-"))
        metrics[1].metric("共同特徵", result.get("features", "-"))
        metrics[2].metric("訓練樣本", f"{int(result.get('train_rows', 0)):,}")
        metrics[3].metric("觀測維度", int(result.get("features", 0)) + 5)
        st.caption(f"環境資料夾：{_format_path(Path(str(result.get('run_dir', ''))))}")
        st.caption(
            f"專家：{result.get('expert', '-')}｜動作範圍 "
            f"{float(result.get('action_min', 0)):.0%} 至 "
            f"+{float(result.get('action_max', 0)):.0%}"
        )


def _render_rl_page() -> None:
    """建立環境、訓練 PPO／SAC 並檢查樣本外結果。"""
    from ai_quant_trading.dashboard.rl_training_page import render_active_rl_job
    from ai_quant_trading.features import SHORT_TERM_FEATURE_COLUMNS
    from ai_quant_trading.reinforcement_learning.feature_contract import (
        attach_expected_return, build_expected_return_contract, compact_feature_metadata,
    )
    from ai_quant_trading.reinforcement_learning.universal import add_universal_rl_features, compact_short_term_feature_columns
    from ai_quant_trading.features.market_context import add_model_market_features
    from ai_quant_trading.reinforcement_learning import (
        PortfolioTradingEnv,
        RLSplitConfig,
        build_expert_profile,
        list_rl_environments,
        prepare_rl_dataset,
        run_environment_diagnostic,
        save_rl_environment,
    )

    _render_header("強化學習")
    phase = st.segmented_control(
        "階段",
        ["環境建構", "模型訓練", "訓練結果"],
        default="環境建構",
        width="stretch",
        key="rl_phase",
    )
    render_active_rl_job(RL_DIR, PROJECT_ROOT)
    if phase in {"模型訓練", "訓練結果"}:
        from ai_quant_trading.dashboard.rl_training_page import (
            render_rl_results,
            render_rl_training,
        )

        if phase == "模型訓練":
            render_rl_training(RL_DIR, PROJECT_ROOT)
        else:
            render_rl_results(RL_DIR, PROJECT_ROOT)
        return

    feature_files = list_processed_files(PROCESSED_DIR)
    raw_files = list_ohlcv_files(RAW_DIR)
    market_files = list(dict.fromkeys([*feature_files, *raw_files]))
    if not market_files:
        st.warning("目前沒有可用的 OHLCV 或 Step 3 feature CSV。")
        return

    if not feature_files:
        st.warning("單一市場環境需要先在「資料研究」建立 Step 3 特徵資料集。")
        return

    feature_files = [
        path
        for path in feature_files
        if path.name.startswith("features_mtf_crypto_binance_futures_BTC-USDT_15m_")
    ]
    if not feature_files:
        st.warning("請先建立 BTC 15m 永續合約多週期特徵資料。")
        return

    selected_path = st.selectbox(
        "BTC 15m 多週期特徵資料",
        feature_files,
        format_func=_format_path,
        key="rl_feature_dataset",
    )
    try:
        source_frame = add_model_market_features(_load_path(selected_path))
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        return
    timestamps = pd.to_datetime(source_frame["timestamp"], utc=True, errors="coerce")
    if pd.isna(timestamps.max()):
        st.error("特徵資料沒有有效 timestamp。")
        return
    profile = build_expert_profile("short_term", initial_capital=1_000.0)

    from ai_quant_trading.ai_pipeline import load_ai_pipeline_config
    from ai_quant_trading.sentiment import FINBERT_CONTEXT_COLUMNS
    from ai_quant_trading.transformer import (
        infer_transformer_context_frame,
        transformer_oos_provenance,
    )

    ai_config = load_ai_pipeline_config(PROJECT_ROOT / "data" / "config" / "ai_pipeline.json")
    transformer_models = sorted(
        TRANSFORMER_DIR.glob("models/*/best_model.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    def compatible_btc_transformer(checkpoint: Path) -> bool:
        try:
            summary = json.loads((checkpoint.parent / "training.json").read_text(encoding="utf-8"))
            trained_intervals = {
                str(source.get("interval", "")).lower() for source in summary.get("sources", [])
            }
            trained_features = {str(column) for column in summary.get("feature_columns", [])}
            return bool(
                summary.get("status") == "complete"
                and "15m" in trained_intervals
                and trained_features.issubset(source_frame.columns)
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False

    transformer_models = [path for path in transformer_models if compatible_btc_transformer(path)]
    selected_transformer: Path | None = None
    if ai_config.ppo_use_transformer:
        selected_transformer = (
            st.selectbox(
                "Transformer 時序模型",
                transformer_models,
                format_func=lambda path: path.parent.name,
                key="btc_rl_transformer_model",
                help="建立環境時會重跑整段歷史推論，SAC 與模擬交易使用同一份模型。",
            )
            if transformer_models
            else None
        )
        if selected_transformer is None:
            st.warning("尚無相容的 BTC 15m 多週期 Transformer，請先完成 Transformer 訓練。")
    finbert_news_path = PROCESSED_DIR / "sentiment" / "finbert_news_latest.csv"
    use_finbert = bool(
        ai_config.ppo_use_finbert
        and finbert_news_path.exists()
        and set(FINBERT_CONTEXT_COLUMNS).issubset(source_frame.columns)
    )
    multitimeframe_features = multitimeframe_numeric_columns(source_frame)
    available_features = compact_short_term_feature_columns(
        source_frame,
        use_transformer=selected_transformer is not None, use_finbert=use_finbert,
    )
    source_intervals = set(multitimeframe_source_intervals(source_frame))
    missing_btc_intervals = [
        value for value in BTC_MULTITIMEFRAME_INTERVALS if value not in source_intervals
    ]
    short_term_available = [
        column for column in SHORT_TERM_FEATURE_COLUMNS if column in source_frame.columns
    ]
    overview = st.columns(5)
    overview[0].metric("標的", str(source_frame.iloc[0].get("symbol", "-")))
    overview[1].metric("K 線週期", str(source_frame.iloc[0].get("interval", "-")))
    overview[2].metric("資料筆數", f"{len(source_frame):,}")
    overview[3].metric("RL 特徵", str(len(available_features)))
    overview[4].metric(
        "短線特徵",
        f"{len(short_term_available)}/{len(SHORT_TERM_FEATURE_COLUMNS)}",
    )
    if source_intervals:
        if missing_btc_intervals:
            st.warning("這份多週期資料缺少：" + "、".join(missing_btc_intervals))
        else:
            st.success(
                f"BTC 15m 多週期 RL · {len(source_intervals)}/"
                f"{len(BTC_MULTITIMEFRAME_INTERVALS)} 週期 · "
                f"{len(multitimeframe_features)} 個多週期指標"
            )
    if len(short_term_available) < len(SHORT_TERM_FEATURE_COLUMNS):
        st.warning("這份特徵資料尚未包含完整短線欄位，請在「資料研究」重新建立特徵資料集。")

    if "rl_feature_columns" in st.session_state:
        retained = [value for value in st.session_state["rl_feature_columns"] if value in available_features]
        st.session_state["rl_feature_columns"] = retained or available_features
    with st.expander("狀態特徵（精簡版）"):
        selected_features = st.multiselect(
            "狀態特徵",
            options=available_features,
            default=available_features,
            key="rl_feature_columns",
            label_visibility="collapsed",
        )
    split_columns = st.columns(4)
    with split_columns[0]:
        train_percent = st.number_input("訓練集 %", 40, 80, 60, 5, key="rl_train_percent")
    with split_columns[1]:
        validation_percent = st.number_input("驗證集 %", 10, 30, 20, 5, key="rl_validation_percent")
    with split_columns[2]:
        max_position_percent = st.number_input(
            "最大名目部位 %", 10, 50, 50, 5, key="rl_max_position"
        )
    with split_columns[3]:
        max_drawdown_percent = st.number_input("最大回撤 %", 5, 20, 10, 1, key="rl_max_drawdown")

    with st.expander("環境設定"):
        environment_columns = st.columns(5)
        with environment_columns[0]:
            initial_capital = st.number_input(
                "初始資金", min_value=100.0, value=1000.0, step=100.0, key="rl_capital"
            )
        with environment_columns[1]:
            fee_percent = st.number_input("Taker 手續費 %", 0.0, 1.0, 0.05, 0.01, key="rl_fee")
        with environment_columns[2]:
            slippage_percent = st.number_input("滑價 %", 0.0, 5.0, 0.05, 0.01, key="rl_slippage")
        with environment_columns[3]:
            drawdown_penalty = st.number_input(
                "回撤懲罰",
                0.0,
                10.0,
                profile.environment.drawdown_penalty,
                0.1,
                key="rl_drawdown_penalty",
            )
        with environment_columns[4]:
            turnover_penalty = st.number_input(
                "換手懲罰",
                0.0,
                1.0,
                profile.environment.turnover_penalty,
                0.001,
                key="rl_turnover_penalty",
            )
        episode_columns = st.columns(4)
        with episode_columns[0]:
            reward_scale = st.number_input(
                "Reward scale", 0.01, 1_000.0, 1.0, 1.0, key="rl_reward_scale"
            )
        with episode_columns[1]:
            episode_length = st.number_input(
                "Episode K 線（0=完整）",
                0,
                max(len(source_frame) - 1, 1),
                min(2_880, max(len(source_frame) - 1, 1)),
                96,
                key="rl_episode_length",
            )
        with episode_columns[2]:
            random_start = st.toggle(
                "隨機 Episode 起點",
                value=True,
                disabled=int(episode_length) == 0,
                key="rl_random_start",
            )
        with episode_columns[3]:
            holding_period_reference = st.number_input(
                "持有期參考 K 線",
                1,
                100,
                profile.environment.holding_period_reference,
                1,
                key="rl_holding_reference",
            )
        short_columns = st.columns(3)
        with short_columns[0]:
            allow_short = st.toggle(
                "允許策略作空",
                value=True,
                key="rl_allow_short",
            )
        with short_columns[1]:
            max_short_percent = st.number_input(
                "最大空頭持倉 %",
                1.0,
                100.0,
                50.0,
                5.0,
                disabled=not allow_short,
                key="rl_max_short",
            )
        with short_columns[2]:
            leverage = st.number_input(
                "逐倉槓桿",
                1.0,
                3.0,
                profile.environment.leverage,
                0.5,
                key="rl_leverage",
            )
        hard_risk_columns = st.columns(4)
        with hard_risk_columns[0]:
            risk_per_trade_percent = st.number_input(
                "單筆風險 %", 0.05, 0.5, 0.25, 0.05, key="rl_risk_per_trade"
            )
        with hard_risk_columns[1]:
            max_margin_percent = st.number_input(
                "最大保證金 %", 5.0, 20.0, 20.0, 1.0, key="rl_max_margin"
            )
        with hard_risk_columns[2]:
            daily_loss_percent = st.number_input(
                "每日停損 %", 0.25, 2.0, 1.0, 0.25, key="rl_daily_loss"
            )
        with hard_risk_columns[3]:
            consecutive_losses = st.number_input(
                "連虧停機筆數", 1, 10, 3, 1, key="rl_consecutive_losses"
            )
        training_randomization = st.columns(5)
        with training_randomization[0]:
            neutral_action_percent = st.number_input(
                "SAC 中性區 %",
                0.0,
                30.0,
                profile.environment.neutral_action_threshold * 100,
                1.0,
                key="rl_neutral_action",
            )
        with training_randomization[1]:
            take_profit_percent = st.number_input(
                "固定停利 %",
                0.1,
                5.0,
                (profile.environment.take_profit_distance or 0.015) * 100,
                0.1,
                key="rl_take_profit",
            )
        with training_randomization[2]:
            minimum_net_risk_reward = st.number_input(
                "最低淨風報比",
                0.0,
                5.0,
                profile.environment.minimum_net_risk_reward,
                0.1,
                key="rl_net_rr",
            )
        with training_randomization[3]:
            capital_randomization_percent = st.number_input(
                "訓練本金隨機 ±%",
                0.0,
                50.0,
                profile.environment.initial_capital_randomization * 100,
                1.0,
                key="rl_capital_randomization",
            )
        with training_randomization[4]:
            slippage_randomization_percent = st.number_input(
                "訓練滑價隨機 ±%",
                0.0,
                0.10,
                profile.environment.slippage_randomization * 100,
                0.005,
                format="%.3f",
                key="rl_slippage_randomization",
            )

    test_percent = 100 - int(train_percent) - int(validation_percent)
    st.caption(
        f"時間切分 {int(train_percent)}%／{int(validation_percent)}%／{test_percent}% · "
        "本期收盤決策，下一根開盤成交"
    )
    if st.button(
        "建立強化學習環境",
        type="primary",
        icon=":material/account_tree:",
        width="stretch",
    ):
        try:
            if test_percent < 10:
                raise ValueError("測試集比例至少需要 10%")
            if not selected_features:
                raise ValueError("至少選擇一個狀態特徵")
            if ai_config.ppo_use_transformer and selected_transformer is None:
                raise ValueError("AI 管線已啟用 Transformer，請先訓練並選擇相容模型")
            split_config = RLSplitConfig(
                train_fraction=float(train_percent) / 100,
                validation_fraction=float(validation_percent) / 100,
                min_rows_per_split=20,
            )
            env_config = replace(
                profile.environment,
                initial_capital=float(initial_capital),
                fee_rate=float(fee_percent) / 100,
                slippage_rate=float(slippage_percent) / 100,
                max_position_fraction=float(max_position_percent) / 100,
                allow_short=bool(allow_short),
                max_short_fraction=(float(max_short_percent) / 100 if allow_short else 0.0),
                drawdown_penalty=float(drawdown_penalty),
                turnover_penalty=float(turnover_penalty),
                max_drawdown_limit=float(max_drawdown_percent) / 100,
                reward_scale=float(reward_scale),
                episode_length=(int(episode_length) if int(episode_length) > 0 else None),
                random_start=bool(random_start) if int(episode_length) > 0 else False,
                holding_period_reference=int(holding_period_reference),
                leverage=float(leverage),
                risk_per_trade=float(risk_per_trade_percent) / 100,
                max_margin_fraction=float(max_margin_percent) / 100,
                daily_loss_limit=float(daily_loss_percent) / 100,
                max_consecutive_losses=int(consecutive_losses),
                neutral_action_threshold=float(neutral_action_percent) / 100,
                take_profit_distance=float(take_profit_percent) / 100,
                minimum_net_risk_reward=float(minimum_net_risk_reward),
                initial_capital_randomization=(float(capital_randomization_percent) / 100),
                slippage_randomization=(float(slippage_randomization_percent) / 100),
            )
            with st.spinner("正在推論 Transformer、切分資料並檢查交易環境..."):
                training_frame = source_frame.copy()
                transformer_provenance: dict[str, object] | None = None
                if selected_transformer is not None:
                    transformer_provenance = transformer_oos_provenance(
                        selected_transformer,
                        training_frame,
                    )
                    training_frame = infer_transformer_context_frame(
                        selected_transformer,
                        training_frame,
                        device="auto",
                    )
                    training_frame = training_frame.copy()
                    safe_start = pd.Timestamp(
                        str(transformer_provenance["safe_rl_start_exclusive"])
                    )
                    training_frame["timestamp"] = pd.to_datetime(
                        training_frame["timestamp"], utc=True, errors="coerce"
                    )
                    training_frame = training_frame.loc[
                        (training_frame["timestamp"] > safe_start)
                        & (
                            pd.to_numeric(
                                training_frame["transformer_available"], errors="coerce"
                            )
                            >= 0.5
                        )
                    ].reset_index(drop=True)
                    training_frame = attach_expected_return(
                        training_frame, build_expected_return_contract(5),
                    )
                training_frame = add_universal_rl_features(training_frame, feature_columns=selected_features)
                training_frame.attrs["rl_feature_contract"] = compact_feature_metadata(selected_features)
                dataset = prepare_rl_dataset(
                    training_frame,
                    selected_features,
                    split_config,
                )
                environment = PortfolioTradingEnv(
                    dataset.train,
                    dataset.feature_columns,
                    env_config,
                )
                diagnostic = run_environment_diagnostic(environment)
                paths = save_rl_environment(
                    dataset,
                    env_config,
                    RL_DIR,
                    source_path=selected_path,
                    diagnostic=diagnostic,
                    transformer_checkpoint=selected_transformer,
                    transformer_provenance=transformer_provenance,
                    finbert_scored_news_path=(finbert_news_path if use_finbert else None),
                    finbert_config=(asdict(ai_config.finbert) if use_finbert else None),
                )
            st.session_state["rl_environment_result"] = {
                "run_dir": str(paths.run_dir),
                "observation_size": int(environment.observation_space.shape[0]),
                "action_max": float(environment.action_space.high[0]),
                "action_min": float(environment.action_space.low[0]),
                "split_rows": [len(dataset.train), len(dataset.validation), len(dataset.test)],
                "diagnostic": diagnostic,
            }
            st.success("環境建立完成，可切換到「模型訓練」開始 PPO／SAC。")
        except (ValueError, OSError, ImportError) as exc:
            st.error(str(exc))

    result = st.session_state.get("rl_environment_result")
    if isinstance(result, dict):
        diagnostic = result.get("diagnostic")
        metrics = st.columns(4)
        metrics[0].metric("觀測維度", str(result.get("observation_size", "-")))
        metrics[1].metric(
            "動作範圍",
            f"{float(result.get('action_min', 0)):.0%} 至 "
            f"+{float(result.get('action_max', 0)):.0%}",
        )
        metrics[2].metric(
            "資料切分",
            "／".join(f"{int(rows):,}" for rows in result.get("split_rows", [])),
        )
        metrics[3].metric(
            "診斷步數", str(len(diagnostic)) if isinstance(diagnostic, pd.DataFrame) else "-"
        )
        if isinstance(diagnostic, pd.DataFrame) and not diagnostic.empty:
            chart_data = diagnostic[["timestamp", "equity"]].copy()
            chart_data["timestamp"] = pd.to_datetime(chart_data["timestamp"], utc=True)
            st.line_chart(chart_data.set_index("timestamp"), height=240)
        st.caption(f"環境資料夾：{_format_path(Path(str(result.get('run_dir', ''))))}")

    recent_rows = []
    for run_dir in list_rl_environments(RL_DIR)[:5]:
        try:
            payload = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
            source = payload.get("source", {})
            split_rows = payload.get("split_rows", {})
            recent_rows.append(
                {
                    "標的": source.get("symbol", "-"),
                    "週期": source.get("interval", "-"),
                    "訓練／驗證／測試": (
                        f"{split_rows.get('train', 0)}／{split_rows.get('validation', 0)}／"
                        f"{split_rows.get('test', 0)}"
                    ),
                    "狀態": "環境就緒" if not payload.get("training_started") else "已訓練",
                    "建立時間": str(payload.get("created_at", ""))[:19].replace("T", " "),
                }
            )
        except (OSError, ValueError, TypeError):
            continue
    if recent_rows:
        st.subheader("最近建立")
        themed_dataframe(
            pd.DataFrame(recent_rows),
            width="stretch",
            hide_index=True,
            key=themed_widget_key("download_recent_created"),
        )


def _render_startup_refresh_settings() -> None:
    """調整每次開啟桌面 App 時的自動更新內容。"""
    from ai_quant_trading.startup_refresh import (
        StartupRefreshConfig,
        load_startup_refresh_config,
        load_startup_refresh_status,
        save_startup_refresh_config,
    )

    saved = load_startup_refresh_config(PROJECT_ROOT)
    status = load_startup_refresh_status(PROJECT_ROOT)
    with st.expander("自動更新設定", expanded=False):
        enabled = st.toggle("每次開啟 App 自動更新", value=saved.enabled)
        options = st.columns(4)
        refresh_market = options[0].toggle("市場資料", value=saved.refresh_market_data)
        collect_news = options[1].toggle("新聞資料", value=saved.collect_news)
        score_finbert = options[2].toggle("FinBERT 評分", value=saved.score_finbert)
        refresh_features = options[3].toggle("特徵資料", value=saved.refresh_features)
        enrich_finbert = st.toggle(
            "情緒附加到特徵",
            value=saved.enrich_features_with_finbert,
        )
        config = StartupRefreshConfig(
            enabled=bool(enabled),
            refresh_market_data=bool(refresh_market),
            collect_news=bool(collect_news),
            score_finbert=bool(score_finbert),
            refresh_features=bool(refresh_features),
            enrich_features_with_finbert=bool(enrich_finbert),
            yahoo_news_per_symbol=saved.yahoo_news_per_symbol,
            news_retention_days=saved.news_retention_days,
            news_max_rows=saved.news_max_rows,
        )
        if config != saved:
            save_startup_refresh_config(PROJECT_ROOT, config)
            st.caption("設定已自動儲存，會在下次開啟 App 時套用。")
        summary = status.get("summary", {})
        if isinstance(summary, dict) and summary:
            st.caption(
                f"最近更新：市場 {summary.get('market_updated', 0)} · "
                f"新聞 {summary.get('news_collected', 0)} · "
                f"FinBERT {summary.get('news_scored', 0)} · "
                f"特徵 {summary.get('features_updated', 0)} · "
                f"多週期 {summary.get('multitimeframe_updated', 0)}"
            )
        errors = status.get("errors", [])
        if isinstance(errors, list) and errors:
            with st.expander("最近更新錯誤"):
                st.code("\n".join(str(item) for item in errors[-20:]), language=None)


def _render_download_page() -> None:
    _render_header("市場資料")
    _render_startup_refresh_settings()

    symbols = ["BTC/USDT"]
    st.caption("Binance USDⓈ-M BTC/USDT 永續合約 · 研究資料與即時交易共用")
    download_mode = st.segmented_control(
        "下載方式",
        options=["單一週期", "BTC 15m 多週期"],
        default="BTC 15m 多週期",
        key="btc_download_mode",
        width="stretch",
    )
    use_btc_bundle = download_mode == "BTC 15m 多週期"

    first_column, second_column = st.columns([2, 1])
    with first_column:
        st.text_input("交易標的", value="BTC/USDT", disabled=True)
        if use_btc_bundle:
            st.caption("BTC 短線專用資料包")
    with second_column:
        intervals = sorted(INTERVAL_TO_MS, key=INTERVAL_TO_MS.get)
        if use_btc_bundle:
            interval = "15m"
            st.text_area(
                "K 線週期",
                value=" · ".join(BTC_MULTITIMEFRAME_INTERVALS),
                height=82,
                disabled=True,
            )
        else:
            interval = st.selectbox(
                "K 線週期",
                intervals,
                index=intervals.index("15m"),
                key="btc_download_interval",
            )

    range_mode = st.segmented_control(
        "資料範圍",
        options=["最近資料", "日期範圍"],
        default="最近資料",
    )
    start: str | None = None
    end: str | None = None

    range_column, option_column = st.columns([1.5, 1])
    with range_column:
        if range_mode == "日期範圍":
            date_columns = st.columns(2)
            with date_columns[0]:
                start_date = st.date_input("開始日期", value=date.today() - timedelta(days=365))
            with date_columns[1]:
                end_date = st.date_input("結束日期", value=date.today())
            start = start_date.isoformat()
            end = end_date.isoformat()
        else:
            limit = st.number_input(
                "K 線筆數",
                min_value=20,
                max_value=1500,
                value=1000,
                step=10,
            )
    with option_column:
        if range_mode == "日期範圍":
            limit = st.number_input(
                "每批下載筆數",
                min_value=20,
                max_value=1500,
                value=1500,
                step=20,
            )
        include_derivatives_context = st.toggle(
            "Funding／OI／Spread",
            value=True,
            help="下載永續合約公開脈絡；OI 歷史受交易所可用期間限制。",
        )

    if symbols:
        themed_dataframe(
            pd.DataFrame(
                {
                    "代碼": symbols,
                    "名稱": ["Bitcoin"],
                }
            ),
            width="stretch",
            hide_index=True,
            key=themed_widget_key("download_selected_symbols"),
        )
    if st.button(
        "下載 BTC 15m 多週期資料" if use_btc_bundle else "下載資料",
        type="primary",
        icon=":material/download:",
        width="stretch",
    ):
        if not symbols:
            st.error("請至少選擇一個交易標的。")
            return
        if start and end and start > end:
            st.error("開始日期不可晚於結束日期。")
            return

        try:
            with st.spinner("正在取得市場資料..."):
                if use_btc_bundle:
                    progress = st.progress(0.0, text="準備下載 BTC 15m 多週期資料")

                    def update_bundle_progress(
                        current_interval: str,
                        completed: int,
                        total: int,
                    ) -> None:
                        progress.progress(
                            completed / total,
                            text=f"下載 {current_interval} · {completed}/{total}",
                        )

                    result = collect_binance_futures_timeframe_bundle(
                        symbols=symbols,
                        raw_dir=RAW_DIR,
                        start=start,
                        end=end,
                        limit=int(limit),
                        include_derivatives_context=include_derivatives_context,
                        progress_callback=update_bundle_progress,
                    )
                else:
                    result = collect_binance_futures_data(
                        symbols=symbols,
                        interval=str(interval),
                        raw_dir=RAW_DIR,
                        start=start,
                        end=end,
                        limit=int(limit),
                        include_derivatives_context=include_derivatives_context,
                    )
            _cached_load_market_frame.clear()
            st.success(f"完成 {len(result.ohlcv_files)} 個 OHLCV 檔案。")
            for output_path in result.ohlcv_files:
                st.code(_format_path(output_path), language=None)
            if result.derivatives_context_files:
                st.caption(
                    f"另更新 {len(result.derivatives_context_files)} 份 Funding／OI／Spread 脈絡。"
                )
        except (BinanceApiError, ValueError, ImportError) as exc:
            st.error(str(exc))

    st.divider()
    recent_files = list_ohlcv_files(RAW_DIR)[:8]
    st.subheader("最近資料")
    if recent_files:
        recent_table = pd.DataFrame(
            {
                "檔案": [_format_path(path) for path in recent_files],
                "大小 KB": [round(path.stat().st_size / 1024, 1) for path in recent_files],
                "更新時間": [
                    pd.Timestamp(path.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M")
                    for path in recent_files
                ],
            }
        )
        themed_dataframe(
            recent_table,
            width="stretch",
            hide_index=True,
            key=themed_widget_key("download_local_files"),
        )
    else:
        st.info("目前沒有 OHLCV CSV。")


def _render_multitimeframe_builder() -> None:
    """用本機既有特徵檔建立 Transformer／SAC 共用的多週期資料集。"""
    candidates = [
        path
        for path in list_processed_files(PROCESSED_DIR)
        if path.name.startswith("features_")
        and not path.name.startswith("features_mtf_")
        and "transformer_" not in path.name
    ]
    with st.expander("多週期 Transformer／SAC 資料集", expanded=False):
        st.caption(
            "BTC 15m 標準組合為 5m、15m、1h、4h、1d。系統只合併各決策時間以前"
            "已收盤的 K 線。決策週期保留 OHLCV 與成交時序，其他週期作為模型觀察。"
        )
        if not candidates:
            st.info("請先為至少一個標的建立兩種以上 K 線週期的 Step 3 特徵資料。")
            return

        identities: dict[Path, tuple[str, str, str]] = {}
        grouped: dict[tuple[str, str], list[Path]] = {}
        for path in candidates:
            try:
                first = pd.read_csv(path, nrows=1).iloc[0]
                identity = (
                    str(first.get("exchange", "unknown")),
                    str(first.get("symbol", path.stem)),
                    str(first.get("interval", "")),
                )
            except (OSError, ValueError, IndexError):
                continue
            identities[path] = identity
            grouped.setdefault((identity[0], identity[1]), []).append(path)
        default_group = max(grouped.values(), key=len, default=[])
        selected = st.multiselect(
            "多週期來源檔案",
            list(identities),
            default=default_group if len(default_group) >= 2 else [],
            format_func=lambda path: (
                f"{identities[path][0]} · {identities[path][1]} · "
                f"{identities[path][2]} · {path.name}"
            ),
            max_selections=50,
            key="multitimeframe_feature_sources",
        )
        intervals = sorted(
            {identities[path][2].lower() for path in selected if identities[path][2]},
            key=lambda value: interval_duration(value) or pd.Timedelta.max,
        )
        if not intervals:
            intervals = ["15m"]
        decision_interval = st.selectbox(
            "SAC 決策週期",
            intervals,
            index=intervals.index("15m") if "15m" in intervals else 0,
            key="multitimeframe_decision_interval",
            help="模型在此週期收盤後做決策；例如 15m 決策時仍可參考 1m、5m、1h、1d。",
        )
        if st.button(
            "建立多週期資料集",
            type="primary",
            icon=":material/account_tree:",
            width="stretch",
            key="build_multitimeframe_features",
        ):
            try:
                artifacts = build_multitimeframe_feature_datasets(
                    selected,
                    str(decision_interval),
                    PROCESSED_DIR,
                )
                _cached_load_market_frame.clear()
                st.session_state["latest_multitimeframe_paths"] = [
                    str(artifact.output_path) for artifact in artifacts
                ]
                st.success(f"已建立 {len(artifacts)} 個多週期市場資料集。")
                themed_dataframe(
                    pd.DataFrame(
                        {
                            "市場": [f"{item.exchange}:{item.symbol}" for item in artifacts],
                            "決策週期": [item.decision_interval for item in artifacts],
                            "觀察週期": ["、".join(item.source_intervals) for item in artifacts],
                            "K 線": [item.rows for item in artifacts],
                            "多週期欄位": [item.feature_columns for item in artifacts],
                            "最低覆蓋": [
                                f"{min(item.coverage.values(), default=0.0):.1%}"
                                for item in artifacts
                            ],
                            "輸出": [_format_path(item.output_path) for item in artifacts],
                        }
                    ),
                    hide_index=True,
                    width="stretch",
                    key=themed_widget_key("multitimeframe_created_files"),
                )
            except (FileNotFoundError, ValueError, OSError) as exc:
                st.error(str(exc))


def _render_feature_page() -> None:
    _render_header("特徵工程")
    symbols = ["BTC/USDT"]
    st.caption("固定使用 Binance USDⓈ-M BTC/USDT，避免模型混入不同市場分布。")
    feature_mode = st.segmented_control(
        "建立方式",
        options=["單一週期", "BTC 15m 多週期"],
        default="BTC 15m 多週期",
        key="btc_feature_mode",
        width="stretch",
    )
    use_btc_bundle = feature_mode == "BTC 15m 多週期"
    selector_columns = st.columns([2, 1, 1])
    with selector_columns[0]:
        st.text_input("交易標的", value="BTC/USDT", disabled=True)
        if use_btc_bundle:
            st.caption("觀察週期：" + "、".join(BTC_MULTITIMEFRAME_INTERVALS))
    with selector_columns[1]:
        intervals = sorted(INTERVAL_TO_MS, key=INTERVAL_TO_MS.get)
        if use_btc_bundle:
            interval = st.selectbox(
                "決策週期",
                BTC_MULTITIMEFRAME_INTERVALS,
                index=BTC_MULTITIMEFRAME_INTERVALS.index("15m"),
                key="btc_bundle_decision_interval",
                help="每隔多久讓 Transformer／SAC 產生一次決策。",
            )
        else:
            interval = st.selectbox(
                "K 線週期",
                intervals,
                index=intervals.index("15m"),
                key="btc_feature_interval",
            )
    with selector_columns[2]:
        if use_btc_bundle:
            history_days = st.number_input(
                "決策資料天數",
                min_value=7,
                max_value=730,
                value=90,
                step=7,
                help="慢週期會自動額外下載至少 250 根作為指標暖機。",
            )
            download_limit = 1500
        else:
            history_days = 0
            download_limit = st.number_input(
                "自動下載 K 線",
                min_value=220,
                max_value=1500,
                value=500,
                step=20,
            )

    control_columns = st.columns(3)
    with control_columns[0]:
        target_horizon = st.number_input(
            "預測期數 N",
            min_value=1,
            max_value=100,
            value=1,
            step=1,
        )
    with control_columns[1]:
        threshold_percent = st.number_input(
            "上漲門檻 %",
            min_value=-20.0,
            max_value=20.0,
            value=0.0,
            step=0.1,
        )
    with control_columns[2]:
        drop_na = st.toggle("移除暖機期缺值", value=True)

    if st.button(
        "建立 BTC 15m 多週期訓練資料" if use_btc_bundle else "自動下載並建立特徵資料集",
        type="primary",
        icon=":material/manufacturing:",
        width="stretch",
    ):
        try:
            if not symbols:
                raise ValueError("請至少選擇一個交易標的")
            minimum_bars = 200 + int(target_horizon)
            if not use_btc_bundle and int(download_limit) < minimum_bars:
                raise ValueError(f"至少需要下載 {minimum_bars} 根 K 線")
            with st.spinner("正在下載資料並計算技術指標..."):
                if use_btc_bundle:
                    progress = st.progress(0.0, text="準備下載 BTC 15m 多週期資料")

                    def update_feature_bundle_progress(
                        current_interval: str,
                        completed: int,
                        total: int,
                    ) -> None:
                        progress.progress(
                            completed / total,
                            text=f"下載 {current_interval} · {completed}/{total}",
                        )

                    collection = collect_binance_futures_timeframe_bundle(
                        symbols=[str(symbol) for symbol in symbols],
                        raw_dir=RAW_DIR,
                        start=(date.today() - timedelta(days=int(history_days))).isoformat(),
                        limit=int(download_limit),
                        include_derivatives_context=True,
                        progress_callback=update_feature_bundle_progress,
                    )
                else:
                    collection = collect_binance_futures_data(
                        symbols=[str(symbol) for symbol in symbols],
                        interval=str(interval),
                        raw_dir=RAW_DIR,
                        limit=int(download_limit),
                        include_derivatives_context=True,
                    )
                artifacts = build_features_from_csv_batch(
                    collection.ohlcv_files,
                    output_dir=PROCESSED_DIR,
                    target_horizon=int(target_horizon),
                    target_threshold=float(threshold_percent) / 100,
                    drop_na=drop_na,
                )
                multitimeframe_artifacts = (
                    build_multitimeframe_feature_datasets(
                        [artifact.output_path for artifact in artifacts],
                        str(interval),
                        PROCESSED_DIR,
                    )
                    if use_btc_bundle
                    else ()
                )
            _cached_load_market_frame.clear()
            st.session_state["latest_feature_paths"] = [
                str(artifact.output_path) for artifact in artifacts
            ]
            if multitimeframe_artifacts:
                st.session_state["latest_multitimeframe_paths"] = [
                    str(artifact.output_path) for artifact in multitimeframe_artifacts
                ]
            st.success(
                f"完成 {len(artifacts)} 個標的、"
                f"共 {sum(artifact.rows for artifact in artifacts):,} 筆特徵資料。"
                + (
                    f" 已建立決策週期 {interval} 的多週期訓練集。"
                    if multitimeframe_artifacts
                    else ""
                )
            )
            themed_dataframe(
                pd.DataFrame(
                    {
                        "Raw 資料": [_format_path(artifact.input_path) for artifact in artifacts],
                        "特徵資料": [_format_path(artifact.output_path) for artifact in artifacts],
                        "輸出筆數": [artifact.rows for artifact in artifacts],
                    }
                ),
                hide_index=True,
                width="stretch",
                key=themed_widget_key("feature_created_files"),
            )
        except (
            BinanceApiError,
            FileNotFoundError,
            ValueError,
            IndexError,
        ) as exc:
            st.error(str(exc))

    _render_multitimeframe_builder()

    st.divider()
    processed_files = list_processed_files(PROCESSED_DIR)[:8]
    st.subheader("已建立資料集")
    if processed_files:
        table = pd.DataFrame(
            {
                "檔案": [_format_path(path) for path in processed_files],
                "大小 KB": [round(path.stat().st_size / 1024, 1) for path in processed_files],
            }
        )
        themed_dataframe(
            table,
            width="stretch",
            hide_index=True,
            key=themed_widget_key("feature_local_files"),
        )
    else:
        st.info("目前沒有 processed CSV。")


def _visible_bar_count(total_bars: int) -> int:
    if total_bars <= 20:
        st.caption(f"顯示全部 {total_bars} 根 K 線")
        return total_bars
    maximum = min(total_bars, 1000)
    return st.slider(
        "顯示最近 K 線",
        min_value=20,
        max_value=maximum,
        value=min(200, maximum),
        step=10,
    )


def _render_chart_page() -> None:
    _render_header("市場圖表")
    raw_files = list_ohlcv_files(RAW_DIR)
    processed_files = list_processed_files(PROCESSED_DIR)
    all_files = processed_files + raw_files
    if not all_files:
        st.warning("目前沒有可繪製的 CSV。")
        return

    selected_path = st.selectbox("資料檔案", all_files, format_func=_format_path)
    try:
        source_frame = _load_path(selected_path)
        feature_frame = ensure_chart_features(source_frame)
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        return

    control_columns = st.columns([1.4, 1, 1])
    with control_columns[0]:
        overlays = st.multiselect(
            "移動平均",
            ["sma_5", "sma_20", "sma_60", "sma_200", "ema_20", "ema_60"],
            default=["sma_20", "sma_60"],
        )
    with control_columns[1]:
        visible_bars = _visible_bar_count(len(feature_frame))
    with control_columns[2]:
        show_volume = st.toggle("顯示成交量", value=True)

    visible_frame = feature_frame.tail(visible_bars).reset_index(drop=True)
    summary = summarize_market(visible_frame)
    symbol = str(visible_frame.iloc[-1].get("symbol", "-"))
    interval = str(visible_frame.iloc[-1].get("interval", "-"))
    metrics = st.columns(5)
    metrics[0].metric("標的", symbol)
    metrics[1].metric("週期", interval)
    metrics[2].metric("最新收盤", f"{summary.latest_close:,.2f}")
    metrics[3].metric("區間報酬", f"{summary.period_return:.2%}")
    metrics[4].metric("K 線數", f"{summary.bars:,}")
    latest_time = pd.Timestamp(summary.end_time)
    st.caption(f"資料最後時間（UTC）：{latest_time.isoformat()}")
    duration = interval_duration(interval)
    stale_limit = max(duration * 3, pd.Timedelta(days=3)) if duration else None
    if stale_limit is not None and pd.Timestamp.now(tz="UTC") - latest_time > stale_limit:
        st.warning("這份圖表資料已過期；請回到市場資料或特徵工程頁自動更新。")

    kline_tab, indicator_tab, ai_context_tab = st.tabs(["K 線", "技術指標", "AI 上下文"])
    with kline_tab:
        figure = make_candlestick_figure(
            visible_frame,
            overlays,
            show_volume,
            _is_dark_mode(),
        )
        st.plotly_chart(figure, width="stretch", config=PLOTLY_CONFIG, key="candlestick")
    with indicator_tab:
        indicator_mode = st.segmented_control(
            "指標",
            options=["RSI", "MACD", "波動率", "成交量"],
            default="RSI",
        )
        indicator_figure = make_indicator_figure(
            visible_frame,
            str(indicator_mode),
            _is_dark_mode(),
        )
        st.plotly_chart(
            indicator_figure,
            width="stretch",
            config=PLOTLY_CONFIG,
            key="indicator",
        )
    with ai_context_tab:
        context_columns = [
            column
            for column in visible_frame.columns
            if column.startswith("finbert_") or column.startswith("transformer_")
        ]
        if not context_columns:
            st.info("這份資料尚未附加 FinBERT 或 Transformer 正式輸出。")
        else:
            latest = visible_frame.iloc[-1]
            metrics = st.columns(4)
            metrics[0].metric(
                "新聞情緒",
                f"{float(latest.get('finbert_sentiment', 0)):.3f}",
            )
            metrics[1].metric(
                "新聞可用",
                "是" if float(latest.get("finbert_available", 0)) > 0 else "否",
            )
            metrics[2].metric(
                "Transformer 多頭",
                f"{float(latest.get('transformer_bull_probability', 0)):.1%}",
            )
            metrics[3].metric(
                "Transformer 可用",
                "是" if float(latest.get("transformer_available", 0)) > 0 else "否",
            )
            chart_columns = [
                column
                for column in [
                    "finbert_sentiment",
                    "transformer_return_1",
                    "transformer_return_5",
                    "transformer_return_20",
                ]
                if column in visible_frame
            ]
            if chart_columns:
                st.line_chart(
                    visible_frame.set_index("timestamp")[chart_columns],
                    height=360,
                )
            themed_dataframe(
                visible_frame[["timestamp", *context_columns]].tail(100),
                width="stretch",
                hide_index=True,
                key=themed_widget_key("chart_ai_context"),
            )

    with st.expander("資料表"):
        themed_dataframe(
            visible_frame.tail(100),
            width="stretch",
            hide_index=True,
            key=themed_widget_key("chart_visible_rows"),
        )
        st.download_button(
            "匯出目前畫面 CSV",
            data=visible_frame.to_csv(index=False).encode("utf-8"),
            file_name=f"chart_{symbol.replace('/', '')}_{interval}.csv",
            mime="text/csv",
            icon=":material/download:",
        )


def _render_paper_account(state: object, paths: object) -> None:
    """顯示模擬帳戶摘要、資產曲線與最近紀錄。"""
    from ai_quant_trading.paper_trading.storage import (
        ORDER_COLUMNS,
        PREDICTION_COLUMNS,
        TRADE_COLUMNS,
        read_account_csv,
    )

    performance = read_account_csv(paths.performance_csv)
    latest = performance.iloc[-1] if not performance.empty else None
    equity = float(latest["equity"]) if latest is not None else float(state.cash)
    total_return = (
        float(latest["total_return"])
        if latest is not None
        else equity / state.config.initial_capital - 1
    )
    drawdown = float(latest["drawdown"]) if latest is not None else 0.0
    signal_labels = {-1: "賣出／開空", 0: "維持", 1: "買進／回補"}

    st.subheader("帳戶狀態")
    metrics = st.columns(6)
    metrics[0].metric("總資產", f"{equity:,.2f}")
    metrics[1].metric("現金", f"{state.cash:,.2f}")
    metrics[2].metric("總報酬", f"{total_return:.2%}")
    metrics[3].metric("目前回撤", f"{drawdown:.2%}")
    position_direction = "做多" if state.quantity > 0 else "做空" if state.quantity < 0 else "空手"
    metrics[4].metric(f"持倉數量 · {position_direction}", f"{state.quantity:.6f}")
    pending_text = signal_labels.get(state.pending_signal, "Hold")
    if state.pending_target_fraction is not None:
        target_direction = (
            "做多"
            if state.pending_target_fraction > 0
            else "做空"
            if state.pending_target_fraction < 0
            else "空手"
        )
        pending_text = f"{target_direction} {abs(state.pending_target_fraction):.1%}"
    metrics[5].metric("待執行", pending_text)
    st.caption(
        f"目前持倉槓桿：{getattr(state, 'position_leverage', 1)}x｜"
        f"下一筆核准槓桿：{getattr(state, 'pending_leverage', None) or '-'}｜"
        f"原因：{getattr(state, 'last_leverage_reason', '-') }"
    )
    st.caption(
        "PPO／SAC · "
        f"{state.symbol} · {state.exchange} · {state.interval}　|　"
        f"最後 K 線：{state.last_processed_timestamp or '-'}　|　"
        f"風控停機：{'已觸發' if state.risk_halted else '未觸發'}　|　"
        f"累計空頭成本：{getattr(state, 'short_carry_paid', 0.0):,.2f}"
    )

    if not performance.empty:
        chart = performance.copy()
        chart["timestamp"] = pd.to_datetime(chart["timestamp"], utc=True, errors="coerce")
        st.subheader("模擬資產曲線")
        st.plotly_chart(
            make_equity_figure(chart, _is_dark_mode()),
            width="stretch",
            config=PLOTLY_CONFIG,
            key="paper_equity",
        )

    order_tab, trade_tab, prediction_tab = st.tabs(["模擬訂單", "完整交易", "RL 目標持倉"])
    with order_tab:
        orders = read_account_csv(paths.orders_csv, ORDER_COLUMNS)
        themed_dataframe(
            orders.tail(100),
            width="stretch",
            hide_index=True,
            key=themed_widget_key("paper_orders"),
        )
    with trade_tab:
        trades = read_account_csv(paths.trades_csv, TRADE_COLUMNS)
        themed_dataframe(
            trades.tail(100),
            width="stretch",
            hide_index=True,
            key=themed_widget_key("paper_trades"),
        )
    with prediction_tab:
        predictions = read_account_csv(paths.predictions_csv, PREDICTION_COLUMNS)
        themed_dataframe(
            predictions.tail(100),
            width="stretch",
            hide_index=True,
            key=themed_widget_key("paper_policy_decisions"),
        )
    st.caption(f"帳戶資料夾：{_format_path(paths.account_dir)}")


def _dynamic_leverage_controls(
    current: DynamicLeverageConfig,
    *,
    key_prefix: str,
    fixed_leverage: int,
    disabled: bool = False,
) -> DynamicLeverageConfig:
    """顯示共用的 1x 到 3x 槓桿風控設定。"""
    enabled = st.toggle(
        "依勝率、損益比與市場風險動態選擇槓桿",
        value=current.enabled,
        key=f"{key_prefix}_enabled",
        disabled=disabled,
    )
    top = st.columns(3)
    hard_max = top[0].selectbox(
        "槓桿硬上限",
        [1, 2, 3],
        index=max(min((current.max_leverage if current.enabled else fixed_leverage) - 1, 2), 0),
        format_func=lambda value: f"{value}x",
        key=f"{key_prefix}_max",
        disabled=disabled,
    )
    uncalibrated_max = top[1].selectbox(
        "機率未校準上限",
        list(range(1, int(hard_max) + 1)),
        index=min(current.uncalibrated_max_leverage, int(hard_max)) - 1,
        format_func=lambda value: f"{value}x",
        key=f"{key_prefix}_uncalibrated",
        disabled=disabled or not enabled,
    )
    positive_expectancy = top[2].toggle(
        "只允許正期望值",
        value=current.require_positive_expectancy,
        key=f"{key_prefix}_expectancy",
        disabled=disabled or not enabled,
    )
    st.caption("模型只決定 1x～硬上限；停損距離與單筆風險仍會先限制名目部位。")

    with st.container(border=True):
        st.markdown("**2x／3x 證據門檻**")
        row = st.columns(4)
        probability_2 = row[0].number_input(
            "2x 最低方向機率",
            0.50,
            0.95,
            float(current.leverage_2_min_probability),
            0.01,
            key=f"{key_prefix}_p2",
            disabled=disabled or not enabled,
        )
        probability_3 = row[1].number_input(
            "3x 最低方向機率",
            float(probability_2),
            0.99,
            max(float(current.leverage_3_min_probability), float(probability_2)),
            0.01,
            key=f"{key_prefix}_p3",
            disabled=disabled or not enabled,
        )
        risk_reward_2 = row[2].number_input(
            "2x 最低淨損益比",
            0.50,
            10.0,
            float(current.leverage_2_min_risk_reward),
            0.05,
            key=f"{key_prefix}_rr2",
            disabled=disabled or not enabled,
        )
        risk_reward_3 = row[3].number_input(
            "3x 最低淨損益比",
            float(risk_reward_2),
            10.0,
            max(float(current.leverage_3_min_risk_reward), float(risk_reward_2)),
            0.05,
            key=f"{key_prefix}_rr3",
            disabled=disabled or not enabled,
        )

        row = st.columns(4)
        uncertainty_2 = row[0].number_input(
            "2x 不確定度上限",
            0.01,
            1.0,
            float(current.leverage_2_max_uncertainty),
            0.01,
            key=f"{key_prefix}_u2",
            disabled=disabled or not enabled,
        )
        uncertainty_3 = row[1].number_input(
            "3x 不確定度上限",
            0.01,
            float(uncertainty_2),
            min(float(current.leverage_3_max_uncertainty), float(uncertainty_2)),
            0.01,
            key=f"{key_prefix}_u3",
            disabled=disabled or not enabled,
        )
        drawdown_2 = row[2].number_input(
            "2x 回撤上限 %",
            0.1,
            50.0,
            float(current.leverage_2_max_drawdown) * 100,
            0.1,
            key=f"{key_prefix}_dd2",
            disabled=disabled or not enabled,
        )
        drawdown_3 = row[3].number_input(
            "3x 回撤上限 %",
            0.1,
            float(drawdown_2),
            min(float(current.leverage_3_max_drawdown) * 100, float(drawdown_2)),
            0.1,
            key=f"{key_prefix}_dd3",
            disabled=disabled or not enabled,
        )

        row = st.columns(4)
        spread_2 = row[0].number_input(
            "2x Spread 上限 bps",
            0.1,
            100.0,
            float(current.leverage_2_max_spread_bps),
            0.1,
            key=f"{key_prefix}_spread2",
            disabled=disabled or not enabled,
        )
        spread_3 = row[1].number_input(
            "3x Spread 上限 bps",
            0.1,
            float(spread_2),
            min(float(current.leverage_3_max_spread_bps), float(spread_2)),
            0.1,
            key=f"{key_prefix}_spread3",
            disabled=disabled or not enabled,
        )
        volatility_2 = row[2].number_input(
            "2x 預測波動上限 %",
            0.1,
            50.0,
            float(current.leverage_2_max_volatility) * 100,
            0.1,
            key=f"{key_prefix}_vol2",
            disabled=disabled or not enabled,
        )
        volatility_3 = row[3].number_input(
            "3x 預測波動上限 %",
            0.1,
            float(volatility_2),
            min(float(current.leverage_3_max_volatility) * 100, float(volatility_2)),
            0.1,
            key=f"{key_prefix}_vol3",
            disabled=disabled or not enabled,
        )

    return DynamicLeverageConfig(
        enabled=bool(enabled),
        max_leverage=int(hard_max),
        uncalibrated_max_leverage=int(uncalibrated_max),
        leverage_2_min_probability=float(probability_2),
        leverage_3_min_probability=float(probability_3),
        leverage_2_min_risk_reward=float(risk_reward_2),
        leverage_3_min_risk_reward=float(risk_reward_3),
        leverage_2_max_uncertainty=float(uncertainty_2),
        leverage_3_max_uncertainty=float(uncertainty_3),
        leverage_2_max_drawdown=float(drawdown_2) / 100,
        leverage_3_max_drawdown=float(drawdown_3) / 100,
        leverage_2_max_spread_bps=float(spread_2),
        leverage_3_max_spread_bps=float(spread_3),
        leverage_2_max_volatility=float(volatility_2) / 100,
        leverage_3_max_volatility=float(volatility_3) / 100,
        require_positive_expectancy=bool(positive_expectancy),
    )


def _render_paper_trading_page() -> None:
    """建立或推進 PPO／SAC 模擬帳戶。"""
    from ai_quant_trading.automation import (
        automation_paths,
        read_automation_status,
        request_automation_stop,
    )
    from ai_quant_trading.dashboard.automation_control import (
        automation_autostart_enabled,
        set_automation_autostart,
        start_automation_worker,
    )
    from ai_quant_trading.paper_trading import (
        PaperTradingConfig,
        account_paths,
        delete_paper_account,
        download_latest_rl_paper_market_data,
        list_paper_accounts,
        load_account_state,
        prepare_rl_paper_market_frame,
        run_rl_paper_cycle,
    )
    from ai_quant_trading.paper_trading.storage import save_account_state
    from ai_quant_trading.reinforcement_learning import (
        assess_rl_training_quality,
        build_expert_profile,
        list_rl_training_runs,
        load_rl_policy,
    )

    _render_header("模擬交易")
    rl_training_dirs = list_rl_training_runs(RL_DIR)
    if not rl_training_dirs:
        st.warning("目前沒有可用 PPO／SAC 模型，請先完成強化學習訓練。")
        return

    def format_rl_model(path: Path) -> str:
        try:
            payload = json.loads((path / "training.json").read_text(encoding="utf-8"))
            source = dict(payload.get("environment_source", {}))
            algorithm = str(payload.get("training_config", {}).get("algorithm", "RL")).upper()
            return (
                f"{algorithm} · {source.get('symbol', '-')} · "
                f"{source.get('interval', '-')} · {path.name[:15]}"
            )
        except (OSError, ValueError, TypeError):
            return path.name

    accounts = []
    for candidate in list_paper_accounts(PAPER_DIR):
        try:
            candidate_state = load_account_state(candidate)
        except (OSError, ValueError, TypeError):
            continue
        if (
            candidate_state.model_kind == "rl"
            and (Path(candidate_state.model_dir) / "training.json").exists()
        ):
            accounts.append(candidate)
    modes = ["建立新帳戶"] + (["繼續既有帳戶"] if accounts else [])
    account_mode = st.segmented_control(
        "帳戶模式",
        options=modes,
        default=modes[-1],
        width="stretch",
    )
    is_new = account_mode == "建立新帳戶"
    state = None
    paths = None
    if is_new:
        model_kind = "rl"
        identity_columns = st.columns([1, 2])
        with identity_columns[0]:
            account_id = st.text_input("帳戶名稱", value="ppo_demo")
        with identity_columns[1]:
            model_dir = st.selectbox(
                "PPO／SAC 模型",
                rl_training_dirs,
                format_func=format_rl_model,
            )
    else:
        selected_account = st.selectbox(
            "模擬帳戶",
            accounts,
            format_func=lambda item: item.account_dir.name,
        )
        if selected_account is None:
            st.info("請選擇要繼續的模擬帳戶。")
            return
        paths = selected_account
        state = load_account_state(paths)
        account_id = state.account_id
        model_kind = state.model_kind
        if model_kind != "rl":
            st.error("此帳戶使用舊版決策格式，請建立新的 PPO 帳戶。")
            return
        model_dir = next(
            (path for path in rl_training_dirs if path.name == Path(state.model_dir).name),
            Path(state.model_dir),
        )
        st.code(_format_path(model_dir), language=None)

    selected_exchange: str | None = None
    selected_symbol: str | None = None
    selected_interval: str | None = None
    rl_position_default = 25.0
    rl_capital_default = 1000.0
    rl_allow_short_default = False
    rl_max_short_default = 0.0
    rl_short_rate_default = 0.0
    expert_kind = "general"
    expert_profile = build_expert_profile("general")
    try:
        training_metadata = json.loads(
            (Path(model_dir) / "training.json").read_text(encoding="utf-8")
        )
        environment_path = Path(model_dir).parent.parent / "environment.json"
        environment_metadata = json.loads(environment_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        st.error(f"強化學習模型中繼資料無法讀取：{exc}")
        return
    quality = assess_rl_training_quality(training_metadata)
    if quality.eligible:
        st.success("RL 模型已通過目前的最低工程品質門檻。")
    else:
        st.warning("此 RL 模型僅供模擬研究：" + "；".join(quality.reasons))
    rl_position_default = (
        float(environment_metadata.get("environment_config", {}).get("max_position_fraction", 0.25))
        * 100
    )
    environment_config = dict(environment_metadata.get("environment_config", {}))
    rl_allow_short_default = bool(environment_config.get("allow_short", False))
    rl_max_short_default = float(environment_config.get("max_short_fraction", 0.0)) * 100
    rl_short_rate_default = float(environment_config.get("short_borrow_rate_annual", 0.0)) * 100
    expert_kind = str(environment_config.get("expert_kind", "general"))
    expert_profile = build_expert_profile(expert_kind)
    rl_capital_default = float(environment_config.get("initial_capital", 1000.0))
    if environment_metadata.get("environment_kind") == "universal":
        markets = [
            dict(values) for values in dict(environment_metadata.get("markets", {})).values()
        ]
        if is_new:
            market = st.selectbox(
                "RL 模擬市場",
                markets,
                format_func=lambda value: (
                    f"{value.get('symbol', '-')} · {value.get('exchange', '-')} · "
                    f"{value.get('interval', '-')}"
                ),
            )
            selected_exchange = str(market.get("exchange", ""))
            selected_symbol = str(market.get("symbol", ""))
            selected_interval = str(market.get("interval", ""))
        else:
            selected_exchange = state.exchange
            selected_symbol = state.symbol
            selected_interval = state.interval
    else:
        source = dict(environment_metadata.get("source", {}))
        selected_exchange = str(source.get("exchange", ""))
        selected_symbol = str(source.get("symbol", ""))
        selected_interval = str(source.get("interval", ""))
    st.caption(
        f"{expert_profile.label} · {selected_symbol} · {selected_exchange} · {selected_interval}"
    )
    download_limit = st.number_input(
        "自動下載 K 線筆數",
        min_value=220,
        max_value=1000,
        value=500,
        step=20,
    )

    if is_new:
        execution_columns = st.columns(4)
        with execution_columns[0]:
            initial_capital = st.number_input(
                "初始資金",
                min_value=1.0,
                value=rl_capital_default,
            )
        with execution_columns[1]:
            fee_percent = st.number_input("手續費 %", 0.0, 10.0, 0.1, 0.01)
        with execution_columns[2]:
            slippage_percent = st.number_input("滑價 %", 0.0, 10.0, 0.05, 0.01)
        with execution_columns[3]:
            position_percent = st.number_input(
                "最大投入 %",
                1.0,
                100.0,
                min(max(rl_position_default, 1.0), 100.0),
                5.0,
            )

        rl_entry_threshold = 0.10
        rl_exit_threshold = 0.02
        short_columns = st.columns(3)
        with short_columns[0]:
            allow_short = st.toggle(
                "允許模型作空",
                value=rl_allow_short_default,
                disabled=not rl_allow_short_default,
            )
        with short_columns[1]:
            max_short_percent = st.number_input(
                "最大空頭持倉 %",
                0.1,
                100.0,
                max(rl_max_short_default, 0.1),
                0.5,
                disabled=not allow_short,
            )
        with short_columns[2]:
            short_borrow_rate_percent = st.number_input(
                "空頭年化持有成本 %",
                0.0,
                200.0,
                rl_short_rate_default,
                0.5,
                disabled=not allow_short,
            )
        st.info(
            "此帳戶會執行帶方向的連續目標持倉，正值做多、負值做空；"
            f"小於 {environment_config.get('rebalance_deadband', 0):.1%} 的變動不下單。"
        )

        risk_columns = st.columns(4)
        with risk_columns[0]:
            max_risk_percent = st.number_input(
                "單筆最大風險 %",
                0.05,
                20.0,
                expert_profile.paper_risk.max_risk_per_trade * 100,
                0.05,
            )
        with risk_columns[1]:
            default_stop_mode = (
                "ATR" if expert_profile.paper_risk.stop_loss_mode == "atr" else "固定比例"
            )
            stop_mode = st.segmented_control(
                "停損方式",
                options=["固定比例", "ATR"],
                default=default_stop_mode,
            )
        with risk_columns[2]:
            stop_value = st.number_input(
                "ATR 倍數" if stop_mode == "ATR" else "固定停損 %",
                min_value=0.1,
                max_value=50.0,
                value=(
                    expert_profile.paper_risk.atr_multiplier
                    if stop_mode == "ATR"
                    else (2.0 if stop_mode == "ATR" else 3.0)
                ),
                step=0.1,
            )
        with risk_columns[3]:
            max_drawdown_percent = st.number_input(
                "最大回撤停機 %",
                1.0,
                100.0,
                expert_profile.paper_risk.max_drawdown_limit * 100,
                1.0,
            )
        default_take_profit = expert_profile.paper_risk.take_profit_pct
        take_profit_enabled = st.toggle(
            "啟用固定停利",
            value=default_take_profit is not None,
        )
        take_profit_percent = st.number_input(
            "固定停利 %",
            min_value=0.1,
            max_value=200.0,
            value=float(default_take_profit or 0.06) * 100,
            step=0.5,
            disabled=not take_profit_enabled,
        )
        with st.expander("動態槓桿風控", expanded=True):
            dynamic_leverage = _dynamic_leverage_controls(
                DynamicLeverageConfig(enabled=True),
                key_prefix="paper_new_dynamic_leverage",
                fixed_leverage=int(round(float(environment_config.get("leverage", 1.0)))),
            )
        paper_config = PaperTradingConfig(
            initial_capital=float(initial_capital),
            fee_rate=float(fee_percent) / 100,
            slippage_rate=float(slippage_percent) / 100,
            position_fraction=float(position_percent) / 100,
            allow_short=bool(allow_short),
            max_short_fraction=(
                min(float(max_short_percent), float(position_percent)) / 100 if allow_short else 0.0
            ),
            short_borrow_rate_annual=(
                float(short_borrow_rate_percent) / 100 if allow_short else 0.0
            ),
            execution_mode=str(environment_config.get("execution_mode", "spot")),
            leverage=float(dynamic_leverage.max_leverage),
            max_margin_fraction=float(environment_config.get("max_margin_fraction", 1.0)),
            daily_loss_limit=float(environment_config.get("daily_loss_limit", 1.0)),
            max_consecutive_losses=int(environment_config.get("max_consecutive_losses", 0)),
        )
        risk_config = RiskConfig(
            max_risk_per_trade=float(max_risk_percent) / 100,
            stop_loss_mode="atr" if stop_mode == "ATR" else "fixed",
            fixed_stop_loss_pct=(0.03 if stop_mode == "ATR" else float(stop_value) / 100),
            atr_multiplier=float(stop_value) if stop_mode == "ATR" else 2.0,
            take_profit_pct=(float(take_profit_percent) / 100 if take_profit_enabled else None),
            max_drawdown_limit=float(max_drawdown_percent) / 100,
            max_position_fraction=float(position_percent) / 100,
            dynamic_leverage=dynamic_leverage,
        )
    else:
        paper_config = state.config
        risk_config = state.risk_config
        rl_entry_threshold = state.rl_entry_threshold
        rl_exit_threshold = state.rl_exit_threshold
        st.caption(
            f"既有設定：初始資金 {paper_config.initial_capital:,.2f}／"
            f"單筆風險 {risk_config.max_risk_per_trade:.2%}／"
            f"最大回撤停機 {risk_config.max_drawdown_limit:.2%}"
        )
        st.caption(
            f"{getattr(state, 'expert_kind', 'general')}／連續目標持倉／"
            f"最大做多 {paper_config.position_fraction:.1%}／"
            f"最大做空 {paper_config.max_short_fraction:.1%}"
        )
        st.caption(
            f"槓桿模式：{'動態' if risk_config.dynamic_leverage.enabled else '固定'}／"
            f"硬上限 {risk_config.dynamic_leverage.max_leverage if risk_config.dynamic_leverage.enabled else int(round(paper_config.leverage))}x"
        )

    st.caption("本期收盤產生訊號，下一根新 K 線開盤才模擬成交；重跑同一根不會重複下單。")
    if st.button(
        "執行一次模擬交易",
        type="primary",
        icon=":material/play_arrow:",
        width="stretch",
    ):
        try:
            if is_new and account_paths(PAPER_DIR, account_id).account_json.exists():
                raise ValueError("帳戶名稱已存在，請改用『繼續既有帳戶』或更換名稱")
            with st.spinner("正在下載資料、更新特徵並執行模擬交易..."):
                policy = load_rl_policy(model_dir, device="cpu")
                market_path = download_latest_rl_paper_market_data(
                    policy,
                    RAW_DIR,
                    exchange=selected_exchange,
                    symbol=selected_symbol,
                    interval=selected_interval,
                    limit=int(download_limit),
                    runtime_feature_dir=(
                        account_paths(PAPER_DIR, account_id).account_dir / "market_cache"
                    ),
                )
                market_frame = prepare_rl_paper_market_frame(market_path, policy)
                result = run_rl_paper_cycle(
                    market_frame,
                    policy,
                    model_dir=model_dir,
                    root_dir=PAPER_DIR,
                    account_id=account_id,
                    exchange=selected_exchange,
                    symbol=selected_symbol,
                    interval=selected_interval,
                    config=paper_config,
                    risk_config=risk_config,
                    entry_threshold=rl_entry_threshold,
                    exit_threshold=rl_exit_threshold,
                )
            state = result.state
            paths = result.paths
            st.success(result.message)
        except (ValueError, OSError, ImportError, RuntimeError) as exc:
            st.error(str(exc))

    if state is not None and paths is not None:
        st.divider()
        _render_paper_account(state, paths)
        st.subheader("無人值守")
        runner_paths = automation_paths(paths.account_dir / "automation")
        runner_status = read_automation_status(runner_paths)
        active = runner_status.state in {"running", "stopping"}
        status_columns = st.columns(4)
        status_columns[0].metric("狀態", runner_status.state)
        status_columns[1].metric("完成輪詢", runner_status.successful_cycles)
        status_columns[2].metric("連續錯誤", runner_status.consecutive_errors)
        status_columns[3].metric("PID", runner_status.pid or "-")
        st.caption(f"最後心跳：{runner_status.heartbeat_at or '-'}｜{runner_status.last_message}")
        with st.expander("調整動態槓桿"):
            updated_dynamic_leverage = _dynamic_leverage_controls(
                state.risk_config.dynamic_leverage,
                key_prefix=f"paper_existing_dynamic_leverage_{state.account_id}",
                fixed_leverage=int(round(state.config.leverage)),
                disabled=active,
            )
            if st.button(
                "儲存槓桿設定",
                icon=":material/save:",
                disabled=active,
                key=f"paper_save_dynamic_leverage_{state.account_id}",
            ):
                state.risk_config = replace(
                    state.risk_config,
                    dynamic_leverage=updated_dynamic_leverage,
                )
                state.config = replace(
                    state.config,
                    leverage=float(updated_dynamic_leverage.max_leverage),
                )
                save_account_state(paths, state)
                st.success("動態槓桿設定已儲存，下一根新 K 線開始套用。")
                st.rerun()
        realtime_capable = (
            state.exchange.lower() == "binance_futures"
            and state.symbol.upper() == "BTC/USDT"
            and state.interval.lower() == "15m"
        )
        transport_options = ["WebSocket 15m"] + (["REST 輪詢"] if realtime_capable else [])
        if not realtime_capable:
            transport_options = ["REST 輪詢"]
        transport = st.segmented_control(
            "行情傳輸",
            transport_options,
            default=transport_options[0],
            width="stretch",
            key=f"paper_transport_{state.account_id}",
        )
        use_websocket = transport == "WebSocket 15m"
        render_realtime_status(runner_paths.root_dir / "realtime_status.json")

        transformer_checkpoint: Path | None = None
        transformer_probability = 0.45
        transformer_uncertainty = 0.62
        max_spread_bps = 5.0
        transformer_device = "cpu"
        poll_seconds = 60
        if use_websocket:
            transformer_models: list[Path] = []
            for checkpoint in sorted(
                TRANSFORMER_DIR.glob("models/*/best_model.pt"),
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            ):
                try:
                    summary = json.loads(
                        (checkpoint.parent / "training.json").read_text(encoding="utf-8")
                    )
                    sources = [dict(item) for item in summary.get("sources", [])]
                    compatible = (
                        summary.get("status") == "complete"
                        and any(
                            str(item.get("symbol", "")).upper() == "BTC/USDT"
                            and str(item.get("interval", "")).lower() == "15m"
                            for item in sources
                        )
                        and any(
                            str(column).startswith("mtf_5m_")
                            for column in summary.get("feature_columns", [])
                        )
                    )
                    if compatible:
                        transformer_models.append(checkpoint)
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    continue
            transformer_checkpoint = (
                st.selectbox(
                    "Transformer 趨勢模型",
                    transformer_models,
                    format_func=lambda item: item.parent.name,
                    key=f"paper_realtime_transformer_{state.account_id}",
                )
                if transformer_models
                else None
            )
            if transformer_checkpoint is None:
                st.error("沒有相容的 BTC 15m 五週期 Transformer，無法啟動即時模型管線。")
            else:
                try:
                    transformer_summary = json.loads(
                        (transformer_checkpoint.parent / "training.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    train_samples = int(
                        dict(transformer_summary.get("sample_counts", {})).get("train", 0)
                    )
                    direction_accuracy = float(
                        dict(transformer_summary.get("test_metrics", {})).get(
                            "direction_accuracy", 0.0
                        )
                    )
                    if train_samples < 5_000 or direction_accuracy < 0.53:
                        st.warning(
                            f"此 Transformer 仍是研究模型：訓練樣本 {train_samples:,}、"
                            f"測試方向準確率 {direction_accuracy:.1%}。請先使用模擬倉。"
                        )
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    st.warning("Transformer 品質資料無法讀取，僅允許模擬研究。")
            with st.expander("即時行情與趨勢門檻"):
                realtime_columns = st.columns(4)
                transformer_probability = realtime_columns[0].number_input(
                    "趨勢最低機率",
                    0.34,
                    0.95,
                    0.45,
                    0.01,
                    key=f"paper_transformer_probability_{state.account_id}",
                )
                transformer_uncertainty = realtime_columns[1].number_input(
                    "不確定性上限",
                    0.05,
                    0.67,
                    0.62,
                    0.01,
                    key=f"paper_transformer_uncertainty_{state.account_id}",
                )
                max_spread_bps = realtime_columns[2].number_input(
                    "最大 Spread bps",
                    0.1,
                    50.0,
                    5.0,
                    0.1,
                    key=f"paper_max_spread_{state.account_id}",
                )
                transformer_device = realtime_columns[3].selectbox(
                    "Transformer 裝置",
                    ["cpu", "cuda"],
                    key=f"paper_transformer_device_{state.account_id}",
                )
            st.caption("WebSocket 持續接收五週期 K 線與 Bid/Ask；只有新 15m K 線收盤才執行模型。")
        else:
            poll_seconds = st.number_input(
                "資料輪詢秒數",
                min_value=10,
                max_value=3600,
                value=60,
                step=10,
                key=f"paper_auto_poll_{state.account_id}",
            )
        max_errors = st.number_input(
            "連續錯誤熔斷",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            key=f"paper_auto_errors_{state.account_id}",
        )
        arguments = [
            "rl-auto",
            "--account",
            state.account_id,
            "--model-dir",
            str(model_dir),
            "--download-latest",
            "--raw-dir",
            str(RAW_DIR),
            "--paper-dir",
            str(PAPER_DIR),
            "--limit",
            str(int(download_limit)),
            "--initial-capital",
            str(state.config.initial_capital),
            "--fee-rate",
            str(state.config.fee_rate),
            "--slippage-rate",
            str(state.config.slippage_rate),
            "--position-fraction",
            str(state.config.position_fraction),
            "--max-risk-per-trade",
            str(state.risk_config.max_risk_per_trade),
            "--stop-loss-mode",
            state.risk_config.stop_loss_mode,
            "--fixed-stop-loss-pct",
            str(state.risk_config.fixed_stop_loss_pct),
            "--atr-multiplier",
            str(state.risk_config.atr_multiplier),
            "--max-drawdown-limit",
            str(state.risk_config.max_drawdown_limit),
            "--poll-seconds",
            str(int(poll_seconds)),
            "--max-consecutive-errors",
            str(int(max_errors)),
            "--transport",
            "websocket" if use_websocket else "rest",
            "--rolling-storage-bars",
            "2000",
            "--persist-rolling-data",
        ]
        if use_websocket and transformer_checkpoint is not None:
            arguments.extend(
                [
                    "--transformer-checkpoint",
                    str(transformer_checkpoint),
                    "--transformer-device",
                    transformer_device,
                    "--transformer-min-probability",
                    str(float(transformer_probability)),
                    "--transformer-max-uncertainty",
                    str(float(transformer_uncertainty)),
                    "--max-spread-bps",
                    str(float(max_spread_bps)),
                ]
            )
        arguments.extend(
            [
                "--exchange",
                state.exchange,
                "--symbol",
                state.symbol,
                "--interval",
                state.interval,
                "--entry-threshold",
                str(state.rl_entry_threshold),
                "--exit-threshold",
                str(state.rl_exit_threshold),
                "--allow-short" if state.config.allow_short else "--no-allow-short",
                "--max-short-fraction",
                str(state.config.max_short_fraction),
                "--short-borrow-rate-annual",
                str(state.config.short_borrow_rate_annual),
            ]
        )
        if state.risk_config.take_profit_pct is None:
            arguments.append("--disable-take-profit")
        else:
            arguments.extend(["--take-profit-pct", str(state.risk_config.take_profit_pct)])

        autostart_enabled = automation_autostart_enabled(runner_paths.root_dir)
        toggle_key = f"paper_continuous_{state.account_id}"
        if toggle_key not in st.session_state:
            st.session_state[toggle_key] = autostart_enabled
        continuous_enabled = st.toggle(
            "機器人持續運行",
            key=toggle_key,
            help="開啟後立即執行，重開 App 時會自動恢復；關閉時會安全停止。",
        )
        st.caption("持續運行時會在背景接收行情；關閉 App 視窗不會中斷，下次開啟 App 也會自動接回。")
        if continuous_enabled != autostart_enabled:
            if continuous_enabled:
                if use_websocket and transformer_checkpoint is None:
                    st.session_state[toggle_key] = False
                    st.error("請先準備相容的 BTC 15m 五週期 Transformer 模型。")
                else:
                    try:
                        pid = runner_status.pid
                        if not active:
                            pid = start_automation_worker(
                                "paper",
                                PROJECT_ROOT,
                                runner_paths.root_dir,
                                arguments,
                            )
                        set_automation_autostart(runner_paths.root_dir, True)
                        st.success(f"機器人已持續運行，PID={pid or '-'}")
                        st.rerun()
                    except (OSError, ValueError) as exc:
                        st.session_state[toggle_key] = False
                        st.error(f"背景模擬啟動失敗：{exc}")
            else:
                set_automation_autostart(runner_paths.root_dir, False)
                if active:
                    request_automation_stop(runner_paths)
                    st.info("已送出安全停止請求。")
                st.rerun()
        with st.expander("重設帳戶"):
            confirmed = st.checkbox("我確認要刪除此模擬帳戶與所有紀錄")
            if st.button(
                "刪除模擬帳戶",
                icon=":material/delete:",
                disabled=not confirmed or active,
            ):
                delete_paper_account(PAPER_DIR, state.account_id)
                st.rerun()


def main() -> None:
    """設定頁面並依側邊選單顯示操作工作區。"""
    st.set_page_config(
        page_title="AI Quant Trading Console",
        page_icon=":material/candlestick_chart:",
        layout="wide",
        initial_sidebar_state="auto",
    )
    initialize_theme_state(PROJECT_ROOT)
    _apply_style(_is_dark_mode())
    center = _render_sidebar()
    if center == "交易中心":
        with st.container(key="center_navigation"):
            page = st.segmented_control(
                "交易功能",
                ["總覽", "模擬機器人", "實盤交易", "機器人監控"],
                default="總覽",
                width="stretch",
                label_visibility="collapsed",
                key="trading_center_page",
            )
        if page == "總覽":
            render_trading_overview_page(
                project_root=PROJECT_ROOT,
                raw_dir=RAW_DIR,
                rl_dir=RL_DIR,
                paper_dir=PAPER_DIR,
                dark_mode=_is_dark_mode(),
                plotly_config=PLOTLY_CONFIG,
            )
        elif page == "模擬機器人":
            _render_paper_trading_page()
        elif page == "實盤交易":
            render_live_trading_page(
                project_root=PROJECT_ROOT,
                raw_dir=RAW_DIR,
                rl_dir=RL_DIR,
                live_dir=LIVE_DIR,
            )
        else:
            from ai_quant_trading.dashboard.robot_monitoring_page import (
                render_robot_monitoring_page,
            )

            render_robot_monitoring_page(
                project_root=PROJECT_ROOT,
                paper_dir=PAPER_DIR,
                live_dir=LIVE_DIR,
                dark_mode=_is_dark_mode(),
                plotly_config=PLOTLY_CONFIG,
            )
    elif center == "模型中心":
        with st.container(key="center_navigation"):
            page = st.segmented_control(
                "模型功能",
                ["強化學習", "Transformer", "模型回測", "AI 管線"],
                default="強化學習",
                width="stretch",
                label_visibility="collapsed",
                key="model_center_page",
            )
        if page == "強化學習":
            _render_rl_page()
        elif page == "Transformer":
            from ai_quant_trading.dashboard.transformer_training_page import (
                render_transformer_training_page,
            )

            render_transformer_training_page(
                project_root=PROJECT_ROOT,
                processed_dir=PROCESSED_DIR,
                transformer_root=TRANSFORMER_DIR,
            )
        elif page == "模型回測":
            from ai_quant_trading.dashboard.model_backtest_page import (
                render_model_backtest_page,
            )

            render_model_backtest_page(
                project_root=PROJECT_ROOT,
                rl_dir=RL_DIR,
                transformer_dir=TRANSFORMER_DIR,
                output_dir=MODEL_BACKTEST_DIR,
                dark_mode=_is_dark_mode(),
            )
        else:
            render_ai_pipeline_page(
                project_root=PROJECT_ROOT,
                processed_dir=PROCESSED_DIR,
            )
    else:
        with st.container(key="center_navigation"):
            page = st.segmented_control(
                "資料功能",
                ["市場資料", "特徵資料", "進階資料"],
                default="市場資料",
                width="stretch",
                label_visibility="collapsed",
                key="data_center_page",
            )
        if page == "市場資料":
            _render_download_page()
        elif page == "特徵資料":
            _render_feature_page()
        else:
            from ai_quant_trading.dashboard.advanced_data_page import (
                render_advanced_data_page,
            )

            render_advanced_data_page(
                project_root=PROJECT_ROOT,
                raw_dir=RAW_DIR,
                processed_dir=PROCESSED_DIR,
            )


if __name__ == "__main__":
    main()
