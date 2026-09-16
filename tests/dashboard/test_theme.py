"""Dashboard 深淺色主題測試。"""

from __future__ import annotations

from unittest.mock import patch

from ai_quant_trading.dashboard.app import _apply_style
from ai_quant_trading.dashboard.ui import (
    initialize_theme_state,
    load_dark_mode_preference,
    save_theme_preference,
    themed_widget_key,
    theme_preference_path,
)


@patch("ai_quant_trading.dashboard.app.st.markdown")
def test_dark_theme_includes_high_contrast_button_rules(mock_markdown):
    """深色模式必須覆蓋 Streamlit 原生按鈕的淺色預設。"""
    _apply_style(True)

    css = mock_markdown.call_args.args[0]
    assert "--on-accent: #071c16" in css
    assert 'button[data-variant="segmented_control"]' in css
    assert 'button[kind="elementToolbar"]' in css
    assert 'button[data-testid="stNumberInputStepDown"]' in css
    assert 'button[data-testid^="stBaseButton-"]' in css
    assert '[data-testid="stDataFrame"] .stDataFrameGlideDataEditor' in css
    assert "--gdg-bg-cell: var(--surface) !important" in css
    assert '[data-testid="stElementToolbarButtonContainer"]' in css
    assert '[data-testid="stSelectbox"] .react-aria-ComboBox > div' in css
    assert '[data-testid="stTextInputRootElement"]' in css
    assert '[data-testid="stDateInputRootElement"]' in css
    assert '[data-testid="stAlert"] p' in css
    assert '[data-testid="stFileUploaderDropzone"]' in css
    assert '[data-testid="stExpander"] summary *' in css
    assert '[data-testid="stPills"] button[aria-pressed="true"]' in css
    assert '[data-testid="stToast"] *' in css
    assert '[data-testid="stMarkdownContainer"] li' in css
    assert '[data-testid="stTooltipIcon"] button' in css
    assert '[role="tooltip"] [data-testid="stTooltipContent"]' in css
    assert '[role="tooltip"] code' in css
    assert "opacity: 1 !important" in css
    assert "visibility: visible !important" in css
    assert '[role="dialog"] *' in css
    assert "padding-top: 4.75rem" in css
    assert ".st-key-center_navigation" in css
    assert "-webkit-text-fill-color: currentColor !important" in css
    assert '[data-baseweb="button-group"] button[aria-pressed="true"] *' in css
    assert mock_markdown.call_args.kwargs["unsafe_allow_html"] is True


@patch("ai_quant_trading.dashboard.app.st.markdown")
def test_light_theme_keeps_accessible_accent_text(mock_markdown):
    """淺色模式的重點按鈕使用白字，避免共用深色模式的前景色。"""
    _apply_style(False)

    css = mock_markdown.call_args.args[0]
    assert "--on-accent: #ffffff" in css
    assert "--ink: #263442" in css


@patch("ai_quant_trading.dashboard.ui.st.session_state", {"dark_mode": True})
def test_canvas_widget_key_changes_with_theme():
    """Canvas 元件使用主題版本 key，切換時才會完整重繪。"""
    assert themed_widget_key("market_table") == "market_table__dark"


def test_dark_theme_preference_survives_a_new_session(tmp_path):
    """切換主題後應保存到可攜式設定檔，下一個工作階段會重新載入。"""
    with patch(
        "ai_quant_trading.dashboard.ui.st.session_state",
        {"dark_mode": True},
    ):
        saved = save_theme_preference(tmp_path)

    assert saved == theme_preference_path(tmp_path)
    assert load_dark_mode_preference(tmp_path) is True

    empty_session: dict[str, bool] = {}
    with patch(
        "ai_quant_trading.dashboard.ui.st.session_state",
        empty_session,
    ):
        initialize_theme_state(tmp_path)
    assert empty_session["dark_mode"] is True


def test_invalid_theme_preference_falls_back_to_light(tmp_path):
    """偏好檔損壞時不可阻止 Dashboard 啟動。"""
    path = theme_preference_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("{invalid", encoding="utf-8")

    assert load_dark_mode_preference(tmp_path) is False
