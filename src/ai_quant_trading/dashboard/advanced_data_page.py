"""BTC 永續合約微結構下載與特徵融合介面。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ai_quant_trading.dashboard.services import (
    list_processed_files,
    relative_file_label,
)
from ai_quant_trading.dashboard.ui import themed_dataframe, themed_widget_key
from ai_quant_trading.data_collection.binance_futures import BinanceFuturesPublicClient
from ai_quant_trading.data_collection.csv_storage import (
    save_derivatives_context_csv,
    save_order_book_context_csv,
)
from ai_quant_trading.features.advanced import enrich_feature_csv


def _relative(path: Path, root: Path) -> str:
    return relative_file_label(path, root)


def _render_microstructure(root: Path, raw_dir: Path) -> None:
    st.subheader("BTC 永續合約微結構")
    st.caption(
        "更新 Binance USD-M Order Book、Funding、OI、多空比、主動買賣比與基差。"
        "深度歷史無法由 REST 回補，會從現在開始累積固定檔案。"
    )
    controls = st.columns(2)
    interval = controls[0].selectbox(
        "統計週期",
        ["5m", "15m", "1h", "4h", "1d"],
        index=1,
        key="btc_advanced_interval",
    )
    depth = controls[1].selectbox(
        "Order Book 檔數",
        [20, 50, 100, 500, 1000],
        index=2,
        key="btc_advanced_depth",
    )
    if not st.button(
        "更新 BTC 微結構",
        type="primary",
        icon=":material/order_approve:",
        width="stretch",
        key="btc_advanced_download",
    ):
        return
    try:
        client = BinanceFuturesPublicClient()
        with st.spinner("正在讀取 Binance Futures 公開資料..."):
            order_book = client.fetch_order_book_snapshot(
                "BTC/USDT", limit=int(depth)
            )
            order_path = save_order_book_context_csv(
                order_book,
                raw_dir,
                "BTC/USDT",
                exchange="binance_futures",
            )
            derivatives = client.fetch_market_context(
                "BTC/USDT",
                str(interval),
                include_advanced=True,
            )
            derivatives_path = save_derivatives_context_csv(
                derivatives,
                raw_dir,
                "BTC/USDT",
                str(interval),
                exchange="binance_futures",
            )
        best = order_book.iloc[0]
        metrics = st.columns(3)
        metrics[0].metric("Spread", f"{float(best['spread_bps']):.3f} bps")
        metrics[1].metric("Bid 深度失衡 20", f"{float(best['imbalance_20']):+.2%}")
        metrics[2].metric(
            "Microprice 偏移",
            f"{float(best['microprice_deviation_bps']):+.3f} bps",
        )
        st.success("BTC 永續微結構已更新。")
        st.code(
            f"{_relative(order_path, root)}\n{_relative(derivatives_path, root)}",
            language="text",
        )
    except (OSError, ValueError, RuntimeError) as exc:
        st.error(str(exc))


def _render_fusion(root: Path, raw_dir: Path, processed_dir: Path) -> None:
    st.subheader("融合到 BTC 特徵")
    st.caption(
        "使用 backward-asof，只把該根 K 線當時已公開的 Funding、OI 與深度快照融合回特徵。"
    )
    files = [
        path
        for path in list_processed_files(processed_dir)
        if path.name.startswith("features_crypto_binance_futures_BTC-USDT_")
        or path.name.startswith("features_mtf_crypto_binance_futures_BTC-USDT_15m_")
    ]
    selected = st.multiselect(
        "BTC 特徵資料",
        files,
        default=files,
        format_func=lambda path: _relative(path, root),
        key="btc_advanced_fusion_files",
    )
    if not st.button(
        "融合微結構特徵",
        type="primary",
        icon=":material/merge:",
        width="stretch",
        disabled=not selected,
        key="btc_advanced_fusion",
    ):
        return
    rows: list[dict[str, object]] = []
    progress = st.progress(0.0, text="準備融合")
    for index, path in enumerate(selected, start=1):
        try:
            artifact = enrich_feature_csv(
                path,
                raw_dir=raw_dir,
                processed_dir=processed_dir,
            )
            rows.append(
                {
                    "檔案": _relative(path, root),
                    "列數": artifact.rows,
                    "新增欄位": len(artifact.added_columns),
                    "有進階資料列": artifact.available_rows,
                    "狀態": "完成",
                }
            )
        except (OSError, ValueError) as exc:
            rows.append(
                {
                    "檔案": _relative(path, root),
                    "列數": 0,
                    "新增欄位": 0,
                    "有進階資料列": 0,
                    "狀態": f"失敗：{exc}",
                }
            )
        progress.progress(index / len(selected), text=f"{index}/{len(selected)}")
    themed_dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
        key=themed_widget_key("btc_advanced_fusion_results"),
    )


def render_advanced_data_page(
    *,
    project_root: str | Path,
    raw_dir: str | Path,
    processed_dir: str | Path,
) -> None:
    """顯示 BTC 永續合約專用的進階資料工作區。"""
    root = Path(project_root)
    raw = Path(raw_dir)
    processed = Path(processed_dir)
    st.title("進階資料")
    phase = st.segmented_control(
        "資料類型",
        ["合約微結構", "特徵融合"],
        default="合約微結構",
        width="stretch",
        label_visibility="collapsed",
        key="btc_advanced_phase",
    )
    if phase == "合約微結構":
        _render_microstructure(root, raw)
    else:
        _render_fusion(root, raw, processed)
