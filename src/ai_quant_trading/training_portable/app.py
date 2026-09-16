"""只包含資料、模型訓練、回測與匯出的 Streamlit 訓練中心。"""

from __future__ import annotations

from datetime import date, timedelta
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from ai_quant_trading.dashboard.model_backtest_page import render_model_backtest_page
from ai_quant_trading.dashboard.rl_training_page import (
    render_active_rl_job,
    render_rl_results,
    render_rl_training,
)
from ai_quant_trading.dashboard.transformer_training_page import (
    render_transformer_training_page,
)
from ai_quant_trading.dashboard.ui import (
    initialize_theme_state,
    save_theme_preference,
    themed_dataframe,
)
from ai_quant_trading.reinforcement_learning import (
    inspect_training_backend,
    list_rl_environments,
    list_rl_training_runs,
)
from ai_quant_trading.training_portable.model_package import (
    export_rl_model_package,
    export_transformer_model_package,
    verify_model_package,
)
from ai_quant_trading.training_portable.workspace import (
    BTCEnvironmentSettings,
    build_btc_rl_environment,
    prepare_btc_training_data,
)
from ai_quant_trading.transformer import transformer_backend_status


PROJECT_ROOT = Path(
    os.environ.get("AI_QUANT_TRAINING_ROOT", Path(__file__).resolve().parents[3])
).resolve()
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRANSFORMER_DIR = PROCESSED_DIR / "transformer"
RL_DIR = PROCESSED_DIR / "rl" / "environments"
BACKTEST_DIR = PROCESSED_DIR / "model_backtests"
EXPORT_DIR = PROJECT_ROOT / "outputs" / "model_packages"


def _prepare_directories() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, TRANSFORMER_DIR, RL_DIR, BACKTEST_DIR, EXPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _transformer_runs() -> list[Path]:
    return sorted(
        (
            path.parent
            for path in TRANSFORMER_DIR.glob("models/*/training.json")
            if _read_json(path).get("status") == "complete"
            and (path.parent / "best_model.pt").is_file()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _mtf_files() -> list[Path]:
    return sorted(
        PROCESSED_DIR.glob(
            "features_mtf_crypto_binance_futures_BTC-USDT_15m_*.csv"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _render_overview() -> None:
    st.title("AI Quant 獨立訓練中心")
    st.caption("BTC 多週期資料 · Transformer · SAC/PPO · 樣本外回測 · 模型匯出")
    transformer = transformer_backend_status()
    rl = inspect_training_backend()
    columns = st.columns(5)
    columns[0].metric("GPU", transformer.device_name or "CPU")
    columns[1].metric("CUDA", "可用" if transformer.cuda_available else "不可用")
    columns[2].metric("Transformer", len(_transformer_runs()))
    columns[3].metric("RL 環境", len(list_rl_environments(RL_DIR)))
    columns[4].metric("RL 模型", len(list_rl_training_runs(RL_DIR)))
    st.info(
        "建議順序：先更新 BTC 資料，再訓練 Transformer，建立 SAC/PPO 環境，"
        "完成強化學習與樣本外回測，最後匯出模型包。"
    )
    st.subheader("硬體狀態")
    status = pd.DataFrame(
        [
            {
                "項目": "PyTorch",
                "狀態": transformer.torch_version,
            },
            {
                "項目": "Stable-Baselines3",
                "狀態": rl.stable_baselines_version or "未安裝",
            },
            {
                "項目": "CUDA 裝置",
                "狀態": "、".join(rl.cuda_devices) if rl.cuda_devices else "CPU",
            },
            {
                "項目": "訓練資料根目錄",
                "狀態": str(PROJECT_ROOT),
            },
        ]
    )
    themed_dataframe(status, hide_index=True, width="stretch")


def _render_data() -> None:
    st.title("BTC 訓練資料")
    st.caption("固定 Binance USD-M BTC/USDT；建立 5m、15m、1h、4h、1d 因果對齊特徵。")
    choices = {
        "90 天（流程測試）": 90,
        "1 年（初步研究）": 365,
        "3 年（正式候選）": 1_095,
        "5 年（完整市場狀態）": 1_825,
    }
    period = st.selectbox("歷史範圍", list(choices), index=2)
    end_date = st.date_input("資料截止日", value=date.today())
    start_date = end_date - timedelta(days=choices[period])
    include_context = st.toggle(
        "下載 Funding／OI／Spread 合約脈絡",
        value=True,
        help="歷史 OI 可用範圍受 Binance 公開端點限制；缺值會由 availability 特徵標記。",
    )
    st.caption(f"下載區間：{start_date.isoformat()} 至 {end_date.isoformat()}")
    if st.button("更新並重建 BTC 訓練資料", type="primary", width="stretch"):
        progress = st.progress(0.0, text="準備下載")

        def update(interval: str, completed: int, total: int) -> None:
            progress.progress(completed / total, text=f"下載 {interval} · {completed}/{total}")

        try:
            result = prepare_btc_training_data(
                PROJECT_ROOT,
                start=start_date.isoformat(),
                end=(end_date + timedelta(days=1)).isoformat(),
                include_derivatives_context=include_context,
                progress_callback=update,
            )
            progress.progress(1.0, text="資料與多週期特徵已完成")
            st.success(f"建立 {result.rows:,} 根 15m 決策資料")
            st.code(_relative(result.multitimeframe_file), language=None)
        except (OSError, RuntimeError, ValueError, ImportError) as exc:
            st.error(f"資料準備失敗：{exc}")

    files = _mtf_files()
    st.subheader("可用資料集")
    if not files:
        st.warning("尚未建立 BTC 15m 多週期資料。")
        return
    rows = []
    for path in files:
        first = pd.read_csv(path, nrows=1)
        rows.append(
            {
                "檔案": _relative(path),
                "大小 MB": round(path.stat().st_size / 1024 / 1024, 1),
                "決策週期": str(first.iloc[0].get("interval", "-")) if not first.empty else "-",
            }
        )
    themed_dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _render_transformer() -> None:
    render_transformer_training_page(
        project_root=PROJECT_ROOT,
        processed_dir=PROCESSED_DIR,
        transformer_root=TRANSFORMER_DIR,
    )


def _render_environment() -> None:
    st.title("建立 SAC/PPO 環境")
    st.caption("將 Transformer 的完整歷史輸出加入 RL 狀態，再做時間切分與訓練集標準化。")
    data_files = _mtf_files()
    models = _transformer_runs()
    if not data_files:
        st.warning("請先在 BTC 訓練資料頁建立多週期資料。")
        return
    if not models:
        st.warning("請先完成至少一個 Transformer 模型。")
        return
    feature_path = st.selectbox("BTC 多週期資料", data_files, format_func=_relative)
    transformer_run = st.selectbox(
        "Transformer 模型", models, format_func=lambda path: path.name
    )
    split = st.columns(3)
    train_percent = split[0].number_input("訓練集 %", 40, 80, 60, 5)
    validation_percent = split[1].number_input("驗證集 %", 10, 30, 20, 5)
    initial_capital = split[2].number_input(
        "初始資金", 100.0, 100_000_000.0, 1_000.0, 100.0
    )
    risk = st.columns(4)
    max_position = risk[0].number_input("最大多頭 %", 5, 100, 50, 5)
    max_short = risk[1].number_input("最大空頭 %", 5, 100, 50, 5)
    leverage = risk[2].number_input("槓桿", 1.0, 3.0, 2.0, 0.5)
    max_drawdown = risk[3].number_input("最大回撤 %", 5.0, 30.0, 10.0, 1.0)
    execution = st.columns(4)
    fee = execution[0].number_input("Taker 手續費 %", 0.0, 1.0, 0.05, 0.01)
    slippage = execution[1].number_input("滑價 %", 0.0, 1.0, 0.05, 0.01)
    take_profit = execution[2].number_input("停利 %", 0.1, 5.0, 1.5, 0.1)
    episode = execution[3].number_input("Episode K 線", 32, 20_000, 2_880, 96)
    if st.button("建立訓練環境", type="primary", width="stretch"):
        try:
            settings = BTCEnvironmentSettings(
                train_fraction=float(train_percent) / 100,
                validation_fraction=float(validation_percent) / 100,
                initial_capital=float(initial_capital),
                max_position_fraction=float(max_position) / 100,
                max_short_fraction=float(max_short) / 100,
                leverage=float(leverage),
                max_drawdown_limit=float(max_drawdown) / 100,
                fee_rate=float(fee) / 100,
                slippage_rate=float(slippage) / 100,
                take_profit_distance=float(take_profit) / 100,
                episode_length=int(episode),
            )
            with st.spinner("正在執行 Transformer 歷史推論與環境診斷..."):
                output = build_btc_rl_environment(
                    PROJECT_ROOT,
                    feature_path,
                    transformer_run / "best_model.pt",
                    settings,
                )
            st.success("SAC/PPO 訓練環境建立完成。")
            st.code(_relative(output), language=None)
        except (OSError, RuntimeError, ValueError, ImportError) as exc:
            st.error(f"建立環境失敗：{exc}")

    environments = list_rl_environments(RL_DIR)
    if environments:
        st.subheader("最近環境")
        themed_dataframe(
            pd.DataFrame({"環境": [_relative(path) for path in environments[:10]]}),
            hide_index=True,
            width="stretch",
        )


def _render_rl() -> None:
    st.title("SAC/PPO 訓練")
    render_active_rl_job(RL_DIR, PROJECT_ROOT)
    phase = st.segmented_control(
        "檢視",
        ["建立訓練", "訓練結果"],
        default="建立訓練",
        width="stretch",
    )
    if phase == "訓練結果":
        render_rl_results(RL_DIR, PROJECT_ROOT)
    else:
        render_rl_training(RL_DIR, PROJECT_ROOT)


def _render_backtest() -> None:
    render_model_backtest_page(
        project_root=PROJECT_ROOT,
        rl_dir=RL_DIR,
        transformer_dir=TRANSFORMER_DIR,
        output_dir=BACKTEST_DIR,
        dark_mode=bool(st.session_state.get("dark_mode", False)),
    )


def _render_export() -> None:
    st.title("匯出模型包")
    st.caption("模型包含特徵順序、標準化、風控環境、Transformer 與 SHA-256 驗證資料。")
    kind = st.segmented_control(
        "模型類型", ["SAC/PPO 完整模型", "Transformer"], default="SAC/PPO 完整模型"
    )
    if kind == "Transformer":
        runs = _transformer_runs()
        if not runs:
            st.warning("沒有可匯出的 Transformer。")
            return
        selected = st.selectbox("訓練結果", runs, format_func=lambda path: path.name)
        exporter = export_transformer_model_package
    else:
        runs = list_rl_training_runs(RL_DIR)
        if not runs:
            st.warning("沒有可匯出的 SAC/PPO 模型。")
            return
        selected = st.selectbox("訓練結果", runs, format_func=lambda path: path.name)
        exporter = export_rl_model_package
    if st.button("建立可攜模型 ZIP", type="primary", width="stretch"):
        try:
            result = exporter(selected, EXPORT_DIR)
            manifest = verify_model_package(result.archive_path)
            st.session_state["latest_model_package"] = str(result.archive_path)
            st.success(
                f"模型包完成：{result.files} 個檔案，"
                f"{result.bytes / 1024 / 1024:.1f} MB"
            )
            st.code(_relative(result.archive_path), language=None)
            st.caption(f"驗證版本：{manifest['schema_version']} · {manifest['package_type']}")
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            st.error(f"模型匯出失敗：{exc}")
    latest = st.session_state.get("latest_model_package")
    if latest and Path(str(latest)).is_file():
        path = Path(str(latest))
        with path.open("rb") as stream:
            st.download_button(
                "下載模型 ZIP",
                data=stream.read(),
                file_name=path.name,
                mime="application/zip",
                width="stretch",
            )


def main() -> None:
    """啟動只服務訓練工作的操作介面。"""
    st.set_page_config(
        page_title="AI Quant 訓練中心",
        page_icon=":material/model_training:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _prepare_directories()
    initialize_theme_state(PROJECT_ROOT)
    from ai_quant_trading.dashboard.app import _apply_style

    _apply_style(bool(st.session_state.get("dark_mode", False)))
    st.sidebar.subheader("AI Quant 訓練")
    st.sidebar.toggle(
        "深色模式",
        key="dark_mode",
        on_change=save_theme_preference,
        args=(PROJECT_ROOT,),
    )
    page = st.sidebar.radio(
        "工作區",
        [
            "總覽",
            "BTC 訓練資料",
            "Transformer",
            "建立 RL 環境",
            "SAC/PPO",
            "模型回測",
            "模型匯出",
        ],
    )
    st.sidebar.caption("此版本不包含 API Key、下單、模擬倉與實盤服務。")
    renderers = {
        "總覽": _render_overview,
        "BTC 訓練資料": _render_data,
        "Transformer": _render_transformer,
        "建立 RL 環境": _render_environment,
        "SAC/PPO": _render_rl,
        "模型回測": _render_backtest,
        "模型匯出": _render_export,
    }
    renderers[page]()


if __name__ == "__main__":
    main()
