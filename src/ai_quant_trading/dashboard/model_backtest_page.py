"""PPO 與 Transformer 歷史回測頁面。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from ai_quant_trading.backtesting import (
    ModelBacktestResult,
    TransformerSignalConfig,
    compare_transformer_horizons,
    filter_history,
    infer_transformer_history,
    list_ppo_backtest_runs,
    list_transformer_backtest_runs,
    load_ppo_history,
    load_transformer_history,
    run_ppo_backtest,
    run_transformer_backtest,
    save_latest_model_backtest,
    transformer_split,
)
from ai_quant_trading.dashboard.model_activity import active_model_training
from ai_quant_trading.dashboard.ui import themed_dataframe, themed_widget_key


PLOTLY_CONFIG = {
    "displaylogo": False,
    "scrollZoom": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}
SPLIT_LABELS = {
    "test": "測試集（樣本外）",
    "validation": "驗證集（模型選擇）",
    "train": "訓練集（樣本內）",
    "all": "全部資料（混合）",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_label(run_dir: Path) -> str:
    try:
        payload = _read_json(run_dir / "training.json")
    except (OSError, ValueError, json.JSONDecodeError):
        return run_dir.name
    source = dict(payload.get("environment_source", {}))
    if not source:
        sources = list(payload.get("sources", []))
        source = dict(sources[0]) if sources else {}
    interval = str(source.get("interval", "-"))
    return f"{run_dir.name}  |  BTC  {interval}"


def _date_range(frame: pd.DataFrame, key: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    minimum = timestamps.min().date()
    maximum = timestamps.max().date()
    selected = st.date_input(
        "歷史區間（UTC）",
        value=(minimum, maximum),
        min_value=minimum,
        max_value=maximum,
        key=key,
    )
    if not isinstance(selected, tuple) or len(selected) != 2:
        raise ValueError("請選擇完整的開始與結束日期")
    start = pd.Timestamp(selected[0], tz="UTC")
    end = pd.Timestamp(selected[1], tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return start, end


def _execution_controls(prefix: str, defaults: dict[str, float]) -> dict[str, float]:
    columns = st.columns(4)
    initial_capital = columns[0].number_input(
        "初始資產 USDT",
        min_value=10.0,
        value=float(defaults.get("initial_capital", 1000.0)),
        step=100.0,
        key=f"{prefix}_capital",
    )
    fee = columns[1].number_input(
        "單邊手續費 %",
        min_value=0.0,
        max_value=2.0,
        value=float(defaults.get("fee_rate", 0.001)) * 100,
        step=0.01,
        key=f"{prefix}_fee",
    )
    slippage = columns[2].number_input(
        "單邊滑價 %",
        min_value=0.0,
        max_value=2.0,
        value=float(defaults.get("slippage_rate", 0.0005)) * 100,
        step=0.01,
        key=f"{prefix}_slippage",
    )
    short_carry = columns[3].number_input(
        "年化空單持有成本 %",
        min_value=0.0,
        max_value=100.0,
        value=float(defaults.get("short_borrow_rate_annual", 0.0)) * 100,
        step=0.1,
        key=f"{prefix}_short_carry",
    )
    return {
        "initial_capital": float(initial_capital),
        "fee_rate": float(fee) / 100,
        "slippage_rate": float(slippage) / 100,
        "short_borrow_rate_annual": float(short_carry) / 100,
    }


def _ratio(value: object) -> str:
    number = float(value or 0.0)
    return f"{number:.2f}" if math.isfinite(number) else "N/A"


def _backtest_figure(result: ModelBacktestResult, dark_mode: bool) -> go.Figure:
    frame = result.evaluation
    text = "#e5e7eb" if dark_mode else "#111827"
    grid = "rgba(148,163,184,0.20)" if dark_mode else "rgba(100,116,139,0.16)"
    background = "#0f172a" if dark_mode else "#ffffff"
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.70, 0.30],
    )
    figure.add_trace(
        go.Scatter(
            x=frame["timestamp"],
            y=frame["equity"],
            name="模型資產",
            line={"color": "#22c55e", "width": 2},
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=frame["timestamp"],
            y=frame["benchmark_equity"],
            name="BTC 持有",
            line={"color": "#f59e0b", "width": 1.5, "dash": "dot"},
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=frame["timestamp"],
            y=frame["position_fraction"],
            name="實際部位",
            line={"color": "#38bdf8", "width": 1.5},
            fill="tozeroy",
            fillcolor="rgba(56,189,248,0.12)",
        ),
        row=2,
        col=1,
    )
    figure.add_hline(y=0, line_width=1, line_color=grid, row=2, col=1)
    figure.update_layout(
        height=520,
        margin={"l": 8, "r": 8, "t": 20, "b": 8},
        paper_bgcolor=background,
        plot_bgcolor=background,
        font={"color": text},
        legend={"orientation": "h", "y": 1.02, "x": 0},
        hovermode="x unified",
    )
    figure.update_xaxes(gridcolor=grid, showgrid=True)
    figure.update_yaxes(gridcolor=grid, showgrid=True, row=1, col=1, title="USDT")
    figure.update_yaxes(gridcolor=grid, showgrid=True, row=2, col=1, title="部位")
    return figure


def _prediction_figure(result: ModelBacktestResult, dark_mode: bool) -> go.Figure:
    frame = result.evaluation.dropna(subset=["predicted_return", "actual_future_return"]).copy()
    background = "#0f172a" if dark_mode else "#ffffff"
    text = "#e5e7eb" if dark_mode else "#111827"
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frame["decision_timestamp"],
            y=frame["predicted_return"] * 100,
            name="預測報酬 %",
            line={"color": "#38bdf8", "width": 1.5},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame["decision_timestamp"],
            y=frame["actual_future_return"] * 100,
            name="實際報酬 %",
            line={"color": "#f59e0b", "width": 1},
        )
    )
    figure.update_layout(
        height=300,
        margin={"l": 8, "r": 8, "t": 20, "b": 8},
        paper_bgcolor=background,
        plot_bgcolor=background,
        font={"color": text},
        legend={"orientation": "h", "y": 1.04, "x": 0},
        hovermode="x unified",
        yaxis_title="報酬 %",
    )
    return figure


def _render_result(result: ModelBacktestResult, run_dir: Path, dark_mode: bool) -> None:
    metrics = result.metrics
    first = st.columns(4)
    first[0].metric("期末資產", f"{float(metrics['final_equity']):,.2f}")
    first[1].metric("總報酬", f"{float(metrics['total_return']):.2%}")
    first[2].metric("最大回撤", f"{float(metrics['max_drawdown']):.2%}")
    first[3].metric("相對 BTC", f"{float(metrics['excess_return']):.2%}")
    second = st.columns(4)
    second[0].metric("Sharpe", _ratio(metrics.get("sharpe_ratio")))
    second[1].metric("Sortino", _ratio(metrics.get("sortino_ratio")))
    second[2].metric("Profit Factor", _ratio(metrics.get("profit_factor")))
    second[3].metric("成交次數", int(metrics.get("trades", 0)))

    benchmarks = dict(result.metadata.get("benchmarks", {}))
    if benchmarks:
        labels = {
            "model_with_costs": "模型（含成本）",
            "model_without_costs": "模型（無成本）",
            "transformer_with_costs": "Transformer（含成本）",
            "transformer_without_costs": "Transformer（無成本）",
            "flat_cash": "空手現金",
            "fixed_ema": "EMA 固定多空",
            "random": "隨機策略",
            "buy_and_hold": "BTC 買入持有",
        }
        comparison = pd.DataFrame.from_dict(benchmarks, orient="index").reset_index()
        comparison["策略"] = comparison.pop("index").map(labels).fillna("其他")
        columns = {
            "total_return": "總報酬",
            "max_drawdown": "最大回撤",
            "profit_factor": "Profit Factor",
            "expectancy": "Expectancy",
            "sharpe_ratio": "Sharpe",
            "trades": "成交次數",
            "fee_to_initial_capital": "成本／本金",
            "liquidation_count": "強平次數",
        }
        comparison = comparison.rename(columns=columns)
        with st.expander("同區間基準比較", expanded=True):
            themed_dataframe(
                comparison[["策略", *[value for value in columns.values() if value in comparison]]],
                hide_index=True,
                width="stretch",
                key=themed_widget_key(f"{result.model_kind}_benchmark_table"),
            )

    st.plotly_chart(
        _backtest_figure(result, dark_mode),
        width="stretch",
        config=PLOTLY_CONFIG,
        key=f"{result.model_kind}_backtest_chart",
    )
    if result.model_kind == "transformer":
        prediction = st.columns(4)
        prediction[0].metric("方向準確率", f"{float(metrics.get('direction_accuracy', 0)):.2%}")
        prediction[1].metric("報酬 MAE", f"{float(metrics.get('return_mae', 0)):.4%}")
        prediction[2].metric("預測相關性", _ratio(metrics.get("return_correlation")))
        prediction[3].metric("訊號覆蓋率", f"{float(metrics.get('signal_coverage', 0)):.2%}")
        st.plotly_chart(
            _prediction_figure(result, dark_mode),
            width="stretch",
            config=PLOTLY_CONFIG,
            key="transformer_prediction_chart",
        )

    trades = result.evaluation.loc[result.evaluation["side"] != "HOLD"].copy()
    with st.expander(f"成交紀錄（{len(trades)}）"):
        if trades.empty:
            st.info("這組條件沒有成交。")
        else:
            columns = [
                "timestamp",
                "side",
                "target_fraction",
                "position_fraction",
                "trade_notional",
                "fee",
                "slippage_cost",
                "equity",
            ]
            themed_dataframe(
                trades[[column for column in columns if column in trades]],
                hide_index=True,
                width="stretch",
                key=themed_widget_key(f"{result.model_kind}_backtest_trades"),
            )
    st.download_button(
        "下載本次逐根結果",
        data=result.evaluation.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{result.model_kind}_backtest_latest.csv",
        mime="text/csv",
        icon=":material/download:",
        key=f"download_{result.model_kind}_backtest",
    )
    st.caption(f"已覆寫保存：{run_dir}")


def _render_ppo(
    rl_dir: Path,
    output_dir: Path,
    dark_mode: bool,
    model_training_active: bool,
) -> None:
    runs = list_ppo_backtest_runs(rl_dir)
    if not runs:
        st.warning("目前沒有可回測的 BTC PPO／SAC 模型。")
        return
    selected = st.selectbox("強化學習模型", runs, format_func=_run_label, key="bt_ppo_run")
    split_label = st.segmented_control(
        "資料區段",
        list(SPLIT_LABELS.values())[:3],
        default=SPLIT_LABELS["test"],
        width="stretch",
        key="bt_ppo_split",
    )
    split = next(key for key, label in SPLIT_LABELS.items() if label == split_label)
    try:
        frame, training, environment = load_ppo_history(selected, split)  # type: ignore[arg-type]
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        st.error(str(exc))
        return
    source = dict(environment.get("source", {}))
    summary = st.columns(4)
    summary[0].metric("K 線週期", str(source.get("interval", "-")))
    summary[1].metric("資料筆數", f"{len(frame):,}")
    summary[2].metric("模型特徵", len(environment.get("feature_columns", [])))
    summary[3].metric(
        "Transformer",
        "有" if dict(environment.get("ai_context", {})).get("transformer_enabled") else "無",
    )
    if split == "test":
        st.success("此區段未參與模型訓練，屬於樣本外測試。")
    elif split == "validation":
        st.warning("驗證集可能參與模型選擇，不可當作最終績效。")
    else:
        st.warning("訓練集結果只用來檢查擬合，不可證明未來獲利。")
    try:
        start, end = _date_range(frame, f"bt_ppo_dates_{selected.name}_{split}")
    except ValueError as exc:
        st.warning(str(exc))
        return
    defaults = dict(environment.get("environment_config", {}))
    execution = _execution_controls("bt_ppo", defaults)
    controls = st.columns(3)
    deterministic = controls[0].toggle("確定性推論", value=True, key="bt_ppo_deterministic")
    seed = controls[1].number_input("Seed", min_value=0, value=42, step=1, key="bt_ppo_seed")
    device_label = controls[2].segmented_control(
        "推論裝置", ["CPU", "自動"], default="CPU", key="bt_ppo_device"
    )
    if st.button(
        "執行 RL 歷史回測",
        type="primary",
        icon=":material/query_stats:",
        width="stretch",
        key="run_ppo_backtest",
        disabled=model_training_active,
    ):
        try:
            selected_frame = filter_history(frame, start=start, end=end)
            with st.spinner("RL 模型正在逐根產生部位並模擬成交..."):
                result = run_ppo_backtest(
                    selected,
                    selected_frame,
                    **execution,
                    deterministic=bool(deterministic),
                    seed=int(seed),
                    device="cpu" if device_label == "CPU" else "auto",
                )
                run_dir = save_latest_model_backtest(
                    result,
                    output_dir,
                    extra_metadata={
                        "dataset_split": split,
                        "training_status": training.get("status"),
                    },
                )
            st.session_state["ppo_model_backtest"] = (result, run_dir)
        except (OSError, RuntimeError, ValueError) as exc:
            st.error(str(exc))
    saved = st.session_state.get("ppo_model_backtest")
    if saved:
        _render_result(saved[0], saved[1], dark_mode)


def _render_transformer(
    transformer_dir: Path,
    output_dir: Path,
    project_root: Path,
    dark_mode: bool,
    model_training_active: bool,
) -> None:
    runs = list_transformer_backtest_runs(transformer_dir)
    if not runs:
        st.warning("目前沒有可回測的 BTC Transformer 模型。")
        return
    selected = st.selectbox(
        "Transformer 模型", runs, format_func=_run_label, key="bt_transformer_run"
    )
    try:
        full_frame, metadata, checkpoint, source_path = load_transformer_history(
            selected, project_root
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        st.error(str(exc))
        return
    horizons = [
        int(value)
        for value in dict(metadata.get("model_config", {})).get("return_horizons", [1, 5, 20])
    ]
    split_label = st.segmented_control(
        "資料區段",
        list(SPLIT_LABELS.values()),
        default=SPLIT_LABELS["test"],
        width="stretch",
        key="bt_transformer_split",
    )
    split = next(key for key, label in SPLIT_LABELS.items() if label == split_label)
    partition = transformer_split(full_frame, metadata, split)  # type: ignore[arg-type]
    overview = st.columns(4)
    overview[0].metric("基礎週期", str(full_frame.get("interval", pd.Series(["-"])).iloc[0]))
    overview[1].metric("區段筆數", f"{len(partition):,}")
    overview[2].metric("模型特徵", len(metadata.get("feature_columns", [])))
    overview[3].metric(
        "序列長度", int(dict(metadata.get("model_config", {})).get("sequence_length", 0))
    )
    if split == "test":
        st.success("此區段未參與 Transformer 權重訓練，屬於樣本外測試。")
    else:
        st.warning("目前不是最終樣本外區段，績效只能用於研究診斷。")
    try:
        start, end = _date_range(partition, f"bt_transformer_dates_{selected.name}_{split}")
    except ValueError as exc:
        st.warning(str(exc))
        return

    signal_columns = st.columns(4)
    horizon = signal_columns[0].selectbox(
        "預測週期（K 線）", horizons, index=min(1, len(horizons) - 1), key="bt_tr_horizon"
    )
    minimum_return = signal_columns[1].number_input(
        "最小預測報酬 %", 0.0, 10.0, 0.03, 0.01, key="bt_tr_min_return"
    )
    minimum_probability = signal_columns[2].number_input(
        "牛熊機率門檻", 0.0, 1.0, 0.35, 0.01, key="bt_tr_probability"
    )
    maximum_uncertainty = signal_columns[3].number_input(
        "不確定性上限", 0.0, 1.0, 0.80, 0.01, key="bt_tr_uncertainty"
    )
    sizing = st.columns(4)
    volatility_multiple = sizing[0].number_input(
        "波動調整倍數", 0.1, 10.0, 1.0, 0.1, key="bt_tr_vol_multiple"
    )
    max_long = sizing[1].number_input("最大多頭 %", 1.0, 100.0, 100.0, 5.0, key="bt_tr_max_long")
    allow_short = sizing[2].toggle("允許作空", value=True, key="bt_tr_allow_short")
    max_short = sizing[3].number_input(
        "最大空頭 %",
        1.0,
        100.0,
        75.0,
        5.0,
        disabled=not allow_short,
        key="bt_tr_max_short",
    )
    cost_gate = st.columns(4)
    cost_multiple = cost_gate[0].number_input(
        "來回成本倍數", 0.0, 10.0, 1.0, 0.25, key="bt_tr_cost_multiple"
    )
    minimum_net_return = cost_gate[1].number_input(
        "最低淨優勢 %", 0.0, 5.0, 0.0, 0.01, key="bt_tr_min_net_return"
    )
    exit_threshold_ratio = cost_gate[2].number_input(
        "續抱門檻比例", 0.0, 1.0, 0.50, 0.05, key="bt_tr_exit_ratio"
    )
    minimum_tradeability = cost_gate[3].number_input(
        "可交易機率門檻", 0.0, 1.0, 0.0, 0.05, key="bt_tr_tradeability"
    )
    execution = _execution_controls("bt_transformer", {})
    risk = st.columns(4)
    deadband = risk[0].number_input("再平衡死區 %", 0.0, 50.0, 3.0, 0.5, key="bt_tr_deadband")
    maximum_drawdown = risk[1].number_input(
        "最大回撤限制 %", 1.0, 100.0, 30.0, 1.0, key="bt_tr_drawdown"
    )
    batch_size = risk[2].number_input("推論 Batch", 1, 4096, 512, 64, key="bt_tr_batch")
    device_label = risk[3].selectbox("推論裝置", ["自動", "CPU", "CUDA"], key="bt_tr_device")
    if st.button(
        "執行 Transformer 歷史回測",
        type="primary",
        icon=":material/query_stats:",
        width="stretch",
        key="run_transformer_backtest",
        disabled=model_training_active,
    ):
        try:
            inference_key = (
                str(checkpoint),
                checkpoint.stat().st_mtime_ns,
                str(source_path),
                source_path.stat().st_mtime_ns,
            )
            cached = st.session_state.get("transformer_backtest_inference")
            if cached and cached[0] == inference_key:
                inferred = cached[1]
            else:
                progress = st.progress(0.0, text="Transformer 正在推論歷史序列...")

                def update(payload: dict[str, object]) -> None:
                    value = float(payload.get("progress", 0.0))
                    progress.progress(
                        min(max(value, 0.0), 1.0),
                        text=f"Transformer 推論 {value:.0%}",
                    )

                inferred = infer_transformer_history(
                    checkpoint,
                    full_frame,
                    device={"自動": "auto", "CPU": "cpu", "CUDA": "cuda"}[device_label],
                    batch_size=int(batch_size),
                    progress_callback=update,
                )
                progress.progress(1.0, text="Transformer 推論完成")
                st.session_state["transformer_backtest_inference"] = (
                    inference_key,
                    inferred,
                )
            selected_frame = transformer_split(inferred, metadata, split)  # type: ignore[arg-type]
            selected_frame = filter_history(selected_frame, start=start, end=end)
            signal_config = TransformerSignalConfig(
                horizon=int(horizon),
                minimum_return=float(minimum_return) / 100,
                minimum_net_return=float(minimum_net_return) / 100,
                round_trip_cost_multiple=float(cost_multiple),
                exit_threshold_ratio=float(exit_threshold_ratio),
                minimum_regime_probability=float(minimum_probability),
                minimum_tradeability=float(minimum_tradeability),
                maximum_uncertainty=float(maximum_uncertainty),
                volatility_multiple=float(volatility_multiple),
                max_long_fraction=float(max_long) / 100,
                allow_short=bool(allow_short),
                max_short_fraction=float(max_short) / 100 if allow_short else 0.0,
            )
            backtest_arguments = {
                **execution,
                "rebalance_deadband": float(deadband) / 100,
                "maximum_drawdown": float(maximum_drawdown) / 100,
            }
            comparison_horizons = tuple(
                value
                for value in (5, 20)
                if f"transformer_return_{value}" in selected_frame
            )
            if len(comparison_horizons) == 2:
                comparison = compare_transformer_horizons(
                    selected_frame,
                    signal_config,
                    horizons=comparison_horizons,
                    **backtest_arguments,
                )
                result = comparison.results.get(int(horizon)) or run_transformer_backtest(
                    selected_frame,
                    signal_config,
                    **backtest_arguments,
                )
                st.session_state["transformer_horizon_comparison"] = comparison.leaderboard
                result.metadata["horizon_comparison"] = json.loads(
                    comparison.leaderboard.to_json(orient="records")
                )
            else:
                result = run_transformer_backtest(
                    selected_frame,
                    signal_config,
                    **backtest_arguments,
                )
                st.session_state.pop("transformer_horizon_comparison", None)
            result.metadata.update(
                {
                    "training_dir": str(selected),
                    "checkpoint": str(checkpoint),
                    "source_path": str(source_path),
                }
            )
            run_dir = save_latest_model_backtest(
                result,
                output_dir,
                extra_metadata={"dataset_split": split},
            )
            st.session_state["transformer_model_backtest"] = (result, run_dir)
        except (OSError, RuntimeError, ValueError) as exc:
            st.error(str(exc))
    saved = st.session_state.get("transformer_model_backtest")
    if saved:
        horizon_comparison = st.session_state.get("transformer_horizon_comparison")
        if isinstance(horizon_comparison, pd.DataFrame):
            with st.expander("5 根與 20 根扣成本比較", expanded=True):
                if split == "test":
                    st.caption("測試集只用於最終報告；請勿看完此表再回頭挑參數。")
                themed_dataframe(
                    horizon_comparison,
                    hide_index=True,
                    width="stretch",
                    key=themed_widget_key("transformer_horizon_comparison"),
                )
        _render_result(saved[0], saved[1], dark_mode)


def render_model_backtest_page(
    *,
    project_root: Path,
    rl_dir: Path,
    transformer_dir: Path,
    output_dir: Path,
    dark_mode: bool,
) -> None:
    """顯示只服務 AI 模型的歷史回測工作區。"""
    st.title("AI 歷史回測")
    activity = active_model_training(rl_dir, transformer_dir)
    if activity is not None:
        st.warning(f"{activity.label}正在執行，模型回測暫時鎖定。")
    model_kind = st.segmented_control(
        "回測模型",
        ["PPO／SAC 模型", "Transformer 模型"],
        default="PPO／SAC 模型",
        width="stretch",
    )
    if model_kind == "PPO／SAC 模型":
        _render_ppo(rl_dir, output_dir, dark_mode, activity is not None)
    else:
        _render_transformer(
            transformer_dir,
            output_dir,
            project_root,
            dark_mode,
            activity is not None,
        )
