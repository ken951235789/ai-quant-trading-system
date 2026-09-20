"""Transformer 背景訓練、進度監控與結果檢查頁。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from ai_quant_trading.ai_pipeline import (
    AIPipelineConfig,
    load_ai_pipeline_config,
    save_ai_pipeline_config,
)
from ai_quant_trading.dashboard.model_activity import active_model_training
from ai_quant_trading.dashboard.services import (
    list_processed_files,
    relative_file_label,
)
from ai_quant_trading.dashboard.ui import themed_dataframe, themed_widget_key
from ai_quant_trading.dashboard.transformer_training_jobs import (
    ACTIVE_TRANSFORMER_JOB_STATUSES,
    TransformerTrainingJob,
    latest_transformer_training_job,
    read_transformer_training_progress,
    submit_transformer_training_job,
)
from ai_quant_trading.features import BTC_MULTITIMEFRAME_INTERVALS
from ai_quant_trading.performance import available_cpu_threads
from ai_quant_trading.transformer import (
    TemporalTransformerConfig,
    TransformerTrainingConfig,
    apply_transformer_checkpoint,
    transformer_backend_status,
)


def _duration_text(value: object) -> str:
    try:
        seconds = max(0, int(float(value)))
    except (TypeError, ValueError):
        return "-"
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours} 小時 {minutes} 分"
    if minutes:
        return f"{minutes} 分 {seconds} 秒"
    return f"{seconds} 秒"


def _parse_horizons(value: str) -> tuple[int, ...]:
    try:
        result = tuple(
            dict.fromkeys(int(item.strip()) for item in value.split(",") if item.strip())
        )
    except ValueError as exc:
        raise ValueError("預測週期請使用逗號分隔的正整數") from exc
    if not result or any(item <= 0 for item in result):
        raise ValueError("預測週期必須是正整數")
    return result


def _parse_quantiles(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("分位數請使用逗號分隔的小數") from exc
    if len(result) < 3 or tuple(sorted(result)) != result:
        raise ValueError("分位數至少三個，並且必須由小到大排列")
    if any(not 0 < item < 1 for item in result):
        raise ValueError("分位數必須介於 0 與 1 之間")
    return result


def _parse_horizon_weights(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("Checkpoint 權重請使用逗號分隔的非負數") from exc
    if result and (any(item < 0 for item in result) or not any(result)):
        raise ValueError("Checkpoint 權重不可為負，且至少需要一個正數")
    return result


def _load_pipeline(path: Path) -> AIPipelineConfig:
    try:
        return load_ai_pipeline_config(path)
    except (OSError, ValueError, TypeError):
        return AIPipelineConfig()


def _relative(path: Path, root: Path) -> str:
    return relative_file_label(path, root)


def _job_history(job: TransformerTrainingJob) -> pd.DataFrame:
    if not job.history_csv.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(job.history_csv)
    except (OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()
    return frame.tail(400)


def _chart_frame(
    frame: pd.DataFrame,
    x_column: str,
    value_columns: list[str],
) -> pd.DataFrame:
    """移除空白監控列，避免圖表在尚無有效資料時產生無限座標警告。"""
    if x_column not in frame or not value_columns:
        return pd.DataFrame()
    result = frame[[x_column, *value_columns]].copy()
    for column in result:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=[x_column]).dropna(
        subset=value_columns,
        how="all",
    )
    return result if len(result) >= 2 else pd.DataFrame()


def _render_job_monitor(transformer_root: Path, project_root: Path) -> None:
    job = latest_transformer_training_job(transformer_root)
    if job is None:
        return
    progress = read_transformer_training_progress(job)
    raw_training_config = job.payload.get("training_config", {})
    training_config = raw_training_config if isinstance(raw_training_config, dict) else {}
    fraction = min(max(float(progress.get("progress", 0.0)), 0.0), 1.0)
    status = job.status
    st.markdown("#### 背景訓練監控")
    if status in ACTIVE_TRANSFORMER_JOB_STATUSES:
        epoch = int(progress.get("epoch", 0))
        epochs = int(progress.get("epochs", 0))
        batch = int(progress.get("batch", 0) or 0)
        batches = int(progress.get("batches", 0) or 0)
        stage = "資料準備中" if progress.get("status") == "preparing" else "訓練中"
        if progress.get("status") == "validating":
            stage = "驗證中"
        label = f"{stage} · Epoch {epoch}/{epochs}"
        if batches:
            label += f" · Batch {batch}/{batches}"
        st.progress(fraction, text=label)
    elif status == "complete":
        st.success("Transformer 訓練已完成，可在下方「訓練結果」檢查模型。")
    elif status == "interrupted":
        st.warning(str(job.payload.get("error", "訓練已中斷")))
    else:
        st.error(f"Transformer 訓練失敗：{job.payload.get('error', '未知錯誤')}")

    raw_metrics = progress.get("metrics", {})
    metrics = raw_metrics if isinstance(raw_metrics, dict) else {}
    samples = progress.get("sample_counts", {})
    samples = samples if isinstance(samples, dict) else {}
    columns = st.columns(6)
    columns[0].metric("完成度", f"{fraction:.1%}")
    columns[1].metric("Train Loss", f"{float(metrics.get('train_loss', 0)):.5f}")
    validation_loss = metrics.get("validation_loss")
    columns[2].metric(
        "Validation Loss",
        "-" if validation_loss is None else f"{float(validation_loss):.5f}",
    )
    regime_accuracy = metrics.get(
        "validation_regime_accuracy",
        metrics.get("test_regime_accuracy"),
    )
    columns[3].metric(
        "狀態準確率",
        ("-" if regime_accuracy is None else f"{float(regime_accuracy):.1%}"),
    )
    columns[4].metric("GPU 記憶體", f"{float(metrics.get('gpu_memory_gb', 0)):.2f} GB")
    columns[5].metric("預估剩餘", _duration_text(progress.get("eta_seconds")))
    st.caption(
        f"工作 {job.job_id} · 裝置 {progress.get('device', '-')} · "
        f"CPU {training_config.get('cpu_threads', '-')} 執行緒 · "
        f"特徵 {int(progress.get('feature_count', 0) or 0)} · "
        f"參數 {int(progress.get('parameters', 0) or 0):,} · "
        f"樣本 Train/Val/Test "
        f"{int(samples.get('train', 0)):,}/"
        f"{int(samples.get('validation', 0)):,}/"
        f"{int(samples.get('test', 0)):,} · "
        f"已執行 {_duration_text(progress.get('elapsed_seconds'))}"
    )
    history = _job_history(job)
    if not history.empty:
        charts = st.columns(2)
        loss_columns = [
            column
            for column in ["train_loss", "validation_loss"]
            if column in history and history[column].notna().any()
        ]
        accuracy_columns = [
            column
            for column in [
                "validation_regime_accuracy",
                "test_regime_accuracy",
                "direction_accuracy",
                "validation_cost_aware_direction_accuracy",
            ]
            if column in history and history[column].notna().any()
        ]
        loss_frame = _chart_frame(history, "progress", loss_columns)
        accuracy_frame = _chart_frame(history, "progress", accuracy_columns)
        if not loss_frame.empty:
            charts[0].line_chart(
                loss_frame,
                x="progress",
                height=220,
            )
        if not accuracy_frame.empty:
            charts[1].line_chart(
                accuracy_frame,
                x="progress",
                height=220,
            )
    result_dir = progress.get("result_dir") or job.payload.get("result_dir")
    if result_dir:
        st.caption(f"結果位置：{_relative(Path(str(result_dir)), project_root)}")
    st.divider()


@st.fragment(run_every=5)
def render_active_transformer_job(
    transformer_root: Path,
    project_root: Path,
) -> None:
    """每兩秒只更新監控區，切換功能頁後仍能從磁碟接回進度。"""
    _render_job_monitor(transformer_root, project_root)


def _list_model_runs(transformer_root: Path) -> list[Path]:
    models_root = transformer_root / "models"
    if not models_root.exists():
        return []
    return sorted(
        (path.parent for path in models_root.glob("*/training.json") if path.is_file()),
        key=lambda path: path.name,
        reverse=True,
    )


def _render_results(transformer_root: Path, project_root: Path) -> None:
    runs = _list_model_runs(transformer_root)
    if not runs:
        st.info("目前沒有 Transformer 訓練結果。")
        return
    selected = st.selectbox(
        "訓練紀錄",
        runs,
        format_func=lambda path: path.name,
        key="transformer_result_run",
    )
    try:
        summary = json.loads((selected / "training.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        st.error(f"無法讀取訓練結果：{exc}")
        return
    metrics = dict(summary.get("test_metrics", {}))
    sample_counts = dict(summary.get("sample_counts", {}))
    model_config = dict(summary.get("model_config", {}))
    cards = st.columns(6)
    cards[0].metric("最佳 Epoch", int(summary.get("best_epoch", 0)))
    cards[1].metric("Test Loss", f"{float(metrics.get('loss', 0)):.5f}")
    cards[2].metric(
        "成本感知方向",
        f"{float(metrics.get('cost_aware_direction_accuracy', metrics.get('direction_accuracy', 0))):.1%}",
    )
    cards[3].metric(
        "狀態準確率",
        f"{float(metrics.get('regime_accuracy', 0)):.1%}",
    )
    cards[4].metric("特徵數", int(model_config.get("input_features", 0)))
    cards[5].metric("訓練時間", _duration_text(summary.get("duration_seconds")))
    mtf_features = [
        column for column in summary.get("feature_columns", []) if str(column).startswith("mtf_")
    ]
    if mtf_features:
        st.success(f"這是多週期 Transformer，共使用 {len(mtf_features)} 個跨週期欄位。")
    history_path = selected / "history.csv"
    if history_path.exists():
        history = pd.read_csv(history_path)
        chart = _chart_frame(
            history,
            "epoch",
            ["train_loss", "validation_loss"],
        )
        if not chart.empty:
            st.line_chart(
                chart,
                x="epoch",
                height=280,
            )
    source_rows = []
    for source in summary.get("sources", []):
        source_rows.append(
            {
                "市場": source.get("symbol", "-"),
                "交易所": source.get("exchange", "-"),
                "週期": source.get("interval", "-"),
                "K線": int(source.get("rows", 0)),
                "開始": source.get("start_at", "-"),
                "結束": source.get("end_at", "-"),
            }
        )
    if source_rows:
        themed_dataframe(
            pd.DataFrame(source_rows),
            width="stretch",
            hide_index=True,
            key=themed_widget_key("transformer_result_sources"),
        )
    st.caption(
        f"Train/Val/Test：{int(sample_counts.get('train', 0)):,}/"
        f"{int(sample_counts.get('validation', 0)):,}/"
        f"{int(sample_counts.get('test', 0)):,}"
    )
    st.caption(f"模型：{_relative(selected / 'best_model.pt', project_root)}")
    with st.expander("實際使用的特徵"):
        st.code("\n".join(summary.get("feature_columns", [])), language="text")


def _render_inference(
    transformer_root: Path,
    project_root: Path,
    processed_dir: Path,
    model_training_active: bool,
) -> None:
    """把正式模型預測與 latent 向量寫回 Step 3 特徵檔。"""
    runs = [run for run in _list_model_runs(transformer_root) if (run / "best_model.pt").exists()]
    if not runs:
        st.warning("請先完成至少一次 Transformer 訓練。")
        return
    files = [
        path
        for path in list_processed_files(processed_dir)
        if path.name.startswith("features_mtf_crypto_binance_futures_BTC-USDT_15m_")
    ]
    if not files:
        st.warning("請先建立 Step 3 特徵 CSV。")
        return
    st.subheader("批次推論到 SAC 特徵")
    st.caption(
        "使用 checkpoint 內保存的特徵順序與訓練期 scaler；"
        "結果直接更新固定特徵檔，SAC 建立環境時會自動讀取。"
    )
    selected_run = st.selectbox(
        "Transformer 模型",
        runs,
        format_func=lambda path: path.name,
        key="transformer_inference_model",
    )
    trained_intervals: set[str] = set()
    trained_paths: set[str] = set()
    model_uses_multitimeframe = False
    try:
        summary = json.loads((selected_run / "training.json").read_text(encoding="utf-8"))
        for source in summary.get("sources", []):
            interval = str(source.get("interval", "")).lower()
            if interval:
                trained_intervals.add(interval)
            trained_paths.add(str(Path(str(source.get("path", ""))).resolve()).lower())
        model_uses_multitimeframe = any(
            str(column).startswith("mtf_") for column in summary.get("feature_columns", [])
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    compatible_files: list[Path] = []
    for path in files:
        try:
            first = pd.read_csv(path, nrows=1).iloc[0]
            interval = str(first.get("interval", "")).lower()
            file_uses_multitimeframe = "mtf_source_intervals" in first.index
        except (OSError, ValueError, IndexError):
            continue
        if (
            not trained_intervals or interval in trained_intervals
        ) and file_uses_multitimeframe == model_uses_multitimeframe:
            compatible_files.append(path)
    default_files = [
        path for path in compatible_files if str(path.resolve()).lower() in trained_paths
    ]
    if not default_files:
        default_files = compatible_files[: min(7, len(compatible_files))]
    if trained_intervals:
        st.caption(
            f"模型訓練週期：{'、'.join(sorted(trained_intervals))}；"
            f"{'多週期融合' if model_uses_multitimeframe else '單週期'}；"
            "只顯示相同資料架構的特徵檔。"
        )
    selected_files = st.multiselect(
        "推論市場",
        compatible_files,
        default=default_files,
        format_func=lambda path: _relative(path, project_root),
        max_selections=50,
        key="transformer_inference_files",
    )
    settings = st.columns(3)
    device = settings[0].selectbox(
        "推論裝置",
        ["auto", "cuda", "cpu"],
        key="transformer_inference_device",
    )
    batch_size = settings[1].number_input(
        "Batch size",
        1,
        8192,
        512,
        1,
        key="transformer_inference_batch",
    )
    mixed_precision = settings[2].toggle(
        "CUDA 混合精度",
        value=True,
        key="transformer_inference_amp",
    )
    if st.button(
        "執行 Transformer 推論",
        type="primary",
        icon=":material/neurology:",
        width="stretch",
        disabled=model_training_active or not selected_files,
        key="transformer_inference_run",
    ):
        overall = st.progress(0.0, text="準備推論")
        rows: list[dict[str, object]] = []
        for file_index, path in enumerate(selected_files):

            def update_file_progress(payload: dict[str, object]) -> None:
                local = float(payload.get("progress", 0.0))
                fraction = (file_index + local) / len(selected_files)
                overall.progress(
                    min(max(fraction, 0.0), 1.0),
                    text=(
                        f"{path.name} · Batch {payload.get('batch', 0)}/{payload.get('batches', 0)}"
                    ),
                )

            try:
                artifact = apply_transformer_checkpoint(
                    selected_run / "best_model.pt",
                    path,
                    device=str(device),
                    batch_size=int(batch_size),
                    mixed_precision=bool(mixed_precision),
                    progress_callback=update_file_progress,
                )
                rows.append(
                    {
                        "市場檔": _relative(path, project_root),
                        "總列數": artifact.rows,
                        "有效預測": artifact.predicted_rows,
                        "裝置": artifact.device,
                        "秒數": round(artifact.duration_seconds, 2),
                        "狀態": "完成",
                    }
                )
            except (OSError, ValueError, RuntimeError) as exc:
                rows.append(
                    {
                        "市場檔": _relative(path, project_root),
                        "總列數": 0,
                        "有效預測": 0,
                        "裝置": str(device),
                        "秒數": 0,
                        "狀態": f"失敗：{exc}",
                    }
                )
        overall.progress(1.0, text="批次推論完成")
        themed_dataframe(
            pd.DataFrame(rows),
            width="stretch",
            hide_index=True,
            key=themed_widget_key("transformer_inference_results"),
        )


def _render_training_form(
    project_root: Path,
    processed_dir: Path,
    transformer_root: Path,
    pipeline_path: Path,
    saved: AIPipelineConfig,
    model_training_active: bool,
) -> None:
    files = list_processed_files(processed_dir)
    files = [
        path
        for path in files
        if path.name.startswith("features_mtf_crypto_binance_futures_BTC-USDT_15m_")
        and "transformer_" not in path.name
    ]
    if not files:
        st.warning("請先在「特徵資料」建立至少一份 Step 3 特徵 CSV。")
        return
    quick_test = st.toggle(
        "快速驗證模式",
        value=True,
        help="限制每個市場的資料量並使用 1～2 Epoch，用來確認流程與 GPU 可正常執行。",
    )
    defaults = files[:1]
    selected_files = st.multiselect(
        "訓練市場特徵",
        files,
        default=defaults,
        format_func=lambda path: _relative(path, project_root),
        max_selections=20,
    )
    uses_multitimeframe = bool(selected_files) and all(
        path.name.startswith("features_mtf_") for path in selected_files
    )
    if (
        selected_files
        and not uses_multitimeframe
        and any(path.name.startswith("features_mtf_") for path in selected_files)
    ):
        st.warning("單週期與多週期資料不可放在同一次 Transformer 訓練。")
    mtf_intervals: set[str] = set()
    mtf_feature_count = 0
    if uses_multitimeframe:
        for path in selected_files:
            try:
                preview = pd.read_csv(path, nrows=1)
            except (OSError, ValueError, pd.errors.ParserError):
                continue
            if preview.empty:
                continue
            mtf_intervals.update(
                value
                for value in str(preview.iloc[0].get("mtf_source_intervals", "")).split("|")
                if value
            )
            mtf_feature_count = max(
                mtf_feature_count,
                sum(
                    column.startswith("mtf_")
                    and column
                    not in {
                        "mtf_decision_interval",
                        "mtf_source_intervals",
                        "mtf_schema_version",
                    }
                    for column in preview.columns
                ),
            )
    run_name = st.text_input(
        "訓練名稱",
        value=(
            "quick_mtf_check"
            if quick_test and uses_multitimeframe
            else "quick_check"
            if quick_test
            else "multitimeframe_transformer"
            if uses_multitimeframe
            else "market_transformer"
        ),
    )
    if uses_multitimeframe:
        missing_intervals = [
            value for value in BTC_MULTITIMEFRAME_INTERVALS if value not in mtf_intervals
        ]
        if missing_intervals:
            st.warning("多週期訓練資料缺少：" + "、".join(missing_intervals))
        else:
            st.success(
                f"BTC 15m 五週期 Transformer · {len(mtf_intervals)}/"
                f"{len(BTC_MULTITIMEFRAME_INTERVALS)} 週期 · "
                f"{mtf_feature_count} 個時間尺度指標"
            )
    else:
        st.caption(
            "多市場訓練必須使用相同 K 線週期；資料依時間切成 Train、Validation、Test，"
            "不會隨機打散或跨市場拼接序列。"
        )

    st.subheader("模型架構")
    architecture = st.columns(4)
    recommended_mtf_features = min(
        384,
        max(192, saved.transformer.input_features),
    )
    maximum_features = architecture[0].number_input(
        "最多輸入特徵",
        1,
        1024,
        recommended_mtf_features
        if uses_multitimeframe
        else 64
        if quick_test
        else saved.transformer.input_features,
        1,
        help=(
            "低硬體模式依固定優先順序保留高價值特徵；提高上限才會納入全部欄位。"
            if uses_multitimeframe
            else None
        ),
    )
    sequence_length = architecture[1].number_input(
        "序列長度",
        8,
        2048,
        saved.transformer.sequence_length,
        8,
    )
    d_model = architecture[2].number_input(
        "d_model",
        16,
        2048,
        min(saved.transformer.d_model, 64) if quick_test else saved.transformer.d_model,
        16,
    )
    n_heads = architecture[3].number_input(
        "注意力頭數",
        1,
        64,
        min(saved.transformer.n_heads, 8),
        1,
    )
    depth = st.columns(4)
    n_layers = depth[0].number_input(
        "Encoder 層數",
        1,
        48,
        min(saved.transformer.n_layers, 2) if quick_test else saved.transformer.n_layers,
        1,
    )
    feedforward = depth[1].number_input(
        "前饋維度",
        16,
        8192,
        min(saved.transformer.feedforward_dim, 128)
        if quick_test
        else saved.transformer.feedforward_dim,
        16,
    )
    dropout = depth[2].number_input(
        "Dropout",
        0.0,
        0.9,
        saved.transformer.dropout,
        0.01,
    )
    latent_dim = depth[3].number_input(
        "SAC 壓縮向量",
        1,
        512,
        saved.transformer.latent_dim,
        1,
    )
    task = st.columns(2)
    horizons = task[0].text_input(
        "報酬預測週期",
        value=",".join(map(str, saved.transformer.return_horizons)),
        help="單位是 K 線根數，例如日線 1,5,20 代表 1、5、20 個交易週期。",
    )
    regime_classes = task[1].number_input(
        "行情狀態數",
        2,
        12,
        saved.transformer.regime_classes,
        1,
    )
    with st.expander("Transformer v3 架構與機率輸出"):
        advanced_architecture = st.columns(5)
        advanced_architecture[0].metric("架構版本", "V3")
        architecture_version = 3
        local_kernel_size = advanced_architecture[1].number_input(
            "局部卷積核",
            1,
            15,
            saved.transformer.local_kernel_size,
            2,
        )
        quantile_levels = advanced_architecture[2].text_input(
            "報酬分位數",
            value=",".join(map(str, saved.transformer.quantile_levels)),
            help="建議 0.1,0.5,0.9，供風控估計下檔、中央與上檔報酬。",
        )
        patch_size = advanced_architecture[3].number_input(
            "Patch 大小",
            1,
            64,
            saved.transformer.patch_size,
            1,
        )
        patch_stride = advanced_architecture[4].number_input(
            "Patch 步幅",
            1,
            64,
            saved.transformer.patch_stride,
            1,
        )
        hierarchical_controls = st.columns(2)
        hierarchical_direction = hierarchical_controls[0].toggle(
            "階層式方向預測",
            value=saved.transformer.hierarchical_direction,
            help="先判斷是否值得交易，再判斷多空方向；正式短線模型建議開啟。",
        )
        horizon_adapter_dim = hierarchical_controls[1].number_input(
            "Horizon adapter 維度",
            1,
            512,
            max(saved.transformer.horizon_adapter_dim, 64),
            1,
            disabled=not hierarchical_direction,
        )

    st.subheader("訓練設定")
    training = st.columns(4)
    epochs = training[0].number_input(
        "Epochs",
        1,
        10_000,
        min(saved.transformer_training.epochs, 2)
        if quick_test
        else saved.transformer_training.epochs,
        1,
    )
    batch_size = training[1].number_input(
        "Batch size",
        1,
        8192,
        saved.transformer_training.batch_size,
        1,
    )
    learning_rate = training[2].number_input(
        "Learning rate",
        1e-7,
        1.0,
        saved.transformer_training.learning_rate,
        format="%.7f",
    )
    weight_decay = training[3].number_input(
        "Weight decay",
        0.0,
        1.0,
        saved.transformer_training.weight_decay,
        format="%.7f",
    )
    runtime = st.columns(5)
    device_options = ["auto", "cuda", "cpu"]
    saved_device = saved.transformer_training.device
    device = runtime[0].selectbox(
        "訓練裝置",
        device_options,
        index=(device_options.index(saved_device) if saved_device in device_options else 0),
    )
    mixed_precision = runtime[1].toggle(
        "混合精度",
        value=saved.transformer_training.mixed_precision,
    )
    workers = runtime[2].number_input(
        "Data workers",
        0,
        64,
        0 if quick_test else saved.transformer_training.num_workers,
        1,
    )
    cpu_threads = runtime[3].number_input(
        "CPU 執行緒",
        1,
        available_cpu_threads(),
        min(
            2 if quick_test else saved.transformer_training.cpu_threads,
            available_cpu_threads(),
        ),
        1,
        help="建議保留 1～2 個核心給 Dashboard 與行情程序。",
    )
    seed = runtime[4].number_input(
        "Random seed",
        0,
        2_147_483_647,
        saved.transformer_training.seed,
        1,
    )
    split = st.columns(4)
    train_fraction = split[0].number_input(
        "訓練比例",
        0.10,
        0.90,
        saved.transformer_training.train_fraction,
        0.05,
    )
    validation_fraction = split[1].number_input(
        "驗證比例",
        0.05,
        0.50,
        saved.transformer_training.validation_fraction,
        0.05,
    )
    patience = split[2].number_input(
        "Early stop",
        1,
        1_000,
        min(saved.transformer_training.early_stopping_patience, 2)
        if quick_test
        else saved.transformer_training.early_stopping_patience,
        1,
    )
    max_rows = split[3].number_input(
        "每市場最多 K 線",
        0,
        10_000_000,
        1_500 if quick_test else int(saved.transformer_training.max_rows_per_source or 0),
        100,
        help="0 代表使用全部資料；快速驗證建議 1,500。",
    )
    optimizer = st.columns(2)
    warmup = optimizer[0].number_input(
        "Warmup ratio",
        0.0,
        0.99,
        saved.transformer_training.warmup_ratio,
        0.01,
    )
    gradient_clip = optimizer[1].number_input(
        "Gradient clip",
        0.01,
        100.0,
        saved.transformer_training.gradient_clip,
        0.1,
    )
    with st.expander("多任務 Loss 權重"):
        weights = st.columns(5)
        return_weight = weights[0].number_input(
            "報酬 Loss",
            0.001,
            100.0,
            saved.transformer_training.return_loss_weight,
            0.1,
        )
        volatility_weight = weights[1].number_input(
            "波動率 Loss",
            0.0,
            100.0,
            saved.transformer_training.volatility_loss_weight,
            0.1,
        )
        regime_weight = weights[2].number_input(
            "行情狀態 Loss",
            0.0,
            100.0,
            saved.transformer_training.regime_loss_weight,
            0.1,
        )
        direction_weight = weights[3].number_input(
            "方向 Loss",
            0.0,
            100.0,
            saved.transformer_training.direction_loss_weight,
            0.1,
        )
        quantile_weight = weights[4].number_input(
            "分位數 Loss",
            0.0,
            100.0,
            saved.transformer_training.quantile_loss_weight,
            0.1,
        )
        v3_weights = st.columns(4)
        edge_weight = v3_weights[0].number_input(
            "多空淨優勢 Loss",
            0.0,
            100.0,
            saved.transformer_training.edge_loss_weight,
            0.1,
        )
        excursion_weight = v3_weights[1].number_input(
            "有利/不利波動 Loss",
            0.0,
            100.0,
            saved.transformer_training.excursion_loss_weight,
            0.1,
        )
        tradeability_weight = v3_weights[2].number_input(
            "可交易性 Loss",
            0.0,
            100.0,
            saved.transformer_training.tradeability_loss_weight,
            0.1,
        )
        volatility_regime_weight = v3_weights[3].number_input(
            "波動狀態 Loss",
            0.0,
            100.0,
            saved.transformer_training.volatility_regime_loss_weight,
            0.1,
        )
        target_controls = st.columns(4)
        direction_threshold_bps = target_controls[0].number_input(
            "方向中性區（bps）",
            0.0,
            500.0,
            saved.transformer_training.direction_threshold_bps,
            1.0,
            help="預測報酬未超過此成本門檻時標為中性，12 bps 適合作為起始研究值。",
        )
        label_smoothing = target_controls[1].number_input(
            "Label smoothing",
            0.0,
            0.5,
            saved.transformer_training.label_smoothing,
            0.01,
        )
        fee_bps = target_controls[2].number_input(
            "單邊手續費（bps）",
            0.0,
            100.0,
            saved.transformer_training.fee_bps_per_side,
            0.1,
        )
        slippage_bps = target_controls[3].number_input(
            "單邊滑價（bps）",
            0.0,
            100.0,
            saved.transformer_training.slippage_bps_per_side,
            0.1,
        )
        probability_calibration = st.toggle(
            "使用驗證集校準預測機率",
            value=saved.transformer_training.probability_calibration,
            help="建議開啟；只使用驗證集估計溫度，不會接觸測試集。",
        )
    with st.expander("Checkpoint 選模與特徵去重"):
        selection = st.columns(4)
        checkpoint_options = [
            "deployment_horizon_skill_score",
            "hierarchical_skill_score",
            "direction_skill_score",
            "cost_aware_direction_balanced_accuracy",
            "validation_loss",
        ]
        saved_metric = saved.transformer_training.checkpoint_metric
        checkpoint_metric = selection[0].selectbox(
            "最佳模型指標",
            checkpoint_options,
            index=(
                checkpoint_options.index(saved_metric) if saved_metric in checkpoint_options else 0
            ),
            help="部署導向指標會依各預測週期權重選擇 checkpoint。",
        )
        horizon_weights = selection[1].text_input(
            "週期權重",
            value=(
                ",".join(map(str, saved.transformer_training.checkpoint_horizon_weights))
                or "0.50,0.35,0.15"
            ),
            help="順序需與報酬預測週期一致；15m 短線建議 5/20/48 使用 0.50/0.35/0.15。",
        )
        drop_constant_features = selection[2].toggle(
            "移除常數特徵",
            value=saved.transformer_training.drop_constant_features,
        )
        correlation_filter = selection[3].toggle(
            "移除高度相關特徵",
            value=saved.transformer_training.max_feature_correlation is not None,
        )
        correlation = st.columns(2)
        max_feature_correlation = correlation[0].number_input(
            "最大絕對相關係數",
            0.90,
            1.00,
            float(saved.transformer_training.max_feature_correlation or 0.985),
            0.005,
            disabled=not correlation_filter,
        )
        correlation_sample_rows = correlation[1].number_input(
            "相關性估計最多列數",
            100,
            1_000_000,
            saved.transformer_training.correlation_sample_rows,
            1_000,
        )

    if st.button(
        "開始快速驗證" if quick_test else "開始 Transformer 訓練",
        type="primary",
        icon=":material/play_arrow:",
        width="stretch",
        disabled=model_training_active or not selected_files,
    ):
        try:
            if any(path.name.startswith("features_mtf_") for path in selected_files) != all(
                path.name.startswith("features_mtf_") for path in selected_files
            ):
                raise ValueError("同一次訓練不可混用單週期與多週期資料")
            model_config = TemporalTransformerConfig(
                input_features=int(maximum_features),
                sequence_length=int(sequence_length),
                d_model=int(d_model),
                n_heads=int(n_heads),
                n_layers=int(n_layers),
                feedforward_dim=int(feedforward),
                dropout=float(dropout),
                latent_dim=int(latent_dim),
                return_horizons=_parse_horizons(str(horizons)),
                regime_classes=int(regime_classes),
                architecture_version=int(architecture_version),
                local_kernel_size=int(local_kernel_size),
                quantile_levels=_parse_quantiles(str(quantile_levels)),
                patch_size=int(patch_size),
                patch_stride=int(patch_stride),
                hierarchical_direction=bool(hierarchical_direction),
                horizon_adapter_dim=(int(horizon_adapter_dim) if hierarchical_direction else 0),
            )
            training_config = TransformerTrainingConfig(
                epochs=int(epochs),
                batch_size=int(batch_size),
                learning_rate=float(learning_rate),
                weight_decay=float(weight_decay),
                warmup_ratio=float(warmup),
                gradient_clip=float(gradient_clip),
                early_stopping_patience=int(patience),
                mixed_precision=bool(mixed_precision),
                device=str(device),
                num_workers=int(workers),
                cpu_threads=int(cpu_threads),
                train_fraction=float(train_fraction),
                validation_fraction=float(validation_fraction),
                seed=int(seed),
                return_loss_weight=float(return_weight),
                volatility_loss_weight=float(volatility_weight),
                regime_loss_weight=float(regime_weight),
                direction_loss_weight=float(direction_weight),
                quantile_loss_weight=float(quantile_weight),
                direction_threshold_bps=float(direction_threshold_bps),
                label_smoothing=float(label_smoothing),
                max_rows_per_source=(int(max_rows) if int(max_rows) > 0 else None),
                fee_bps_per_side=float(fee_bps),
                slippage_bps_per_side=float(slippage_bps),
                max_observed_spread_bps=(saved.transformer_training.max_observed_spread_bps),
                edge_loss_weight=float(edge_weight),
                excursion_loss_weight=float(excursion_weight),
                tradeability_loss_weight=float(tradeability_weight),
                volatility_regime_loss_weight=float(volatility_regime_weight),
                probability_calibration=bool(probability_calibration),
                checkpoint_metric=str(checkpoint_metric),
                checkpoint_horizon_weights=_parse_horizon_weights(str(horizon_weights)),
                drop_constant_features=bool(drop_constant_features),
                max_feature_correlation=(
                    float(max_feature_correlation) if correlation_filter else None
                ),
                correlation_sample_rows=int(correlation_sample_rows),
            )
            pipeline = AIPipelineConfig(
                schema_version=saved.schema_version,
                finbert_enabled=saved.finbert_enabled,
                transformer_enabled=True,
                ppo_use_finbert=saved.ppo_use_finbert,
                ppo_use_transformer=saved.ppo_use_transformer,
                finbert=saved.finbert,
                transformer=model_config,
                transformer_training=training_config,
            )
            if not quick_test:
                save_ai_pipeline_config(pipeline, pipeline_path)
            job = submit_transformer_training_job(
                transformer_root,
                selected_files,
                model_config,
                training_config,
                run_name=str(run_name),
            )
            st.session_state["transformer_active_job_id"] = job.job_id
            st.success("背景訓練已開始；切換其他頁面後進度仍會保存。")
            st.rerun()
        except (OSError, ValueError, RuntimeError) as exc:
            st.error(str(exc))


def render_transformer_training_page(
    *,
    project_root: str | Path,
    processed_dir: str | Path,
    transformer_root: str | Path,
) -> None:
    """顯示 Transformer 訓練、監控與結果頁面。"""
    root = Path(project_root)
    processed = Path(processed_dir)
    transformer = Path(transformer_root)
    pipeline_path = root / "data" / "config" / "ai_pipeline.json"
    saved = _load_pipeline(pipeline_path)
    backend = transformer_backend_status()

    st.title("Transformer 訓練")
    st.caption("市場序列編碼 · 多週期報酬 · 波動率 · 行情狀態")
    status = st.columns(4)
    status[0].metric("PyTorch", backend.torch_version)
    status[1].metric("CUDA", "可用" if backend.cuda_available else "不可用")
    status[2].metric("運算裝置", backend.device_name)
    status[3].metric(
        "GPU 記憶體",
        f"{backend.total_memory_gb:.1f} GB" if backend.cuda_available else "-",
    )
    render_active_transformer_job(transformer, root)
    activity = active_model_training(
        transformer.parent / "rl" / "environments",
        transformer,
    )
    if activity is not None:
        st.info(f"{activity.label}正在執行；訓練與批次推論暫時鎖定。")
    phase = st.segmented_control(
        "階段",
        ["建立訓練", "訓練結果", "批次推論"],
        default="建立訓練",
        width="stretch",
        key="transformer_training_phase",
    )
    if phase == "訓練結果":
        _render_results(transformer, root)
    elif phase == "批次推論":
        _render_inference(
            transformer,
            root,
            processed,
            activity is not None,
        )
    else:
        _render_training_form(
            root,
            processed,
            transformer,
            pipeline_path,
            saved,
            activity is not None,
        )
