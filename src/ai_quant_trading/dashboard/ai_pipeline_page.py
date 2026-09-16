"""SAC、FinBERT 與 Transformer 的操作與參數設定頁。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ai_quant_trading.ai_pipeline import (
    AIPipelineConfig,
    load_ai_pipeline_config,
    save_ai_pipeline_config,
)
from ai_quant_trading.sentiment import (
    FinBERTAnalyzer,
    FinBERTConfig,
    aggregate_finbert_features,
    finbert_backend_status,
    load_news_history,
    load_scored_news,
    news_history_path,
    recent_news,
)
from ai_quant_trading.dashboard.services import (
    list_ohlcv_files,
    list_processed_files,
    load_market_frame,
    relative_file_label,
)
from ai_quant_trading.dashboard.ui import themed_dataframe, themed_widget_key
from ai_quant_trading.performance import available_cpu_threads
from ai_quant_trading.transformer import (
    TemporalTransformerConfig,
    TransformerTrainingConfig,
)
from ai_quant_trading.startup_refresh import (
    load_startup_refresh_status,
    scored_news_path,
)


def _device_index(device: str) -> int:
    options = ["auto", "cpu", "cuda"]
    return options.index(device) if device in options else 0


def _parse_horizons(value: str) -> tuple[int, ...]:
    try:
        horizons = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",")))
    except ValueError as exc:
        raise ValueError("預測週期請使用逗號分隔的正整數") from exc
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError("預測週期必須包含正整數")
    return horizons


def _load_config(path: Path) -> AIPipelineConfig:
    try:
        return load_ai_pipeline_config(path)
    except (OSError, ValueError, TypeError):
        return AIPipelineConfig()


def _render_news_scoring(
    config: FinBERTConfig,
    project_root: Path,
    processed_dir: Path,
) -> None:
    uploaded = st.file_uploader(
        "新聞 CSV",
        type=["csv"],
        help="需要 published_at，以及 title 或 text；symbol 可選。",
    )
    if uploaded is None:
        return
    try:
        news = pd.read_csv(uploaded)
    except (OSError, ValueError) as exc:
        st.error(f"新聞 CSV 無法讀取：{exc}")
        return
    themed_dataframe(
        news.tail(20),
        width="stretch",
        hide_index=True,
        key=themed_widget_key("finbert_news_preview"),
    )
    if st.button(
        "執行 FinBERT 評分",
        type="primary",
        icon=":material/psychology:",
        width="stretch",
    ):
        try:
            if "published_at" not in news:
                raise ValueError("新聞 CSV 缺少 published_at")
            with st.spinner("正在載入 FinBERT 並評分新聞..."):
                scored = FinBERTAnalyzer(config).score_news(news)
                output_dir = processed_dir / "sentiment"
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / "manual_finbert_news_latest.csv"
                scored.to_csv(output_path, index=False, encoding="utf-8")
            st.session_state["latest_finbert_result"] = scored
            st.session_state["latest_finbert_path"] = str(output_path)
            st.success(f"完成 {len(scored):,} 筆新聞評分。")
        except (OSError, ValueError, RuntimeError) as exc:
            st.error(str(exc))
    result = st.session_state.get("latest_finbert_result")
    if isinstance(result, pd.DataFrame):
        columns = [
            column
            for column in [
                "published_at",
                "symbol",
                "title",
                "finbert_sentiment",
                "finbert_positive",
                "finbert_negative",
                "finbert_confidence",
            ]
            if column in result
        ]
        themed_dataframe(
            result[columns].tail(100),
            width="stretch",
            hide_index=True,
            key=themed_widget_key("finbert_scored_news"),
        )
        st.caption(str(st.session_state.get("latest_finbert_path", "")))
        market_files = list_processed_files(processed_dir) + list_ohlcv_files(
            project_root / "data" / "raw"
        )
        if market_files:
            market_path = st.selectbox(
                "附加情緒的市場資料",
                market_files,
                format_func=lambda path: relative_file_label(path, project_root),
                key="finbert_market_path",
            )
            if st.button(
                "建立 FinBERT 市場特徵",
                icon=":material/merge:",
                width="stretch",
            ):
                try:
                    market = load_market_frame(market_path)
                    enriched = aggregate_finbert_features(market, result, config)
                    output_path = (
                        processed_dir / f"{market_path.stem}_finbert_features.csv"
                    )
                    enriched.to_csv(output_path, index=False, encoding="utf-8")
                    st.success(f"已保存 {len(enriched):,} 根 K 線：{output_path}")
                except (OSError, ValueError) as exc:
                    st.error(str(exc))


def _render_automatic_news(project_root: Path, processed_dir: Path) -> None:
    """顯示啟動更新工作自動收集與評分的新聞。"""
    raw_path = news_history_path(project_root / "data" / "raw")
    scored_path = scored_news_path(processed_dir)
    try:
        raw = load_news_history(raw_path)
        scored = load_scored_news(scored_path)
    except (OSError, ValueError, RuntimeError) as exc:
        st.warning(f"自動新聞資料無法讀取：{exc}")
        return

    refresh = load_startup_refresh_status(project_root)
    pending = max(len(raw) - raw["news_id"].isin(scored.get("news_id", [])).sum(), 0)
    metrics = st.columns(4)
    metrics[0].metric("已收集新聞", f"{len(raw):,}")
    metrics[1].metric("FinBERT 已評分", f"{len(scored):,}")
    metrics[2].metric("等待評分", f"{pending:,}")
    metrics[3].metric("自動更新", str(refresh.get("status", "idle")))

    if raw.empty:
        st.info("尚無自動新聞；可從左側按「立即更新」，或下次開啟 App 時自動收集。")
        return
    preview = recent_news(scored if not scored.empty else raw, days=7)
    columns = [
        column
        for column in [
            "published_at",
            "symbol",
            "asset_class",
            "source",
            "title",
            "finbert_sentiment",
            "finbert_confidence",
        ]
        if column in preview
    ]
    themed_dataframe(
        preview[columns].sort_values("published_at", ascending=False).head(100),
        width="stretch",
        hide_index=True,
        key=themed_widget_key("automatic_finbert_news"),
    )
    st.caption(f"Raw：{raw_path} · FinBERT：{scored_path}")


def render_ai_pipeline_page(
    *,
    project_root: str | Path,
    processed_dir: str | Path,
) -> None:
    """顯示三模型狀態與可保存的訓練方案，不啟動 Transformer 訓練。"""
    root = Path(project_root)
    output_root = Path(processed_dir)
    config_path = root / "data" / "config" / "ai_pipeline.json"
    saved = _load_config(config_path)
    backend = finbert_backend_status()

    st.title("AI 管線")
    st.caption("SAC 執行決策 · FinBERT 解析新聞 · Transformer 編碼市場序列")
    status = st.columns(4)
    status[0].metric("FinBERT", "可用" if backend.ready else "缺少套件")
    status[1].metric("FinBERT 裝置", "CUDA" if backend.cuda_available else "CPU")
    status[2].metric("Transformer", "架構就緒")
    status[3].metric("SAC 融合", "已設定" if config_path.exists() else "預設值")

    finbert_tab, transformer_tab, fusion_tab = st.tabs(
        ["FinBERT 新聞", "Transformer 時序", "SAC 特徵融合"]
    )
    with finbert_tab:
        enabled = st.toggle("啟用 FinBERT", value=saved.finbert_enabled)
        finbert_columns = st.columns(4)
        model_name = finbert_columns[0].text_input(
            "Hugging Face 模型",
            value=saved.finbert.model_name,
        )
        finbert_device = finbert_columns[1].selectbox(
            "推論裝置",
            ["auto", "cpu", "cuda"],
            index=_device_index(saved.finbert.device),
        )
        finbert_batch = finbert_columns[2].number_input(
            "批次大小",
            min_value=1,
            max_value=512,
            value=saved.finbert.batch_size,
            step=1,
        )
        max_length = finbert_columns[3].number_input(
            "最大 Token",
            min_value=8,
            max_value=4096,
            value=saved.finbert.max_length,
            step=8,
        )
        aggregation_columns = st.columns(4)
        confidence = aggregation_columns[0].number_input(
            "最低信心",
            min_value=0.0,
            max_value=1.0,
            value=saved.finbert.confidence_threshold,
            step=0.01,
        )
        window_hours = aggregation_columns[1].number_input(
            "聚合視窗（小時）",
            min_value=1,
            max_value=720,
            value=saved.finbert.aggregation_window_hours,
            step=1,
        )
        half_life = aggregation_columns[2].number_input(
            "衰減半衰期（小時）",
            min_value=0.1,
            max_value=720.0,
            value=saved.finbert.decay_half_life_hours,
            step=0.5,
        )
        minimum_coverage = aggregation_columns[3].number_input(
            "正式訓練最低覆蓋率",
            min_value=0.01,
            max_value=1.0,
            value=saved.finbert.minimum_training_coverage,
            step=0.05,
            help="未達此比例時不允許把 FinBERT 納入正式 SAC 模型。",
        )
        finbert_config = FinBERTConfig(
            model_name=str(model_name),
            revision=saved.finbert.revision,
            device=str(finbert_device),
            batch_size=int(finbert_batch),
            max_length=int(max_length),
            confidence_threshold=float(confidence),
            aggregation_window_hours=int(window_hours),
            decay_half_life_hours=float(half_life),
            minimum_training_coverage=float(minimum_coverage),
            cache_dir=saved.finbert.cache_dir,
        )
        if backend.ready and enabled:
            _render_automatic_news(root, output_root)
            with st.expander("匯入自訂新聞 CSV"):
                _render_news_scoring(finbert_config, root, output_root)
        elif enabled:
            st.warning(backend.message)

    with transformer_tab:
        transformer_enabled = st.toggle(
            "啟用 Transformer",
            value=saved.transformer_enabled,
        )
        architecture = st.columns(4)
        input_features = architecture[0].number_input(
            "輸入特徵數",
            min_value=1,
            max_value=1024,
            value=saved.transformer.input_features,
            step=1,
        )
        sequence_length = architecture[1].number_input(
            "序列長度",
            min_value=8,
            max_value=2048,
            value=saved.transformer.sequence_length,
            step=8,
        )
        d_model = architecture[2].number_input(
            "d_model",
            min_value=16,
            max_value=2048,
            value=saved.transformer.d_model,
            step=16,
        )
        n_heads = architecture[3].number_input(
            "注意力頭數",
            min_value=1,
            max_value=64,
            value=saved.transformer.n_heads,
            step=1,
        )
        depth = st.columns(4)
        n_layers = depth[0].number_input(
            "Encoder 層數",
            min_value=1,
            max_value=48,
            value=saved.transformer.n_layers,
            step=1,
        )
        feedforward = depth[1].number_input(
            "前饋維度",
            min_value=16,
            max_value=8192,
            value=saved.transformer.feedforward_dim,
            step=16,
        )
        dropout = depth[2].number_input(
            "Dropout",
            min_value=0.0,
            max_value=0.9,
            value=saved.transformer.dropout,
            step=0.01,
        )
        latent_dim = depth[3].number_input(
            "SAC 壓縮向量",
            min_value=1,
            max_value=512,
            value=saved.transformer.latent_dim,
            step=1,
        )
        task_columns = st.columns(2)
        horizons_text = task_columns[0].text_input(
            "報酬預測週期",
            value=",".join(map(str, saved.transformer.return_horizons)),
        )
        regime_classes = task_columns[1].number_input(
            "行情狀態數",
            min_value=2,
            max_value=12,
            value=saved.transformer.regime_classes,
            step=1,
        )

        st.subheader("訓練方案")
        training = st.columns(4)
        epochs = training[0].number_input(
            "Epochs", 1, 10_000, saved.transformer_training.epochs, 1
        )
        transformer_batch = training[1].number_input(
            "Batch size", 1, 8192, saved.transformer_training.batch_size, 1
        )
        learning_rate = training[2].number_input(
            "Learning rate",
            min_value=1e-7,
            max_value=1.0,
            value=saved.transformer_training.learning_rate,
            format="%.7f",
        )
        weight_decay = training[3].number_input(
            "Weight decay",
            min_value=0.0,
            max_value=1.0,
            value=saved.transformer_training.weight_decay,
            format="%.7f",
        )
        optimizer = st.columns(4)
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
        patience = optimizer[2].number_input(
            "Early stop",
            1,
            1_000,
            saved.transformer_training.early_stopping_patience,
            1,
        )
        workers = optimizer[3].number_input(
            "Data workers",
            0,
            64,
            saved.transformer_training.num_workers,
            1,
        )
        runtime = st.columns(5)
        transformer_device = runtime[0].selectbox(
            "訓練裝置",
            ["auto", "cpu", "cuda"],
            index=_device_index(saved.transformer_training.device),
        )
        mixed_precision = runtime[1].toggle(
            "混合精度",
            value=saved.transformer_training.mixed_precision,
        )
        cpu_threads = runtime[2].number_input(
            "CPU 執行緒",
            1,
            available_cpu_threads(),
            min(saved.transformer_training.cpu_threads, available_cpu_threads()),
            1,
        )
        train_fraction = runtime[3].number_input(
            "訓練比例",
            0.10,
            0.90,
            saved.transformer_training.train_fraction,
            0.05,
        )
        validation_fraction = runtime[4].number_input(
            "驗證比例",
            0.05,
            0.50,
            saved.transformer_training.validation_fraction,
            0.05,
        )
        seed = st.number_input(
            "隨機種子",
            min_value=0,
            max_value=2_147_483_647,
            value=saved.transformer_training.seed,
            step=1,
        )

    with fusion_tab:
        ppo_use_finbert = st.toggle(
            "SAC 使用 FinBERT 特徵",
            value=saved.ppo_use_finbert,
            disabled=not enabled,
        )
        ppo_use_transformer = st.toggle(
            "SAC 使用 Transformer 特徵",
            value=saved.ppo_use_transformer,
            disabled=not transformer_enabled,
        )
        themed_dataframe(
            pd.DataFrame(
                [
                    {
                        "模組": "技術與市場結構",
                        "狀態": "已接入",
                        "SAC 輸入": "趨勢、動能、波動、成交量、持倉與風控",
                    },
                    {
                        "模組": "FinBERT",
                        "狀態": "啟用" if ppo_use_finbert else "停用",
                        "SAC 輸入": "情緒、信心、新聞量、變化與新鮮度",
                    },
                    {
                        "模組": "Transformer",
                        "狀態": "啟用" if ppo_use_transformer else "停用",
                        "SAC 輸入": "多週期報酬、波動、行情機率與不確定度",
                    },
                ]
            ),
            width="stretch",
            hide_index=True,
            key=themed_widget_key("ai_pipeline_fusion_status"),
        )

    if st.button(
        "儲存 AI 管線設定",
        type="primary",
        icon=":material/save:",
        width="stretch",
    ):
        try:
            transformer_config = TemporalTransformerConfig(
                input_features=int(input_features),
                sequence_length=int(sequence_length),
                d_model=int(d_model),
                n_heads=int(n_heads),
                n_layers=int(n_layers),
                feedforward_dim=int(feedforward),
                dropout=float(dropout),
                latent_dim=int(latent_dim),
                return_horizons=_parse_horizons(str(horizons_text)),
                regime_classes=int(regime_classes),
                architecture_version=saved.transformer.architecture_version,
                local_kernel_size=saved.transformer.local_kernel_size,
                quantile_levels=saved.transformer.quantile_levels,
                patch_size=saved.transformer.patch_size,
                patch_stride=saved.transformer.patch_stride,
                feature_group_ids=saved.transformer.feature_group_ids,
                feature_group_names=saved.transformer.feature_group_names,
                volatility_regime_classes=(
                    saved.transformer.volatility_regime_classes
                ),
            )
            training_config = TransformerTrainingConfig(
                epochs=int(epochs),
                batch_size=int(transformer_batch),
                learning_rate=float(learning_rate),
                weight_decay=float(weight_decay),
                warmup_ratio=float(warmup),
                gradient_clip=float(gradient_clip),
                early_stopping_patience=int(patience),
                mixed_precision=bool(mixed_precision),
                device=str(transformer_device),
                num_workers=int(workers),
                cpu_threads=int(cpu_threads),
                train_fraction=float(train_fraction),
                validation_fraction=float(validation_fraction),
                seed=int(seed),
                return_loss_weight=saved.transformer_training.return_loss_weight,
                volatility_loss_weight=(
                    saved.transformer_training.volatility_loss_weight
                ),
                regime_loss_weight=saved.transformer_training.regime_loss_weight,
                direction_loss_weight=(
                    saved.transformer_training.direction_loss_weight
                ),
                quantile_loss_weight=(
                    saved.transformer_training.quantile_loss_weight
                ),
                direction_threshold_bps=(
                    saved.transformer_training.direction_threshold_bps
                ),
                label_smoothing=saved.transformer_training.label_smoothing,
                max_rows_per_source=saved.transformer_training.max_rows_per_source,
                fee_bps_per_side=saved.transformer_training.fee_bps_per_side,
                slippage_bps_per_side=(
                    saved.transformer_training.slippage_bps_per_side
                ),
                max_observed_spread_bps=(
                    saved.transformer_training.max_observed_spread_bps
                ),
                edge_loss_weight=saved.transformer_training.edge_loss_weight,
                excursion_loss_weight=(
                    saved.transformer_training.excursion_loss_weight
                ),
                tradeability_loss_weight=(
                    saved.transformer_training.tradeability_loss_weight
                ),
                volatility_regime_loss_weight=(
                    saved.transformer_training.volatility_regime_loss_weight
                ),
                probability_calibration=(
                    saved.transformer_training.probability_calibration
                ),
            )
            pipeline = AIPipelineConfig(
                schema_version=max(saved.schema_version, 3),
                finbert_enabled=bool(enabled),
                transformer_enabled=bool(transformer_enabled),
                ppo_use_finbert=bool(ppo_use_finbert and enabled),
                ppo_use_transformer=bool(
                    ppo_use_transformer and transformer_enabled
                ),
                finbert=finbert_config,
                transformer=transformer_config,
                transformer_training=training_config,
            )
            target = save_ai_pipeline_config(pipeline, config_path)
            st.success(f"設定已保存：{target}")
        except (OSError, ValueError) as exc:
            st.error(str(exc))
