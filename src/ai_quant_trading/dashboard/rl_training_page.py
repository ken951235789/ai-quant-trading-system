"""Step 9 PPO／SAC 訓練設定、進度與結果介面。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from ai_quant_trading.dashboard.model_activity import active_model_training
from ai_quant_trading.dashboard.rl_training_jobs import (
    ACTIVE_JOB_STATUSES,
    RLTrainingJob,
    latest_rl_training_job,
    read_rl_training_progress,
    submit_rl_training_job,
)
from ai_quant_trading.dashboard.ui import themed_dataframe, themed_widget_key
from ai_quant_trading.performance import available_cpu_threads
from ai_quant_trading.reinforcement_learning import (
    RLResearchConfig,
    RLTrainingConfig,
    assess_rl_environment_preflight,
    assess_rl_training_quality,
    inspect_training_backend,
    list_rl_checkpoints,
    list_rl_environments,
    list_rl_training_runs,
    list_rl_research_experiments,
    load_model_registry,
    parse_seed_list,
    promote_model_challenger,
    register_model_challenger,
    rollback_model_champion,
)


@st.cache_resource(show_spinner=False)
def _cached_training_backend() -> Any:
    return inspect_training_backend()


@st.cache_data(show_spinner=False)
def _cached_environment_preflight(environment_dir: str):
    """環境成品建立後不再變動，避免每次切頁重讀大型訓練 CSV。"""
    return assess_rl_environment_preflight(Path(environment_dir))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _environment_label(path: Path) -> str:
    try:
        payload = _read_json(path / "environment.json")
        source = payload.get("source", {})
        rows = payload.get("split_rows", {})
        expert = {
            "long_term": "長期專家",
            "short_term": "短期專家",
            "general": "一般模型",
        }.get(payload.get("expert", {}).get("kind"), "一般模型")
        return (
            f"{expert} · {source.get('symbol', '-')} · {source.get('interval', '-')} · "
            f"訓練 {rows.get('train', 0):,} 筆 · {path.name[:15]}"
        )
    except (OSError, ValueError, TypeError):
        return path.name


def _training_label(path: Path) -> str:
    try:
        payload = _read_json(path / "training.json")
        source = payload.get("environment_source", {})
        config = payload.get("training_config", {})
        expert = {
            "long_term": "長期專家",
            "short_term": "短期專家",
            "general": "一般模型",
        }.get(payload.get("environment_expert", {}).get("kind"), "一般模型")
        status = {"complete": "完成", "running": "執行中", "failed": "失敗"}.get(
            payload.get("status"), "未知"
        )
        return (
            f"{expert} · {source.get('symbol', '-')} · "
            f"{str(config.get('algorithm', '-')).upper()} · {status} · {path.name[:15]}"
        )
    except (OSError, ValueError, TypeError):
        return path.name


def _relative_label(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path)


def _result_training_runs(rl_dir: Path, project_root: Path) -> list[Path]:
    """合併目前資料根目錄與本機打包版的訓練紀錄。"""
    roots = [rl_dir]
    packaged = (
        project_root
        / "dist"
        / "AIQuantTradingSystem"
        / "data"
        / "processed"
        / "rl"
        / "environments"
    )
    if packaged.exists() and packaged.resolve() != rl_dir.resolve():
        roots.append(packaged)
    unique: dict[tuple[str, str], Path] = {}
    for root in roots:
        for run in list_rl_training_runs(root):
            environment_name = run.parent.parent.name
            unique.setdefault((environment_name, run.name), run)
    return sorted(
        unique.values(),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _result_research_experiments(rl_dir: Path, project_root: Path) -> list[Path]:
    """合併原始碼與打包桌面版產生的穩健性研究。"""
    roots = [rl_dir]
    packaged = (
        project_root
        / "dist"
        / "AIQuantTradingSystem"
        / "data"
        / "processed"
        / "rl"
        / "environments"
    )
    if packaged.exists() and packaged.resolve() != rl_dir.resolve():
        roots.append(packaged)
    unique: dict[str, Path] = {}
    for root in roots:
        for experiment in list_rl_research_experiments(root):
            try:
                source = _read_json(experiment / "summary.json").get(
                    "source_environment", ""
                )
            except (OSError, ValueError, TypeError):
                continue
            key = f"{Path(str(source)).name}:{experiment.name}"
            unique.setdefault(key, experiment)
    return sorted(unique.values(), key=lambda path: path.stat().st_mtime, reverse=True)


def _parse_network_architecture(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("神經網路層請使用逗號分隔的正整數，例如 128,128") from exc
    if not result or any(width <= 0 for width in result):
        raise ValueError("神經網路至少需要一層正整數寬度")
    return result


def _parse_auto_number(
    value: str,
    name: str,
    *,
    allow_auto_initial: bool = True,
) -> str | float:
    normalized = value.strip().lower()
    if normalized == "auto" or (allow_auto_initial and normalized.startswith("auto_")):
        return normalized
    try:
        return float(normalized)
    except ValueError as exc:
        allowed = "auto、auto_N 或數值" if allow_auto_initial else "auto 或數值"
        raise ValueError(f"{name} 必須是 {allowed}") from exc


def _optional_positive(enabled: bool, value: float) -> float | None:
    return float(value) if enabled else None


def _render_backend_status() -> tuple[Any, dict[str, str]]:
    status = _cached_training_backend()
    if not status.available:
        st.error(f"訓練後端尚未安裝：{status.error}")
        return status, {}
    labels = {
        "auto": "自動（CUDA 優先）",
        "cpu": "CPU",
        **{
            f"cuda:{index}": f"CUDA:{index} · {name}"
            for index, name in zip(
                status.cuda_device_indices or range(len(status.cuda_devices)),
                status.cuda_devices,
                strict=True,
            )
        },
    }
    backend_text = (
        f"PyTorch {status.torch_version} · Stable-Baselines3 {status.stable_baselines_version}"
    )
    if status.cuda_available:
        backend_text += f" · GPU {len(status.cuda_devices)} 張"
    else:
        backend_text += " · 此環境目前只有 CPU"
    st.caption(backend_text)
    if status.cuda_errors:
        st.warning("偵測到無法執行 PyTorch kernel 的 GPU：" + "；".join(status.cuda_errors))
    return status, labels


def _render_completed_metrics(payload: dict[str, Any]) -> None:
    metrics = payload.get("metrics", {})
    test = metrics.get("test", {})
    validation = metrics.get("validation", {})
    columns = st.columns(8)
    columns[0].metric(
        "狀態", {"complete": "完成", "failed": "失敗"}.get(payload.get("status"), "執行中")
    )
    columns[1].metric("裝置", str(payload.get("resolved_device", "-")))
    columns[2].metric("測試報酬", f"{float(test.get('total_return', 0)):.2%}")
    columns[3].metric("測試回撤", f"{float(test.get('max_drawdown', 0)):.2%}")
    columns[4].metric("測試交易", f"{int(test.get('trades', 0)):,}")
    columns[5].metric("驗證報酬", f"{float(validation.get('total_return', 0)):.2%}")
    columns[6].metric("測試 Sharpe", f"{float(test.get('sharpe_ratio', 0)):.2f}")
    columns[7].metric(
        "成本／本金",
        f"{float(test.get('fee_to_initial_capital', 0)):.2%}",
    )


def _evaluation_curve(path: Path) -> pd.DataFrame:
    """將每個市場的權益換算成獨立報酬曲線，避免不同市場被平均在一起。"""
    frame = pd.read_csv(path)
    required = {"timestamp", "equity"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=["timestamp", "market", "累積報酬"])
    columns = [column for column in ["timestamp", "market", "equity"] if column in frame]
    result = frame[columns].copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    if "market" not in result:
        result["market"] = "單一市場"
    initial_equity = result.groupby("market")["equity"].transform("first")
    result["累積報酬"] = result["equity"] / initial_equity - 1.0
    return result[["timestamp", "market", "累積報酬"]]


def _position_exposure_summary(path: Path) -> dict[str, float | int]:
    """用實際部位判斷多空曝險，避免把一般減碼誤算成放空。"""
    if not path.exists():
        return {}
    frame = pd.read_csv(path, usecols=lambda column: column == "position_fraction")
    if frame.empty or "position_fraction" not in frame:
        return {}
    positions = pd.to_numeric(frame["position_fraction"], errors="coerce").dropna()
    if positions.empty:
        return {}
    # 極小部位在實際下單時會落在交易死區，UI 應視為空手而非多空曝險。
    tolerance = 1e-3
    total = len(positions)
    long_rows = int((positions > tolerance).sum())
    short_rows = int((positions < -tolerance).sum())
    return {
        "rows": total,
        "long_rows": long_rows,
        "short_rows": short_rows,
        "flat_rows": total - long_rows - short_rows,
        "long_ratio": long_rows / total,
        "short_ratio": short_rows / total,
        "flat_ratio": (total - long_rows - short_rows) / total,
        "min_position": float(positions.min()),
        "max_position": float(positions.max()),
    }


def _market_metrics_frame(payload: dict[str, Any]) -> pd.DataFrame:
    """整理通用模型的逐市場樣本外指標，供 UI 與測試共用。"""
    markets = payload.get("metrics", {}).get("test_markets", {})
    rows = []
    for market, metrics in markets.items():
        rows.append(
            {
                "市場": market,
                "測試報酬": float(metrics.get("total_return", 0)),
                "最大回撤": float(metrics.get("max_drawdown", 0)),
                "期末權益": float(metrics.get("final_equity", 0)),
                "交易次數": int(metrics.get("trades", 0)),
                "手續費": float(metrics.get("total_fees", 0)),
                "滑價成本": float(metrics.get("total_slippage", 0)),
                "空頭持有成本": float(metrics.get("total_short_carry", 0)),
                "成本／本金": float(metrics.get("fee_to_initial_capital", 0)),
                "Sharpe": float(metrics.get("sharpe_ratio", 0)),
                "基準報酬": float(metrics.get("buy_and_hold_return", 0)),
                "超額報酬": float(metrics.get("excess_return", 0)),
            }
        )
    return pd.DataFrame(rows)


def _duration_text(seconds: object) -> str:
    """將秒數整理成適合即時監控的短文字。"""
    try:
        value = max(float(seconds), 0.0)
    except (TypeError, ValueError):
        return "-"
    if value < 60:
        return f"{value:.0f} 秒"
    hours, remainder = divmod(int(value), 3_600)
    minutes, seconds_value = divmod(remainder, 60)
    if hours:
        return f"{hours} 小時 {minutes} 分"
    return f"{minutes} 分 {seconds_value} 秒"


def _live_training_record(payload: dict[str, object]) -> dict[str, float]:
    """將訓練 callback 攤平成可直接畫線的單列資料。"""
    raw_metrics = payload.get("metrics", {})
    metrics = raw_metrics if isinstance(raw_metrics, dict) else {}
    aliases = {
        "recent_reward_mean": "近期 Reward",
        "episode_reward_mean": "Episode Reward",
        "evaluation_reward": "驗證 Reward",
        "value_loss": "Value Loss",
        "policy_loss": "Policy Loss",
        "actor_loss": "Actor Loss",
        "critic_loss": "Critic Loss",
        "entropy_loss": "Entropy Loss",
        "approx_kl": "Approx KL",
        "explained_variance": "Explained Variance",
    }
    result = {"步數": float(payload.get("completed_timesteps", 0))}
    for source, target in aliases.items():
        value = metrics.get(source)
        if value is not None:
            result[target] = float(value)
    return result


def _job_history_frame(job: RLTrainingJob) -> pd.DataFrame:
    """讀取背景工作歷史，讓圖表在頁面重跑後仍可重建。"""
    if not job.history_csv.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(job.history_csv).tail(200)
    except (OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()
    aliases = {
        "recent_reward_mean": "近期 Reward",
        "episode_reward_mean": "Episode Reward",
        "evaluation_reward": "驗證 Reward",
        "value_loss": "Value Loss",
        "policy_loss": "Policy Loss",
        "actor_loss": "Actor Loss",
        "critic_loss": "Critic Loss",
    }
    frame = frame.rename(columns=aliases)
    if "experiment_runs" in frame and frame["experiment_runs"].notna().any():
        frame["進度"] = pd.to_numeric(frame.get("progress"), errors="coerce") * 100
    else:
        frame["進度"] = pd.to_numeric(frame.get("completed_timesteps"), errors="coerce")
    return frame


def _render_rl_job_monitor(rl_dir: Path, project_root: Path) -> None:
    """依落盤資料重建訓練畫面，不依賴單次 Streamlit 執行狀態。"""
    job = latest_rl_training_job(rl_dir)
    if job is None:
        return
    progress_payload = read_rl_training_progress(job)
    fraction = min(max(float(progress_payload.get("progress", 0)), 0.0), 1.0)
    status = job.status
    config = job.payload.get("training_config", {})
    config = config if isinstance(config, dict) else {}
    algorithm = str(config.get("algorithm", "-")).upper()
    environment_name = Path(str(job.payload.get("environment_dir", ""))).name

    st.markdown("#### 背景訓練監控")
    if status in ACTIVE_JOB_STATUSES:
        completed = int(progress_payload.get("completed_timesteps", 0))
        total = int(progress_payload.get("total_timesteps", 0))
        progress_label = (
            "樣本外評估中"
            if progress_payload.get("status") == "evaluating"
            else f"訓練中 {completed:,}／{total:,} 步"
        )
        st.progress(fraction, text=progress_label)
    elif status == "complete":
        st.success(f"{algorithm} 背景訓練已完成；切換到「訓練結果」即可檢查績效。")
    elif status == "interrupted":
        st.warning(str(job.payload.get("error", "背景訓練已中斷。")))
    else:
        st.error(f"背景訓練失敗：{job.payload.get('error', '未知錯誤')}")

    raw_metrics = progress_payload.get("metrics", {})
    metrics = raw_metrics if isinstance(raw_metrics, dict) else {}
    metric_columns = st.columns(5)
    metric_columns[0].metric("完成度", f"{fraction:.1%}")
    metric_columns[1].metric(
        "速度", f"{float(metrics.get('steps_per_second', 0)):,.0f} 步／秒"
    )
    metric_columns[2].metric("預估剩餘", _duration_text(metrics.get("eta_seconds")))
    metric_columns[3].metric(
        "近期 Reward", f"{float(metrics.get('recent_reward_mean', 0)):.5f}"
    )
    evaluation_reward = metrics.get("evaluation_reward")
    metric_columns[4].metric(
        "驗證 Reward",
        "-" if evaluation_reward is None else f"{float(evaluation_reward):.5f}",
    )

    run_context = ""
    if progress_payload.get("experiment_runs"):
        run_context = (
            f" · Run {progress_payload.get('experiment_run')}/"
            f"{progress_payload.get('experiment_runs')}"
            f" · Seed {progress_payload.get('experiment_seed')}"
            f" · Fold {progress_payload.get('experiment_fold')}"
        )
    st.caption(
        f"工作 {job.job_id} · {algorithm} · 環境 {environment_name}"
        f" · 裝置 {progress_payload.get('device', config.get('device', '-'))}"
        f" · CPU {config.get('cpu_threads', '-')} 執行緒"
        f" · 已執行 {_duration_text(progress_payload.get('elapsed_seconds'))}{run_context}"
    )

    history = _job_history_frame(job)
    if not history.empty:
        reward_columns = [
            column
            for column in ["近期 Reward", "Episode Reward", "驗證 Reward"]
            if column in history and history[column].notna().any()
        ]
        loss_columns = [
            column
            for column in ["Value Loss", "Policy Loss", "Actor Loss", "Critic Loss"]
            if column in history and history[column].notna().any()
        ]
        charts = st.columns(2)
        if reward_columns:
            charts[0].line_chart(history[["進度", *reward_columns]], x="進度", height=220)
        if loss_columns:
            charts[1].line_chart(history[["進度", *loss_columns]], x="進度", height=220)

    raw_snapshot = progress_payload.get("snapshot", {})
    snapshot = raw_snapshot if isinstance(raw_snapshot, dict) else {}
    if snapshot:
        st.caption(
            f"最近環境 · {snapshot.get('market', '單一市場')} · "
            f"{snapshot.get('side', 'HOLD')} · 目標部位 "
            f"{float(snapshot.get('target_fraction') or 0):.1%} · "
            f"權益 {float(snapshot.get('equity') or 0):,.2f} · "
            f"回撤 {abs(float(snapshot.get('drawdown') or 0)):.2%}"
        )
    result_dir = job.payload.get("result_dir")
    if result_dir:
        st.caption(f"結果位置：{_relative_label(Path(str(result_dir)), project_root)}")
    st.divider()


@st.fragment(run_every=5)
def render_active_rl_job(rl_dir: Path, project_root: Path) -> None:
    """每五秒只更新監控區，不重新執行整個強化學習頁。"""
    _render_rl_job_monitor(rl_dir, project_root)


def _training_history_frame(
    progress: pd.DataFrame,
    aliases: dict[str, str],
) -> pd.DataFrame:
    """選取存在的 SB3 指標並改成中文顯示名稱。"""
    step_column = "time/total_timesteps"
    columns = [column for column in aliases if column in progress.columns]
    if step_column not in progress or not columns:
        return pd.DataFrame()
    result = progress[[step_column, *columns]].copy()
    result = result.rename(columns={step_column: "步數", **aliases})
    return result.dropna(how="all", subset=[aliases[column] for column in columns])


def _render_decision_replay(run_dir: Path) -> None:
    """用滑桿重播模型在樣本外測試中的部位、動作與帳戶狀態。"""
    path = run_dir / "test_evaluation.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    required = {"timestamp", "side", "position_fraction", "equity", "drawdown"}
    if frame.empty or not required.issubset(frame.columns):
        return
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    if "market" not in frame:
        frame["market"] = "單一市場"

    with st.expander("模型決策重播"):
        market = st.selectbox(
            "市場",
            frame["market"].drop_duplicates().tolist(),
            key=f"replay_market_{run_dir.name}",
        )
        market_frame = frame.loc[frame["market"] == market].reset_index(drop=True)
        replay_index = st.slider(
            "時間位置",
            0,
            len(market_frame) - 1,
            len(market_frame) - 1,
            key=f"replay_index_{run_dir.name}_{market}",
        )
        visible = market_frame.iloc[: replay_index + 1]
        latest = visible.iloc[-1]
        metrics = st.columns(5)
        metrics[0].metric("時間", str(latest["timestamp"])[:10])
        metrics[1].metric("動作", str(latest["side"]))
        metrics[2].metric("目前部位", f"{float(latest['position_fraction']):.1%}")
        metrics[3].metric("權益", f"{float(latest['equity']):,.2f}")
        metrics[4].metric("回撤", f"{abs(float(latest['drawdown'])):.2%}")

        position_columns = ["position_fraction"]
        if "target_fraction" in visible:
            position_columns.insert(0, "target_fraction")
        position_chart = visible[["timestamp", *position_columns]].rename(
            columns={"target_fraction": "目標部位", "position_fraction": "實際部位"}
        )
        st.line_chart(position_chart, x="timestamp", height=230)
        account_chart = visible[["timestamp", "equity", "drawdown"]].rename(
            columns={"equity": "權益", "drawdown": "回撤"}
        )
        chart_columns = st.columns(2)
        chart_columns[0].line_chart(account_chart, x="timestamp", y="權益", height=220)
        chart_columns[1].line_chart(account_chart, x="timestamp", y="回撤", height=220)

        trades = visible.loc[visible["side"] != "HOLD"].tail(20)
        if not trades.empty:
            display_columns = [
                column
                for column in ["timestamp", "side", "target_fraction", "fee", "equity"]
                if column in trades
            ]
            themed_dataframe(
                trades[display_columns],
                hide_index=True,
                width="stretch",
                key=themed_widget_key("rl_live_trades"),
            )


def _render_training_result(run_dir: Path, project_root: Path) -> None:
    payload = _read_json(run_dir / "training.json")
    _render_completed_metrics(payload)
    if payload.get("status") == "failed":
        st.error(str(payload.get("error", "訓練失敗")))
        return

    expert = {
        "long_term": "長期專家",
        "short_term": "短期專家",
        "general": "一般模型",
    }.get(payload.get("environment_expert", {}).get("kind"), "一般模型")
    exposure = _position_exposure_summary(run_dir / "test_evaluation.csv")
    test_metrics = dict(payload.get("metrics", {}).get("test", {}))
    if "long_exposure_ratio" in test_metrics:
        exposure["long_ratio"] = float(test_metrics["long_exposure_ratio"])
        exposure["short_ratio"] = float(test_metrics.get("short_exposure_ratio", 0))
        exposure["flat_ratio"] = float(test_metrics.get("flat_exposure_ratio", 0))
    exposure_metrics = st.columns(5)
    exposure_metrics[0].metric("模型類型", expert)
    exposure_metrics[1].metric("多頭曝險時間", f"{float(exposure.get('long_ratio', 0)):.1%}")
    exposure_metrics[2].metric("空頭曝險時間", f"{float(exposure.get('short_ratio', 0)):.1%}")
    exposure_metrics[3].metric("空手時間", f"{float(exposure.get('flat_ratio', 0)):.1%}")
    exposure_metrics[4].metric(
        "實際部位範圍",
        f"{float(exposure.get('min_position', 0)):.1%} ～ "
        f"{float(exposure.get('max_position', 0)):.1%}",
    )

    quality = assess_rl_training_quality(payload)
    if quality.eligible:
        st.success("此模型已通過目前的樣本外品質門檻。")
    else:
        st.warning("此模型目前僅供研究與模擬：" + "；".join(quality.reasons))

    market_metrics = _market_metrics_frame(payload)
    if not market_metrics.empty:
        market_count = int(test_metrics.get("market_count") or len(market_metrics))
        st.markdown(f"#### {market_count} 市場樣本外結果")
        themed_dataframe(
            market_metrics,
            hide_index=True,
            width="stretch",
            key=themed_widget_key(f"rl_market_metrics_{run_dir.name}"),
            column_config={
                "測試報酬": st.column_config.NumberColumn(format="percent"),
                "最大回撤": st.column_config.NumberColumn(format="percent"),
                "期末權益": st.column_config.NumberColumn(format="%.2f"),
                "手續費": st.column_config.NumberColumn(format="%.2f"),
                "滑價成本": st.column_config.NumberColumn(format="%.2f"),
                "成本／本金": st.column_config.NumberColumn(format="percent"),
                "Sharpe": st.column_config.NumberColumn(format="%.2f"),
                "基準報酬": st.column_config.NumberColumn(format="percent"),
                "超額報酬": st.column_config.NumberColumn(format="percent"),
            },
        )

    curves: list[tuple[str, pd.DataFrame]] = []
    for split, filename in [("驗證", "validation_evaluation.csv"), ("測試", "test_evaluation.csv")]:
        path = run_dir / filename
        if not path.exists():
            continue
        curve = _evaluation_curve(path)
        if not curve.empty:
            curves.append((split, curve))
    if curves:
        tabs = st.tabs([f"{split}曲線" for split, _ in curves])
        for tab, (_, curve) in zip(tabs, curves, strict=True):
            with tab:
                st.line_chart(
                    curve,
                    x="timestamp",
                    y="累積報酬",
                    color="market",
                    height=280,
                )

    _render_decision_replay(run_dir)

    progress_csv = run_dir / "logs" / "progress.csv"
    if progress_csv.exists():
        progress = pd.read_csv(progress_csv)
        reward_history = _training_history_frame(
            progress,
            {
                "rollout/ep_rew_mean": "Episode Reward",
                "eval/mean_reward": "驗證 Reward",
            },
        )
        loss_history = _training_history_frame(
            progress,
            {
                "train/value_loss": "Value Loss",
                "train/policy_gradient_loss": "Policy Loss",
                "train/actor_loss": "Actor Loss",
                "train/critic_loss": "Critic Loss",
            },
        )
        stability_history = _training_history_frame(
            progress,
            {
                "train/entropy_loss": "Entropy Loss",
                "train/approx_kl": "Approx KL",
                "train/explained_variance": "Explained Variance",
            },
        )
        histories = [
            ("Reward", reward_history),
            ("Loss", loss_history),
            ("穩定性", stability_history),
        ]
        histories = [(label, frame) for label, frame in histories if not frame.empty]
        if histories:
            st.markdown("#### 訓練學習過程")
            history_tabs = st.tabs([label for label, _ in histories])
            for tab, (_, history) in zip(history_tabs, histories, strict=True):
                with tab:
                    st.line_chart(history, x="步數", height=240)

    st.caption(f"模型資料夾：{_relative_label(run_dir, project_root)}")
    with st.expander("完整訓練設定"):
        st.json(payload.get("training_config", {}))


def _render_research_result(
    experiment_dir: Path,
    rl_dir: Path,
    project_root: Path,
) -> None:
    """顯示多 seed Walk-forward 摘要與模型治理操作。"""
    summary = _read_json(experiment_dir / "summary.json")
    metrics = st.columns(6)
    metrics[0].metric("Seeds", int(summary.get("seed_count", 0)))
    metrics[1].metric("時間窗", int(summary.get("fold_count", 0)))
    metrics[2].metric("選模窗中位報酬", f"{float(summary.get('median_test_return', 0)):.2%}")
    metrics[3].metric("正報酬 Run", f"{float(summary.get('positive_run_ratio', 0)):.0%}")
    metrics[4].metric("最差回撤", f"{float(summary.get('worst_max_drawdown', 0)):.2%}")
    metrics[5].metric("成本中位數", f"{float(summary.get('median_cost_to_capital', 0)):.2%}")
    if summary.get("eligible"):
        st.success(
            "此實驗已通過多 seed、Walk-forward 與獨立 final holdout，"
            "可進入模擬倉驗證。"
        )
    else:
        st.warning("目前不可升級為 Champion：" + "；".join(summary.get("reasons", [])))

    final_holdout = dict(summary.get("final_holdout", {}))
    if final_holdout.get("status") == "complete":
        holdout_metrics = dict(final_holdout.get("metrics", {}))
        holdout_columns = st.columns(4)
        holdout_columns[0].metric(
            "Holdout 報酬",
            f"{float(holdout_metrics.get('total_return', 0)):.2%}",
        )
        holdout_columns[1].metric(
            "Holdout Sharpe",
            f"{float(holdout_metrics.get('sharpe_ratio', 0)):.2f}",
        )
        holdout_columns[2].metric(
            "Holdout 最大回撤",
            f"{float(holdout_metrics.get('max_drawdown', 0)):.2%}",
        )
        holdout_columns[3].metric(
            "Holdout 調倉",
            int(holdout_metrics.get("trades", 0)),
        )
        st.caption("Final holdout 位於資料尾端，不參與 seed、超參數或候選模型選擇。")
    elif final_holdout:
        st.info("Final holdout 狀態：" + str(final_holdout.get("status", "unknown")))

    runs_path = experiment_dir / "runs.csv"
    if runs_path.exists():
        runs = pd.read_csv(runs_path)
        themed_dataframe(
            runs,
            hide_index=True,
            width="stretch",
            key=themed_widget_key(f"rl_research_runs_{experiment_dir.name}"),
            column_config={
                "validation_return": st.column_config.NumberColumn(format="percent"),
                "test_return": st.column_config.NumberColumn("選模窗報酬", format="percent"),
                "max_drawdown": st.column_config.NumberColumn(format="percent"),
                "cost_to_capital": st.column_config.NumberColumn(format="percent"),
                "positive_market_ratio": st.column_config.NumberColumn(format="percent"),
                "test_sharpe": st.column_config.NumberColumn("選模窗 Sharpe", format="%.2f"),
            },
        )

    candidate_relative = str(summary.get("candidate_run_relative", ""))
    candidate_text = str(summary.get("candidate_run", ""))
    candidate = (
        experiment_dir / candidate_relative
        if candidate_relative
        else (Path(candidate_text) if candidate_text else None)
    )
    if candidate is None or not (candidate / "training.json").exists():
        st.info("此實驗目前沒有可登錄的候選模型。")
        return
    training = _read_json(candidate / "training.json")
    kind = str(dict(training.get("environment_expert", {})).get("kind", "general"))
    registry_path = rl_dir / "model_registry.json"
    registry = load_model_registry(registry_path)
    champion = dict(dict(registry.get("experts", {})).get(kind, {})).get("champion")
    st.caption(f"候選模型：{_relative_label(candidate, project_root)}")
    if champion:
        st.caption(f"目前 Champion：{_relative_label(Path(champion['run_dir']), project_root)}")
    controls = st.columns(3)
    if controls[0].button(
        "登錄 Challenger",
        icon=":material/add_task:",
        width="stretch",
        key=f"register_{experiment_dir.name}",
    ):
        try:
            register_model_challenger(
                registry_path,
                candidate,
                research_summary=experiment_dir / "summary.json",
            )
            st.success("已登錄 Challenger；目前交易模型沒有被更換。")
            st.rerun()
        except (OSError, ValueError) as exc:
            st.error(str(exc))
    if controls[1].button(
        "升級 Champion",
        icon=":material/verified:",
        width="stretch",
        key=f"promote_{experiment_dir.name}",
    ):
        try:
            promote_model_challenger(
                registry_path,
                candidate,
                paper_root=project_root / "data" / "paper_trading",
            )
            st.success("已通過研究與模擬證據，候選模型已升級為 Champion。")
            st.rerun()
        except (OSError, ValueError) as exc:
            st.error(str(exc))
    if controls[2].button(
        "回滾 Champion",
        icon=":material/undo:",
        width="stretch",
        key=f"rollback_{experiment_dir.name}",
        disabled=not champion,
    ):
        try:
            rollback_model_champion(registry_path, kind)
            st.success("已還原上一個 Champion。")
            st.rerun()
        except (OSError, ValueError) as exc:
            st.error(str(exc))


def render_rl_results(rl_dir: Path, project_root: Path) -> None:
    """顯示所有 PPO／SAC 訓練 Run 與樣本外曲線。"""
    runs = _result_training_runs(rl_dir, project_root)
    experiments = _result_research_experiments(rl_dir, project_root)
    if not runs and not experiments:
        st.info("目前沒有強化學習訓練紀錄。")
        return
    modes = (["單次訓練"] if runs else []) + (["穩健性實驗"] if experiments else [])
    mode = st.segmented_control("結果類型", modes, default=modes[0], width="stretch")
    if mode == "穩健性實驗":
        selected_experiment = st.selectbox(
            "研究實驗",
            experiments,
            format_func=lambda path: path.name,
            key="rl_research_result",
        )
        _render_research_result(selected_experiment, rl_dir, project_root)
        return
    st.caption(f"目前資料位置：{_relative_label(runs[0], project_root)}")
    selected = st.selectbox("訓練紀錄", runs, format_func=_training_label, key="rl_result_run")
    _render_training_result(selected, project_root)


def render_rl_training(rl_dir: Path, project_root: Path) -> None:
    """顯示完整 PPO／SAC 設定並同步執行一次訓練。"""
    environments = list_rl_environments(rl_dir)
    compatible_environments: list[Path] = []
    for environment in environments:
        try:
            payload = json.loads(
                (environment / "environment.json").read_text(encoding="utf-8")
            )
            source = dict(payload.get("source", {}))
            if (
                str(source.get("exchange", "")).lower() == "binance_futures"
                and str(source.get("symbol", "")).upper() == "BTC/USDT"
                and str(source.get("interval", "")).lower() == "15m"
            ):
                compatible_environments.append(environment)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    environments = compatible_environments
    if not environments:
        st.warning("請先建立至少一個強化學習環境。")
        return
    status, device_labels = _render_backend_status()
    if not status.available:
        return

    environment_dir = st.selectbox(
        "訓練環境",
        environments,
        format_func=_environment_label,
        key="rl_training_environment",
    )
    try:
        environment_payload = json.loads(
            (environment_dir / "environment.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError, json.JSONDecodeError):
        environment_payload = {}
    try:
        preflight = _cached_environment_preflight(str(environment_dir.resolve()))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        preflight = None
        st.error(f"訓練前檢查無法執行：{exc}")
    if preflight is not None:
        if not preflight.eligible:
            st.error("訓練前完整性阻擋：" + "；".join(preflight.errors))
        elif not preflight.formal_research_ready:
            st.warning(
                f"目前只適合 smoke：{preflight.total_rows:,} 根、"
                f"{preflight.observed_days:.1f} 天。" + "；".join(preflight.warnings)
            )
        else:
            st.success(
                f"正式訓練前檢查通過：{preflight.total_rows:,} 根、"
                f"{preflight.observed_days:.1f} 天。"
            )
    expert_kind = str(dict(environment_payload.get("expert", {})).get("kind", "general"))
    recommended_algorithm = "SAC" if expert_kind == "short_term" else "PPO"
    algorithm_label = st.segmented_control(
        "演算法",
        ["PPO", "SAC"],
        default=recommended_algorithm,
        width="stretch",
        key="rl_algorithm",
    )
    algorithm = str(algorithm_label).lower()

    common_columns = st.columns(5)
    with common_columns[0]:
        total_timesteps = st.number_input(
            "訓練步數", 1_000, 100_000_000, 100_000, 10_000, key="rl_total_timesteps"
        )
    with common_columns[1]:
        device = st.selectbox(
            "運算裝置",
            status.device_options,
            format_func=lambda value: device_labels[value],
            key="rl_training_device",
        )
    with common_columns[2]:
        n_envs = st.number_input("平行環境", 1, 32, 1, 1, key="rl_n_envs")
    with common_columns[3]:
        cpu_threads = st.number_input(
            "CPU 執行緒",
            1,
            available_cpu_threads(),
            min(4, available_cpu_threads()),
            1,
            help="建議保留 1～2 個核心給介面與即時行情。",
            key="rl_cpu_threads",
        )
    with common_columns[4]:
        seed = st.number_input("Random seed", 0, 2_147_483_647, 42, 1, key="rl_seed")

    robust_mode = st.toggle(
        "多 Seed＋Walk-forward 穩健性實驗",
        help="依時間前推並逐一訓練；為避免 GPU 記憶體競爭，不會同時執行多個模型。",
        key="rl_robust_mode",
    )
    research_seeds = "11,23,42,67,101"
    research_folds = 3
    research_holdout_percent = 10
    research_holdout_rows = 100
    if robust_mode:
        research_columns = st.columns(4)
        with research_columns[0]:
            research_seeds = st.text_input(
                "Seeds",
                research_seeds,
                help="至少 3 個；正式研究建議 5 個。",
                key="rl_research_seeds",
            )
        with research_columns[1]:
            research_folds = st.number_input(
                "Walk-forward 時間窗",
                2,
                8,
                3,
                1,
                key="rl_research_folds",
            )
        with research_columns[2]:
            research_holdout_percent = st.number_input(
                "Final holdout 比例 (%)",
                5,
                30,
                10,
                1,
                help="封存資料尾端；選 seed 與調參期間完全不使用。",
                key="rl_research_holdout_percent",
            )
        with research_columns[3]:
            research_holdout_rows = st.number_input(
                "Holdout 最少 K 線",
                20,
                1_000_000,
                100,
                20,
                key="rl_research_holdout_rows",
            )
        try:
            run_count = len(parse_seed_list(research_seeds)) * int(research_folds)
            st.caption(
                f"本次會串行訓練 {run_count} 個模型；每一個都使用上方的訓練步數。"
            )
        except ValueError as exc:
            st.warning(str(exc))

    with st.expander("共用模型設定"):
        common = st.columns(6)
        with common[0]:
            learning_rate = st.number_input(
                "Learning rate", 0.000001, 1.0, 0.0003, 0.0001, format="%.6f", key="rl_lr"
            )
        with common[1]:
            gamma = st.number_input("Gamma", 0.0, 1.0, 0.99, 0.005, key="rl_gamma")
        with common[2]:
            batch_size = st.number_input("Batch size", 2, 65_536, 256, 2, key="rl_batch")
        with common[3]:
            net_arch_text = st.text_input("網路層", "128,128", key="rl_net_arch")
        with common[4]:
            activation_fn = st.selectbox(
                "Activation", ["relu", "tanh", "elu", "leaky_relu"], key="rl_activation"
            )
        with common[5]:
            stats_window_size = st.number_input(
                "統計視窗", 1, 10_000, 100, 10, key="rl_stats_window"
            )

    ppo_values: dict[str, Any] = {}
    sac_values: dict[str, Any] = {}
    if algorithm == "ppo":
        with st.expander("PPO 完整參數"):
            row1 = st.columns(6)
            with row1[0]:
                ppo_values["n_steps"] = st.number_input(
                    "n_steps", 2, 1_000_000, 1024, 128, key="rl_ppo_steps"
                )
            with row1[1]:
                ppo_values["n_epochs"] = st.number_input(
                    "n_epochs", 1, 1_000, 10, 1, key="rl_ppo_epochs"
                )
            with row1[2]:
                ppo_values["gae_lambda"] = st.number_input(
                    "GAE lambda", 0.0, 1.0, 0.95, 0.01, key="rl_ppo_gae"
                )
            with row1[3]:
                ppo_values["clip_range"] = st.number_input(
                    "Clip range", 0.001, 10.0, 0.2, 0.05, key="rl_ppo_clip"
                )
            with row1[4]:
                ppo_values["ent_coef"] = st.number_input(
                    "Entropy coef", 0.0, 10.0, 0.005, 0.001, format="%.4f", key="rl_ppo_ent"
                )
            with row1[5]:
                ppo_values["vf_coef"] = st.number_input(
                    "Value coef", 0.0, 10.0, 0.5, 0.1, key="rl_ppo_vf"
                )
            row2 = st.columns(6)
            with row2[0]:
                ppo_values["max_grad_norm"] = st.number_input(
                    "Max grad norm", 0.001, 100.0, 0.5, 0.1, key="rl_ppo_grad"
                )
            with row2[1]:
                ppo_values["normalize_advantage"] = st.toggle(
                    "Normalize advantage", value=True, key="rl_ppo_normalize"
                )
            with row2[2]:
                ppo_values["use_sde"] = st.toggle("Use SDE", key="rl_ppo_sde")
            with row2[3]:
                ppo_values["sde_sample_freq"] = st.number_input(
                    "SDE sample freq", -1, 1_000_000, -1, 1, key="rl_ppo_sde_freq"
                )
            with row2[4]:
                clip_vf_enabled = st.toggle("Value clip", key="rl_ppo_clip_vf_enabled")
                clip_vf_value = st.number_input(
                    "Value clip 值",
                    0.001,
                    10.0,
                    0.2,
                    0.05,
                    disabled=not clip_vf_enabled,
                    key="rl_ppo_clip_vf",
                )
            with row2[5]:
                target_kl_enabled = st.toggle("Target KL", key="rl_ppo_kl_enabled")
                target_kl_value = st.number_input(
                    "Target KL 值",
                    0.0001,
                    10.0,
                    0.03,
                    0.01,
                    disabled=not target_kl_enabled,
                    key="rl_ppo_kl",
                )
            ppo_values["clip_range_vf"] = _optional_positive(clip_vf_enabled, clip_vf_value)
            ppo_values["target_kl"] = _optional_positive(target_kl_enabled, target_kl_value)
    else:
        with st.expander("SAC 完整參數"):
            row1 = st.columns(6)
            with row1[0]:
                sac_values["buffer_size"] = st.number_input(
                    "Replay buffer", 100, 100_000_000, 200_000, 10_000, key="rl_sac_buffer"
                )
            with row1[1]:
                sac_values["learning_starts"] = st.number_input(
                    "Learning starts", 0, 100_000_000, 5_000, 1_000, key="rl_sac_starts"
                )
            with row1[2]:
                sac_values["tau"] = st.number_input(
                    "Tau", 0.000001, 1.0, 0.005, 0.001, format="%.6f", key="rl_sac_tau"
                )
            with row1[3]:
                sac_values["train_freq"] = st.number_input(
                    "Train freq", 1, 1_000_000, 1, 1, key="rl_sac_train_freq"
                )
            with row1[4]:
                sac_values["train_freq_unit"] = st.selectbox(
                    "Train freq unit", ["step", "episode"], key="rl_sac_train_unit"
                )
            with row1[5]:
                sac_values["gradient_steps"] = st.number_input(
                    "Gradient steps", -1, 1_000_000, 1, 1, key="rl_sac_gradient"
                )
            row2 = st.columns(6)
            with row2[0]:
                sac_values["n_steps"] = st.number_input(
                    "N-step return", 1, 1_000, 1, 1, key="rl_sac_n_steps"
                )
            with row2[1]:
                sac_values["ent_coef"] = st.text_input("Entropy coef", "auto", key="rl_sac_ent")
            with row2[2]:
                sac_values["target_entropy"] = st.text_input(
                    "Target entropy", "auto", key="rl_sac_target_entropy"
                )
            with row2[3]:
                sac_values["target_update_interval"] = st.number_input(
                    "Target update", 1, 1_000_000, 1, 1, key="rl_sac_target_update"
                )
            with row2[4]:
                sac_values["optimize_memory_usage"] = st.toggle(
                    "節省 Replay 記憶體", key="rl_sac_memory"
                )
            with row2[5]:
                sac_values["use_sde"] = st.toggle("Use SDE", key="rl_sac_sde")
            row3 = st.columns(5)
            with row3[0]:
                sac_values["sde_sample_freq"] = st.number_input(
                    "SDE sample freq", -1, 1_000_000, -1, 1, key="rl_sac_sde_freq"
                )
            with row3[1]:
                sac_values["use_sde_at_warmup"] = st.toggle(
                    "Warmup 使用 SDE", key="rl_sac_sde_warmup"
                )
            with row3[2]:
                sac_values["action_noise"] = st.selectbox(
                    "Action noise", ["none", "normal", "ornstein_uhlenbeck"], key="rl_sac_noise"
                )
            with row3[3]:
                sac_values["action_noise_sigma"] = st.number_input(
                    "Noise sigma", 0.0, 10.0, 0.1, 0.05, key="rl_sac_sigma"
                )
            with row3[4]:
                sac_values["save_replay_buffer"] = st.toggle(
                    "保存 Replay buffer", value=True, key="rl_sac_save_replay"
                )

    with st.expander("Checkpoint 與評估"):
        evaluation = st.columns(4)
        with evaluation[0]:
            checkpoint_freq = st.number_input(
                "Checkpoint 每 N 步", 0, 100_000_000, 25_000, 5_000, key="rl_checkpoint"
            )
        with evaluation[1]:
            evaluation_freq = st.number_input(
                "驗證每 N 步", 0, 100_000_000, 10_000, 5_000, key="rl_eval_freq"
            )
        with evaluation[2]:
            n_eval_episodes = st.number_input(
                "驗證 Episodes", 1, 1_000, 1, 1, key="rl_eval_episodes"
            )
        with evaluation[3]:
            deterministic_eval = st.toggle("確定性評估", value=True, key="rl_deterministic")

    checkpoints = list_rl_checkpoints(rl_dir)
    resume_enabled = st.toggle(
        "從既有模型續跑",
        key="rl_resume_enabled",
        disabled=robust_mode,
    )
    resume_from = None
    if resume_enabled:
        if checkpoints:
            resume_from = st.selectbox(
                "續跑模型",
                checkpoints,
                format_func=lambda path: _relative_label(path, project_root),
                key="rl_resume_path",
            )
            st.caption(
                "續跑會保留模型原有網路與演算法參數；本頁的步數、裝置、評估與保存設定仍會套用。"
            )
        else:
            st.warning("目前沒有可續跑的模型檔。")

    activity = active_model_training(
        rl_dir,
        Path(rl_dir).resolve().parent.parent / "transformer",
    )
    training_active = activity is not None
    if activity is not None:
        st.info(
            f"{activity.label}正在執行；目前調整的參數只會套用到下一次工作。"
        )

    if st.button(
        "開始穩健性實驗" if robust_mode else "開始訓練",
        type="primary",
        icon=":material/play_arrow:",
        width="stretch",
        disabled=training_active or (resume_enabled and resume_from is None),
    ):
        try:
            config = RLTrainingConfig(
                algorithm=algorithm,
                total_timesteps=int(total_timesteps),
                device=str(device),
                seed=int(seed),
                n_envs=int(n_envs),
                cpu_threads=int(cpu_threads),
                learning_rate=float(learning_rate),
                gamma=float(gamma),
                batch_size=int(batch_size),
                net_arch=_parse_network_architecture(net_arch_text),
                activation_fn=str(activation_fn),
                stats_window_size=int(stats_window_size),
                checkpoint_freq=int(checkpoint_freq),
                evaluation_freq=int(evaluation_freq),
                n_eval_episodes=int(n_eval_episodes),
                deterministic_eval=bool(deterministic_eval),
                save_replay_buffer=bool(sac_values.get("save_replay_buffer", True)),
                ppo_n_steps=int(ppo_values.get("n_steps", 1024)),
                ppo_n_epochs=int(ppo_values.get("n_epochs", 10)),
                ppo_gae_lambda=float(ppo_values.get("gae_lambda", 0.95)),
                ppo_clip_range=float(ppo_values.get("clip_range", 0.2)),
                ppo_clip_range_vf=ppo_values.get("clip_range_vf"),
                ppo_normalize_advantage=bool(ppo_values.get("normalize_advantage", True)),
                ppo_ent_coef=float(ppo_values.get("ent_coef", 0.005)),
                ppo_vf_coef=float(ppo_values.get("vf_coef", 0.5)),
                ppo_max_grad_norm=float(ppo_values.get("max_grad_norm", 0.5)),
                ppo_use_sde=bool(ppo_values.get("use_sde", False)),
                ppo_sde_sample_freq=int(ppo_values.get("sde_sample_freq", -1)),
                ppo_target_kl=ppo_values.get("target_kl"),
                sac_buffer_size=int(sac_values.get("buffer_size", 200_000)),
                sac_learning_starts=int(sac_values.get("learning_starts", 5_000)),
                sac_tau=float(sac_values.get("tau", 0.005)),
                sac_train_freq=int(sac_values.get("train_freq", 1)),
                sac_train_freq_unit=str(sac_values.get("train_freq_unit", "step")),
                sac_gradient_steps=int(sac_values.get("gradient_steps", 1)),
                sac_optimize_memory_usage=bool(sac_values.get("optimize_memory_usage", False)),
                sac_n_steps=int(sac_values.get("n_steps", 1)),
                sac_ent_coef=_parse_auto_number(
                    str(sac_values.get("ent_coef", "auto")), "Entropy coef"
                ),
                sac_target_update_interval=int(sac_values.get("target_update_interval", 1)),
                sac_target_entropy=_parse_auto_number(
                    str(sac_values.get("target_entropy", "auto")),
                    "Target entropy",
                    allow_auto_initial=False,
                ),
                sac_use_sde=bool(sac_values.get("use_sde", False)),
                sac_sde_sample_freq=int(sac_values.get("sde_sample_freq", -1)),
                sac_use_sde_at_warmup=bool(sac_values.get("use_sde_at_warmup", False)),
                sac_action_noise=str(sac_values.get("action_noise", "none")),
                sac_action_noise_sigma=float(sac_values.get("action_noise_sigma", 0.1)),
            )
            research_config = None
            if robust_mode:
                research_config = RLResearchConfig(
                    seeds=parse_seed_list(research_seeds),
                    walk_forward_folds=int(research_folds),
                    final_holdout_fraction=float(research_holdout_percent) / 100,
                    min_final_holdout_rows=int(research_holdout_rows),
                )
            job = submit_rl_training_job(
                rl_dir,
                environment_dir,
                config,
                research_config=research_config,
                resume_from=resume_from,
                backend_status=status,
            )
            st.session_state["rl_active_job_id"] = job.job_id
            st.success("背景訓練已開始；可繼續操作其他功能，進度不會消失。")
            st.rerun()
        except (ValueError, OSError, RuntimeError, ImportError) as exc:
            st.error(str(exc))
