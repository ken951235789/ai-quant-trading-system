"""Step 8 Binance 實盤交易操作頁。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from ai_quant_trading.database.config import DatabaseSettings
from ai_quant_trading.database.engine import check_database_health, engine_from_url
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
from ai_quant_trading.dashboard.ui import themed_dataframe, themed_widget_key
from ai_quant_trading.data_collection.assets import asset_display_name
from ai_quant_trading.live_trading import (
    LiveTradingConfig,
    build_live_gateway,
    credentials_available,
)
from ai_quant_trading.live_trading.credentials import credential_variable_names
from ai_quant_trading.live_trading.notification import (
    configure_live_logger,
    email_notification_status,
    notify_event,
    run_email_failure_recovery_drill,
    send_test_email,
)
from ai_quant_trading.live_trading.operations import OperationalRiskLimits
from ai_quant_trading.live_trading.readiness import (
    assess_operational_evidence,
    assess_testnet_evidence,
)
from ai_quant_trading.live_trading.reporting import (
    build_daily_trading_report,
    save_daily_trading_report,
)
from ai_quant_trading.live_trading.storage import (
    CYCLE_COLUMNS,
    ORDER_COLUMNS,
    PROTECTION_COLUMNS,
    SNAPSHOT_COLUMNS,
    activate_emergency_halt,
    clear_emergency_halt,
    emergency_halt_reason,
    live_trading_paths,
    load_positions,
    read_live_csv,
)
from ai_quant_trading.risk import DynamicLeverageConfig, RiskConfig
from ai_quant_trading.runtime_secrets import runtime_env_path


ENVIRONMENT_LABELS = {"Testnet": "testnet", "Demo": "demo", "Live": "live"}
LOGGER = logging.getLogger("ai_quant_trading.live_trading")


@dataclass(frozen=True, slots=True)
class LivePageSettings:
    """頁面共用且不包含 API Secret 的交易限制。"""

    environment: str
    allowed_symbols: tuple[str, ...]
    max_order_quote: float
    max_balance_fraction: float
    leverage: int

    def config(self, enabled: bool = False) -> LiveTradingConfig:
        return LiveTradingConfig(
            environment=self.environment,
            market_type="usd_m_futures",
            trading_enabled=enabled,
            allowed_symbols=self.allowed_symbols,
            max_order_quote=self.max_order_quote,
            max_balance_fraction=self.max_balance_fraction,
            leverage=self.leverage,
        )


def _rl_training_directories(rl_dir: Path) -> list[Path]:
    """只列出已完成且實際包含模型檔的 PPO／SAC 訓練。"""
    from ai_quant_trading.reinforcement_learning import (
        list_rl_training_runs,
        registered_champion,
    )

    results = []
    for run_dir in list_rl_training_runs(rl_dir):
        try:
            payload = json.loads((run_dir / "training.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if payload.get("status") == "complete" and any(run_dir.rglob("*.zip")):
            results.append(run_dir)
    champion_paths = {
        path.resolve()
        for kind in ("long_term", "short_term", "general")
        if (path := registered_champion(rl_dir / "model_registry.json", kind)) is not None
    }
    return sorted(
        results,
        key=lambda path: (
            path.resolve() not in champion_paths,
            -path.stat().st_mtime,
        ),
    )


def _render_live_dynamic_leverage() -> DynamicLeverageConfig:
    """顯示實盤用動態槓桿門檻；3x 永遠保留給最嚴格的一級。"""
    with st.expander("動態槓桿風控", expanded=True):
        enabled = st.toggle(
            "依 Transformer 證據在 1x～硬上限間調整",
            value=True,
            key="live_dynamic_leverage_enabled",
        )
        top = st.columns(3)
        uncalibrated_max = top[0].selectbox(
            "機率未校準上限",
            [1, 2, 3],
            index=0,
            format_func=lambda value: f"{value}x",
            key="live_dynamic_uncalibrated",
            disabled=not enabled,
        )
        positive_expectancy = top[1].toggle(
            "只允許正期望值",
            value=True,
            key="live_dynamic_expectancy",
            disabled=not enabled,
        )
        top[2].caption("硬上限使用頁面上方的逐倉槓桿設定。")
        evidence = st.columns(4)
        probability_2 = evidence[0].number_input(
            "2x 最低方向機率", 0.50, 0.95, 0.58, 0.01,
            key="live_dynamic_p2", disabled=not enabled,
        )
        probability_3 = evidence[1].number_input(
            "3x 最低方向機率", float(probability_2), 0.99,
            max(0.65, float(probability_2)), 0.01,
            key="live_dynamic_p3", disabled=not enabled,
        )
        risk_reward_2 = evidence[2].number_input(
            "2x 最低淨損益比", 0.50, 10.0, 1.25, 0.05,
            key="live_dynamic_rr2", disabled=not enabled,
        )
        risk_reward_3 = evidence[3].number_input(
            "3x 最低淨損益比", float(risk_reward_2), 10.0,
            max(1.75, float(risk_reward_2)), 0.05,
            key="live_dynamic_rr3", disabled=not enabled,
        )
        market = st.columns(4)
        uncertainty_2 = market[0].number_input(
            "2x 不確定度上限", 0.01, 1.0, 0.45, 0.01,
            key="live_dynamic_u2", disabled=not enabled,
        )
        uncertainty_3 = market[1].number_input(
            "3x 不確定度上限", 0.01, float(uncertainty_2),
            min(0.30, float(uncertainty_2)), 0.01,
            key="live_dynamic_u3", disabled=not enabled,
        )
        drawdown_2 = market[2].number_input(
            "2x 回撤上限 %", 0.1, 50.0, 5.0, 0.1,
            key="live_dynamic_dd2", disabled=not enabled,
        )
        drawdown_3 = market[3].number_input(
            "3x 回撤上限 %", 0.1, float(drawdown_2),
            min(2.5, float(drawdown_2)), 0.1,
            key="live_dynamic_dd3", disabled=not enabled,
        )
        execution = st.columns(4)
        spread_2 = execution[0].number_input(
            "2x Spread 上限 bps", 0.1, 100.0, 8.0, 0.1,
            key="live_dynamic_spread2", disabled=not enabled,
        )
        spread_3 = execution[1].number_input(
            "3x Spread 上限 bps", 0.1, float(spread_2),
            min(4.0, float(spread_2)), 0.1,
            key="live_dynamic_spread3", disabled=not enabled,
        )
        volatility_2 = execution[2].number_input(
            "2x 預測波動上限 %", 0.1, 50.0, 3.0, 0.1,
            key="live_dynamic_vol2", disabled=not enabled,
        )
        volatility_3 = execution[3].number_input(
            "3x 預測波動上限 %", 0.1, float(volatility_2),
            min(2.0, float(volatility_2)), 0.1,
            key="live_dynamic_vol3", disabled=not enabled,
        )
    return DynamicLeverageConfig(
        enabled=bool(enabled),
        max_leverage=3,
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


def _rl_environment_metadata(training_dir: Path) -> dict[str, object]:
    """使用訓練資料夾旁的可攜路徑讀取環境設定。"""
    local = training_dir.parent.parent / "environment.json"
    training = json.loads((training_dir / "training.json").read_text(encoding="utf-8"))
    configured = Path(str(training.get("environment_dir", ""))) / "environment.json"
    target = local if local.exists() else configured
    return json.loads(target.read_text(encoding="utf-8"))


def _format_rl_label(training_dir: Path) -> str:
    try:
        training = json.loads((training_dir / "training.json").read_text(encoding="utf-8"))
        environment = _rl_environment_metadata(training_dir)
        source = dict(environment.get("source", {}))
        algorithm = str(training.get("training_config", {}).get("algorithm", "RL")).upper()
        kind = (
            "通用模型"
            if environment.get("environment_kind") == "universal"
            else source.get("symbol", "未知標的")
        )
        result = dict(training.get("metrics", {}).get("test", {}))
        return (
            f"{kind} · {algorithm} · 測試 {float(result.get('total_return', 0)):.2%} · "
            f"{training_dir.name[:15]}"
        )
    except (OSError, ValueError, TypeError):
        return training_dir.name


def _render_account(account: dict[str, object]) -> None:
    is_futures = "assets" in account
    balances = pd.DataFrame(
        account.get("assets", []) if is_futures else account.get("balances", [])
    )
    if not balances.empty:
        numeric = (
            ["walletBalance", "marginBalance", "availableBalance", "unrealizedProfit"]
            if is_futures
            else ["free", "locked"]
        )
        available_columns = [column for column in numeric if column in balances.columns]
        balances[available_columns] = balances[available_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        balances = balances.loc[balances[available_columns].fillna(0).ne(0).any(axis=1)]
    metrics = st.columns(3)
    metrics[0].metric("可交易", "是" if account.get("canTrade") else "否")
    metrics[1].metric(
        "帳戶權益" if is_futures else "可提領",
        (
            f"{float(account.get('totalMarginBalance', 0)):.2f} USDT"
            if is_futures
            else "是"
            if account.get("canWithdraw")
            else "否"
        ),
    )
    metrics[2].metric("非零資產", len(balances))
    if not balances.empty:
        themed_dataframe(
            balances[["asset", *available_columns]],
            width="stretch",
            hide_index=True,
            key=themed_widget_key("live_balances"),
        )


def _render_email_notifications(
    *,
    env_file: Path,
    live_dir: Path,
    environment: str,
) -> None:
    """顯示不含憑證的 Email 告警狀態，並提供實際送達測試。"""
    status = email_notification_status(env_file)
    with st.expander("Email 告警", expanded=False, icon=":material/mail:"):
        columns = st.columns(3)
        columns[0].metric("告警通道", "可使用" if status["configured"] else "未完成")
        columns[1].metric("收件人", int(status["recipient_count"]))
        columns[2].metric("加密", str(status["security"]).upper())
        if status["enabled"]:
            st.caption(
                f"SMTP：{status['smtp_host']}:{status['smtp_port']} · "
                f"層級：{', '.join(status['alert_levels'])}"
            )
        else:
            display_path = env_file.as_posix()
            st.caption(f"在 {display_path} 設定 Email 參數並重新啟動 App。")
        errors = tuple(status["errors"])
        if errors:
            st.warning("；".join(str(item) for item in errors))
        if st.button(
            "寄送測試 Email",
            icon=":material/send:",
            disabled=not bool(status["configured"]),
            key=f"live_email_test_{environment}",
        ):
            try:
                paths = live_trading_paths(live_dir, environment)
                send_test_email(paths.notifications_log, env_file)
                st.success("測試信已交給 SMTP 伺服器，請確認收件匣與垃圾郵件匣。")
            except (OSError, ValueError, RuntimeError) as exc:
                LOGGER.exception("Dashboard Email 告警測試失敗：%s", exc)
                st.error(f"測試寄送失敗：{type(exc).__name__}")
        drill_confirmed = st.checkbox(
            "我了解演練會寄出一封故障信和一封恢復信，但不會送單或停機",
            key=f"live_email_drill_confirm_{environment}",
        )
        if st.button(
            "執行 Email 故障與恢復演練",
            icon=":material/health_and_safety:",
            disabled=not bool(status["configured"]) or not drill_confirmed,
            key=f"live_email_drill_{environment}",
        ):
            try:
                paths = live_trading_paths(live_dir, environment)
                run_email_failure_recovery_drill(paths.notifications_log, env_file)
                st.success("演練信已寄出：請確認收到故障與恢復兩封 Email。")
            except (OSError, ValueError, RuntimeError) as exc:
                LOGGER.exception("Dashboard Email 故障演練失敗：%s", exc)
                st.error(f"Email 演練失敗：{exc}")


def _render_execution_controls(key_prefix: str, settings: LivePageSettings) -> tuple[bool, str]:
    mode = st.segmented_control(
        "執行模式",
        options=["只驗證", "送到目前環境"],
        default="只驗證",
        key=f"{key_prefix}_execution_mode",
        width="stretch",
    )
    execute = mode == "送到目前環境"
    confirmation = ""
    if execute:
        confirmed = st.checkbox(
            "我已確認交易環境、標的、數量與風險上限",
            key=f"{key_prefix}_confirmed",
        )
        confirmation = st.text_input(
            "確認文字",
            placeholder=settings.config().confirmation_phrase,
            key=f"{key_prefix}_confirmation",
        )
        if not confirmed:
            confirmation = ""
    return execute, confirmation


def _render_rl_cycle(
    settings: LivePageSettings,
    *,
    env_file: Path,
    raw_dir: Path,
    rl_dir: Path,
    live_dir: Path,
    project_root: Path,
) -> None:
    """顯示 RL 目標持倉、單次輪次與無人值守控制。"""
    from ai_quant_trading.reinforcement_learning import assess_rl_training_quality

    training_dirs = _rl_training_directories(rl_dir)
    if not training_dirs:
        st.warning("目前沒有已完成的 PPO／SAC 訓練模型。")
        return
    training_dir = st.selectbox(
        "RL 訓練成品",
        training_dirs,
        format_func=_format_rl_label,
        key="live_rl_training",
    )
    training = json.loads((training_dir / "training.json").read_text(encoding="utf-8"))
    environment = _rl_environment_metadata(training_dir)
    environment_config = dict(environment.get("environment_config", {}))
    if not environment_config.get("allow_short", False):
        st.info("此模型沒有空頭動作；USD-M 下單層仍會維持 long-only 模型語意。")
    st.caption(
        f"執行市場：Binance USD-M 永續 · One-way · ISOLATED · "
        f"槓桿硬上限 {settings.leverage}x"
    )
    universal = environment.get("environment_kind") == "universal"
    quality = assess_rl_training_quality(training)
    source = dict(environment.get("source", {}))
    if universal:
        market_columns = st.columns(2)
        with market_columns[0]:
            symbol = st.selectbox(
                "套用標的",
                settings.allowed_symbols,
                format_func=lambda value: asset_display_name(value, "crypto"),
                key="live_rl_symbol",
            )
        with market_columns[1]:
            interval = st.selectbox(
                "K 線週期",
                ["5m", "15m", "30m", "1h", "4h", "1d"],
                index=3,
                key="live_rl_interval",
            )
    else:
        symbol = str(source.get("symbol", ""))
        interval = str(source.get("interval", ""))
        st.caption(f"模型市場：{symbol} · {interval}")
    evidence = assess_testnet_evidence(
        live_dir,
        model_dir=training_dir,
        symbol=symbol,
        market_type="usd_m_futures",
    )
    operational_evidence = assess_operational_evidence(
        live_dir,
        "testnet",
        market_type="usd_m_futures",
    )
    if universal:
        trained_binance_symbols = {
            str(values.get("symbol", ""))
            for values in dict(environment.get("markets", {})).values()
            if str(values.get("exchange", "")).lower() == "binance_futures"
        }
    else:
        trained_binance_symbols = (
            {str(source.get("symbol", ""))}
            if str(source.get("exchange", "")).lower() == "binance_futures"
            else set()
        )
    trained_market = symbol in trained_binance_symbols
    status_columns = st.columns(3)
    status_columns[0].metric("RL 品質", "通過" if quality.eligible else "研究中")
    status_columns[1].metric(
        "Testnet 證據",
        f"{evidence.successful_cycles}/{evidence.minimum_cycles} 輪次",
        f"{evidence.observed_days:.1f}/{evidence.minimum_days} 天",
    )
    status_columns[2].metric(
        "營運安全",
        "通過" if operational_evidence.eligible else "待完成",
        f"稽核 {operational_evidence.audit_events} 筆",
    )
    if not quality.eligible:
        st.warning("Live 已鎖定；Testnet 仍可驗證：" + "；".join(quality.reasons))
    elif not trained_market:
        st.warning("此標的未出現在 RL 訓練環境；可做 Testnet 研究，但不可直接 Live。")
    elif not evidence.eligible:
        st.info("模型品質已通過，Live 仍需完成：" + "；".join(evidence.reasons))
    elif not operational_evidence.eligible:
        st.info("Testnet 營運驗收仍需完成：" + "；".join(operational_evidence.reasons))
    else:
        st.success("模型品質、訓練市場、Testnet 證據與營運安全都已通過。")
    symbol_allowed = symbol in settings.allowed_symbols
    if not symbol_allowed:
        st.error("RL 模型標的不在目前交易白名單。")

    controls = st.columns(4)
    with controls[0]:
        download_limit = st.number_input("最新 K 線筆數", 220, 1000, 500, 20, key="live_rl_limit")
    with controls[1]:
        entry_percent = st.number_input(
            "進場目標持倉 %", 1.0, 100.0, 10.0, 1.0, key="live_rl_entry"
        )
    with controls[2]:
        exit_percent = st.number_input("出場目標持倉 %", 0.0, 99.0, 2.0, 1.0, key="live_rl_exit")
    with controls[3]:
        position_percent = st.number_input(
            "硬性持倉上限 %", 1.0, 100.0, 25.0, 1.0, key="live_rl_position"
        )
    risk_columns = st.columns(5)
    with risk_columns[0]:
        max_risk = st.number_input("單筆最大風險 %", 0.1, 20.0, 1.0, 0.1, key="live_rl_risk")
    with risk_columns[1]:
        stop_loss = st.number_input("固定停損 %", 0.1, 50.0, 1.0, 0.1, key="live_rl_stop")
    with risk_columns[2]:
        take_profit = st.number_input("固定停利 %", 0.1, 200.0, 1.5, 0.1, key="live_rl_take")
    with risk_columns[3]:
        daily_loss = st.number_input(
            "單日虧損上限 %", 0.1, 10.0, 1.0, 0.1, key="live_rl_daily_loss"
        )
    with risk_columns[4]:
        maximum_drawdown = st.number_input(
            "最大回撤 %", 1.0, 30.0, 8.0, 1.0, key="live_rl_drawdown"
        )
    thresholds_valid = float(exit_percent) < float(entry_percent)
    if not thresholds_valid:
        st.error("出場門檻必須小於進場門檻。")
    dynamic_leverage = _render_live_dynamic_leverage()
    dynamic_leverage = DynamicLeverageConfig(
        **{
            **dynamic_leverage.to_dict(),
            "max_leverage": int(settings.leverage),
            "uncalibrated_max_leverage": min(
                dynamic_leverage.uncalibrated_max_leverage,
                int(settings.leverage),
            ),
        }
    )

    execute, confirmation = _render_execution_controls("rl_cycle", settings)
    live_blocked = settings.environment == "live" and (
        not quality.eligible
        or not trained_market
        or not evidence.eligible
        or not operational_evidence.eligible
    )
    disabled = (
        not credentials_available(settings.environment, env_file)
        or not symbol_allowed
        or not thresholds_valid
        or live_blocked
        or (execute and confirmation != settings.config().confirmation_phrase)
    )
    if st.button(
        "執行 RL 交易輪次",
        type="primary",
        icon=":material/model_training:",
        width="stretch",
        disabled=disabled,
    ):
        try:
            from ai_quant_trading.live_trading.rl_service import (
                download_latest_rl_market_data,
                run_live_rl_cycle,
            )
            from ai_quant_trading.reinforcement_learning import load_rl_policy

            gateway = build_live_gateway(settings.config(enabled=execute), env_path=env_file)
            policy = load_rl_policy(training_dir, device="cpu")
            with st.spinner("正在下載資料、建立 observation 並檢查訂單..."):
                market_path = download_latest_rl_market_data(
                    policy,
                    raw_dir,
                    symbol=symbol if universal else None,
                    interval=interval if universal else None,
                    limit=int(download_limit),
                )
                result = run_live_rl_cycle(
                    pd.read_csv(market_path),
                    policy,
                    gateway=gateway,
                    storage_root=live_dir,
                    risk_config=RiskConfig(
                        max_risk_per_trade=float(max_risk) / 100,
                        fixed_stop_loss_pct=float(stop_loss) / 100,
                        take_profit_pct=float(take_profit) / 100,
                        max_drawdown_limit=float(maximum_drawdown) / 100,
                        max_position_fraction=float(position_percent) / 100,
                        dynamic_leverage=dynamic_leverage,
                    ),
                    operational_limits=OperationalRiskLimits(
                        maximum_daily_loss=float(daily_loss) / 100,
                        maximum_drawdown=float(maximum_drawdown) / 100,
                    ),
                    execute=execute,
                    confirmation=confirmation,
                    entry_threshold=float(entry_percent) / 100,
                    exit_threshold=float(exit_percent) / 100,
                )
            st.success(result.message)
            st.json(
                {
                    "signal_time": result.signal.timestamp,
                    "symbol": result.signal.symbol,
                    "target_position": round(result.signal.target_fraction, 6),
                    "signal": result.signal.action_signal,
                    "selected_leverage": result.signal.leverage,
                    "leverage_reason": result.signal.leverage_reason,
                    "reason": result.reason,
                }
            )
        except (ValueError, OSError, PermissionError, RuntimeError) as exc:
            LOGGER.exception("Dashboard RL 實盤輪次失敗：%s", exc)
            st.error(str(exc))

    st.subheader("RL 無人值守")
    live_paths = live_trading_paths(live_dir, settings.environment)
    runner_paths = automation_paths(live_paths.environment_dir / "automation")
    runner_status = read_automation_status(runner_paths)
    summary = st.columns(3)
    summary[0].metric("狀態", runner_status.state)
    summary[1].metric("成功輪詢", runner_status.successful_cycles)
    summary[2].metric("連續錯誤", runner_status.consecutive_errors)
    poll_seconds = st.number_input(
        "資料輪詢秒數", 10, 3600, 60, 10, key=f"live_rl_poll_{settings.environment}"
    )
    running = runner_status.state == "running"
    active = runner_status.state in {"running", "stopping"}
    autostart_enabled = automation_autostart_enabled(runner_paths.root_dir)
    autostart_requested = st.toggle(
        "Windows 登入後自動恢復 Testnet Worker",
        value=autostart_enabled,
        disabled=settings.environment != "testnet",
        key=f"live_rl_autostart_{settings.environment}",
        help="只支援 Testnet；Supervisor 不會把設定切換成 Live。",
    )
    if settings.environment != "testnet":
        st.caption("開機自動恢復只開放 Testnet，Demo 與 Live 必須人工啟動。")
    elif autostart_requested:
        st.caption("啟動並送到 Testnet 後會保存授權；重開機不需要開啟 App。")
    else:
        st.caption("目前不會在 Windows 登入後自動送出 Testnet 訂單。")
    if autostart_enabled and not autostart_requested:
        set_automation_autostart(runner_paths.root_dir, False)
        autostart_enabled = False
    actions = st.columns(2)
    if actions[0].button(
        "啟動 RL 背景輪詢",
        type="primary",
        icon=":material/schedule:",
        width="stretch",
        disabled=disabled or active,
        key=f"live_rl_auto_start_{settings.environment}",
    ):
        arguments = [
            "rl-auto",
            "--environment",
            settings.environment,
            "--env-file",
            str(env_file),
            "--allowed-symbols",
            *settings.allowed_symbols,
            "--max-order-quote",
            str(settings.max_order_quote),
            "--max-balance-fraction",
            str(settings.max_balance_fraction),
            "--leverage",
            str(settings.leverage),
            "--training-dir",
            str(training_dir),
            "--download-latest",
            "--download-limit",
            str(int(download_limit)),
            "--entry-threshold",
            str(float(entry_percent) / 100),
            "--exit-threshold",
            str(float(exit_percent) / 100),
            "--max-risk-per-trade",
            str(float(max_risk) / 100),
            "--fixed-stop-loss-pct",
            str(float(stop_loss) / 100),
            "--take-profit-pct",
            str(float(take_profit) / 100),
            "--max-position-fraction",
            str(float(position_percent) / 100),
            "--maximum-daily-loss",
            str(float(daily_loss) / 100),
            "--max-drawdown-limit",
            str(float(maximum_drawdown) / 100),
            "--dynamic-leverage" if dynamic_leverage.enabled else "--no-dynamic-leverage",
            "--dynamic-uncalibrated-max",
            str(dynamic_leverage.uncalibrated_max_leverage),
            "--dynamic-probability-2",
            str(dynamic_leverage.leverage_2_min_probability),
            "--dynamic-probability-3",
            str(dynamic_leverage.leverage_3_min_probability),
            "--dynamic-risk-reward-2",
            str(dynamic_leverage.leverage_2_min_risk_reward),
            "--dynamic-risk-reward-3",
            str(dynamic_leverage.leverage_3_min_risk_reward),
            "--dynamic-uncertainty-2",
            str(dynamic_leverage.leverage_2_max_uncertainty),
            "--dynamic-uncertainty-3",
            str(dynamic_leverage.leverage_3_max_uncertainty),
            "--dynamic-drawdown-2",
            str(dynamic_leverage.leverage_2_max_drawdown),
            "--dynamic-drawdown-3",
            str(dynamic_leverage.leverage_3_max_drawdown),
            "--dynamic-spread-2",
            str(dynamic_leverage.leverage_2_max_spread_bps),
            "--dynamic-spread-3",
            str(dynamic_leverage.leverage_3_max_spread_bps),
            "--dynamic-volatility-2",
            str(dynamic_leverage.leverage_2_max_volatility),
            "--dynamic-volatility-3",
            str(dynamic_leverage.leverage_3_max_volatility),
            "--dynamic-positive-expectancy"
            if dynamic_leverage.require_positive_expectancy
            else "--no-dynamic-positive-expectancy",
            "--poll-seconds",
            str(int(poll_seconds)),
        ]
        if universal:
            arguments.extend(["--symbol", symbol, "--interval", interval])
        if execute:
            arguments.extend(["--execute", "--confirm", confirmation])
        try:
            if autostart_requested and not execute:
                raise ValueError("開機自動恢復必須選擇『送到目前環境』並確認 Testnet")
            pid = start_automation_worker("live", project_root, runner_paths.root_dir, arguments)
            set_automation_autostart(
                runner_paths.root_dir,
                bool(autostart_requested and settings.environment == "testnet"),
            )
            st.success(f"RL 背景輪詢已啟動，PID={pid}")
        except (OSError, ValueError) as exc:
            st.error(f"RL 背景輪詢啟動失敗：{exc}")
    if actions[1].button(
        "安全停止",
        icon=":material/stop_circle:",
        width="stretch",
        disabled=not running,
        key=f"live_rl_auto_stop_{settings.environment}",
    ):
        set_automation_autostart(runner_paths.root_dir, False)
        request_automation_stop(runner_paths)
        st.info("已送出安全停止請求，並取消下次開機自動恢復。")


def _render_manual_order(
    settings: LivePageSettings,
    *,
    env_file: Path,
    live_dir: Path,
) -> None:
    st.info("手動區只提供緊急 reduce-only 全平，不允許繞過模型與保護單直接開倉。")
    symbol = st.selectbox(
        "交易標的",
        settings.allowed_symbols,
        format_func=lambda value: asset_display_name(value, "crypto"),
        key="live_manual_symbol",
    )
    execute, confirmation = _render_execution_controls("manual_order", settings)
    disabled = not credentials_available(settings.environment, env_file) or (
        execute and confirmation != settings.config().confirmation_phrase
    )
    if st.button(
        "檢查並全平目前持倉",
        icon=":material/close:",
        width="stretch",
        disabled=disabled,
    ):
        try:
            from ai_quant_trading.live_trading.service import record_order_submission

            config = settings.config(enabled=execute)
            gateway = build_live_gateway(config, env_path=env_file)
            snapshot = gateway.portfolio(symbol)
            position_quantity = float(getattr(snapshot, "position_quantity", 0.0))
            if abs(position_quantity) <= 1e-12:
                raise ValueError("目前沒有 USD-M 持倉可供平倉")
            side = "SELL" if position_quantity > 0 else "BUY"
            preview = gateway.preview_market_order(
                symbol,
                side,
                full_exit=True,
                reduce_only=True,
            )
            submission = (
                gateway.submit_order(preview, confirmation)
                if execute
                else gateway.validate_order(preview)
            )
            paths = live_trading_paths(live_dir, settings.environment)
            record_order_submission(paths, settings.environment, submission, None, None)
            notify_event(
                paths.notifications_log,
                "warning" if execute else "info",
                f"緊急全平 {settings.environment} {preview.symbol} "
                f"{preview.side} {preview.quantity}",
                env_path=env_file,
            )
            st.success("訂單已送出。" if execute else "訂單驗證通過，未送入撮合引擎。")
            st.json(
                {
                    "symbol": preview.symbol,
                    "side": preview.side,
                    "quantity": str(preview.quantity),
                    "estimated_notional": str(preview.estimated_notional),
                    "client_order_id": submission.client_order_id,
                }
            )
        except (ValueError, OSError, PermissionError, RuntimeError) as exc:
            LOGGER.exception("Dashboard 手動實盤訂單失敗：%s", exc)
            st.error(str(exc))


def _render_records(
    settings: LivePageSettings,
    live_dir: Path,
    database_settings: DatabaseSettings,
) -> None:
    paths = live_trading_paths(live_dir, settings.environment)
    st.caption(
        f"交易狀態來源：{'PostgreSQL' if database_settings.enabled else 'CSV/JSON'} · "
        f"本機紀錄路徑：{paths.environment_dir}"
    )
    report = build_daily_trading_report(paths)
    operations = assess_operational_evidence(
        live_dir,
        settings.environment,
        market_type="usd_m_futures",
    )
    summary = st.columns(5)
    summary[0].metric("今日損益", f"{report.pnl:+,.2f}", f"{report.return_rate:+.2%}")
    summary[1].metric("今日最大回撤", f"{report.maximum_drawdown:.2%}")
    summary[2].metric("決策／實際單", f"{report.decision_cycles}/{report.submitted_orders}")
    summary[3].metric("未確認訂單", report.unresolved_orders)
    summary[4].metric("稽核鏈", "正常" if report.audit_valid else "異常")
    if report.reasons:
        st.warning("；".join(report.reasons))
    elif operations.eligible:
        st.success("帳戶快照、訂單、保護單與稽核鏈目前正常。")

    actions = st.columns(3)
    if actions[0].button(
        "產生今日報告",
        icon=":material/description:",
        width="stretch",
        key=f"live_report_{settings.environment}",
    ):
        json_path, markdown_path, _ = save_daily_trading_report(paths)
        st.success(f"已保存：{json_path.name}、{markdown_path.name}")
    actions[1].download_button(
        "下載報告 JSON",
        data=json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        file_name=f"{settings.environment}_{report.report_date}_report.json",
        mime="application/json",
        width="stretch",
        key=f"live_report_download_{settings.environment}",
    )
    halt = emergency_halt_reason(paths)
    if actions[2].button(
        "啟動緊急停機" if halt is None else "解除緊急停機",
        icon=":material/emergency:" if halt is None else ":material/restart_alt:",
        width="stretch",
        key=f"live_halt_{settings.environment}",
    ):
        if halt is None:
            activate_emergency_halt(paths, "由操作介面手動啟動")
            request_automation_stop(automation_paths(paths.environment_dir / "automation"))
            notify_event(
                paths.notifications_log,
                "critical",
                f"{settings.environment} 由操作介面啟動緊急停機",
            )
            st.error("已停止新增風險並要求背景程序安全停止；既有交易所保護單不會取消。")
        elif report.unresolved_orders or report.unprotected_positions:
            st.error("仍有未確認訂單或未保護部位，禁止解除緊急停機。")
        else:
            clear_emergency_halt(paths)
            notify_event(
                paths.notifications_log,
                "info",
                f"{settings.environment} 緊急停機已解除；背景程序仍需手動重新啟動",
            )
            st.success("緊急停機已解除；背景程序仍需由你重新啟動。")

    orders = read_live_csv(paths.orders_csv, ORDER_COLUMNS, limit=100)
    cycles = read_live_csv(paths.cycles_csv, CYCLE_COLUMNS, limit=100)
    snapshots = read_live_csv(paths.account_snapshots_csv, SNAPSHOT_COLUMNS, limit=100)
    protections = read_live_csv(paths.protections_csv, PROTECTION_COLUMNS, limit=100)
    st.subheader("AI 輪次")
    themed_dataframe(
        cycles.tail(100),
        width="stretch",
        hide_index=True,
        key=themed_widget_key("live_cycles"),
    )
    st.subheader("訂單")
    themed_dataframe(
        orders.tail(100),
        width="stretch",
        hide_index=True,
        key=themed_widget_key("live_orders"),
    )
    st.subheader("帳戶快照")
    themed_dataframe(
        snapshots.tail(100),
        width="stretch",
        hide_index=True,
        key=themed_widget_key("live_snapshots"),
    )
    st.subheader("交易所保護單")
    themed_dataframe(
        protections.tail(100),
        width="stretch",
        hide_index=True,
        key=themed_widget_key("live_protections"),
    )
    if paths.emergency_halt_json.exists():
        st.error("緊急停機已啟用")
        st.json(json.loads(paths.emergency_halt_json.read_text(encoding="utf-8")))
    positions = load_positions(paths)
    if positions:
        st.subheader("本系統管理的部位")
        st.json({key: asdict(value) for key, value in positions.items()})


def render_live_trading_page(
    *,
    project_root: Path,
    raw_dir: Path,
    rl_dir: Path,
    live_dir: Path,
) -> None:
    """顯示憑證狀態、RL 輪次、手動單與持久化紀錄。"""
    configure_live_logger(project_root / "logs")
    st.title("實盤交易")
    st.caption("BTC/USDT USD-M 永續合約；真實下單預設關閉")
    st.warning(
        "研究與下單都固定為 Binance USD-M 永續。先使用 Testnet；"
        "Live 仍需要模型品質、營運證據、`.env` 總開關與每次確認文字。"
    )

    environment_label = st.segmented_control(
        "Binance 環境",
        list(ENVIRONMENT_LABELS),
        default="Testnet",
        width="stretch",
    )
    environment = ENVIRONMENT_LABELS[str(environment_label)]
    env_file = runtime_env_path(project_root)
    available = credentials_available(environment, env_file)
    key_name, secret_name = credential_variable_names(environment)
    status_columns = st.columns([1, 2])
    status_columns[0].metric("API 憑證", "已設定" if available else "未設定")
    status_columns[1].code(f"{key_name}\n{secret_name}", language=None)

    database_settings = DatabaseSettings.from_environment(env_file)
    database_columns = st.columns([1, 2])
    database_columns[0].metric(
        "交易狀態儲存",
        "PostgreSQL" if database_settings.enabled else "CSV / JSON",
    )
    if database_settings.enabled:
        database_columns[1].caption(database_settings.redacted_url)
        if database_columns[1].button(
            "測試資料庫",
            icon=":material/database:",
            key="live_database_health",
        ):
            engine = engine_from_url(
                database_settings.database_url,
                database_settings.pool_size,
                database_settings.max_overflow,
                database_settings.connect_timeout_seconds,
            )
            healthy, message = check_database_health(engine)
            st.session_state["live_database_health_result"] = (healthy, message)
        database_result = st.session_state.get("live_database_health_result")
        if database_result:
            healthy, message = database_result
            st.success(message) if healthy else st.error(message)
    else:
        database_columns[1].caption(
            f"在 {env_file} 設定 AI_QUANT_STORAGE_BACKEND=postgres 後才會切換。"
        )

    _render_email_notifications(
        env_file=env_file,
        live_dir=live_dir,
        environment=environment,
    )

    allowed_symbols = ("BTC/USDT",)
    st.text_input("交易白名單", value="BTC/USDT", disabled=True)
    limit_columns = st.columns(3)
    with limit_columns[0]:
        max_order_quote = st.number_input(
            "單筆最大報價資產金額",
            min_value=1.0,
            value=100.0,
            step=10.0,
        )
    with limit_columns[1]:
        max_balance_fraction = st.number_input(
            "單筆最多使用可用餘額 %",
            min_value=1.0,
            max_value=100.0,
            value=25.0,
            step=1.0,
        )
    with limit_columns[2]:
        leverage = st.selectbox(
            "逐倉槓桿硬上限",
            [1, 2, 3],
            index=1,
            format_func=lambda value: f"{value}x",
        )
    settings = LivePageSettings(
        environment,
        tuple(allowed_symbols),
        float(max_order_quote),
        float(max_balance_fraction) / 100,
        int(leverage),
    )

    if st.button(
        "測試連線並讀取帳戶",
        icon=":material/account_balance_wallet:",
        disabled=not available,
    ):
        try:
            gateway = build_live_gateway(settings.config(), env_path=env_file)
            st.session_state[f"live_account_{environment}"] = gateway.account_status()
        except (ValueError, OSError, RuntimeError) as exc:
            LOGGER.exception("Dashboard 實盤帳戶連線失敗：%s", exc)
            st.error(str(exc))
    account = st.session_state.get(f"live_account_{environment}")
    if isinstance(account, dict):
        _render_account(account)

    st.divider()
    rl_tab, manual_tab, records_tab = st.tabs(["RL 目標持倉", "緊急全平", "交易紀錄"])
    with rl_tab:
        _render_rl_cycle(
            settings,
            env_file=env_file,
            raw_dir=raw_dir,
            rl_dir=rl_dir,
            live_dir=live_dir,
            project_root=project_root,
        )
    with manual_tab:
        _render_manual_order(settings, env_file=env_file, live_dir=live_dir)
    with records_tab:
        _render_records(settings, live_dir, database_settings)
