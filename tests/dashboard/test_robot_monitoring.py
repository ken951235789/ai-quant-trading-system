"""機器人監控績效口徑與健康狀態測試。"""

from __future__ import annotations

import pandas as pd

from ai_quant_trading.automation import AutomationStatus
from ai_quant_trading.dashboard.robot_monitoring import (
    assess_robot_health,
    calculate_position_exposure,
    calculate_profit_metrics,
)
from ai_quant_trading.dashboard.robot_monitoring_page import _model_path


def test_profit_metrics_use_taipei_day_and_monday_boundary() -> None:
    performance = pd.DataFrame(
        {
            "timestamp": [
                "2026-07-26T10:00:00Z",
                "2026-07-27T00:00:00Z",
                "2026-07-28T15:00:00Z",
                "2026-07-28T18:00:00Z",
                "2026-07-29T02:00:00Z",
            ],
            "equity": [1000, 1020, 1030, 1040, 1050],
        }
    )
    trades = pd.DataFrame({"net_pnl": [20, -5]})

    metrics, curve = calculate_profit_metrics(
        performance,
        initial_equity=1000,
        trades=trades,
        now=pd.Timestamp("2026-07-29T12:00:00", tz="Asia/Taipei"),
    )

    assert metrics.today_pnl == 20
    assert metrics.week_pnl == 50
    assert metrics.cumulative_pnl == 50
    assert metrics.today_return == 20 / 1030
    assert metrics.trade_count == 2
    assert metrics.win_rate == 0.5
    assert curve.iloc[-1]["cumulative_pnl"] == 50


def test_stale_curve_reports_zero_today_and_week_profit() -> None:
    performance = pd.DataFrame(
        {
            "timestamp": ["2026-07-20T00:00:00Z"],
            "equity": [1100],
        }
    )

    metrics, _ = calculate_profit_metrics(
        performance,
        initial_equity=1000,
        now=pd.Timestamp("2026-07-29T12:00:00", tz="Asia/Taipei"),
    )

    assert metrics.today_pnl == 0
    assert metrics.week_pnl == 0
    assert metrics.cumulative_pnl == 100


def test_position_exposure_reports_long_short_capital_and_leverage() -> None:
    long_position = calculate_position_exposure(
        quantity=0.002,
        entry_price=60_000,
        market_price=61_000,
        equity=1_000,
    )
    short_position = calculate_position_exposure(
        quantity=-0.001,
        entry_price=60_000,
        market_price=59_000,
        equity=1_000,
    )

    assert long_position.direction == "多單"
    assert long_position.capital_used == 120
    assert long_position.position_notional == 122
    assert long_position.account_exposure == 0.122
    assert long_position.leverage == 1
    assert short_position.direction == "空單"
    assert short_position.capital_used == 60
    assert short_position.position_notional == 59


def test_position_exposure_reports_unopened_account() -> None:
    exposure = calculate_position_exposure(
        quantity=0,
        entry_price=None,
        market_price=60_000,
        equity=1_000,
    )

    assert exposure.direction == "空手"
    assert exposure.capital_used == 0
    assert exposure.leverage == 0


def test_robot_health_prioritizes_risk_halt_and_missing_model() -> None:
    running = AutomationStatus(
        state="running",
        heartbeat_at="2026-07-29T03:59:00Z",
        last_message="本輪完成",
    )
    now = pd.Timestamp("2026-07-29T04:00:00Z")

    halted = assess_robot_health(running, risk_halted=True, now=now)
    missing = assess_robot_health(running, model_available=False, now=now)
    healthy = assess_robot_health(running, now=now)

    assert halted.status == "風控停機"
    assert missing.status == "模型遺失"
    assert healthy.status == "運行中"
    assert healthy.heartbeat_age_minutes == 1


def test_robot_health_detects_stale_heartbeat_and_failure() -> None:
    stale = AutomationStatus(
        state="running",
        heartbeat_at="2026-07-29T03:50:00Z",
        last_message="等待下一輪",
    )
    failed = AutomationStatus(
        state="failed",
        last_message="熔斷",
        last_error="network error",
    )
    now = pd.Timestamp("2026-07-29T04:00:00Z")

    assert assess_robot_health(stale, now=now).status == "心跳逾時"
    assert assess_robot_health(failed, now=now).message == "network error"


def test_robot_health_accepts_recent_websocket_activity() -> None:
    stale_cycle = AutomationStatus(
        state="running",
        heartbeat_at="2026-07-29T03:40:00Z",
        last_message="等待 15m 收盤",
    )

    health = assess_robot_health(
        stale_cycle,
        activity_at="2026-07-29T03:59:58Z",
        now=pd.Timestamp("2026-07-29T04:00:00Z"),
    )

    assert health.status == "運行中"
    assert health.heartbeat_age_minutes is not None
    assert health.heartbeat_age_minutes < 0.1


def test_model_path_resolves_legacy_rl_folder_name(tmp_path) -> None:
    expected = (
        tmp_path
        / "data"
        / "processed"
        / "rl"
        / "environments"
        / "environment_a"
        / "training"
        / "20260802T052007550212Z_ppo"
    )
    expected.mkdir(parents=True)

    resolved = _model_path("20260802T052007550212Z_ppo", tmp_path)

    assert resolved == expected.resolve()


def test_model_path_keeps_missing_reference_for_health_check(tmp_path) -> None:
    resolved = _model_path("missing_model", tmp_path)

    assert resolved == (tmp_path / "missing_model").resolve()
