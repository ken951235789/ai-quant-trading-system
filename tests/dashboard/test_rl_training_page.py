"""強化學習結果頁的資料整理測試。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ai_quant_trading.dashboard.rl_training_page import (
    _duration_text,
    _evaluation_curve,
    _live_training_record,
    _market_metrics_frame,
    _parse_auto_number,
    _position_exposure_summary,
    _result_training_runs,
    _training_label,
    _training_history_frame,
)


def test_evaluation_curve_normalizes_each_market_independently(tmp_path: Path) -> None:
    path = tmp_path / "evaluation.csv"
    pd.DataFrame(
        {
            "timestamp": ["2026-01-01", "2026-01-02", "2026-01-01", "2026-01-02"],
            "market": ["BTC", "BTC", "AAPL", "AAPL"],
            "equity": [50_000, 51_000, 50_000, 49_500],
        }
    ).to_csv(path, index=False)

    curve = _evaluation_curve(path)

    assert curve.loc[curve["market"] == "BTC", "累積報酬"].iloc[-1] == pytest.approx(0.02)
    assert curve.loc[curve["market"] == "AAPL", "累積報酬"].iloc[-1] == pytest.approx(-0.01)


def test_market_metrics_frame_keeps_all_test_markets() -> None:
    payload = {
        "metrics": {
            "test_markets": {
                "binance:BTC/USDT:1d": {
                    "total_return": 0.03,
                    "max_drawdown": 0.05,
                    "final_equity": 51_500,
                    "trades": 8,
                    "total_fees": 12.5,
                },
                "yahoo_finance:AAPL:1d": {
                    "total_return": -0.01,
                    "max_drawdown": 0.02,
                    "final_equity": 49_500,
                    "trades": 3,
                    "total_fees": 4.0,
                },
            }
        }
    }

    frame = _market_metrics_frame(payload)

    assert frame["市場"].tolist() == ["binance:BTC/USDT:1d", "yahoo_finance:AAPL:1d"]
    assert frame["測試報酬"].tolist() == pytest.approx([0.03, -0.01])


def test_live_progress_helpers_prepare_charts_and_eta() -> None:
    payload = {
        "completed_timesteps": 2_000,
        "metrics": {
            "recent_reward_mean": 0.012,
            "value_loss": 0.25,
        },
    }
    record = _live_training_record(payload)
    history = _training_history_frame(
        pd.DataFrame(
            {
                "time/total_timesteps": [1_000, 2_000],
                "rollout/ep_rew_mean": [-0.1, 0.1],
            }
        ),
        {"rollout/ep_rew_mean": "Episode Reward", "eval/mean_reward": "驗證 Reward"},
    )

    assert record == {"步數": 2_000.0, "近期 Reward": 0.012, "Value Loss": 0.25}
    assert history.columns.tolist() == ["步數", "Episode Reward"]
    assert _duration_text(125) == "2 分 5 秒"


def test_target_entropy_parser_rejects_auto_initial_value() -> None:
    with pytest.raises(ValueError, match="Target entropy"):
        _parse_auto_number("auto_0.1", "Target entropy", allow_auto_initial=False)


def test_result_training_runs_includes_packaged_desktop_data(tmp_path: Path) -> None:
    """原始碼介面也應看見同專案打包版產生的訓練結果。"""
    source_root = tmp_path / "data" / "processed" / "rl" / "environments"
    packaged_root = (
        tmp_path
        / "dist"
        / "AIQuantTradingSystem"
        / "data"
        / "processed"
        / "rl"
        / "environments"
    )
    source_run = source_root / "source_env" / "training" / "source_run"
    packaged_run = packaged_root / "packaged_env" / "training" / "packaged_run"
    for run in (source_run, packaged_run):
        run.mkdir(parents=True)
        (run / "training.json").write_text('{"status":"complete"}', encoding="utf-8")

    runs = _result_training_runs(source_root, tmp_path)

    assert {path.resolve() for path in runs} == {
        source_run.resolve(),
        packaged_run.resolve(),
    }


def test_training_label_distinguishes_long_and_short_experts(tmp_path: Path) -> None:
    run = tmp_path / "20260719T090547430330Z_ppo"
    run.mkdir()
    (run / "training.json").write_text(
        """{
          "status": "complete",
          "environment_source": {"symbol": "UNIVERSAL"},
          "environment_expert": {"kind": "short_term"},
          "training_config": {"algorithm": "ppo"}
        }""",
        encoding="utf-8",
    )

    assert _training_label(run).startswith("短期專家 · UNIVERSAL · PPO · 完成")


def test_position_exposure_uses_actual_position_instead_of_sell_action(tmp_path: Path) -> None:
    path = tmp_path / "test_evaluation.csv"
    pd.DataFrame(
        {
            "side": ["BUY", "SELL", "SELL", "HOLD"],
            "position_fraction": [0.10, 0.05, -0.05, 0.0],
        }
    ).to_csv(path, index=False)

    summary = _position_exposure_summary(path)

    assert summary["long_ratio"] == pytest.approx(0.5)
    assert summary["short_ratio"] == pytest.approx(0.25)
    assert summary["flat_ratio"] == pytest.approx(0.25)
    assert summary["short_rows"] == 1
    assert summary["min_position"] == pytest.approx(-0.05)
