"""故障注入與可重現研究報告測試。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ai_quant_trading.operations.integrity import verify_artifact_manifest
from ai_quant_trading.research.faults import run_fault_injection_experiments
from ai_quant_trading.research.replay import (
    HistoricalMarketEventSource,
    InjectedTransportDisconnect,
    MarketEventReplayer,
    ReplayCheckpointStore,
)
from ai_quant_trading.research.reporting import run_replay_research
from ai_quant_trading.trading.market_events import MarketEventBus


def _frame(rows: int = 12) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=rows, freq="15min")
    opened = timestamps.view("int64") // 1_000_000
    close = pd.Series([60_000.0 + index for index in range(rows)])
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "exchange": "binance_futures",
            "symbol": "BTC/USDT",
            "interval": "15m",
            "open": close,
            "high": close + 10,
            "low": close - 10,
            "close": close + 2,
            "volume": 10.0,
            "open_time_ms": opened,
            "close_time_ms": opened + 899_999,
            "collected_at": timestamps + pd.Timedelta(minutes=15),
        }
    )


def test_disconnect_resumes_from_last_successful_checkpoint(tmp_path: Path) -> None:
    source = HistoricalMarketEventSource(_frame())
    checkpoint = ReplayCheckpointStore(tmp_path / "checkpoint.json")
    delivered: list[str] = []
    bus = MarketEventBus([lambda event: delivered.append(event.event_id)])
    replayer = MarketEventReplayer(source.events, source_id=source.source_id)

    with pytest.raises(InjectedTransportDisconnect):
        replayer.run(bus, checkpoint_store=checkpoint, disconnect_before_index=5)
    result = replayer.run(bus, checkpoint_store=checkpoint)

    assert result.resumed
    assert len(delivered) == len(source.events)
    assert len(set(delivered)) == len(source.events)
    assert checkpoint.load(source.source_id).last_index == len(source.events) - 1


def test_batched_checkpoint_replays_tail_idempotently(tmp_path: Path) -> None:
    source = HistoricalMarketEventSource(_frame())
    checkpoint = ReplayCheckpointStore(tmp_path / "batched.json")
    delivered: list[str] = []
    bus = MarketEventBus([lambda event: delivered.append(event.event_id)])

    with pytest.raises(InjectedTransportDisconnect):
        MarketEventReplayer(source.events, source_id=source.source_id).run(
            bus,
            checkpoint_store=checkpoint,
            disconnect_before_index=5,
            checkpoint_every=3,
        )
    result = MarketEventReplayer(source.events, source_id=source.source_id).run(
        bus,
        checkpoint_store=checkpoint,
        checkpoint_every=3,
    )

    assert result.resumed
    assert len(set(delivered)) == len(source.events)
    assert bus.metrics.duplicates == 2


def test_source_id_changes_when_selected_event_window_changes() -> None:
    frame = _frame()

    full = HistoricalMarketEventSource(frame)
    shorter = HistoricalMarketEventSource(frame.tail(10).reset_index(drop=True))

    assert full.source_id != shorter.source_id


def test_all_fault_injection_scenarios_pass(tmp_path: Path) -> None:
    source = HistoricalMarketEventSource(_frame())

    results = run_fault_injection_experiments(
        source.events,
        source_id=source.source_id,
        output_dir=tmp_path,
    )

    assert len(results) == 7
    assert all(result.passed for result in results)
    assert {result.scenario for result in results} >= {
        "duplicate_event",
        "disconnect_resume",
        "consumer_crash_recovery",
        "out_of_order_detection",
        "missing_bar_detection",
        "corrupted_ohlcv_rejection",
    }


def test_research_report_records_hash_environment_and_reproduction(tmp_path: Path) -> None:
    source_path = tmp_path / "btc_15m.csv"
    _frame().to_csv(source_path, index=False)

    artifacts = run_replay_research(
        source_path,
        output_root=tmp_path / "results",
        limit=10,
        project_root=tmp_path,
    )

    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    report = artifacts.report_path.read_text(encoding="utf-8")
    assert artifacts.passed
    assert manifest["source"]["sha256"]
    assert len(manifest["result_sha256"]) == 64
    assert manifest["environment"]["python"]
    assert "run_replay_research.py" in manifest["reproduce_command"]
    assert "故障注入結果" in report
    assert "研究能力與驗證矩陣" in report
    assert verify_artifact_manifest(artifacts.experiment_dir)
