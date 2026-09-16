"""離線故障注入；驗證重播可靠性，不接觸任何交易 API。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

from ai_quant_trading.research.replay import (
    InjectedTransportDisconnect,
    MarketEventReplayer,
    ReplayCheckpointStore,
)
from ai_quant_trading.trading.market_events import (
    ConsumerDeliveryError,
    MarketEvent,
    MarketEventBus,
    MarketEventGap,
    OutOfOrderMarketEvent,
)


@dataclass(frozen=True, slots=True)
class FaultExperimentResult:
    scenario: str
    passed: bool
    expected: str
    observed: str
    delivered_events: int
    duplicate_events: int = 0
    gap_events: int = 0
    consumer_failures: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class _RecordingConsumer:
    """以 event_id 冪等保存，模擬正式 consumer 的必要行為。"""

    def __init__(self) -> None:
        self.event_ids: dict[str, MarketEvent] = {}

    def __call__(self, event: MarketEvent) -> None:
        self.event_ids[event.event_id] = event


class _FailOnceConsumer:
    def __init__(self, recorder: _RecordingConsumer, target_event_id: str) -> None:
        self.recorder = recorder
        self.target_event_id = target_event_id
        self.failed = False

    def __call__(self, event: MarketEvent) -> None:
        if event.event_id == self.target_event_id and not self.failed:
            self.failed = True
            raise RuntimeError("注入 consumer 單次崩潰")
        self.recorder(event)


def _result(
    scenario: str,
    passed: bool,
    expected: str,
    observed: str,
    bus: MarketEventBus,
) -> FaultExperimentResult:
    return FaultExperimentResult(
        scenario=scenario,
        passed=passed,
        expected=expected,
        observed=observed,
        delivered_events=bus.metrics.delivered,
        duplicate_events=bus.metrics.duplicates,
        gap_events=bus.metrics.gaps,
        consumer_failures=bus.metrics.consumer_failures,
    )


def run_fault_injection_experiments(
    events: Sequence[MarketEvent],
    *,
    source_id: str,
    output_dir: str | Path,
) -> tuple[FaultExperimentResult, ...]:
    """執行六個可重現情境，全部只使用記憶體與實驗 checkpoint。"""
    if len(events) < 5:
        raise ValueError("故障注入至少需要 5 筆連續事件")
    if len({event.stream_key for event in events}) != 1:
        raise ValueError("故障注入必須使用同一標的與同一週期事件")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    midpoint = len(events) // 2
    # 離線研究允許批次 checkpoint；正式交易呼叫仍維持每事件一次的預設值。
    checkpoint_every = max(1, min(25, len(events) // 4))
    results: list[FaultExperimentResult] = []

    recorder = _RecordingConsumer()
    bus = MarketEventBus([recorder])
    MarketEventReplayer(events, source_id=source_id).run(bus)
    results.append(
        _result(
            "baseline",
            len(recorder.event_ids) == len(events),
            "所有事件各投遞一次",
            f"保存 {len(recorder.event_ids)}/{len(events)} 筆",
            bus,
        )
    )

    recorder = _RecordingConsumer()
    bus = MarketEventBus([recorder])
    MarketEventReplayer(events, source_id=source_id).run(
        bus,
        duplicate_at_index=midpoint,
    )
    results.append(
        _result(
            "duplicate_event",
            len(recorder.event_ids) == len(events) and bus.metrics.duplicates == 1,
            "重複事件被 event_id 擋下且不重複寫入",
            f"重複 {bus.metrics.duplicates} 筆，保存 {len(recorder.event_ids)} 筆",
            bus,
        )
    )

    recorder = _RecordingConsumer()
    bus = MarketEventBus([recorder])
    checkpoint = ReplayCheckpointStore(root / "disconnect_checkpoint.json")
    interrupted = False
    try:
        MarketEventReplayer(events, source_id=source_id).run(
            bus,
            checkpoint_store=checkpoint,
            disconnect_before_index=midpoint,
            checkpoint_every=checkpoint_every,
        )
    except InjectedTransportDisconnect:
        interrupted = True
    resumed = MarketEventReplayer(events, source_id=source_id).run(
        bus,
        checkpoint_store=checkpoint,
        checkpoint_every=checkpoint_every,
    )
    results.append(
        _result(
            "disconnect_resume",
            interrupted and resumed.resumed and len(recorder.event_ids) == len(events),
            "中斷後從最後成功 checkpoint 續傳且不遺失",
            f"resumed={resumed.resumed}，保存 {len(recorder.event_ids)} 筆",
            bus,
        )
    )

    recorder = _RecordingConsumer()
    flaky = _FailOnceConsumer(recorder, events[midpoint].event_id)
    bus = MarketEventBus([flaky])
    checkpoint = ReplayCheckpointStore(root / "consumer_checkpoint.json")
    failed_once = False
    try:
        MarketEventReplayer(events, source_id=source_id).run(
            bus,
            checkpoint_store=checkpoint,
            checkpoint_every=checkpoint_every,
        )
    except ConsumerDeliveryError:
        failed_once = True
    resumed = MarketEventReplayer(events, source_id=source_id).run(
        bus,
        checkpoint_store=checkpoint,
        checkpoint_every=checkpoint_every,
    )
    results.append(
        _result(
            "consumer_crash_recovery",
            failed_once and resumed.resumed and len(recorder.event_ids) == len(events),
            "Consumer 失敗時 checkpoint 不前進，重啟後重試同一事件",
            f"consumer_failures={bus.metrics.consumer_failures}，保存 {len(recorder.event_ids)} 筆",
            bus,
        )
    )

    recorder = _RecordingConsumer()
    bus = MarketEventBus([recorder], strict_ordering=True)
    reordered = list(events)
    reordered[midpoint], reordered[midpoint + 1] = (
        reordered[midpoint + 1],
        reordered[midpoint],
    )
    detected = False
    try:
        MarketEventReplayer(reordered, source_id=source_id).run(bus)
    except OutOfOrderMarketEvent:
        detected = True
    results.append(
        _result(
            "out_of_order_detection",
            detected and bus.metrics.out_of_order == 1,
            "事件時間倒退時 fail closed",
            f"detected={detected}，out_of_order={bus.metrics.out_of_order}",
            bus,
        )
    )

    recorder = _RecordingConsumer()
    bus = MarketEventBus([recorder], strict_continuity=True)
    dropped = [event for index, event in enumerate(events) if index != midpoint]
    detected = False
    try:
        MarketEventReplayer(dropped, source_id=source_id).run(bus)
    except MarketEventGap:
        detected = True
    results.append(
        _result(
            "missing_bar_detection",
            detected and bus.metrics.gaps == 1,
            "少一根 K 線時偵測時間缺口",
            f"detected={detected}，gaps={bus.metrics.gaps}",
            bus,
        )
    )

    original = events[midpoint]
    corrupted_payload = dict(original.payload)
    corrupted_payload["high"] = min(
        float(corrupted_payload["open"]),
        float(corrupted_payload["close"]),
    ) - 1.0
    rejected = False
    try:
        replace(original, payload=corrupted_payload)
    except ValueError:
        rejected = True
    empty_bus = MarketEventBus()
    results.append(
        _result(
            "corrupted_ohlcv_rejection",
            rejected,
            "不合理 high/low 在投遞前被拒絕",
            f"rejected={rejected}",
            empty_bus,
        )
    )
    return tuple(results)
