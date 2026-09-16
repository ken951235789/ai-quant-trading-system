"""不連線真實交易所的實盤安全演練。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from ai_quant_trading.live_trading.binance_client import BinancePrivateApiError
from ai_quant_trading.live_trading.config import LiveTradingConfig
from ai_quant_trading.live_trading.futures_gateway import FuturesTradingGateway
from ai_quant_trading.live_trading.readiness import (
    assess_operational_evidence,
    assess_testnet_evidence,
)
from ai_quant_trading.live_trading.storage import (
    activate_emergency_halt,
    clear_emergency_halt,
    emergency_halt_reason,
    live_trading_paths,
)
from ai_quant_trading.live_trading.user_stream import (
    BinanceFuturesUserStream,
    UserStreamConfig,
)


@dataclass(frozen=True, slots=True)
class DrillResult:
    """單項安全演練結果。"""

    name: str
    passed: bool
    details: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class _StreamRepository:
    def __init__(self) -> None:
        self.states: list[str] = []
        self.events: list[dict[str, Any]] = []
        self.projected: list[str] = []

    def replay_unprocessed_exchange_events(self, *, limit: int = 1_000) -> int:
        return 0

    def update_stream_checkpoint(self, *_args: Any, **values: Any) -> None:
        self.states.append(str(values["state"]))

    def append_exchange_event(
        self,
        _environment: str,
        _stream: str,
        payload: dict[str, Any],
    ) -> tuple[bool, str]:
        key = f"event-{len(self.events) + 1}"
        self.events.append(payload)
        return True, key

    def project_exchange_event(self, event_key: str) -> None:
        self.projected.append(event_key)


class _StreamClient:
    def __init__(self) -> None:
        self.created = 0
        self.closed = 0

    def create_listen_key(self) -> str:
        self.created += 1
        return f"listen-key-{self.created}"

    def close_listen_key(self) -> None:
        self.closed += 1

    def keepalive_listen_key(self) -> None:
        return None


class _DrillSocket:
    def __init__(
        self,
        _url: str,
        *,
        on_open: Any,
        on_message: Any,
        on_error: Any,
        on_pong: Any,
        attempt: int,
        stream: BinanceFuturesUserStream,
    ) -> None:
        self.on_open = on_open
        self.on_message = on_message
        self.on_error = on_error
        self.on_pong = on_pong
        self.attempt = attempt
        self.stream = stream
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def run_forever(self, **_kwargs: Any) -> None:
        self.on_open(self)
        if self.attempt == 1:
            return
        self.on_message(
            self,
            '{"e":"ACCOUNT_UPDATE","E":1788800000000,"a":{"m":"ORDER"}}',
        )
        self.stream.stop()


class _ProtectionClient:
    def __init__(self, position: str = "0.002") -> None:
        self.position = Decimal(position)
        self.algos: dict[str, dict[str, object]] = {}
        self.cancelled: list[str] = []

    def sync_server_time(self) -> int:
        return 0

    def fetch_account(self) -> dict[str, object]:
        return {
            "canTrade": True,
            "totalMarginBalance": "1000",
            "availableBalance": "900",
            "assets": [
                {
                    "asset": "USDT",
                    "walletBalance": "1000",
                    "marginBalance": "1000",
                    "availableBalance": "900",
                }
            ],
        }

    def fetch_position_risk(self, _symbol: str) -> list[dict[str, object]]:
        return [
            {
                "symbol": "BTCUSDT",
                "positionSide": "BOTH",
                "positionAmt": str(self.position),
                "entryPrice": "50000",
                "markPrice": "50000",
                "unRealizedProfit": "0",
                "liquidationPrice": "30000",
                "leverage": "2",
                "marginType": "isolated",
            }
        ]

    def fetch_exchange_info(self, _symbol: str) -> dict[str, object]:
        return {
            "symbol": "BTCUSDT",
            "status": "TRADING",
            "baseAsset": "BTC",
            "quoteAsset": "USDT",
            "filters": [
                {
                    "filterType": "MARKET_LOT_SIZE",
                    "minQty": "0.0001",
                    "maxQty": "100",
                    "stepSize": "0.0001",
                },
                {"filterType": "MIN_NOTIONAL", "notional": "5"},
                {
                    "filterType": "PRICE_FILTER",
                    "minPrice": "1",
                    "maxPrice": "1000000",
                    "tickSize": "0.1",
                },
            ],
        }

    def fetch_price(self, _symbol: str) -> Decimal:
        return Decimal("50000")

    def query_algo_order(self, client_id: str) -> dict[str, object]:
        if client_id not in self.algos:
            raise BinancePrivateApiError("not found", -2013)
        return self.algos[client_id]

    def place_close_position_algo_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        trigger_price: Decimal,
        client_id: str,
    ) -> dict[str, object]:
        response = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "triggerPrice": str(trigger_price),
            "closePosition": True,
            "workingType": "MARK_PRICE",
            "algoStatus": "NEW",
            "clientAlgoId": client_id,
        }
        self.algos[client_id] = response
        return response

    def cancel_algo_order(self, client_id: str) -> dict[str, object]:
        self.cancelled.append(client_id)
        return {"algoStatus": "CANCELED", "clientAlgoId": client_id}


def run_disconnect_reconnect_drill() -> DrillResult:
    """模擬首次連線中斷，驗證重連、對帳與事件投影。"""
    repository = _StreamRepository()
    client = _StreamClient()
    reconciliations = 0
    attempts = 0

    def reconcile() -> object:
        nonlocal reconciliations
        reconciliations += 1
        return type("ReconcileResult", (), {"consistent": True})()

    stream = BinanceFuturesUserStream(
        client,  # type: ignore[arg-type]
        repository,  # type: ignore[arg-type]
        UserStreamConfig(
            environment="testnet",
            reconnect_initial_seconds=0.001,
            reconnect_max_seconds=0.002,
        ),
        reconcile=reconcile,
    )

    def socket_factory(url: str, **callbacks: Any) -> _DrillSocket:
        nonlocal attempts
        attempts += 1
        return _DrillSocket(
            url,
            attempt=attempts,
            stream=stream,
            **callbacks,
        )

    stream.websocket_factory = socket_factory
    stream.run()
    passed = bool(
        attempts == 2
        and stream.reconnect_count == 1
        and reconciliations == 2
        and repository.projected == ["event-1"]
        and "reconnecting" in repository.states
        and repository.states[-1] == "stopped"
    )
    return DrillResult(
        "disconnect_reconnect",
        passed,
        {
            "connection_attempts": attempts,
            "reconnect_count": stream.reconnect_count,
            "reconciliations": reconciliations,
            "projected_events": len(repository.projected),
            "checkpoint_states": repository.states,
        },
    )


def run_protection_order_drill() -> DrillResult:
    """驗證多單停損、停利方向、Mark Price 觸發與取消。"""
    client = _ProtectionClient()
    gateway = FuturesTradingGateway(
        client,  # type: ignore[arg-type]
        LiveTradingConfig(
            environment="testnet",
            market_type="usd_m_futures",
            trading_enabled=True,
            allowed_symbols=("BTC/USDT",),
            exchange_protection_enabled=True,
        ),
    )
    submission = gateway.submit_position_protection(
        symbol="BTC/USDT",
        position_side="LONG",
        stop_price=49_000,
        take_profit_price=51_000,
        confirmation="EXECUTE TESTNET",
        seed="offline-protection-drill",
    )
    responses = [submission.stop_response, submission.take_profit_response]
    cancellation = gateway.cancel_position_protection(
        submission.stop_client_algo_id,
        submission.take_profit_client_algo_id,
        "EXECUTE TESTNET",
    )
    passed = bool(
        {str(item["type"]) for item in responses}
        == {"STOP_MARKET", "TAKE_PROFIT_MARKET"}
        and all(item["side"] == "SELL" for item in responses)
        and all(item["closePosition"] is True for item in responses)
        and all(item["workingType"] == "MARK_PRICE" for item in responses)
        and set(cancellation) == {"stop", "take_profit"}
    )
    return DrillResult(
        "protection_orders",
        passed,
        {
            "stop": submission.stop_response,
            "take_profit": submission.take_profit_response,
            "cancelled_ids": client.cancelled,
        },
    )


def run_kill_switch_drill() -> DrillResult:
    """在隔離目錄開關緊急停機，驗證 Live 證據門檻會拒絕。"""
    with TemporaryDirectory(prefix="aiquant_kill_switch_") as temporary:
        paths = live_trading_paths(temporary, "testnet")
        activate_emergency_halt(paths, "離線 Kill Switch 演練")
        reason = emergency_halt_reason(paths)
        evidence = assess_testnet_evidence(temporary, minimum_cycles=1, minimum_days=0)
        blocked_by_halt = any("緊急停機" in item for item in evidence.reasons)
        clear_emergency_halt(paths)
        cleared = emergency_halt_reason(paths) is None
    return DrillResult(
        "kill_switch",
        bool(reason and blocked_by_halt and cleared),
        {
            "activation_reason": reason,
            "live_gate_blocked": not evidence.eligible,
            "gate_reasons": list(evidence.reasons),
            "cleared": cleared,
        },
    )


def run_live_safety_drills(storage_root: str | Path) -> dict[str, object]:
    """執行三項離線演練，再檢查真實本機的 Live 門檻狀態。"""
    drills = [
        run_disconnect_reconnect_drill(),
        run_protection_order_drill(),
        run_kill_switch_drill(),
    ]
    testnet = assess_testnet_evidence(storage_root)
    operational = assess_operational_evidence(storage_root, "testnet")
    return {
        "mode": "offline_simulation_no_exchange_orders",
        "passed": all(item.passed for item in drills),
        "drills": [item.to_dict() for item in drills],
        "current_live_gate": {
            "locked": not (testnet.eligible and operational.eligible),
            "testnet_evidence": asdict(testnet),
            "operational_evidence": asdict(operational),
        },
    }
