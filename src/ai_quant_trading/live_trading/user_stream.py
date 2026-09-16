"""Binance USD-M 私有 User Data Stream 常駐消費器。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import random
from threading import Event
from time import monotonic
from typing import Any, Callable
from urllib.parse import urlsplit

from ai_quant_trading.data_collection.http_client import create_secure_ssl_context
from ai_quant_trading.database.repository import PostgresTradingRepository
from ai_quant_trading.live_trading.binance_futures_client import (
    BinanceFuturesPrivateClient,
)


USER_STREAM_BASE_URLS = {
    "testnet": "wss://fstream.binancefuture.com/ws",
    "demo": "wss://fstream.binancefuture.com/ws",
    "live": "wss://fstream.binance.com/ws",
}


@dataclass(frozen=True, slots=True)
class UserStreamConfig:
    """私有事件流重連與健康檢查設定。"""

    environment: str
    stream_name: str = "usd_m_user_data"
    keepalive_seconds: float = 30 * 60
    reconnect_initial_seconds: float = 1.0
    reconnect_max_seconds: float = 30.0
    stale_after_seconds: float = 120.0
    replay_batch_size: int = 1_000
    replay_max_batches: int = 20

    def __post_init__(self) -> None:
        if self.environment not in USER_STREAM_BASE_URLS:
            raise ValueError("私有事件流環境不合法")
        if min(
            self.keepalive_seconds,
            self.reconnect_initial_seconds,
            self.reconnect_max_seconds,
            self.stale_after_seconds,
        ) <= 0:
            raise ValueError("私有事件流秒數設定必須大於 0")
        if self.replay_batch_size <= 0 or self.replay_max_batches <= 0:
            raise ValueError("事件重播批次設定必須大於 0")


class BinanceFuturesUserStream:
    """事件先落 PostgreSQL，再冪等投影訂單與成交。"""

    def __init__(
        self,
        client: BinanceFuturesPrivateClient,
        repository: PostgresTradingRepository,
        config: UserStreamConfig,
        *,
        reconcile: Callable[[], Any] | None = None,
        websocket_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.client = client
        self.repository = repository
        self.config = config
        self.reconcile = reconcile
        self.websocket_factory = websocket_factory
        self.stop_event = Event()
        self.reconnect_count = 0
        self.gap_count = 0
        self.last_event_key: str | None = None
        self.last_event_time: datetime | None = None
        self.connected_at: datetime | None = None
        self.last_message_monotonic: float | None = None
        self.connection_ready = False
        self.projection_ready = False
        self._sync_error: str | None = None
        self._socket: Any = None

    def stop(self) -> None:
        """要求連線安全停止。"""
        self.stop_event.set()
        if self._socket is not None:
            self._socket.close()

    def _checkpoint(self, state: str, *, error: str | None = None) -> None:
        now = datetime.now(timezone.utc)
        self.repository.update_stream_checkpoint(
            self.config.environment,
            self.config.stream_name,
            state=state,
            heartbeat_at=now,
            last_event_key=self.last_event_key,
            last_event_time=self.last_event_time,
            connected_at=self.connected_at,
            reconnect_count=self.reconnect_count,
            gap_count=self.gap_count,
            details={
                "last_error": error,
                "schema_version": 2,
                "socket_connected": self.connected_at is not None,
                "projection_ready": self.projection_ready,
            },
        )

    def _mark_degraded(self, error: Exception | str, socket: Any | None = None) -> None:
        """保留原始錯誤並確保關閉連線；資料庫故障時 checkpoint 採 best effort。"""
        self.connection_ready = False
        self.projection_ready = False
        self._sync_error = (
            f"{type(error).__name__}: {error}" if isinstance(error, Exception) else str(error)
        )[:500]
        try:
            self._checkpoint("degraded", error=self._sync_error)
        except Exception:
            pass
        if socket is not None:
            socket.close()

    def _drain_unprocessed_events(self) -> int:
        """分批清空未投影事件；達到上限時拒絕宣告同步完成。"""
        total = 0
        for _ in range(self.config.replay_max_batches):
            replayed = self.repository.replay_unprocessed_exchange_events(
                limit=self.config.replay_batch_size
            )
            total += replayed
            if replayed < self.config.replay_batch_size:
                return total
        raise RuntimeError(
            "未投影交易所事件超過單次恢復上限，保持 degraded 等待人工檢查"
        )

    @staticmethod
    def _event_time(payload: dict[str, Any]) -> datetime | None:
        value = payload.get("E")
        if value in {None, ""}:
            return None
        return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc)

    def _on_open(self, _socket: Any) -> None:
        self.connected_at = datetime.now(timezone.utc)
        self.last_message_monotonic = monotonic()
        self.connection_ready = False
        self.projection_ready = False
        self._sync_error = None
        self._checkpoint("syncing")
        try:
            self._drain_unprocessed_events()
            if self.reconcile is not None:
                result = self.reconcile()
                if getattr(result, "consistent", True) is False:
                    raise RuntimeError(getattr(result, "reason", "重連對帳失敗"))
        except Exception as exc:
            self._mark_degraded(exc, _socket)
            return
        self.projection_ready = True
        self.connection_ready = True
        self._checkpoint("connected")

    def _on_message(self, socket: Any, message: str | bytes) -> None:
        if isinstance(message, bytes):
            message = message.decode("utf-8")
        payload = json.loads(message)
        if not isinstance(payload, dict):
            return
        self.last_message_monotonic = monotonic()
        event_time = self._event_time(payload)
        event_time_regressed = (
            event_time is not None
            and self.last_event_time is not None
            and event_time < self.last_event_time
        )
        if event_time_regressed:
            self.gap_count += 1
        _inserted, event_key = self.repository.append_exchange_event(
            self.config.environment,
            self.config.stream_name,
            payload,
        )
        self.last_event_key = event_key
        if event_time is not None:
            self.last_event_time = max(event_time, self.last_event_time or event_time)
        try:
            # 即使事件已存在仍執行冪等投影，讓先前失敗的事件可以靠重送恢復。
            self.repository.project_exchange_event(event_key)
        except Exception as exc:
            self._mark_degraded(exc, socket)
            raise
        if event_time_regressed and self.reconcile is not None:
            result = self.reconcile()
            if getattr(result, "consistent", True) is False:
                socket.close()
                raise RuntimeError(getattr(result, "reason", "事件倒序後對帳失敗"))
        self._checkpoint("connected")
        if payload.get("e") == "listenKeyExpired":
            socket.close()

    def _on_error(self, _socket: Any, error: Any) -> None:
        self._mark_degraded(error)

    def _on_pong(self, _socket: Any, _payload: Any) -> None:
        self.last_message_monotonic = monotonic()
        if self.connection_ready and self.projection_ready:
            self._checkpoint("connected")
        else:
            self._checkpoint("degraded", error=self._sync_error or "事件投影尚未同步完成")

    def _keepalive(self, connection_closed: Event) -> None:
        while not connection_closed.wait(self.config.keepalive_seconds):
            if self.stop_event.is_set():
                return
            try:
                self.client.keepalive_listen_key()
            except Exception as exc:
                self._mark_degraded(exc, self._socket)
                return

    def run(self) -> None:
        """持續連線；每次中斷建立新 listenKey 並重新做 REST 對帳。"""
        import threading
        import websocket

        factory = self.websocket_factory or websocket.WebSocketApp
        delay = self.config.reconnect_initial_seconds
        while not self.stop_event.is_set():
            self.connection_ready = False
            self.projection_ready = False
            self._checkpoint("connecting")
            listen_key = ""
            connection_closed = Event()
            try:
                listen_key = self.client.create_listen_key()
                base_url = USER_STREAM_BASE_URLS[self.config.environment]
                host = urlsplit(base_url).hostname or ""
                self._socket = factory(
                    f"{base_url}/{listen_key}",
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_pong=self._on_pong,
                )
                keeper = threading.Thread(
                    target=self._keepalive,
                    args=(connection_closed,),
                    daemon=True,
                )
                keeper.start()
                self._socket.run_forever(
                    sslopt={"context": create_secure_ssl_context()},
                    ping_interval=30,
                    ping_timeout=10,
                    http_no_proxy=[host],
                )
            except Exception as exc:
                self._checkpoint("degraded", error=f"{type(exc).__name__}: {exc}"[:500])
            finally:
                connection_closed.set()
                if listen_key:
                    try:
                        self.client.close_listen_key()
                    except Exception:
                        pass
                self._socket = None
            if self.stop_event.is_set():
                break
            if self.connection_ready:
                # 成功連線並完成對帳後，下次斷線從短延遲重新開始。
                delay = self.config.reconnect_initial_seconds
            self.reconnect_count += 1
            self._checkpoint("reconnecting")
            self.stop_event.wait(delay + random.uniform(0, min(delay * 0.2, 1.0)))
            delay = min(delay * 2, self.config.reconnect_max_seconds)
        self._checkpoint("stopped")
