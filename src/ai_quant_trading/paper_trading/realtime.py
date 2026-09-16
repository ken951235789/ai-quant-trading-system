"""以 Binance Futures WebSocket 驅動 BTC 15m Transformer＋SAC/PPO 模擬交易。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from queue import Empty, Queue
from threading import Lock
import tempfile
from time import monotonic, sleep
from typing import Callable

import numpy as np
import pandas as pd

from ai_quant_trading.automation import AutomationConfig, AutomationPaths, run_automation
from ai_quant_trading.data_collection.binance_futures import (
    BinanceFuturesPublicClient,
    attach_derivatives_context,
)
from ai_quant_trading.data_collection.binance_stream import (
    BinanceKlineUpdate,
    BinanceRealtimeStream,
    BinanceStreamConfig,
)
from ai_quant_trading.data_collection.csv_storage import (
    canonical_ohlcv_path,
    save_ohlcv_csv,
)
from ai_quant_trading.data_collection.schemas import DERIVATIVES_CONTEXT_COLUMNS
from ai_quant_trading.data_collection.validators import validate_ohlcv_dataframe
from ai_quant_trading.features import build_feature_dataset, build_multitimeframe_frame
from ai_quant_trading.features.builder import infer_annualization_periods
from ai_quant_trading.market_clock import (
    BTC_MULTITIMEFRAME_INTERVALS,
    completed_bars_only,
    interval_duration,
)
from ai_quant_trading.paper_trading.config import PaperTradingConfig
from ai_quant_trading.paper_trading.rl_engine import (
    _policy_multitimeframe_intervals,
    run_rl_paper_cycle,
)
from ai_quant_trading.paper_trading.storage import (
    PREDICTION_COLUMNS,
    account_paths,
    read_account_csv,
)
from ai_quant_trading.reinforcement_learning import LoadedRLPolicy
from ai_quant_trading.risk import RiskConfig
from ai_quant_trading.trading import (
    CapitalManager,
    CapitalManagerConfig,
    MarketDataGuard,
    MarketDataGuardConfig,
    MarketEvent,
    MarketEventBus,
    MarketRegimeConfig,
    MarketRegimeDetector,
    ModelForecast,
    TradeIntent,
    TradingDecisionOrchestrator,
)
from ai_quant_trading.transformer import infer_transformer_context_frame


@dataclass(frozen=True, slots=True)
class TransformerTrendGateConfig:
    """Transformer 趨勢確認門檻；只限制增加風險，不阻擋減倉。"""

    minimum_probability: float = 0.45
    maximum_uncertainty: float = 0.62
    minimum_absolute_return_5: float = 0.0
    high_volatility_threshold: float = 0.03
    high_volatility_multiplier: float = 0.50
    target_forecast_volatility: float | None = 0.015
    confidence_reference: float = 0.40

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_probability <= 1.0:
            raise ValueError("Transformer 趨勢機率門檻必須介於 0 與 1")
        if not 0.0 <= self.maximum_uncertainty <= 1.0:
            raise ValueError("Transformer 不確定性上限必須介於 0 與 1")
        if self.minimum_absolute_return_5 < 0:
            raise ValueError("Transformer 最小預期報酬不可小於 0")
        if self.high_volatility_threshold <= 0:
            raise ValueError("Transformer 高波動門檻必須大於 0")
        if not 0 < self.high_volatility_multiplier <= 1:
            raise ValueError("高波動資金倍率必須介於 0 與 1")
        if self.target_forecast_volatility is not None and self.target_forecast_volatility <= 0:
            raise ValueError("資金管理目標波動必須大於 0 或設為 None")
        if not 0 < self.confidence_reference <= 1:
            raise ValueError("資金管理信心基準必須介於 0 與 1")

    def regime_config(self) -> MarketRegimeConfig:
        """轉成交易核心使用的市場狀態設定。"""
        return MarketRegimeConfig(
            minimum_probability=self.minimum_probability,
            maximum_uncertainty=self.maximum_uncertainty,
            minimum_absolute_return_5=self.minimum_absolute_return_5,
            high_volatility_threshold=self.high_volatility_threshold,
            high_volatility_multiplier=self.high_volatility_multiplier,
        )


def latest_completed_bar_open(
    now: datetime | pd.Timestamp,
    interval: str,
    *,
    settle_seconds: float = 0.0,
) -> pd.Timestamp:
    """計算已越過結算緩衝時間的最新完整 K 線開盤時間。"""
    if settle_seconds < 0:
        raise ValueError("K 線結算等待秒數不可小於 0")
    duration = interval_duration(interval)
    if duration is None:
        raise ValueError(f"無法計算 K 線週期：{interval}")
    current = pd.Timestamp(now)
    if current.tzinfo is None:
        current = current.tz_localize("UTC")
    else:
        current = current.tz_convert("UTC")
    settled = current - pd.Timedelta(seconds=settle_seconds)
    duration_seconds = int(duration.total_seconds())
    current_open_seconds = int(settled.timestamp()) // duration_seconds * duration_seconds
    return pd.Timestamp(current_open_seconds - duration_seconds, unit="s", tz="UTC")


@dataclass(frozen=True, slots=True)
class RealtimePaperConfig:
    """BTC 15m 即時模擬工作設定。"""

    account_id: str
    model_dir: Path
    raw_dir: Path
    paper_dir: Path
    transformer_checkpoint: Path
    symbol: str = "BTC/USDT"
    exchange: str = "binance_futures"
    decision_interval: str = "15m"
    history_bars: int = 512
    rolling_storage_bars: int = 2_000
    persist_rolling_data: bool = True
    settle_seconds: float = 2.0
    max_spread_bps: float = 5.0
    transformer_device: str = "cpu"
    stream_connect_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.exchange.lower() not in {"binance", "binance_futures"} or (
            self.symbol.upper() != "BTC/USDT"
        ):
            raise ValueError("即時管線目前只支援 Binance BTC/USDT 永續合約")
        if self.decision_interval.lower() not in BTC_MULTITIMEFRAME_INTERVALS:
            raise ValueError("即時模型決策週期不在 BTC 多週期資料契約內")
        if self.history_bars < 260:
            raise ValueError("即時多週期特徵至少需要 260 根暖機資料")
        if self.rolling_storage_bars < self.history_bars:
            raise ValueError("滾動資料上限不可小於模型暖機資料量")
        if self.settle_seconds < 0:
            raise ValueError("K 線收盤等待秒數不可小於 0")
        if self.max_spread_bps <= 0:
            raise ValueError("最大 Spread 必須大於 0")
        if self.stream_connect_timeout_seconds <= 0:
            raise ValueError("WebSocket 連線等待秒數必須大於 0")
        checkpoint = Path(self.transformer_checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"找不到 Transformer 模型：{checkpoint}")


class RealtimeStatusStore:
    """讓背景 Worker 與 Streamlit 透過 JSON 交換即時狀態。"""

    _REPLACE_ATTEMPTS = 12
    _INITIAL_RETRY_SECONDS = 0.02
    _MAX_RETRY_SECONDS = 0.25

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = Lock()
        self._payload: dict[str, object] = {
            "schema_version": 1,
            "transport": "binance_websocket",
            "state": "starting",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def update(self, **values: object) -> None:
        with self._lock:
            self._payload.update(values)
            self._payload["updated_at"] = datetime.now(timezone.utc).isoformat()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            serialized = json.dumps(
                self._payload,
                ensure_ascii=False,
                indent=2,
                default=self._json_default,
            )
            descriptor, temporary_name = tempfile.mkstemp(
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                text=True,
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(
                    descriptor,
                    mode="w",
                    encoding="utf-8",
                    newline="\n",
                ) as handle:
                    handle.write(serialized)
                    handle.flush()
                    os.fsync(handle.fileno())
                self._replace_with_retry(temporary)
            finally:
                temporary.unlink(missing_ok=True)

    def _replace_with_retry(self, temporary: Path) -> None:
        """處理 Windows 短暫鎖定狀態檔時的存取被拒。"""
        for attempt in range(self._REPLACE_ATTEMPTS):
            try:
                os.replace(temporary, self.path)
                return
            except PermissionError:
                if attempt + 1 >= self._REPLACE_ATTEMPTS:
                    raise
                delay = min(
                    self._INITIAL_RETRY_SECONDS * (2**attempt),
                    self._MAX_RETRY_SECONDS,
                )
                sleep(delay)

    def update_stream(self, values: dict[str, object]) -> None:
        self.update(stream=values)

    @staticmethod
    def _json_default(value: object) -> object:
        """將 Pandas／NumPy 純量轉成 JSON 可保存的 Python 型別。"""
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"無法序列化即時狀態型別：{type(value).__name__}")


class RealtimeMarketBuffer:
    """直接以 Binance 暖機與 WebSocket 更新固定長度的記憶體 K 線。"""

    def __init__(
        self,
        symbol: str,
        intervals: tuple[str, ...],
        *,
        max_bars: int,
        raw_dir: str | Path,
        client: BinanceFuturesPublicClient | None = None,
        persist_rolling_data: bool = True,
    ) -> None:
        if max_bars < 260:
            raise ValueError("即時市場記憶體至少需要保留 260 根 K 線")
        self.symbol = symbol
        self.intervals = intervals
        self.max_bars = max_bars
        self.raw_dir = Path(raw_dir)
        self.client = client or BinanceFuturesPublicClient()
        self.persist_rolling_data = persist_rolling_data
        self._frames: dict[str, pd.DataFrame] = {}
        self._lock = Lock()

    def _merge(self, interval: str, update: pd.DataFrame) -> None:
        if update.empty:
            return
        validate_ohlcv_dataframe(update)
        with self._lock:
            existing = self._frames.get(interval, pd.DataFrame())
            merged = pd.concat([existing, update], ignore_index=True, sort=False)
            merged["timestamp"] = pd.to_datetime(
                merged["timestamp"],
                utc=True,
                errors="coerce",
                format="mixed",
            )
            merged = merged.dropna(subset=["timestamp"])
            merged = merged.sort_values("timestamp").drop_duplicates(
                "timestamp",
                keep="last",
            )
            self._frames[interval] = merged.tail(self.max_bars).reset_index(drop=True)

    def append_closed_kline(self, event: BinanceKlineUpdate) -> None:
        """保留舊介面，內部仍統一轉成 MarketEvent。"""
        self.consume_market_event(event.to_market_event())

    def consume_market_event(self, event: MarketEvent) -> None:
        """歷史重播與 WebSocket 共用的市場資料 consumer。"""
        if event.kind != "kline_closed":
            return
        if event.symbol.upper() != self.symbol.upper():
            raise ValueError(f"市場事件標的不一致：{event.symbol} != {self.symbol}")
        if event.interval not in self.intervals:
            return
        self._merge(
            str(event.interval),
            pd.DataFrame.from_records([event.to_ohlcv_row()]),
        )

    def _fallback_local_frame(self, interval: str) -> pd.DataFrame:
        path = canonical_ohlcv_path(self.raw_dir, "binance_futures", self.symbol, interval)
        if not path.is_file():
            return pd.DataFrame()
        try:
            return pd.read_csv(path).tail(self.max_bars).reset_index(drop=True)
        except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError):
            return pd.DataFrame()

    def _attach_latest_derivatives_context(
        self,
        interval: str,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> None:
        """把 Funding、OI 與 Spread 因果對齊到指定週期 K 線。"""
        context = self.client.fetch_market_context(
            self.symbol,
            interval,
            start=start.isoformat(),
            end=end.isoformat(),
            include_advanced=False,
        )
        with self._lock:
            existing = self._frames.get(interval, pd.DataFrame()).copy()
        if existing.empty:
            return
        context_columns = [
            column
            for column in DERIVATIVES_CONTEXT_COLUMNS
            if column not in {"timestamp", "symbol", "exchange", "interval", "collected_at"}
        ]
        # 同一個常駐程序會反覆更新脈絡；先移除舊欄，避免 merge_asof
        # 產生 funding_rate_x / funding_rate_y 而讓最新 K 線失去資料。
        base = existing.drop(
            columns=[
                column
                for column in [*context_columns, "derivatives_context_available"]
                if column in existing
            ]
        )
        enriched = attach_derivatives_context(base, context)
        with self._lock:
            self._frames[interval] = enriched.tail(self.max_bars).reset_index(drop=True)

    def sync_from_binance(self) -> None:
        """補齊 K 線與永續合約脈絡，之後由 WebSocket 接手即時行情。"""
        now = pd.Timestamp.now(tz="UTC")
        for interval in self.intervals:
            duration = interval_duration(interval)
            if duration is None:
                raise ValueError(f"無法辨識 {interval} 的 K 線長度")
            with self._lock:
                existing = self._frames.get(interval, pd.DataFrame()).copy()
            if existing.empty:
                start = now - duration * (self.max_bars + 5)
            else:
                timestamps = pd.to_datetime(
                    existing["timestamp"],
                    utc=True,
                    errors="coerce",
                    format="mixed",
                ).dropna()
                start = (
                    timestamps.iloc[-1] - duration * 2
                    if not timestamps.empty
                    else now - duration * (self.max_bars + 5)
                )
            try:
                latest = self.client.fetch_ohlcv(
                    symbol=self.symbol,
                    interval=interval,
                    start=start.isoformat(),
                    end=now.isoformat(),
                    limit=1000,
                )
                latest = completed_bars_only(latest).reset_index(drop=True)
            except Exception:
                if not existing.empty:
                    # 暫時無法使用 REST 補漏時，WebSocket 記憶體仍可繼續供模型判斷。
                    continue
                latest = completed_bars_only(self._fallback_local_frame(interval)).reset_index(
                    drop=True
                )
                if latest.empty:
                    raise
            self._merge(interval, latest)
            # Binance 公開衍生品統計只保留最近 30 天；即時決策也只需要
            # 近期序列，避免每 15 分鐘重抓多年 Funding 歷史。
            context_start = now - pd.Timedelta(days=30)
            try:
                self._attach_latest_derivatives_context(
                    interval,
                    start=context_start,
                    end=now,
                )
            except Exception:
                with self._lock:
                    current = self._frames.get(interval, pd.DataFrame())
                    has_cached_context = "funding_rate" in current.columns
                if not has_cached_context:
                    raise

    def snapshot(self) -> dict[str, pd.DataFrame]:
        with self._lock:
            return {interval: frame.copy() for interval, frame in self._frames.items()}

    def row_counts(self) -> dict[str, int]:
        with self._lock:
            return {interval: len(frame) for interval, frame in self._frames.items()}

    def persist_rolling_snapshots(self) -> tuple[Path, ...]:
        """只保存固定長度的近期原始 K 線，模型推論不會再讀取這些 CSV。"""
        if not self.persist_rolling_data:
            return ()
        paths: list[Path] = []
        for interval, frame in self.snapshot().items():
            if frame.empty:
                continue
            paths.append(
                save_ohlcv_csv(
                    frame,
                    self.raw_dir,
                    "binance_futures",
                    self.symbol,
                    interval,
                    max_rows=self.max_bars,
                )
            )
        return tuple(paths)


def backfill_binance_timeframes(
    symbol: str,
    raw_dir: str | Path,
    intervals: tuple[str, ...],
    *,
    minimum_bars: int = 260,
    client: BinanceFuturesPublicClient | None = None,
) -> tuple[Path, ...]:
    """從每個本機檔案最後一根開始補漏，不重複下載完整歷史。"""
    futures = client or BinanceFuturesPublicClient()
    now = pd.Timestamp.now(tz="UTC")
    paths: list[Path] = []
    for interval in intervals:
        duration = interval_duration(interval)
        if duration is None:
            raise ValueError(f"無法計算 {interval} 的補漏範圍")
        path = canonical_ohlcv_path(raw_dir, "binance_futures", symbol, interval)
        start = now - duration * minimum_bars
        if path.is_file():
            try:
                stored = pd.read_csv(path, usecols=["timestamp"])
                timestamps = pd.to_datetime(
                    stored["timestamp"], utc=True, errors="coerce", format="mixed"
                ).dropna()
                if len(timestamps) >= minimum_bars:
                    start = timestamps.iloc[-1] - duration
            except (OSError, ValueError, pd.errors.ParserError):
                pass
        latest = futures.fetch_ohlcv(
            symbol=symbol,
            interval=interval,
            start=start.isoformat(),
            end=now.isoformat(),
            limit=1000,
        )
        validate_ohlcv_dataframe(latest)
        paths.append(save_ohlcv_csv(latest, raw_dir, "binance_futures", symbol, interval))
    return tuple(paths)


def build_realtime_multitimeframe_frame(
    raw_paths: tuple[Path, ...],
    *,
    decision_interval: str = "15m",
    history_bars: int = 512,
) -> pd.DataFrame:
    """只讀取推論所需尾端資料，重建同訓練版的多週期特徵。"""
    raw_frames = {
        path.name: pd.read_csv(path).tail(history_bars).reset_index(drop=True) for path in raw_paths
    }
    return build_realtime_multitimeframe_frame_from_frames(
        raw_frames,
        decision_interval=decision_interval,
        history_bars=history_bars,
    )


def build_realtime_multitimeframe_frame_from_frames(
    raw_frames: dict[str, pd.DataFrame],
    *,
    decision_interval: str = "15m",
    history_bars: int = 512,
) -> pd.DataFrame:
    """直接從記憶體 K 線建立多週期特徵，不依賴推論用 CSV。"""
    feature_frames: dict[str, pd.DataFrame] = {}
    for label, frame in raw_frames.items():
        raw = frame.tail(history_bars).reset_index(drop=True)
        raw = completed_bars_only(raw).reset_index(drop=True)
        if raw.empty:
            raise ValueError(f"{label} 沒有已收盤 K 線")
        interval = str(raw.iloc[-1].get("interval", "")).lower()
        if not interval:
            raise ValueError(f"{label} 缺少 interval")
        feature_frames[interval] = build_feature_dataset(
            raw,
            target_horizon=1,
            annualization_periods=infer_annualization_periods(raw),
            drop_na=False,
        )
    fused, _ = build_multitimeframe_frame(feature_frames, decision_interval)
    return fused.reset_index(drop=True)


def classify_transformer_trend(
    row: pd.Series,
    config: TransformerTrendGateConfig,
) -> tuple[str, dict[str, object]]:
    """把 Transformer 多任務輸出轉成可解釋的多／空／中性趨勢。"""
    forecast = ModelForecast.from_mapping(row)
    regime = MarketRegimeDetector(config.regime_config()).classify(forecast)
    trend = {
        "bull_trend": "偏多",
        "high_volatility_bull": "偏多",
        "bear_trend": "偏空",
        "high_volatility_bear": "偏空",
        "range": "中性",
        "uncertain": "中性",
        "unavailable": "不可用",
    }[regime.regime]
    return trend, {
        "transformer_trend": trend,
        "transformer_return_1": forecast.return_1,
        "transformer_return_5": forecast.return_5,
        "transformer_return_20": forecast.return_20,
        "transformer_volatility": forecast.volatility,
        "transformer_bull_probability": forecast.bull_probability,
        "transformer_bear_probability": forecast.bear_probability,
        "transformer_uncertainty": forecast.uncertainty,
        "market_regime": regime.regime,
        "regime_risk_multiplier": regime.risk_multiplier,
    }


def build_transformer_target_adjuster(
    gate_config: TransformerTrendGateConfig,
    stream_snapshot: Callable[[], dict[str, object]],
    *,
    max_spread_bps: float,
) -> Callable[[float, float, pd.Series], tuple[float, dict[str, object]]]:
    """建立 RL 目標的趨勢、市場健康與動態資金防護函式。"""
    orchestrator = TradingDecisionOrchestrator(
        regime_detector=MarketRegimeDetector(gate_config.regime_config()),
        capital_manager=CapitalManager(
            CapitalManagerConfig(
                target_forecast_volatility=gate_config.target_forecast_volatility,
                confidence_reference=gate_config.confidence_reference,
            )
        ),
    )

    def adjust(
        proposed_target: float,
        current_position: float,
        row: pd.Series,
    ) -> tuple[float, dict[str, object]]:
        trend, details = classify_transformer_trend(row, gate_config)
        forecast = ModelForecast.from_mapping(row)
        timestamp = pd.Timestamp(
            row.get("timestamp", pd.Timestamp.now(tz="UTC"))
        ).isoformat()
        symbol = str(row.get("symbol", "BTC/USDT"))
        intent = TradeIntent(
            decision_id=f"{symbol}-{timestamp}",
            timestamp=timestamp,
            symbol=symbol,
            proposed_target=float(np.clip(proposed_target, -1.0, 1.0)),
            current_position=float(np.clip(current_position, -1.0, 1.0)),
        )
        trace = orchestrator.evaluate(intent, forecast)
        details.update(trace.to_dict())
        stream = stream_snapshot()
        spread_value = stream.get("spread_bps")
        spread = float(spread_value) if spread_value is not None else float("inf")
        reducing_same_direction = (
            proposed_target * current_position >= 0
            and abs(proposed_target) <= abs(current_position) + 1e-9
        )

        approved = trace.capital.allocated_target
        governance_reason = str(details.get("governance_reasons", "")).strip()
        reason = governance_reason or "RL 與 Transformer 趨勢一致"
        if not np.isfinite(spread) or spread > max_spread_bps:
            if not reducing_same_direction:
                approved = current_position
            reason = "Spread 過大或尚未取得報價，不增加風險"
        elif trend == "不可用":
            if not reducing_same_direction:
                approved = current_position
            reason = "Transformer 不可用，不增加風險"
        elif trend == "中性":
            if not reducing_same_direction:
                approved = current_position
            reason = "Transformer 趨勢中性，不建立或加碼"
        elif current_position > 0 and trend == "偏空":
            approved = min(proposed_target, 0.0)
            reason = "Transformer 轉為偏空，退出多頭或確認空頭"
        elif current_position < 0 and trend == "偏多":
            approved = max(proposed_target, 0.0)
            reason = "Transformer 轉為偏多，退出空頭或確認多頭"
        elif abs(current_position) <= 1e-9:
            if proposed_target > 0 and trend != "偏多":
                approved = 0.0
                reason = "RL 多頭未獲 Transformer 確認"
            elif proposed_target < 0 and trend != "偏空":
                approved = 0.0
                reason = "RL 空頭未獲 Transformer 確認"
        elif proposed_target * current_position < 0:
            if proposed_target > 0 and trend != "偏多":
                approved = 0.0
                reason = "先平空，Transformer 尚未確認反手做多"
            elif proposed_target < 0 and trend != "偏空":
                approved = 0.0
                reason = "先平多，Transformer 尚未確認反手做空"

        details["decision_guard_reason"] = reason
        return float(approved), details

    return adjust


class RealtimePaperTradingSession:
    """常駐 WebSocket，僅在新決策週期收盤後執行一次完整模型流程。"""

    def __init__(
        self,
        policy: LoadedRLPolicy,
        config: RealtimePaperConfig,
        *,
        paper_config: PaperTradingConfig,
        risk_config: RiskConfig,
        entry_threshold: float,
        exit_threshold: float,
        trend_gate: TransformerTrendGateConfig | None = None,
        rest_client: BinanceFuturesPublicClient | None = None,
    ) -> None:
        self.policy = policy
        self.config = config
        self.paper_config = paper_config
        self.risk_config = risk_config
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.trend_gate = trend_gate or TransformerTrendGateConfig()
        self.rest_client = rest_client
        required = _policy_multitimeframe_intervals(policy)
        if required and set(required) != set(BTC_MULTITIMEFRAME_INTERVALS):
            raise ValueError("即時模式要求 RL 模型使用完整 BTC 多週期特徵")
        self.intervals = tuple(BTC_MULTITIMEFRAME_INTERVALS)
        self.market_buffer = RealtimeMarketBuffer(
            config.symbol,
            self.intervals,
            max_bars=config.rolling_storage_bars,
            raw_dir=config.raw_dir,
            client=rest_client,
            persist_rolling_data=config.persist_rolling_data,
        )
        paths = account_paths(config.paper_dir, config.account_id)
        self.automation_dir = paths.account_dir / "automation"
        self.status = RealtimeStatusStore(self.automation_dir / "realtime_status.json")
        self.data_guard = MarketDataGuard(
            MarketDataGuardConfig(
                required_intervals=self.intervals,
                minimum_contiguous_bars=20,
                settle_seconds=config.settle_seconds,
                maximum_spread_bps=config.max_spread_bps,
                require_spread=False,
            )
        )
        self._decisions: Queue[int] = Queue()
        self._last_decision_close_ms: int | None = None
        self._last_processed_timestamp: pd.Timestamp | None = None
        self.market_event_bus = MarketEventBus(
            consumers=[self._consume_market_event],
            strict_ordering=True,
        )
        self.stream = BinanceRealtimeStream(
            BinanceStreamConfig(symbol=config.symbol, intervals=self.intervals),
            on_closed_kline=lambda _event: None,
            on_status=self.status.update_stream,
            on_market_event=self._on_market_event,
        )

    def _on_closed_kline(self, event: BinanceKlineUpdate) -> None:
        """保留測試與舊呼叫介面，實際串流改走統一事件匯流排。"""
        self._on_market_event(event.to_market_event())

    def _on_market_event(self, event: MarketEvent) -> None:
        self.market_event_bus.publish(event)

    def _consume_market_event(self, event: MarketEvent) -> None:
        self.market_buffer.consume_market_event(event)
        if event.kind == "kline_closed" and event.interval == self.config.decision_interval:
            self._decisions.put(int(event.payload["close_time_ms"]))

    def _market_frame(self, *, sync_latest: bool = True) -> pd.DataFrame:
        if sync_latest:
            self.market_buffer.sync_from_binance()
        self.market_buffer.persist_rolling_snapshots()
        raw_frames = self.market_buffer.snapshot()
        missing = sorted(set(self.intervals) - set(raw_frames))
        if missing:
            raise ValueError(f"即時市場記憶體缺少週期：{', '.join(missing)}")
        stream = self.stream.snapshot()
        health = self.data_guard.assess(
            raw_frames,
            spread_bps=stream.get("spread_bps"),
        )
        self.status.update(data_health=health.to_dict())
        if not health.ready:
            raise ValueError("即時資料守門拒絕交易：" + "；".join(health.reasons))
        fused = build_realtime_multitimeframe_frame_from_frames(
            raw_frames,
            decision_interval=self.config.decision_interval,
            history_bars=self.config.history_bars,
        )
        return infer_transformer_context_frame(
            self.config.transformer_checkpoint,
            fused,
            device=self.config.transformer_device,
            mixed_precision=self.config.transformer_device != "cpu",
        )

    def run_decision(self, trigger: str, *, sync_latest: bool = True) -> object:
        """補漏、建立最新特徵、執行 Transformer 與 RL，再保存帳戶。"""
        started = datetime.now(timezone.utc)
        self.status.update(state="processing", trigger=trigger, last_error=None)
        try:
            frame = self._market_frame(sync_latest=sync_latest)
            adjuster = build_transformer_target_adjuster(
                self.trend_gate,
                self.stream.snapshot,
                max_spread_bps=self.config.max_spread_bps,
            )
            result = run_rl_paper_cycle(
                frame,
                self.policy,
                model_dir=self.config.model_dir,
                root_dir=self.config.paper_dir,
                account_id=self.config.account_id,
                exchange=self.config.exchange,
                symbol=self.config.symbol,
                interval=self.config.decision_interval,
                config=self.paper_config,
                risk_config=self.risk_config,
                entry_threshold=self.entry_threshold,
                exit_threshold=self.exit_threshold,
                target_adjuster=adjuster,
            )
            processed = pd.to_datetime(
                result.state.last_processed_timestamp,
                utc=True,
                errors="coerce",
            )
            if pd.notna(processed):
                self._last_processed_timestamp = pd.Timestamp(processed)
            predictions = read_account_csv(result.paths.predictions_csv, PREDICTION_COLUMNS)
            latest = predictions.iloc[-1].to_dict() if not predictions.empty else {}
            elapsed = (datetime.now(timezone.utc) - started).total_seconds()
            self.status.update(
                state="waiting_decision_close",
                last_decision_at=datetime.now(timezone.utc).isoformat(),
                last_market_timestamp=result.state.last_processed_timestamp,
                decision_seconds=elapsed,
                latest_decision=latest,
                account_id=result.state.account_id,
                model_dir=str(self.config.model_dir),
                transformer_checkpoint=str(self.config.transformer_checkpoint),
                market_source="binance_memory_buffer",
                buffer_rows=self.market_buffer.row_counts(),
                rolling_storage_bars=self.config.rolling_storage_bars,
                market_event_metrics=self.market_event_bus.metrics.to_dict(),
            )
            return result
        except Exception as exc:
            self.status.update(state="error", last_error=f"{type(exc).__name__}: {exc}")
            raise

    def _wait_for_stream_ready(self, paths: AutomationPaths) -> None:
        """等待 WebSocket 與第一筆 Bid/Ask，避免啟動時和 REST 同時握手。"""
        deadline = monotonic() + self.config.stream_connect_timeout_seconds
        while not paths.stop_file.exists():
            snapshot = self.stream.snapshot()
            if snapshot.get("connected") and snapshot.get("spread_bps") is not None:
                return
            if snapshot.get("state") == "failed":
                raise RuntimeError(str(snapshot.get("last_error") or "WebSocket 啟動失敗"))
            if monotonic() >= deadline:
                raise TimeoutError("WebSocket 連線或 Bid/Ask 行情等待逾時")
            sleep(0.1)
        raise RuntimeError("收到安全停止請求")

    def _run_initial_decision(self, paths: AutomationPaths) -> object:
        """依序完成 REST 暖機、WebSocket 連線與首輪模型決策。"""
        self.status.update(state="warming_market", last_error=None)
        self.market_buffer.sync_from_binance()
        self.status.update(state="connecting")
        self.stream.start()
        self._wait_for_stream_ready(paths)
        return self.run_decision("startup", sync_latest=False)

    def _wait_for_decision(self, paths: AutomationPaths) -> object:
        while not paths.stop_file.exists():
            try:
                close_time_ms = self._decisions.get(timeout=1.0)
            except Empty:
                snapshot = self.stream.snapshot()
                if snapshot.get("state") == "failed":
                    raise RuntimeError(str(snapshot.get("last_error") or "WebSocket 啟動失敗"))
                latest_completed = latest_completed_bar_open(
                    datetime.now(timezone.utc),
                    self.config.decision_interval,
                    settle_seconds=self.config.settle_seconds,
                )
                if (
                    self._last_processed_timestamp is not None
                    and latest_completed > self._last_processed_timestamp
                ):
                    # 部分網路環境只持續收到 BookTicker；用 REST 補漏確保收盤決策不中斷。
                    return self.run_decision("rest_close_fallback", sync_latest=True)
                continue
            if close_time_ms == 0:
                return self.run_decision("startup")
            if close_time_ms == self._last_decision_close_ms:
                continue
            self._last_decision_close_ms = close_time_ms
            if self.config.settle_seconds > 0 and paths.stop_file.exists() is False:
                paths.stop_file.parent.mkdir(parents=True, exist_ok=True)
                # 讓同一時刻收盤的慢週期訊息先落盤，再由 REST 做最後補漏。
                sleep(self.config.settle_seconds)
            return self.run_decision("websocket_decision_close")
        return "收到安全停止請求"

    def run(
        self,
        automation_paths: AutomationPaths,
        *,
        max_consecutive_errors: int = 5,
    ) -> object:
        """啟動串流並交由既有 Automation runner 管理鎖與熔斷。"""
        initial_decision_pending = True

        def cycle() -> object:
            nonlocal initial_decision_pending
            if initial_decision_pending:
                result = self._run_initial_decision(automation_paths)
                initial_decision_pending = False
                return result
            return self._wait_for_decision(automation_paths)

        try:
            return run_automation(
                cycle,
                automation_paths,
                AutomationConfig(
                    poll_seconds=0,
                    max_cycles=0,
                    max_consecutive_errors=max_consecutive_errors,
                ),
            )
        finally:
            self.stream.stop()
            self.status.update(state="stopped")
