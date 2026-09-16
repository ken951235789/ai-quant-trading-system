"""即時多週期 K 線的資料品質守門與 fail-closed 判定。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math

import pandas as pd

from ai_quant_trading.data_collection.validators import validate_ohlcv_dataframe
from ai_quant_trading.market_clock import BTC_MULTITIMEFRAME_INTERVALS, interval_duration


class MarketDataRejected(RuntimeError):
    """市場資料不足以安全產生新交易時拋出的例外。"""


@dataclass(frozen=True, slots=True)
class MarketDataGuardConfig:
    """多週期資料完整性、時效與異常價格門檻。"""

    required_intervals: tuple[str, ...] = BTC_MULTITIMEFRAME_INTERVALS
    minimum_contiguous_bars: int = 20
    freshness_multiple: float = 2.5
    settle_seconds: float = 2.0
    maximum_single_bar_return: float = 0.35
    maximum_spread_bps: float = 5.0
    require_spread: bool = False

    def __post_init__(self) -> None:
        normalized = tuple(dict.fromkeys(value.strip().lower() for value in self.required_intervals))
        if not normalized:
            raise ValueError("required_intervals 不可為空")
        if self.minimum_contiguous_bars < 2:
            raise ValueError("minimum_contiguous_bars 至少為 2")
        if self.freshness_multiple <= 0 or self.settle_seconds < 0:
            raise ValueError("資料時效倍率必須大於 0，結算秒數不可小於 0")
        if not 0 < self.maximum_single_bar_return < 1:
            raise ValueError("maximum_single_bar_return 必須介於 0 與 1")
        if self.maximum_spread_bps <= 0:
            raise ValueError("maximum_spread_bps 必須大於 0")
        object.__setattr__(self, "required_intervals", normalized)


@dataclass(frozen=True, slots=True)
class MarketDataHealth:
    """資料守門的檢查結果，包含 UI 與稽核需要的原因。"""

    ready: bool
    checked_at: str
    row_counts: dict[str, int]
    latest_closed_by_interval: dict[str, str]
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "checked_at": self.checked_at,
            "row_counts": self.row_counts,
            "latest_closed_by_interval": self.latest_closed_by_interval,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
        }


class MarketDataGuard:
    """檢查缺列、重複、斷 K、未收盤、過期、跳價與 Spread。"""

    def __init__(self, config: MarketDataGuardConfig | None = None) -> None:
        self.config = config or MarketDataGuardConfig()

    def assess(
        self,
        frames: dict[str, pd.DataFrame],
        *,
        now: datetime | pd.Timestamp | None = None,
        spread_bps: float | None = None,
    ) -> MarketDataHealth:
        current = pd.Timestamp(now or datetime.now(timezone.utc))
        current = current.tz_localize("UTC") if current.tzinfo is None else current.tz_convert("UTC")
        settled_at = current - pd.Timedelta(seconds=self.config.settle_seconds)
        reasons: list[str] = []
        warnings: list[str] = []
        counts: dict[str, int] = {}
        latest: dict[str, str] = {}

        normalized_frames = {str(key).lower(): value for key, value in frames.items()}
        for interval in self.config.required_intervals:
            frame = normalized_frames.get(interval)
            if frame is None or frame.empty:
                reasons.append(f"{interval} 缺少 K 線資料")
                counts[interval] = 0
                continue
            duration = interval_duration(interval)
            if duration is None:
                reasons.append(f"無法辨識 K 線週期 {interval}")
                continue
            candidate = frame.copy()
            counts[interval] = len(candidate)
            try:
                validate_ohlcv_dataframe(candidate)
            except (KeyError, TypeError, ValueError) as exc:
                reasons.append(f"{interval} OHLCV 驗證失敗：{exc}")
                continue
            timestamps = pd.to_datetime(candidate["timestamp"], utc=True, errors="coerce")
            candidate = candidate.assign(timestamp=timestamps).sort_values("timestamp")
            if len(candidate) < self.config.minimum_contiguous_bars:
                reasons.append(
                    f"{interval} 只有 {len(candidate)} 根，至少需要 "
                    f"{self.config.minimum_contiguous_bars} 根"
                )
                continue
            tail = candidate.tail(self.config.minimum_contiguous_bars)
            diffs = tail["timestamp"].diff().dropna()
            if (diffs != duration).any():
                reasons.append(f"{interval} 最近 K 線有缺口或時間錯位")
            last_open = pd.Timestamp(tail.iloc[-1]["timestamp"])
            last_close = last_open + duration
            latest[interval] = last_close.isoformat()
            if last_close > settled_at:
                reasons.append(f"{interval} 最新 K 線尚未完成收盤")
            elif settled_at - last_close > duration * self.config.freshness_multiple:
                reasons.append(f"{interval} 最新 K 線已過期")
            returns = pd.to_numeric(tail["close"], errors="coerce").pct_change().abs()
            if (returns > self.config.maximum_single_bar_return).any():
                reasons.append(f"{interval} 發現超過門檻的單根異常跳價")

        if spread_bps is None or not math.isfinite(float(spread_bps)):
            message = "尚未取得有效 Bid-Ask Spread"
            (reasons if self.config.require_spread else warnings).append(message)
        elif float(spread_bps) > self.config.maximum_spread_bps:
            reasons.append(
                f"Spread {float(spread_bps):.2f} bps 超過 "
                f"{self.config.maximum_spread_bps:.2f} bps"
            )
        return MarketDataHealth(
            ready=not reasons,
            checked_at=current.isoformat(),
            row_counts=counts,
            latest_closed_by_interval=latest,
            reasons=tuple(reasons),
            warnings=tuple(warnings),
        )

    def ensure_healthy(
        self,
        frames: dict[str, pd.DataFrame],
        *,
        now: datetime | pd.Timestamp | None = None,
        spread_bps: float | None = None,
    ) -> MarketDataHealth:
        health = self.assess(frames, now=now, spread_bps=spread_bps)
        if not health.ready:
            raise MarketDataRejected("；".join(health.reasons))
        return health
