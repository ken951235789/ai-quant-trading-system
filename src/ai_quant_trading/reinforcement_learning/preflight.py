"""在耗時訓練前檢查資料契約、時間切分、成本與風控設定。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ai_quant_trading.market_clock import interval_duration
from ai_quant_trading.reinforcement_learning.config import PortfolioEnvConfig


FrameCollection = pd.DataFrame | Mapping[str, pd.DataFrame]
_REQUIRED_MARKET_COLUMNS = frozenset(
    {"timestamp", "open", "high", "low", "close", "volume"}
)
_LEAKAGE_PATTERN = re.compile(
    r"(^target($|_)|_target$|future_|^label$|next_(close|open|high|low|return))",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class PreflightFinding:
    """單一訓練前檢查結果。"""

    severity: str
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class PretrainingReadinessReport:
    """硬錯誤會阻擋訓練，警告則要求研究者確認後再做正式長訓練。"""

    eligible: bool
    formal_research_ready: bool
    total_rows: int
    observed_days: float
    findings: tuple[PreflightFinding, ...]

    @property
    def errors(self) -> tuple[str, ...]:
        return tuple(item.message for item in self.findings if item.severity == "error")

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(item.message for item in self.findings if item.severity == "warning")

    def to_dict(self) -> dict[str, object]:
        return {
            "eligible": self.eligible,
            "formal_research_ready": self.formal_research_ready,
            "total_rows": self.total_rows,
            "observed_days": self.observed_days,
            "findings": [asdict(item) for item in self.findings],
        }


def _collection(frame: FrameCollection) -> dict[str, pd.DataFrame]:
    if isinstance(frame, pd.DataFrame):
        return {"default": frame}
    return {str(name): value for name, value in frame.items()}


def _timestamps(frame: pd.DataFrame) -> pd.Series:
    if "timestamp" not in frame:
        return pd.Series(dtype="datetime64[ns, UTC]")
    return pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")


def assess_pretraining_readiness(
    *,
    metadata: Mapping[str, Any],
    feature_columns: list[str] | tuple[str, ...],
    env_config: PortfolioEnvConfig,
    train: FrameCollection,
    validation: FrameCollection,
    test: FrameCollection,
    minimum_formal_rows: int = 50_000,
    minimum_formal_days: float = 730.0,
) -> PretrainingReadinessReport:
    """檢查不可逆的資料錯誤，並標示正式研究仍不足的證據。"""
    if minimum_formal_rows <= 0 or minimum_formal_days <= 0:
        raise ValueError("正式研究資料門檻必須大於 0")
    findings: list[PreflightFinding] = []
    features = list(feature_columns)
    if not features:
        findings.append(PreflightFinding("error", "features.empty", "沒有模型特徵欄位"))
    duplicates = sorted({name for name in features if features.count(name) > 1})
    if duplicates:
        findings.append(
            PreflightFinding("error", "features.duplicate", f"模型特徵重複：{duplicates[:8]}")
        )
    leaked = [name for name in features if _LEAKAGE_PATTERN.search(name)]
    if leaked:
        findings.append(
            PreflightFinding(
                "error",
                "features.lookahead",
                f"模型特徵疑似包含未來答案：{leaked[:8]}",
            )
        )

    split_frames = {
        "train": _collection(train),
        "validation": _collection(validation),
        "test": _collection(test),
    }
    market_names = set(split_frames["train"])
    if not market_names or any(set(values) != market_names for values in split_frames.values()):
        findings.append(
            PreflightFinding("error", "split.markets", "訓練、驗證、測試的市場集合不一致")
        )

    all_timestamps: list[pd.Series] = []
    total_rows = 0
    for split_name, markets in split_frames.items():
        for market_name, frame in markets.items():
            label = f"{market_name}/{split_name}"
            total_rows += len(frame)
            missing_market = sorted(_REQUIRED_MARKET_COLUMNS.difference(frame.columns))
            missing_features = sorted(set(features).difference(frame.columns))
            if frame.empty:
                findings.append(PreflightFinding("error", "data.empty", f"{label} 沒有資料"))
                continue
            if missing_market:
                findings.append(
                    PreflightFinding(
                        "error", "data.columns", f"{label} 缺少行情欄位：{missing_market}"
                    )
                )
            if missing_features:
                findings.append(
                    PreflightFinding(
                        "error", "features.missing", f"{label} 缺少模型特徵：{missing_features[:8]}"
                    )
                )
            timestamps = _timestamps(frame)
            if timestamps.isna().any():
                findings.append(PreflightFinding("error", "time.invalid", f"{label} 含無效時間"))
            elif timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
                findings.append(
                    PreflightFinding("error", "time.order", f"{label} 時間重複或未遞增")
                )
            else:
                all_timestamps.append(timestamps)
            if not missing_features:
                numeric = frame[features].apply(pd.to_numeric, errors="coerce")
                values = numeric.to_numpy(dtype=float)
                if numeric.isna().any().any() or not np.isfinite(values).all():
                    findings.append(
                        PreflightFinding("error", "features.nonfinite", f"{label} 特徵含 NaN 或無限值")
                    )
            if not missing_market:
                prices = frame[["open", "high", "low", "close"]].apply(
                    pd.to_numeric, errors="coerce"
                )
                invalid_price = (
                    prices.isna().any().any()
                    or (prices <= 0).any().any()
                    or (prices["high"] < prices[["open", "close", "low"]].max(axis=1)).any()
                    or (prices["low"] > prices[["open", "close", "high"]].min(axis=1)).any()
                )
                if invalid_price:
                    findings.append(
                        PreflightFinding("error", "price.invalid", f"{label} OHLC 價格關係不合法")
                    )

    for market_name in market_names:
        ranges: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
        for split_name in ("train", "validation", "test"):
            frame = split_frames[split_name].get(market_name)
            timestamps = _timestamps(frame) if frame is not None else pd.Series(dtype="datetime64[ns, UTC]")
            if not timestamps.empty and not timestamps.isna().any():
                ranges.append((split_name, timestamps.iloc[0], timestamps.iloc[-1]))
        for previous, current in zip(ranges, ranges[1:], strict=False):
            if previous[2] >= current[1]:
                findings.append(
                    PreflightFinding(
                        "error",
                        "split.overlap",
                        f"{market_name} 的 {previous[0]} 與 {current[0]} 時間重疊",
                    )
                )

    observed_days = 0.0
    if all_timestamps:
        earliest = min(values.iloc[0] for values in all_timestamps)
        latest = max(values.iloc[-1] for values in all_timestamps)
        observed_days = float((latest - earliest).total_seconds() / 86_400)
    if total_rows < minimum_formal_rows:
        findings.append(
            PreflightFinding(
                "warning",
                "coverage.rows",
                f"總資料只有 {total_rows:,} 根，正式研究建議至少 {minimum_formal_rows:,} 根",
            )
        )
    if observed_days < minimum_formal_days:
        findings.append(
            PreflightFinding(
                "warning",
                "coverage.days",
                f"市場跨度只有 {observed_days:.1f} 天，建議至少 {minimum_formal_days:.0f} 天",
            )
        )

    source = dict(metadata.get("source", {}))
    exchange = str(source.get("exchange", "")).lower()
    symbol = str(source.get("symbol", "")).upper().replace("/", "")
    interval = str(source.get("interval", "")).lower()
    if exchange == "binance_futures" and env_config.execution_mode != "perpetual":
        findings.append(
            PreflightFinding("error", "market.execution", "永續合約資料不可使用 Spot 帳務訓練")
        )
    if exchange == "binance_futures" and not env_config.allow_short:
        findings.append(
            PreflightFinding("warning", "market.short", "永續合約環境尚未啟用空頭動作")
        )
    if symbol and symbol != "BTCUSDT":
        findings.append(
            PreflightFinding("warning", "market.symbol", f"目前研究目標不是 BTC/USDT：{symbol}")
        )
    if interval and interval_duration(interval) is None:
        findings.append(PreflightFinding("error", "market.interval", f"無法辨識決策週期：{interval}"))

    if env_config.fee_rate <= 0 or env_config.slippage_rate <= 0:
        findings.append(
            PreflightFinding("error", "cost.zero", "手續費與滑價必須使用非零保守估計")
        )
    if env_config.execution_mode == "perpetual" and env_config.spread_rate <= 0:
        findings.append(
            PreflightFinding("warning", "cost.spread", "永續合約尚未設定 Bid-Ask Spread 成本")
        )
    if env_config.max_leverage > 3:
        findings.append(
            PreflightFinding("warning", "risk.leverage", "研究階段最大槓桿高於 3x")
        )
    if env_config.risk_per_trade > 0.01:
        findings.append(
            PreflightFinding("warning", "risk.trade", "每筆風險高於 1%，不適合初期小額驗證")
        )
    if env_config.max_drawdown_limit > 0.15:
        findings.append(
            PreflightFinding("warning", "risk.drawdown", "硬性最大回撤高於 15%")
        )

    ai_context = dict(metadata.get("ai_context", {}))
    aggregate = dict(dict(ai_context.get("coverage", {})).get("aggregate", {}))
    if bool(ai_context.get("transformer_enabled", False)):
        coverage = float(aggregate.get("transformer", 0.0) or 0.0)
        if coverage < 0.50:
            findings.append(
                PreflightFinding(
                    "warning" if total_rows < minimum_formal_rows else "error",
                    "ai.transformer",
                    f"Transformer 歷史覆蓋只有 {coverage:.2%}",
                )
            )
    if bool(ai_context.get("finbert_enabled", False)):
        coverage = float(aggregate.get("finbert", 0.0) or 0.0)
        finbert_config = dict(ai_context.get("finbert_config", {}))
        minimum_coverage = float(
            finbert_config.get("minimum_training_coverage", 0.30) or 0.30
        )
        if coverage < minimum_coverage:
            findings.append(
                PreflightFinding(
                    "warning" if total_rows < minimum_formal_rows else "error",
                    "ai.finbert",
                    f"FinBERT 歷史覆蓋只有 {coverage:.2%}，"
                    f"啟用時至少需要 {minimum_coverage:.0%}；"
                    "覆蓋不足時請關閉 FinBERT 特徵",
                )
            )

    errors = [item for item in findings if item.severity == "error"]
    warnings = [item for item in findings if item.severity == "warning"]
    return PretrainingReadinessReport(
        eligible=not errors,
        formal_research_ready=not errors and not warnings,
        total_rows=total_rows,
        observed_days=observed_days if math.isfinite(observed_days) else 0.0,
        findings=tuple(findings),
    )


def ensure_pretraining_integrity(report: PretrainingReadinessReport) -> None:
    """只阻擋資料洩漏、時間重疊等硬錯誤；資料量警告保留給 smoke 測試。"""
    if not report.eligible:
        raise ValueError("訓練前完整性檢查失敗：" + "；".join(report.errors))
