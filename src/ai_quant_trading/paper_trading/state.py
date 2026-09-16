"""可跨次啟動延續的模擬帳戶狀態。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from ai_quant_trading.paper_trading.config import PaperTradingConfig
from ai_quant_trading.risk import RiskConfig


def utc_now_text() -> str:
    """回傳可寫入 JSON 的 UTC 時間。"""
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class PaperAccountState:
    """模擬帳戶的現金、持倉、待執行訊號與風控狀態。"""

    account_id: str
    model_dir: str
    exchange: str
    symbol: str
    interval: str
    config: PaperTradingConfig
    risk_config: RiskConfig
    model_kind: str = "rl"
    expert_kind: str = "general"
    rl_entry_threshold: float = 0.10
    rl_exit_threshold: float = 0.02
    created_at: str = field(default_factory=utc_now_text)
    updated_at: str = field(default_factory=utc_now_text)
    cash: float = 0.0
    quantity: float = 0.0
    entry_time: str | None = None
    entry_price: float | None = None
    entry_fee: float = 0.0
    entry_cost: float = 0.0
    holding_periods: int = 0
    stop_loss: float | None = None
    take_profit: float | None = None
    risk_budget: float | None = None
    position_leverage: int = 1
    pending_signal: int = 0
    pending_target_fraction: float | None = None
    pending_leverage: int | None = None
    pending_time: str | None = None
    pending_atr: float | None = None
    pending_reason: str = "hold"
    last_processed_timestamp: str | None = None
    equity_peak: float = 0.0
    risk_halted: bool = False
    short_carry_paid: float = 0.0
    position_carry_cost: float = 0.0
    realized_pnl: float = 0.0
    day_start_equity: float = 0.0
    current_day: str | None = None
    consecutive_losses: int = 0
    funding_paid: float = 0.0
    last_leverage_reason: str = "尚未產生槓桿決策"

    @classmethod
    def create(
        cls,
        *,
        account_id: str,
        model_dir: str,
        exchange: str,
        symbol: str,
        interval: str,
        config: PaperTradingConfig,
        risk_config: RiskConfig,
        model_kind: str = "rl",
        expert_kind: str = "general",
        rl_entry_threshold: float = 0.10,
        rl_exit_threshold: float = 0.02,
    ) -> "PaperAccountState":
        """建立尚未處理任何 K 線的新帳戶。"""
        return cls(
            account_id=account_id,
            model_dir=model_dir,
            exchange=exchange,
            symbol=symbol,
            interval=interval,
            config=config,
            risk_config=risk_config,
            model_kind=model_kind,
            expert_kind=expert_kind,
            rl_entry_threshold=rl_entry_threshold,
            rl_exit_threshold=rl_exit_threshold,
            cash=config.initial_capital,
            equity_peak=config.initial_capital,
            day_start_equity=config.initial_capital,
            position_leverage=(
                risk_config.dynamic_leverage.min_leverage
                if risk_config.dynamic_leverage.enabled
                else max(1, int(round(config.leverage)))
            ),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PaperAccountState":
        """從 account.json 還原帳戶。"""
        values = dict(payload)
        if "model_kind" not in values:
            model_reference = str(values.get("model_dir", "")).lower().replace("\\", "/")
            legacy_tokens = ("xgboost", "lightgbm", "/models/")
            values["model_kind"] = (
                "legacy"
                if any(token in model_reference for token in legacy_tokens)
                else "rl"
            )
        values.setdefault("expert_kind", "general")
        values.setdefault("rl_entry_threshold", 0.10)
        values.setdefault("rl_exit_threshold", 0.02)
        legacy_target = values.pop("pending_probability", None)
        values.setdefault("pending_target_fraction", legacy_target)
        values.setdefault("pending_leverage", None)
        values.setdefault("short_carry_paid", 0.0)
        values.setdefault("position_carry_cost", 0.0)
        values.setdefault("realized_pnl", 0.0)
        values.setdefault("day_start_equity", float(values.get("config", {}).get("initial_capital", 0.0)))
        values.setdefault("current_day", None)
        values.setdefault("consecutive_losses", 0)
        values.setdefault("funding_paid", 0.0)
        values["config"] = PaperTradingConfig(**dict(values["config"]))
        values["risk_config"] = RiskConfig(**dict(values["risk_config"]))
        values.setdefault(
            "position_leverage",
            (
                values["risk_config"].dynamic_leverage.min_leverage
                if values["risk_config"].dynamic_leverage.enabled
                and abs(float(values.get("quantity", 0.0))) <= 1e-12
                else max(1, int(round(values["config"].leverage)))
            ),
        )
        values.setdefault("last_leverage_reason", "舊帳戶沿用固定槓桿")
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        """轉成可序列化的 JSON 字典。"""
        return asdict(self)
