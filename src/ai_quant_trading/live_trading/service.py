"""RL 訊號、風控與 Binance 實盤安全閘門的單次交易流程。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from ai_quant_trading.live_trading.gateway import (
    LiveTradingGateway,
    OrderPreview,
    OrderSubmission,
    PortfolioSnapshot,
    ProtectionSubmission,
    make_client_order_id,
)
from ai_quant_trading.live_trading.audit import append_audit_event
from ai_quant_trading.live_trading.notification import notify_event
from ai_quant_trading.live_trading.operations import (
    OperationalRiskLimits,
    assess_operational_risk,
)
from ai_quant_trading.live_trading.readiness import ensure_live_operational_readiness
from ai_quant_trading.live_trading.storage import (
    CYCLE_COLUMNS,
    ORDER_COLUMNS,
    PROTECTION_COLUMNS,
    SNAPSHOT_COLUMNS,
    LivePosition,
    LiveTradingPaths,
    activate_emergency_halt,
    append_csv_row,
    cycle_was_executed,
    emergency_halt_reason,
    live_trading_paths,
    load_positions,
    read_live_csv,
    save_positions,
)
from ai_quant_trading.risk import (
    RiskConfig,
    calculate_position_size,
    calculate_stop_loss,
    calculate_take_profit,
)


@dataclass(frozen=True, slots=True)
class LiveSignal:
    """最新已收盤 K 線的 RL 目標持倉與動作。"""

    timestamp: str
    symbol: str
    target_fraction: float
    action_signal: int
    close: float
    atr: float | None
    leverage: int | None = None
    required_leverage: int | None = None
    leverage_evidence_cap: int | None = None
    leverage_win_probability: float | None = None
    leverage_risk_reward: float | None = None
    leverage_net_expectancy: float | None = None
    leverage_reason: str = "固定槓桿"


@dataclass(frozen=True, slots=True)
class LiveCycleResult:
    """一次 AI 實盤輪次的完整摘要。"""

    signal: LiveSignal
    paths: LiveTradingPaths
    portfolio: Any | None
    preview: OrderPreview | None
    submission: OrderSubmission | None
    processed: bool
    reason: str
    message: str


@dataclass(frozen=True, slots=True)
class ManagedPositionReconciliation:
    """本機受管理部位與交易所現貨餘額的保守對帳結果。"""

    consistent: bool
    managed_quantity: float
    exchange_quantity: float
    shortage: float
    reason: str


def utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record_snapshot(
    paths: LiveTradingPaths, environment: str, snapshot: Any
) -> None:
    append_csv_row(
        paths.account_snapshots_csv,
        SNAPSHOT_COLUMNS,
        {
            "timestamp": utc_now_text(),
            "environment": environment,
            "market_type": getattr(snapshot, "market_type", None)
            or ("usd_m_futures" if hasattr(snapshot, "position_quantity") else "spot"),
            "symbol": snapshot.symbol,
            "base_asset": snapshot.base_asset,
            "quote_asset": snapshot.quote_asset,
            "base_free": snapshot.base_free,
            "base_locked": snapshot.base_locked,
            "quote_free": snapshot.quote_free,
            "quote_locked": snapshot.quote_locked,
            "price": snapshot.price,
            "estimated_equity": snapshot.estimated_equity,
            "can_trade": snapshot.can_trade,
            "wallet_balance": getattr(snapshot, "wallet_balance", None),
            "available_balance": getattr(snapshot, "available_balance", None),
            "margin_balance": getattr(snapshot, "margin_balance", None),
            "position_quantity": getattr(snapshot, "position_quantity", None),
            "entry_price": getattr(snapshot, "entry_price", None),
            "unrealized_pnl": getattr(snapshot, "unrealized_pnl", None),
            "liquidation_price": getattr(snapshot, "liquidation_price", None),
            "leverage": getattr(snapshot, "leverage", None),
            "margin_type": getattr(snapshot, "margin_type", None),
        },
    )
    append_audit_event(
        paths.audit_jsonl,
        "account_snapshot",
        {
            "environment": environment,
            "market_type": (
                "usd_m_futures" if hasattr(snapshot, "position_quantity") else "spot"
            ),
            "symbol": snapshot.symbol,
            "estimated_equity": str(snapshot.estimated_equity),
            "price": str(snapshot.price),
            "can_trade": snapshot.can_trade,
        },
    )


def record_order_submission(
    paths: LiveTradingPaths,
    environment: str,
    submission: OrderSubmission,
    stop_loss: float | None,
    take_profit: float | None,
) -> None:
    response = submission.response
    append_csv_row(
        paths.orders_csv,
        ORDER_COLUMNS,
        {
            "timestamp": utc_now_text(),
            "environment": environment,
            "market_type": submission.preview.market_type,
            "symbol": submission.preview.symbol,
            "side": submission.preview.side,
            "position_side": submission.preview.position_side,
            "reduce_only": submission.preview.reduce_only,
            "quantity": submission.preview.quantity,
            "reference_price": submission.preview.reference_price,
            "estimated_notional": submission.preview.estimated_notional,
            "client_order_id": submission.client_order_id,
            "validated_only": submission.validated_only,
            "recovered_after_unknown": submission.recovered_after_unknown,
            "status": response.get(
                "status", "VALIDATED" if submission.validated_only else "UNKNOWN"
            ),
            "order_id": response.get("orderId"),
            "executed_quantity": response.get("executedQty"),
            "cumulative_quote_quantity": response.get("cummulativeQuoteQty")
            or response.get("cumQuote"),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        },
    )
    append_audit_event(
        paths.audit_jsonl,
        "order_submission",
        {
            "environment": environment,
            "symbol": submission.preview.symbol,
            "side": submission.preview.side,
            "quantity": str(submission.preview.quantity),
            "client_order_id": submission.client_order_id,
            "validated_only": submission.validated_only,
            "recovered_after_unknown": submission.recovered_after_unknown,
            "status": str(response.get("status", "UNKNOWN")),
        },
        severity="warning" if not submission.validated_only else "info",
    )


def record_protection_submission(
    paths: LiveTradingPaths,
    environment: str,
    submission: ProtectionSubmission,
    message: str,
) -> None:
    """保存交易所 OCO 識別碼與目前狀態，供人工稽核及重啟對帳。"""
    response = submission.response
    append_csv_row(
        paths.protections_csv,
        PROTECTION_COLUMNS,
        {
            "timestamp": utc_now_text(),
            "environment": environment,
            "symbol": submission.symbol,
            "quantity": submission.quantity,
            "take_profit_price": submission.take_profit_price,
            "stop_price": submission.stop_price,
            "list_client_order_id": submission.list_client_order_id,
            "order_list_id": response.get("orderListId"),
            "list_status_type": response.get("listStatusType"),
            "list_order_status": response.get("listOrderStatus"),
            "recovered_after_unknown": submission.recovered_after_unknown,
            "message": message,
        },
    )
    append_audit_event(
        paths.audit_jsonl,
        "protection_submission",
        {
            "environment": environment,
            "symbol": submission.symbol,
            "quantity": str(submission.quantity),
            "list_client_order_id": submission.list_client_order_id,
            "list_order_status": str(response.get("listOrderStatus", "UNKNOWN")),
            "message": message,
        },
        severity="warning",
    )


def _record_cycle(
    paths: LiveTradingPaths,
    environment: str,
    model_dir: str,
    signal: LiveSignal,
    effective_signal: int,
    reason: str,
    execution_mode: str,
    status: str,
    client_order_id: str | None,
    message: str,
    market_type: str = "spot",
) -> None:
    append_csv_row(
        paths.cycles_csv,
        CYCLE_COLUMNS,
        {
            "timestamp": utc_now_text(),
            "environment": environment,
            "market_type": market_type,
            "model_dir": model_dir,
            "signal_time": signal.timestamp,
            "symbol": signal.symbol,
            "target_fraction": signal.target_fraction,
            "leverage": signal.leverage,
            "required_leverage": signal.required_leverage,
            "leverage_evidence_cap": signal.leverage_evidence_cap,
            "leverage_win_probability": signal.leverage_win_probability,
            "leverage_risk_reward": signal.leverage_risk_reward,
            "leverage_net_expectancy": signal.leverage_net_expectancy,
            "leverage_reason": signal.leverage_reason,
            "policy_signal": signal.action_signal,
            "effective_signal": effective_signal,
            "reason": reason,
            "execution_mode": execution_mode,
            "status": status,
            "client_order_id": client_order_id,
            "message": message,
        },
    )
    append_audit_event(
        paths.audit_jsonl,
        "decision_cycle",
        {
            "environment": environment,
            "market_type": market_type,
            "model_dir": model_dir,
            "signal_time": signal.timestamp,
            "symbol": signal.symbol,
            "target_fraction": signal.target_fraction,
            "leverage": signal.leverage,
            "leverage_reason": signal.leverage_reason,
            "policy_signal": signal.action_signal,
            "effective_signal": effective_signal,
            "reason": reason,
            "execution_mode": execution_mode,
            "status": status,
            "client_order_id": client_order_id,
            "message": message,
        },
        severity="warning" if status in {"failed", "submitted"} else "info",
    )
    try:
        from ai_quant_trading.live_trading.reporting import save_daily_trading_report

        save_daily_trading_report(paths)
    except (OSError, TypeError, ValueError) as exc:
        notify_event(
            paths.notifications_log,
            "warning",
            f"每日營運報告更新失敗：{type(exc).__name__}",
        )


def _fill_values(submission: OrderSubmission) -> tuple[float, float]:
    response = submission.response
    quantity = float(response.get("executedQty") or 0)
    quote = float(response.get("cummulativeQuoteQty") or response.get("cumQuote") or 0)
    fills = list(response.get("fills", []))
    if quantity <= 0 and fills:
        quantity = sum(float(fill.get("qty") or 0) for fill in fills)
    if quote <= 0 and fills:
        quote = sum(
            float(fill.get("qty") or 0) * float(fill.get("price") or 0)
            for fill in fills
        )
    if quantity <= 0:
        return 0.0, 0.0
    average = float(response.get("avgPrice") or 0)
    return quantity, (
        average
        if average > 0
        else quote / quantity
        if quote > 0
        else float(submission.preview.reference_price)
    )


def reconcile_managed_spot_position(
    snapshot: PortfolioSnapshot,
    position: LivePosition | None,
    *,
    relative_tolerance: float = 0.001,
) -> ManagedPositionReconciliation:
    """確認交易所可用加鎖定數量足以覆蓋本系統記錄的現貨部位。"""
    if not 0 <= relative_tolerance < 1:
        raise ValueError("relative_tolerance 必須介於 0 與 1")
    managed = 0.0 if position is None else max(float(position.quantity), 0.0)
    exchange = max(float(snapshot.base_free + snapshot.base_locked), 0.0)
    shortage = max(managed - exchange, 0.0)
    tolerance = max(managed * relative_tolerance, 1e-12)
    consistent = shortage <= tolerance
    reason = "position_reconciled" if consistent else "exchange_position_shortage"
    return ManagedPositionReconciliation(consistent, managed, exchange, shortage, reason)


def _protectable_quantity(submission: OrderSubmission) -> float:
    """扣除以買入幣種支付的手續費，避免 OCO 數量超過實際可用餘額。"""
    quantity, _ = _fill_values(submission)
    base_asset = submission.preview.base_asset
    base_commission = sum(
        float(fill.get("commission") or 0)
        for fill in submission.response.get("fills", [])
        if str(fill.get("commissionAsset")) == base_asset
    )
    return max(quantity - base_commission, 0.0)


def run_live_signal_cycle(
    signal: LiveSignal,
    *,
    model_dir: str | Path,
    gateway: LiveTradingGateway,
    storage_root: str | Path,
    risk_config: RiskConfig,
    execute: bool = False,
    confirmation: str = "",
    execution_eligibility: Callable[[], None] | None = None,
    portfolio_override: PortfolioSnapshot | None = None,
    sizing_equity_override: float | None = None,
    operational_limits: OperationalRiskLimits | None = None,
) -> LiveCycleResult:
    """執行一次 RL 最新收盤訊號；預設只驗證訂單，不實際成交。"""
    if execute and gateway.config.environment == "live":
        if execution_eligibility is None:
            raise ValueError("Live 執行缺少 RL 品質檢查")
        execution_eligibility()
    if execute and gateway.config.environment == "live":
        ensure_live_operational_readiness(
            storage_root,
            model_dir=model_dir,
            symbol=signal.symbol,
        )
    model_text = str(Path(model_dir))
    environment = gateway.config.environment
    paths = live_trading_paths(storage_root, environment)
    mode = "execute" if execute else "validate"
    if execute and cycle_was_executed(paths, model_text, signal.symbol, signal.timestamp):
        return LiveCycleResult(
            signal,
            paths,
            None,
            None,
            None,
            False,
            "duplicate",
            "同一模型與收盤 K 線已執行過，不重複送單。",
        )

    snapshot = portfolio_override or gateway.portfolio(signal.symbol)
    _record_snapshot(paths, environment, snapshot)
    positions = load_positions(paths)
    position = positions.get(snapshot.symbol)

    if position is not None and execute and gateway.config.exchange_protection_enabled:
        protection_id = position.protection_list_client_order_id
        if position.protection_status == "active" and protection_id:
            protection_state = gateway.query_oco_protection(protection_id)
            list_status = str(
                protection_state.get("listOrderStatus")
                or protection_state.get("listStatusType")
                or ""
            ).upper()
            if list_status == "ALL_DONE":
                reports = list(protection_state.get("orderReports", []))
                protection_filled = any(
                    str(report.get("status", "")).upper() == "FILLED"
                    or float(report.get("executedQty") or 0) > 0
                    for report in reports
                )
                if not protection_filled:
                    halt_message = "OCO 已結束但沒有成交紀錄，部位可能失去交易所保護"
                    positions[snapshot.symbol] = replace(
                        position,
                        protection_status="failed",
                        updated_at=utc_now_text(),
                    )
                    save_positions(paths, positions)
                    activate_emergency_halt(paths, halt_message)
                    raise RuntimeError(halt_message)
                positions.pop(snapshot.symbol, None)
                save_positions(paths, positions)
                message = "交易所 OCO 已成交，本地持倉已完成對帳。"
                _record_cycle(
                    paths,
                    environment,
                    model_text,
                    signal,
                    0,
                    "exchange_protection_filled",
                    mode,
                    "hold",
                    None,
                    message,
                )
                return LiveCycleResult(
                    signal,
                    paths,
                    snapshot,
                    None,
                    None,
                    True,
                    "exchange_protection_filled",
                    message,
                )
        elif position.protection_status in {"pending", "failed"}:
            if position.take_profit is None:
                raise RuntimeError("既有部位缺少停利價，無法建立 OCO 保護單")
            protection = gateway.submit_oco_protection(
                symbol=position.symbol,
                quantity=position.quantity,
                take_profit_price=position.take_profit,
                stop_price=position.stop_loss,
                confirmation=confirmation,
                seed=position.entry_order_id,
            )
            positions[position.symbol] = replace(
                position,
                protection_list_client_order_id=protection.list_client_order_id,
                protection_order_list_id=protection.response.get("orderListId"),
                protection_status="active",
                updated_at=utc_now_text(),
            )
            save_positions(paths, positions)
            record_protection_submission(paths, environment, protection, "保護單恢復成功")
            message = "既有部位的交易所 OCO 保護單已恢復。"
            _record_cycle(
                paths,
                environment,
                model_text,
                signal,
                0,
                "protection_recovered",
                mode,
                "hold",
                protection.list_client_order_id,
                message,
            )
            return LiveCycleResult(
                signal,
                paths,
                snapshot,
                None,
                None,
                True,
                "protection_recovered",
                message,
            )

    reconciliation = reconcile_managed_spot_position(snapshot, position)
    if execute and not reconciliation.consistent:
        halt_message = (
            "交易所 BTC 餘額少於本機受管理部位："
            f"本機 {reconciliation.managed_quantity:.8f}、"
            f"交易所 {reconciliation.exchange_quantity:.8f}"
        )
        activate_emergency_halt(paths, halt_message)
        notify_event(paths.notifications_log, "critical", halt_message)
        raise RuntimeError(halt_message)

    operational = assess_operational_risk(
        read_live_csv(paths.account_snapshots_csv, SNAPSHOT_COLUMNS),
        current_equity=float(snapshot.estimated_equity),
        has_open_position=position is not None,
        can_trade=snapshot.can_trade,
        emergency_halt=emergency_halt_reason(paths),
        limits=(
            operational_limits
            or OperationalRiskLimits(maximum_drawdown=risk_config.max_drawdown_limit)
        ),
    )
    append_audit_event(
        paths.audit_jsonl,
        "operational_risk_assessment",
        {
            "symbol": snapshot.symbol,
            **operational.to_dict(),
        },
        severity="critical" if operational.hard_halt else "info",
    )
    if execute and operational.hard_halt:
        halt_message = "；".join(operational.reasons)
        activate_emergency_halt(paths, halt_message)
        notify_event(paths.notifications_log, "critical", halt_message)

    effective_signal = signal.action_signal
    reason = "rl_policy"
    exchange_protected = position is not None and position.protection_status == "active"
    if position is not None and not exchange_protected and snapshot.price <= Decimal(
        str(position.stop_loss)
    ):
        effective_signal = -1
        reason = "stop_loss"
    elif (
        position is not None
        and not exchange_protected
        and position.take_profit is not None
        and snapshot.price >= Decimal(str(position.take_profit))
    ):
        effective_signal = -1
        reason = "take_profit"
    if execute and operational.force_reduce:
        effective_signal = -1
        reason = "operational_risk_exit"
    elif execute and not operational.allow_new_risk and effective_signal > 0:
        effective_signal = 0
        reason = "operational_risk_halt"

    if effective_signal == 0 or (effective_signal > 0 and position is not None):
        message = "本期維持 Hold。" if effective_signal == 0 else "已有受管理部位，不重複加碼。"
        _record_cycle(
            paths, environment, model_text, signal, 0, reason, mode, "hold", None, message
        )
        return LiveCycleResult(signal, paths, snapshot, None, None, True, reason, message)
    if effective_signal < 0 and position is None:
        message = "沒有由本系統管理的部位，本期不送出賣單。"
        _record_cycle(
            paths,
            environment,
            model_text,
            signal,
            0,
            "no_position",
            mode,
            "no_position",
            None,
            message,
        )
        return LiveCycleResult(signal, paths, snapshot, None, None, True, "no_position", message)

    stop_loss: float | None = None
    take_profit: float | None = None
    if effective_signal > 0:
        entry_price = float(snapshot.price)
        stop_loss = calculate_stop_loss(entry_price, risk_config, signal.atr)
        take_profit = calculate_take_profit(entry_price, risk_config)
        if execute and gateway.config.exchange_protection_enabled and take_profit is None:
            raise ValueError("已要求交易所 OCO 保護，但風控設定沒有停利價")
        sizing = calculate_position_size(
            equity=(
                float(snapshot.estimated_equity)
                if sizing_equity_override is None
                else sizing_equity_override
            ),
            cash=float(snapshot.quote_free),
            entry_price=entry_price,
            stop_loss=stop_loss,
            fee_rate=gateway.config.estimated_fee_rate,
            max_risk_per_trade=risk_config.max_risk_per_trade,
            max_position_fraction=risk_config.max_position_fraction,
        )
        preview = gateway.preview_market_order(
            signal.symbol,
            "BUY",
            quote_amount=min(sizing.position_notional, gateway.config.max_order_quote),
        )
    else:
        if position is None:
            raise RuntimeError("平倉訊號缺少本系統管理的部位，已拒絕送單")
        if execute and position.protection_status == "active":
            if position.protection_list_client_order_id is None:
                raise RuntimeError("保護單狀態為 active，但缺少交易所清單 ID")
            gateway.cancel_oco_protection(
                position.symbol,
                position.protection_list_client_order_id,
                confirmation,
            )
            positions[position.symbol] = replace(
                position,
                protection_status="cancelled",
                updated_at=utc_now_text(),
            )
            save_positions(paths, positions)
            snapshot = gateway.portfolio(signal.symbol)
        preview = gateway.preview_market_order(
            signal.symbol,
            "SELL",
            quantity=min(position.quantity, float(snapshot.base_free)),
            full_exit=True,
        )

    order_seed = f"{model_text}|{signal.timestamp}|{reason}"
    client_order_id = make_client_order_id(preview.symbol, preview.side, order_seed)
    submission = (
        gateway.submit_order(preview, confirmation, client_order_id)
        if execute
        else gateway.validate_order(preview, client_order_id)
    )
    record_order_submission(paths, environment, submission, stop_loss, take_profit)

    if execute:
        filled_quantity, average_price = _fill_values(submission)
        if preview.side == "BUY" and filled_quantity > 0 and stop_loss is not None:
            positions[preview.symbol] = LivePosition(
                symbol=preview.symbol,
                quantity=_protectable_quantity(submission),
                entry_price=average_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                entry_order_id=submission.client_order_id,
                opened_at=utc_now_text(),
                updated_at=utc_now_text(),
                protection_status=(
                    "pending" if gateway.config.exchange_protection_enabled else "disabled"
                ),
            )
            save_positions(paths, positions)
            if gateway.config.exchange_protection_enabled:
                try:
                    if take_profit is None:
                        raise RuntimeError("交易所保護已啟用，但停利價格遺失")
                    protection = gateway.submit_oco_protection(
                        symbol=preview.symbol,
                        quantity=positions[preview.symbol].quantity,
                        take_profit_price=take_profit,
                        stop_price=stop_loss,
                        confirmation=confirmation,
                        seed=submission.client_order_id,
                    )
                except Exception as exc:
                    positions[preview.symbol] = replace(
                        positions[preview.symbol],
                        protection_status="failed",
                        updated_at=utc_now_text(),
                    )
                    save_positions(paths, positions)
                    halt_message = f"買入已成交，但 OCO 保護單建立失敗：{exc}"
                    activate_emergency_halt(paths, halt_message)
                    notify_event(paths.notifications_log, "critical", halt_message)
                    raise RuntimeError(halt_message) from exc
                positions[preview.symbol] = replace(
                    positions[preview.symbol],
                    protection_list_client_order_id=protection.list_client_order_id,
                    protection_order_list_id=protection.response.get("orderListId"),
                    protection_status="active",
                    updated_at=utc_now_text(),
                )
                save_positions(paths, positions)
                record_protection_submission(paths, environment, protection, "保護單建立成功")
        elif preview.side == "SELL" and position is not None and filled_quantity > 0:
            remaining = max(position.quantity - filled_quantity, 0.0)
            if remaining <= max(position.quantity * 1e-6, 1e-12):
                positions.pop(preview.symbol, None)
            else:
                positions[preview.symbol] = replace(
                    position,
                    quantity=remaining,
                    protection_list_client_order_id=None,
                    protection_order_list_id=None,
                    protection_status=(
                        "pending"
                        if gateway.config.exchange_protection_enabled
                        else "disabled"
                    ),
                    updated_at=utc_now_text(),
                )
            save_positions(paths, positions)

    status = "submitted" if execute else "validated"
    message = "訂單已送出。" if execute else "訂單參數與簽章驗證通過，未送入撮合引擎。"
    _record_cycle(
        paths,
        environment,
        model_text,
        signal,
        effective_signal,
        reason,
        mode,
        status,
        submission.client_order_id,
        message,
    )
    notify_event(
        paths.notifications_log,
        "warning" if execute else "info",
        f"{environment} {preview.symbol} {preview.side} {preview.quantity}：{message}",
    )
    return LiveCycleResult(
        signal,
        paths,
        snapshot,
        preview,
        submission,
        True,
        reason,
        message,
    )
