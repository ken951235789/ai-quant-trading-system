"""SAC／PPO 帶方向目標曝險的 USD-M 永續合約單次執行流程。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import time

from ai_quant_trading.live_trading.audit import append_audit_event
from ai_quant_trading.live_trading.futures_gateway import (
    FuturesPortfolioSnapshot,
    FuturesProtectionSubmission,
    FuturesTradingGateway,
)
from ai_quant_trading.live_trading.gateway import make_client_order_id
from ai_quant_trading.live_trading.notification import notify_event
from ai_quant_trading.live_trading.operations import (
    OperationalRiskLimits,
    assess_operational_risk,
)
from ai_quant_trading.live_trading.readiness import ensure_live_operational_readiness
from ai_quant_trading.live_trading.service import (
    LiveCycleResult,
    LiveSignal,
    _record_cycle,
    _record_snapshot,
    record_order_submission,
    utc_now_text,
)
from ai_quant_trading.live_trading.storage import (
    PROTECTION_COLUMNS,
    SNAPSHOT_COLUMNS,
    LivePosition,
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


def _signed_local_quantity(position: LivePosition | None) -> float:
    return 0.0 if position is None else position.signed_quantity


def _position_consistent(
    snapshot: FuturesPortfolioSnapshot,
    position: LivePosition | None,
    *,
    relative_tolerance: float = 0.001,
) -> bool:
    managed = _signed_local_quantity(position)
    exchange = float(snapshot.position_quantity)
    tolerance = max(abs(managed) * relative_tolerance, 1e-8)
    return abs(managed - exchange) <= tolerance


def _algo_status(response: dict[str, object]) -> str:
    return str(
        response.get("algoStatus")
        or response.get("status")
        or response.get("orderStatus")
        or "UNKNOWN"
    ).upper()


def _record_futures_protection(
    paths,
    environment: str,
    submission: FuturesProtectionSubmission,
    quantity: float,
    message: str,
) -> None:
    stop_status = _algo_status(submission.stop_response)
    take_status = _algo_status(submission.take_profit_response)
    append_csv_row(
        paths.protections_csv,
        PROTECTION_COLUMNS,
        {
            "timestamp": utc_now_text(),
            "environment": environment,
            "market_type": "usd_m_futures",
            "symbol": submission.symbol,
            "quantity": quantity,
            "position_side": submission.side,
            "take_profit_price": submission.take_profit_price,
            "stop_price": submission.stop_price,
            "stop_client_algo_id": submission.stop_client_algo_id,
            "take_profit_client_algo_id": submission.take_profit_client_algo_id,
            "stop_status": stop_status,
            "take_profit_status": take_status,
            "recovered_after_unknown": submission.recovered_after_unknown,
            "message": message,
        },
    )
    append_audit_event(
        paths.audit_jsonl,
        "futures_protection_submission",
        {
            "environment": environment,
            "symbol": submission.symbol,
            "position_side": submission.side,
            "stop_client_algo_id": submission.stop_client_algo_id,
            "take_profit_client_algo_id": submission.take_profit_client_algo_id,
            "stop_status": stop_status,
            "take_profit_status": take_status,
        },
        severity="warning",
    )


def _wait_for_position_change(
    gateway: FuturesTradingGateway,
    symbol: str,
    before: float,
    *,
    attempts: int = 6,
) -> FuturesPortfolioSnapshot:
    """市場單回報後短暫輪詢 Position Risk，避免用舊持倉覆寫本地狀態。"""
    latest = gateway.portfolio(symbol)
    for attempt in range(attempts):
        if abs(float(latest.position_quantity) - before) > 1e-8:
            return latest
        if attempt + 1 < attempts:
            time.sleep(0.2 * (attempt + 1))
            latest = gateway.portfolio(symbol)
    return latest


def _cancel_existing_protection(
    gateway: FuturesTradingGateway,
    position: LivePosition,
    confirmation: str,
) -> None:
    if position.protection_status != "active":
        return
    gateway.cancel_position_protection(
        position.stop_client_algo_id,
        position.take_profit_client_algo_id,
        confirmation,
    )


def _restore_existing_protection(
    gateway: FuturesTradingGateway,
    position: LivePosition,
    confirmation: str,
) -> FuturesProtectionSubmission:
    if position.take_profit is None:
        raise RuntimeError("既有永續持倉缺少停利價格，無法恢復交易所保護")
    return gateway.submit_position_protection(
        symbol=position.symbol,
        position_side=position.side,
        stop_price=position.stop_loss,
        take_profit_price=position.take_profit,
        confirmation=confirmation,
        seed=position.entry_order_id,
    )


def run_futures_live_target_cycle(
    signal: LiveSignal,
    *,
    model_dir: str | Path,
    gateway: FuturesTradingGateway,
    storage_root: str | Path,
    risk_config: RiskConfig,
    execute: bool = False,
    confirmation: str = "",
    execution_eligibility=None,
    portfolio_override: FuturesPortfolioSnapshot | None = None,
    operational_limits: OperationalRiskLimits | None = None,
) -> LiveCycleResult:
    """把正負目標曝險轉成多空增減倉，並在交易所端重建保護單。"""
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
    position_key = f"usd_m_futures:{snapshot.symbol}"
    position = positions.get(position_key)

    # 本地有部位但交易所已歸零，通常表示交易所保護單成交；以交易所為準完成對帳。
    if execute and position is not None and abs(float(snapshot.position_quantity)) <= 1e-8:
        _cancel_existing_protection(gateway, position, confirmation)
        positions.pop(position_key, None)
        save_positions(paths, positions)
        message = "交易所持倉已歸零，本地永續持倉與保護單已完成對帳。"
        _record_cycle(
            paths,
            environment,
            model_text,
            signal,
            0,
            "exchange_position_closed",
            mode,
            "hold",
            None,
            message,
            market_type="usd_m_futures",
        )
        return LiveCycleResult(
            signal, paths, snapshot, None, None, True, "exchange_position_closed", message
        )

    if execute and not _position_consistent(snapshot, position):
        halt_message = (
            "本機受管理持倉與 Binance USD-M 不一致："
            f"本機 {_signed_local_quantity(position):.8f} BTC、"
            f"交易所 {float(snapshot.position_quantity):.8f} BTC"
        )
        activate_emergency_halt(paths, halt_message)
        notify_event(paths.notifications_log, "critical", halt_message)
        raise RuntimeError(halt_message)

    snapshot_history = read_live_csv(paths.account_snapshots_csv, SNAPSHOT_COLUMNS)
    if "market_type" in snapshot_history:
        snapshot_history = snapshot_history.loc[
            snapshot_history["market_type"].astype(str).eq("usd_m_futures")
        ]
    operational = assess_operational_risk(
        snapshot_history,
        current_equity=float(snapshot.estimated_equity),
        has_open_position=abs(float(snapshot.position_quantity)) > 1e-8,
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
        {"symbol": snapshot.symbol, **operational.to_dict()},
        severity="critical" if operational.hard_halt else "info",
    )

    target = float(signal.target_fraction)
    reason = "rl_target_rebalance"
    if execute and operational.force_reduce:
        target = 0.0
        reason = "operational_risk_exit"
    elif execute and not operational.allow_new_risk:
        current_fraction = snapshot.position_fraction
        increases = not (
            target * current_fraction >= 0 and abs(target) <= abs(current_fraction)
        )
        if increases:
            target = current_fraction
            reason = "operational_risk_halt"

    current_quantity = float(snapshot.position_quantity)
    current_fraction = snapshot.position_fraction
    if current_quantity * target < 0:
        target = 0.0
        reason = "reverse_flatten_first"

    reducing_risk = (
        abs(target) <= 1e-12
        or current_quantity != 0
        and target * current_fraction >= 0
        and abs(target) <= abs(current_fraction) + 1e-12
    )
    if execute and gateway.config.environment == "live" and not reducing_risk:
        if execution_eligibility is None:
            raise ValueError("Live 新增風險缺少 RL 品質檢查")
        execution_eligibility()
        ensure_live_operational_readiness(
            storage_root,
            model_dir=model_dir,
            symbol=signal.symbol,
            market_type="usd_m_futures",
        )

    if execute and position is not None and position.protection_status == "active":
        protection_error: Exception | None = None
        states: set[str] = set()
        try:
            protection = gateway.query_position_protection(
                position.stop_client_algo_id or "",
                position.take_profit_client_algo_id or "",
            )
            states = {_algo_status(item) for item in protection.values()}
            open_statuses = {"NEW", "ACCEPTED", "WORKING", "TRIGGERING"}
            if not states or not states.issubset(open_statuses):
                protection_error = RuntimeError(
                    f"持倉仍存在，但保護單狀態異常：{sorted(states)}"
                )
        except Exception as exc:
            protection_error = exc
        if protection_error is not None:
            halt_message = (
                str(protection_error)
                if states
                else f"無法確認永續停損停利保護單：{protection_error}"
            )
            activate_emergency_halt(paths, halt_message)
            notify_event(paths.notifications_log, "critical", halt_message)
            position = replace(
                position,
                protection_status="failed",
                updated_at=utc_now_text(),
            )
            positions[position_key] = position
            save_positions(paths, positions)
            if not reducing_risk:
                raise RuntimeError(halt_message) from protection_error

    equity = max(float(snapshot.estimated_equity), 0.0)
    if equity <= 0:
        raise ValueError("USD-M 帳戶缺少可用權益")
    side_name = "short" if target < 0 else "long" if target > 0 else (
        "short" if current_quantity < 0 else "long"
    )
    entry_price = float(snapshot.mark_price)
    stop_loss = calculate_stop_loss(
        entry_price,
        risk_config,
        signal.atr,
        side=side_name,
    )
    take_profit = calculate_take_profit(entry_price, risk_config, side=side_name)
    if (
        execute
        and abs(target) > 1e-12
        and gateway.config.exchange_protection_enabled
        and take_profit is None
    ):
        raise ValueError("已要求交易所保護，但風控設定沒有停利價")

    target_notional = abs(target) * equity
    if abs(target) > abs(current_fraction) or target * current_fraction < 0:
        sizing = calculate_position_size(
            equity=equity,
            cash=float(snapshot.available_balance) * gateway.config.leverage,
            entry_price=entry_price,
            stop_loss=stop_loss,
            fee_rate=gateway.config.estimated_fee_rate,
            max_risk_per_trade=risk_config.max_risk_per_trade,
            max_position_fraction=risk_config.max_position_fraction,
            side=side_name,
        )
        margin_cap = (
            equity * gateway.config.max_balance_fraction * gateway.config.leverage
        )
        target_notional = min(target_notional, sizing.position_notional, margin_cap)
    desired_quantity = (target_notional / entry_price) * (-1 if target < 0 else 1)
    delta = desired_quantity - current_quantity
    if abs(delta * entry_price) < 0.01:
        message = "目標曝險與目前持倉相同，本期維持 Hold。"
        _record_cycle(
            paths,
            environment,
            model_text,
            signal,
            0,
            reason,
            mode,
            "hold",
            None,
            message,
            market_type="usd_m_futures",
        )
        return LiveCycleResult(signal, paths, snapshot, None, None, True, reason, message)

    reducing = current_quantity != 0 and (
        desired_quantity == 0
        or desired_quantity * current_quantity > 0
        and abs(desired_quantity) < abs(current_quantity)
    )
    side = "BUY" if delta > 0 else "SELL"
    if reducing:
        preview = gateway.preview_market_order(
            signal.symbol,
            side,
            quantity=abs(delta),
            full_exit=abs(desired_quantity) <= 1e-12,
            reduce_only=True,
        )
    else:
        preview = gateway.preview_market_order(
            signal.symbol,
            side,
            quote_amount=min(abs(delta) * entry_price, gateway.config.max_order_quote),
        )

    order_seed = f"{model_text}|{signal.timestamp}|{reason}|{target:.8f}"
    client_order_id = make_client_order_id(preview.symbol, preview.side, order_seed)
    protection_cancelled = False
    if execute and position is not None and position.protection_status == "active":
        _cancel_existing_protection(gateway, position, confirmation)
        protection_cancelled = True
        positions[position_key] = replace(
            position,
            protection_status="cancelled",
            updated_at=utc_now_text(),
        )
        save_positions(paths, positions)

    try:
        submission = (
            gateway.submit_order(preview, confirmation, client_order_id)
            if execute
            else gateway.validate_order(preview, client_order_id)
        )
    except Exception as exc:
        if protection_cancelled and position is not None:
            try:
                restored = _restore_existing_protection(gateway, position, confirmation)
                positions[position_key] = replace(
                    position,
                    stop_client_algo_id=restored.stop_client_algo_id,
                    take_profit_client_algo_id=restored.take_profit_client_algo_id,
                    protection_status="active",
                    updated_at=utc_now_text(),
                )
                save_positions(paths, positions)
            except Exception as restore_exc:
                halt_message = f"調倉失敗且原保護單無法恢復：{restore_exc}"
                activate_emergency_halt(paths, halt_message)
                raise RuntimeError(halt_message) from exc
        raise

    record_order_submission(paths, environment, submission, stop_loss, take_profit)
    if execute:
        updated = _wait_for_position_change(
            gateway,
            signal.symbol,
            current_quantity,
        )
        updated_quantity = float(updated.position_quantity)
        if abs(updated_quantity - current_quantity) <= 1e-8:
            halt_message = "訂單已回報，但 Position Risk 未反映任何持倉變化"
            activate_emergency_halt(paths, halt_message)
            raise RuntimeError(halt_message)
        if current_quantity > 0 and updated_quantity < -1e-8:
            raise RuntimeError("多單減倉意外翻成空單，已停止自動交易")
        if current_quantity < 0 and updated_quantity > 1e-8:
            raise RuntimeError("空單減倉意外翻成多單，已停止自動交易")
        snapshot = updated
        _record_snapshot(paths, environment, snapshot)
        if abs(updated_quantity) <= 1e-8:
            positions.pop(position_key, None)
            save_positions(paths, positions)
        else:
            actual_side = "SHORT" if updated_quantity < 0 else "LONG"
            risk_side = actual_side.lower()
            actual_entry = float(snapshot.entry_price or snapshot.mark_price)
            actual_stop = calculate_stop_loss(
                actual_entry,
                risk_config,
                signal.atr,
                side=risk_side,
            )
            actual_take = calculate_take_profit(actual_entry, risk_config, side=risk_side)
            opened_at = (
                position.opened_at
                if position is not None and position.side == actual_side
                else utc_now_text()
            )
            positions[position_key] = LivePosition(
                symbol=signal.symbol,
                quantity=abs(updated_quantity),
                entry_price=actual_entry,
                stop_loss=actual_stop,
                take_profit=actual_take,
                entry_order_id=submission.client_order_id,
                opened_at=opened_at,
                updated_at=utc_now_text(),
                protection_status=(
                    "pending" if gateway.config.exchange_protection_enabled else "disabled"
                ),
                market_type="usd_m_futures",
                side=actual_side,
                leverage=snapshot.leverage,
                margin_type=snapshot.margin_type,
            )
            save_positions(paths, positions)
            if gateway.config.exchange_protection_enabled:
                try:
                    if actual_take is None:
                        raise ValueError("停利價不存在")
                    protected = gateway.submit_position_protection(
                        symbol=signal.symbol,
                        position_side=actual_side,
                        stop_price=actual_stop,
                        take_profit_price=actual_take,
                        confirmation=confirmation,
                        seed=submission.client_order_id,
                    )
                except Exception as exc:
                    positions[position_key] = replace(
                        positions[position_key],
                        protection_status="failed",
                        updated_at=utc_now_text(),
                    )
                    save_positions(paths, positions)
                    halt_message = f"合約調倉已成交，但交易所保護單建立失敗：{exc}"
                    activate_emergency_halt(paths, halt_message)
                    notify_event(paths.notifications_log, "critical", halt_message)
                    raise RuntimeError(halt_message) from exc
                positions[position_key] = replace(
                    positions[position_key],
                    stop_client_algo_id=protected.stop_client_algo_id,
                    take_profit_client_algo_id=protected.take_profit_client_algo_id,
                    protection_status="active",
                    updated_at=utc_now_text(),
                )
                save_positions(paths, positions)
                _record_futures_protection(
                    paths,
                    environment,
                    protected,
                    abs(updated_quantity),
                    "永續停損停利建立成功",
                )

    status = "submitted" if execute else "validated"
    message = (
        "USD-M 永續調倉已送出並完成持倉對帳。"
        if execute
        else "USD-M 永續訂單參數與簽章驗證通過，未送入撮合引擎。"
    )
    _record_cycle(
        paths,
        environment,
        model_text,
        signal,
        1 if preview.side == "BUY" else -1,
        reason,
        mode,
        status,
        submission.client_order_id,
        message,
        market_type="usd_m_futures",
    )
    notify_event(
        paths.notifications_log,
        "warning" if execute else "info",
        f"{environment} USD-M {preview.symbol} {preview.side} {preview.quantity}：{message}",
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
