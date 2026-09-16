"""以最新已收盤 K 線推進一次模擬交易帳戶。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from ai_quant_trading.features.builder import infer_annualization_periods
from ai_quant_trading.market_clock import completed_bars_only, interval_duration
from ai_quant_trading.paper_trading.state import PaperAccountState, utc_now_text
from ai_quant_trading.paper_trading.storage import (
    ORDER_COLUMNS,
    PERFORMANCE_COLUMNS,
    POSITION_COLUMNS,
    PREDICTION_COLUMNS,
    TRADE_COLUMNS,
    PaperAccountPaths,
    append_csv_row,
)
from ai_quant_trading.risk import (
    RiskConfig,
    calculate_position_size,
    calculate_stop_loss,
    calculate_take_profit,
)


@dataclass(frozen=True, slots=True)
class PaperCycleResult:
    """一次模擬交易執行結果。"""

    state: PaperAccountState
    paths: PaperAccountPaths
    processed: bool
    initialized: bool
    executed_signal: int
    policy_signal: int
    target_fraction: float | None
    message: str


def _account_equity(state: PaperAccountState, price: float) -> float:
    """依帳戶模式計算淨值；永續合約只把未實現損益加回餘額。"""
    if state.config.execution_mode == "perpetual":
        unrealized = (
            state.quantity * (price - float(state.entry_price or price))
            if abs(state.quantity) > 1e-12
            else 0.0
        )
        return state.cash + unrealized
    return state.cash + state.quantity * price


def _margin_used(state: PaperAccountState, price: float) -> float:
    if state.config.execution_mode != "perpetual":
        return abs(state.quantity) * price
    return abs(state.quantity) * price / max(state.position_leverage, 1)


def _record_trade_result(state: PaperAccountState, net_pnl: float) -> None:
    state.consecutive_losses = state.consecutive_losses + 1 if net_pnl < 0 else 0


def _prepare_market_frame(frame: pd.DataFrame, risk_config: RiskConfig) -> pd.DataFrame:
    """驗證最新 K 線及模型所需資料。"""
    required = ["timestamp", "open", "high", "low", "close"]
    if risk_config.stop_loss_mode == "atr":
        required.append("atr_14")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"模擬交易資料缺少必要欄位：{missing}")
    if frame.empty:
        raise ValueError("模擬交易資料不可為空")

    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
    numeric = ["open", "high", "low", "close"]
    if "atr_14" in required:
        numeric.append("atr_14")
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="coerce")
    if (
        result["timestamp"].isna().any()
        or result[["open", "high", "low", "close"]].isna().any().any()
    ):
        raise ValueError("模擬交易資料含有無效的時間或價格")
    if (result[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("模擬交易價格必須大於 0")
    result = result.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    result = completed_bars_only(result)
    if result.empty:
        raise ValueError("資料中沒有已收盤的 K 線")
    return result.reset_index(drop=True)


def _validate_account_source(state: PaperAccountState, row: pd.Series) -> None:
    """避免把不同標的或週期的資料誤送進同一帳戶。"""
    actual = {
        "exchange": str(row.get("exchange", state.exchange)),
        "symbol": str(row.get("symbol", state.symbol)),
        "interval": str(row.get("interval", state.interval)),
    }
    expected = {
        "exchange": state.exchange,
        "symbol": state.symbol,
        "interval": state.interval,
    }
    mismatches = [key for key in expected if actual[key].lower() != expected[key].lower()]
    if mismatches:
        raise ValueError(f"市場資料與帳戶來源不一致：{mismatches}")


def _close_position(
    state: PaperAccountState,
    row: pd.Series,
    paths: PaperAccountPaths,
    *,
    base_price: float,
    reason: str,
) -> None:
    """依目前方向平掉多頭或空頭，並保存訂單與完整交易。"""
    if state.entry_time is None or state.entry_price is None:
        raise RuntimeError("帳戶持倉缺少進場時間或價格，拒絕平倉")
    signed_quantity = state.quantity
    direction = 1.0 if signed_quantity > 0 else -1.0
    quantity = abs(signed_quantity)
    fill_price = base_price * (
        1 - state.config.slippage_rate
        if direction > 0
        else 1 + state.config.slippage_rate
    )
    notional = quantity * fill_price
    exit_fee = notional * state.config.fee_rate
    gross_pnl = direction * (fill_price - state.entry_price) * quantity
    net_pnl = gross_pnl - state.entry_fee - exit_fee - state.position_carry_cost
    timestamp = pd.Timestamp(row["timestamp"]).isoformat()

    if state.config.execution_mode == "perpetual":
        state.cash += gross_pnl - exit_fee
        state.realized_pnl += gross_pnl - exit_fee
        order_side = "SELL" if direction > 0 else "BUY_TO_COVER"
    elif direction > 0:
        state.cash += notional - exit_fee
        order_side = "SELL"
    else:
        state.cash -= notional + exit_fee
        order_side = "BUY_TO_COVER"
    append_csv_row(
        paths.orders_csv,
        ORDER_COLUMNS,
        {
            "timestamp": timestamp,
            "side": order_side,
            "reason": reason,
            "signal_time": state.pending_time,
            "target_fraction": state.pending_target_fraction,
            "leverage": state.position_leverage,
            "market_open": float(row["open"]),
            "fill_price": fill_price,
            "quantity": quantity,
            "notional": notional,
            "fee": exit_fee,
            "cash_after": state.cash,
        },
    )
    _record_trade_result(state, net_pnl)
    append_csv_row(
        paths.trades_csv,
        TRADE_COLUMNS,
        {
            "entry_time": state.entry_time,
            "exit_time": timestamp,
            "entry_price": state.entry_price,
            "exit_price": fill_price,
            "quantity": quantity,
            "entry_fee": state.entry_fee,
            "exit_fee": exit_fee,
            "total_fee": state.entry_fee + exit_fee + state.position_carry_cost,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "return_pct": net_pnl / state.entry_cost if state.entry_cost else 0.0,
            "holding_periods": state.holding_periods,
            "exit_reason": reason,
            "stop_loss": state.stop_loss,
            "take_profit": state.take_profit,
            "risk_budget": state.risk_budget,
            "leverage": state.position_leverage,
        },
    )
    state.quantity = 0.0
    state.entry_time = None
    state.entry_price = None
    state.entry_fee = 0.0
    state.entry_cost = 0.0
    state.holding_periods = 0
    state.stop_loss = None
    state.take_profit = None
    state.risk_budget = None
    state.position_carry_cost = 0.0
    state.position_leverage = (
        state.risk_config.dynamic_leverage.min_leverage
        if state.risk_config.dynamic_leverage.enabled
        else max(1, int(round(state.config.leverage)))
    )


def _position_side(quantity: float) -> str:
    """把帶正負號數量轉成風控函式使用的方向。"""
    return "long" if quantity >= 0 else "short"


def _set_position_protection(state: PaperAccountState, atr: float | None) -> None:
    """依目前方向更新停損與停利價。"""
    if abs(state.quantity) <= 1e-12 or state.entry_price is None:
        state.stop_loss = None
        state.take_profit = None
        return
    side = _position_side(state.quantity)
    state.stop_loss = calculate_stop_loss(
        state.entry_price,
        state.risk_config,
        atr,
        side=side,
    )
    state.take_profit = calculate_take_profit(
        state.entry_price,
        state.risk_config,
        side=side,
    )


def _execute_pending_signal(
    state: PaperAccountState,
    row: pd.Series,
    paths: PaperAccountPaths,
) -> int:
    """在新 K 線開盤執行前一根收盤後保存的訊號。"""
    if state.pending_target_fraction is None:
        return 0
    executed_signal = _execute_rl_target(state, row, paths)
    state.pending_signal = 0
    state.pending_target_fraction = None
    state.pending_leverage = None
    state.pending_time = None
    state.pending_atr = None
    state.pending_reason = "hold"
    return executed_signal


def _execute_rl_target(
    state: PaperAccountState,
    row: pd.Series,
    paths: PaperAccountPaths,
) -> int:
    """把帶正負號的 RL 目標部位轉成加減碼、回補或反手。"""
    short_limit = state.config.max_short_fraction if state.config.allow_short else 0.0
    long_limit = min(
        state.config.position_fraction,
        state.risk_config.max_position_fraction,
    )
    if state.config.execution_mode == "perpetual":
        execution_leverage = max(
            1,
            int(state.pending_leverage or state.position_leverage or state.config.leverage),
        )
        margin_limit = state.config.max_margin_fraction * execution_leverage
        long_limit = min(long_limit, margin_limit)
        short_limit = min(short_limit, margin_limit)
    target = min(
        max(float(state.pending_target_fraction or 0.0), -short_limit),
        long_limit,
    )
    if state.risk_halted:
        target = 0.0

    open_price = float(row["open"])
    equity_at_open = _account_equity(state, open_price)
    if equity_at_open <= 0:
        return 0
    current_fraction = state.quantity * open_price / equity_at_open
    if abs(target - current_fraction) <= 1e-9:
        return 0

    target_direction = 0.0 if abs(target) <= 1e-12 else (1.0 if target > 0 else -1.0)
    current_direction = (
        0.0
        if abs(state.quantity) <= 1e-12
        else (1.0 if state.quantity > 0 else -1.0)
    )
    if current_direction and current_direction != target_direction:
        close_signal = -1 if current_direction > 0 else 1
        _close_position(
            state,
            row,
            paths,
            base_price=open_price,
            reason=state.pending_reason,
        )
        # 反手一律先平倉；下一根仍維持反向訊號才建立新方向。
        return close_signal

    desired_abs_quantity = equity_at_open * abs(target) / open_price
    current_abs_quantity = abs(state.quantity)
    if desired_abs_quantity < current_abs_quantity - 1e-12:
        return _reduce_rl_position(
            state,
            row,
            paths,
            desired_abs_quantity=desired_abs_quantity,
            target=target,
        )
    if desired_abs_quantity <= current_abs_quantity + 1e-12:
        return 0

    side = "long" if target > 0 else "short"
    fill_price = open_price * (
        1 + state.config.slippage_rate
        if side == "long"
        else 1 - state.config.slippage_rate
    )
    stop_loss = calculate_stop_loss(
        fill_price,
        state.risk_config,
        state.pending_atr,
        side=side,
    )
    sizing = calculate_position_size(
        equity=equity_at_open,
        cash=equity_at_open,
        entry_price=fill_price,
        stop_loss=stop_loss,
        fee_rate=state.config.fee_rate,
        max_risk_per_trade=state.risk_config.max_risk_per_trade,
        max_position_fraction=abs(target),
        side=side,
    )
    desired_total_quantity = sizing.quantity
    added_quantity = max(desired_total_quantity - current_abs_quantity, 0.0)
    if side == "long" and state.config.execution_mode != "perpetual":
        affordable_quantity = state.cash / (fill_price * (1 + state.config.fee_rate))
        added_quantity = min(added_quantity, affordable_quantity)
    if added_quantity <= 1e-12:
        return 0

    notional = added_quantity * fill_price
    fee = notional * state.config.fee_rate
    previous_abs_quantity = abs(state.quantity)
    previous_cost_basis = previous_abs_quantity * float(state.entry_price or 0.0)
    if state.config.execution_mode == "perpetual":
        state.cash -= fee
        state.realized_pnl -= fee
        order_side = "BUY" if side == "long" else "SELL_SHORT"
        direction = 1.0 if side == "long" else -1.0
    elif side == "long":
        state.cash -= notional + fee
        order_side = "BUY"
        direction = 1.0
    else:
        state.cash += notional - fee
        order_side = "SELL_SHORT"
        direction = -1.0
    new_abs_quantity = previous_abs_quantity + added_quantity
    state.quantity = direction * new_abs_quantity
    state.entry_price = (
        previous_cost_basis + added_quantity * fill_price
    ) / new_abs_quantity
    state.entry_fee += fee
    state.entry_cost += notional + fee
    if previous_abs_quantity <= 1e-12:
        state.entry_time = pd.Timestamp(row["timestamp"]).isoformat()
        state.holding_periods = 0
    if state.config.execution_mode == "perpetual":
        state.position_leverage = max(
            1,
            int(state.pending_leverage or state.position_leverage or state.config.leverage),
        )
    _set_position_protection(state, state.pending_atr)
    state.risk_budget = sizing.risk_budget
    append_csv_row(
        paths.orders_csv,
        ORDER_COLUMNS,
        {
            "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
            "side": order_side,
            "reason": state.pending_reason,
            "signal_time": state.pending_time,
            "target_fraction": target,
            "leverage": state.position_leverage,
            "market_open": open_price,
            "fill_price": fill_price,
            "quantity": added_quantity,
            "notional": notional,
            "fee": fee,
            "cash_after": state.cash,
        },
    )
    return 1 if side == "long" else -1


def _reduce_rl_position(
    state: PaperAccountState,
    row: pd.Series,
    paths: PaperAccountPaths,
    *,
    desired_abs_quantity: float,
    target: float,
) -> int:
    """部分平掉同方向部位；完整平倉交由共用平倉函式處理。"""
    current_abs_quantity = abs(state.quantity)
    if desired_abs_quantity <= 1e-12:
        signal = -1 if state.quantity > 0 else 1
        _close_position(
            state,
            row,
            paths,
            base_price=float(row["open"]),
            reason=state.pending_reason,
        )
        return signal

    if state.entry_time is None or state.entry_price is None:
        raise RuntimeError("帳戶持倉缺少進場時間或價格，拒絕部分平倉")
    direction = 1.0 if state.quantity > 0 else -1.0
    closed_quantity = current_abs_quantity - desired_abs_quantity
    open_price = float(row["open"])
    fill_price = open_price * (
        1 - state.config.slippage_rate
        if direction > 0
        else 1 + state.config.slippage_rate
    )
    notional = closed_quantity * fill_price
    exit_fee = notional * state.config.fee_rate
    closed_fraction = closed_quantity / current_abs_quantity
    allocated_entry_cost = state.entry_cost * closed_fraction
    allocated_entry_fee = state.entry_fee * closed_fraction
    allocated_carry_cost = state.position_carry_cost * closed_fraction
    gross_pnl = direction * (fill_price - state.entry_price) * closed_quantity
    net_pnl = gross_pnl - allocated_entry_fee - exit_fee - allocated_carry_cost
    if state.config.execution_mode == "perpetual":
        state.cash += gross_pnl - exit_fee
        state.realized_pnl += gross_pnl - exit_fee
        order_side = "SELL" if direction > 0 else "BUY_TO_COVER"
        signal = -1 if direction > 0 else 1
    elif direction > 0:
        state.cash += notional - exit_fee
        order_side = "SELL"
        signal = -1
    else:
        state.cash -= notional + exit_fee
        order_side = "BUY_TO_COVER"
        signal = 1
    timestamp = pd.Timestamp(row["timestamp"]).isoformat()
    append_csv_row(
        paths.orders_csv,
        ORDER_COLUMNS,
        {
            "timestamp": timestamp,
            "side": order_side,
            "reason": state.pending_reason,
            "signal_time": state.pending_time,
            "target_fraction": target,
            "leverage": state.position_leverage,
            "market_open": open_price,
            "fill_price": fill_price,
            "quantity": closed_quantity,
            "notional": notional,
            "fee": exit_fee,
            "cash_after": state.cash,
        },
    )
    append_csv_row(
        paths.trades_csv,
        TRADE_COLUMNS,
        {
            "entry_time": state.entry_time,
            "exit_time": timestamp,
            "entry_price": state.entry_price,
            "exit_price": fill_price,
            "quantity": closed_quantity,
            "entry_fee": allocated_entry_fee,
            "exit_fee": exit_fee,
            "total_fee": allocated_entry_fee + exit_fee + allocated_carry_cost,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "return_pct": net_pnl / allocated_entry_cost if allocated_entry_cost else 0.0,
            "holding_periods": state.holding_periods,
            "exit_reason": state.pending_reason,
            "stop_loss": state.stop_loss,
            "take_profit": state.take_profit,
            "risk_budget": state.risk_budget,
            "leverage": state.position_leverage,
        },
    )
    state.quantity = direction * desired_abs_quantity
    state.entry_cost -= allocated_entry_cost
    state.entry_fee -= allocated_entry_fee
    state.position_carry_cost -= allocated_carry_cost
    _record_trade_result(state, net_pnl)
    return signal


def _apply_intrabar_risk(
    state: PaperAccountState,
    row: pd.Series,
    paths: PaperAccountPaths,
) -> int:
    """在同一根 K 線內檢查停損與停利；同時碰到時先採停損。"""
    if abs(state.quantity) <= 1e-12 or state.stop_loss is None:
        return 0
    if state.quantity > 0 and float(row["low"]) <= state.stop_loss:
        _close_position(
            state,
            row,
            paths,
            base_price=min(float(row["open"]), state.stop_loss),
            reason="stop_loss",
        )
        return -1
    if (
        state.quantity > 0
        and state.take_profit is not None
        and float(row["high"]) >= state.take_profit
    ):
        _close_position(
            state,
            row,
            paths,
            base_price=max(float(row["open"]), state.take_profit),
            reason="take_profit",
        )
        return -1
    if state.quantity < 0 and float(row["high"]) >= state.stop_loss:
        _close_position(
            state,
            row,
            paths,
            base_price=max(float(row["open"]), state.stop_loss),
            reason="stop_loss",
        )
        return 1
    if (
        state.quantity < 0
        and state.take_profit is not None
        and float(row["low"]) <= state.take_profit
    ):
        _close_position(
            state,
            row,
            paths,
            base_price=min(float(row["open"]), state.take_profit),
            reason="take_profit",
        )
        return 1
    return 0


def _charge_short_carry(state: PaperAccountState, row: pd.Series) -> float:
    """依 K 線週期向未平空單收取借券或永續合約資金成本。"""
    if state.config.execution_mode == "perpetual":
        if abs(state.quantity) <= 1e-12:
            return 0.0
        funding_rate = pd.to_numeric(row.get("funding_rate"), errors="coerce")
        if pd.isna(funding_rate):
            return 0.0
        duration = interval_duration(str(row.get("interval", state.interval)))
        elapsed_hours = (
            duration.total_seconds() / 3_600 if duration is not None else 0.25
        )
        direction = 1.0 if state.quantity > 0 else -1.0
        carry = (
            abs(state.quantity)
            * float(row["close"])
            * float(funding_rate)
            * direction
            * elapsed_hours
            / 8.0
        )
        state.cash -= carry
        state.realized_pnl -= carry
        state.funding_paid += carry
        state.position_carry_cost += carry
        return carry
    if state.quantity >= 0 or state.config.short_borrow_rate_annual <= 0:
        return 0.0
    periods = max(infer_annualization_periods(pd.DataFrame([row])), 1)
    carry = (
        abs(state.quantity)
        * float(row["close"])
        * state.config.short_borrow_rate_annual
        / periods
    )
    state.cash -= carry
    state.short_carry_paid += carry
    state.position_carry_cost += carry
    return carry


def _process_bar(
    state: PaperAccountState,
    row: pd.Series,
    prediction_row: pd.Series | None,
    paths: PaperAccountPaths,
    *,
    execute_pending: bool,
    prediction_provider: Callable[[PaperAccountState, pd.Series], pd.Series] | None = None,
) -> tuple[int, int, float]:
    """依序處理一根新 K 線，並將該期帳戶結果寫入紀錄。"""
    current_day = pd.Timestamp(row["timestamp"]).date().isoformat()
    equity_at_open = _account_equity(state, float(row["open"]))
    if state.current_day != current_day:
        state.current_day = current_day
        state.day_start_equity = max(equity_at_open, 1e-12)
        state.consecutive_losses = 0
    current_daily_return = (
        equity_at_open / max(state.day_start_equity, 1e-12) - 1
    )
    daily_halted = current_daily_return <= -state.config.daily_loss_limit
    loss_halted = bool(
        state.config.max_consecutive_losses > 0
        and state.consecutive_losses >= state.config.max_consecutive_losses
    )
    if execute_pending and (daily_halted or loss_halted):
        state.pending_target_fraction = 0.0
        state.pending_reason = (
            "daily_loss_limit" if daily_halted else "consecutive_loss_limit"
        )
    executed_signal = _execute_pending_signal(state, row, paths) if execute_pending else 0
    intrabar_signal = _apply_intrabar_risk(state, row, paths)
    if intrabar_signal:
        executed_signal = intrabar_signal
    elif abs(state.quantity) > 1e-12:
        state.holding_periods += 1
    short_carry_cost = _charge_short_carry(state, row)

    if prediction_provider is not None:
        prediction_row = prediction_provider(state, row)
    if prediction_row is None:
        raise ValueError("模擬交易缺少模型輸出")

    policy_signal = int(prediction_row["action_signal"])
    target_fraction = float(prediction_row["target_fraction"])
    model_target_fraction = float(
        prediction_row.get("model_target_fraction", target_fraction)
    )
    transformer_target_fraction = float(
        prediction_row.get("transformer_target_fraction", model_target_fraction)
    )
    threshold_target_fraction = float(
        prediction_row.get("threshold_target_fraction", target_fraction)
    )
    close_price = float(row["close"])
    position_value = state.quantity * close_price
    equity = _account_equity(state, close_price)
    state.equity_peak = max(state.equity_peak, equity)
    drawdown = equity / state.equity_peak - 1 if state.equity_peak > 0 else 0.0
    if drawdown <= -state.risk_config.max_drawdown_limit:
        state.risk_halted = True

    pending_target_fraction = target_fraction
    current_fraction = position_value / max(equity, 1e-12)
    daily_return = equity / max(state.day_start_equity, 1e-12) - 1
    daily_halted = daily_return <= -state.config.daily_loss_limit
    loss_halted = bool(
        state.config.max_consecutive_losses > 0
        and state.consecutive_losses >= state.config.max_consecutive_losses
    )
    if state.risk_halted or daily_halted or loss_halted:
        pending_target_fraction = 0.0
        pending_signal = -1 if state.quantity > 0 else 1 if state.quantity < 0 else 0
        if state.risk_halted:
            pending_reason = (
                "max_drawdown" if abs(state.quantity) > 1e-12 else "risk_halted"
            )
        elif daily_halted:
            pending_reason = "daily_loss_limit"
        else:
            pending_reason = "consecutive_loss_limit"
    elif pending_target_fraction > current_fraction + 1e-9:
        pending_signal = 1
        pending_reason = (
            "rl_increase_long" if pending_target_fraction > 0 else "rl_reduce_short"
        )
    elif pending_target_fraction < current_fraction - 1e-9:
        pending_signal = -1
        pending_reason = (
            "rl_increase_short" if pending_target_fraction < 0 else "rl_reduce_long"
        )
    else:
        pending_signal = 0
        pending_reason = "rl_hold"

    timestamp = pd.Timestamp(row["timestamp"]).isoformat()
    atr_value: float | None = None
    if state.risk_config.stop_loss_mode == "atr":
        atr_value = float(prediction_row["atr_14"])

    state.pending_signal = pending_signal
    state.pending_target_fraction = pending_target_fraction
    state.pending_leverage = max(
        1,
        int(prediction_row.get("selected_leverage", state.position_leverage)),
    )
    state.last_leverage_reason = str(
        prediction_row.get("leverage_reason", state.last_leverage_reason)
    )
    state.pending_time = timestamp
    state.pending_atr = atr_value
    state.pending_reason = pending_reason
    state.last_processed_timestamp = timestamp
    state.updated_at = utc_now_text()

    append_csv_row(
        paths.predictions_csv,
        PREDICTION_COLUMNS,
        {
            "timestamp": timestamp,
            "symbol": state.symbol,
            "close": close_price,
            "model_target_fraction": model_target_fraction,
            "transformer_target_fraction": transformer_target_fraction,
            "threshold_target_fraction": threshold_target_fraction,
            "approved_target_fraction": target_fraction,
            "target_fraction": target_fraction,
            "action_signal": policy_signal,
            "finbert_sentiment": prediction_row.get("finbert_sentiment", 0.0),
            "finbert_confidence": prediction_row.get("finbert_confidence", 0.0),
            "finbert_available": prediction_row.get("finbert_available", 0.0),
            "transformer_trend": prediction_row.get("transformer_trend", "未啟用"),
            "transformer_return_1": prediction_row.get("transformer_return_1"),
            "transformer_return_5": prediction_row.get("transformer_return_5"),
            "transformer_return_20": prediction_row.get("transformer_return_20"),
            "transformer_volatility": prediction_row.get("transformer_volatility"),
            "transformer_bull_probability": prediction_row.get(
                "transformer_bull_probability"
            ),
            "transformer_bear_probability": prediction_row.get(
                "transformer_bear_probability"
            ),
            "transformer_uncertainty": prediction_row.get("transformer_uncertainty"),
            "transformer_probability_calibrated": prediction_row.get(
                "transformer_probability_calibrated", False
            ),
            "market_regime": prediction_row.get("market_regime", "未啟用"),
            "regime_risk_multiplier": prediction_row.get("regime_risk_multiplier"),
            "capital_multiplier": prediction_row.get("capital_multiplier"),
            "confidence_multiplier": prediction_row.get("confidence_multiplier"),
            "volatility_multiplier": prediction_row.get("volatility_multiplier"),
            "model_input_drifted": prediction_row.get("model_input_drifted", False),
            "model_input_severe_drift": prediction_row.get(
                "model_input_severe_drift", False
            ),
            "model_input_extreme_fraction": prediction_row.get(
                "model_input_extreme_fraction", 0.0
            ),
            "model_input_drift_score": prediction_row.get(
                "model_input_drift_score", 0.0
            ),
            "model_input_drift_multiplier": prediction_row.get(
                "model_input_drift_multiplier", 1.0
            ),
            "target_risk_multiplier": prediction_row.get("target_risk_multiplier"),
            "portfolio_gross_exposure": prediction_row.get(
                "portfolio_gross_exposure"
            ),
            "portfolio_net_exposure": prediction_row.get("portfolio_net_exposure"),
            "portfolio_halted": prediction_row.get("portfolio_halted", False),
            "dynamic_leverage_enabled": prediction_row.get(
                "dynamic_leverage_enabled", False
            ),
            "selected_leverage": state.pending_leverage,
            "required_leverage": prediction_row.get("required_leverage"),
            "leverage_evidence_cap": prediction_row.get("leverage_evidence_cap"),
            "leverage_win_probability": prediction_row.get(
                "leverage_win_probability"
            ),
            "leverage_risk_reward": prediction_row.get("leverage_risk_reward"),
            "leverage_net_expectancy": prediction_row.get(
                "leverage_net_expectancy"
            ),
            "leverage_reason": state.last_leverage_reason,
            "decision_guard_reason": prediction_row.get(
                "decision_guard_reason", "RL policy"
            ),
            "pending_reason": pending_reason,
        },
    )
    append_csv_row(
        paths.performance_csv,
        PERFORMANCE_COLUMNS,
        {
            "timestamp": timestamp,
            "open": float(row["open"]),
            "close": close_price,
            "executed_signal": executed_signal,
            "policy_signal": policy_signal,
            "cash": state.cash,
            "quantity": state.quantity,
            "position_value": position_value,
            "equity": equity,
            "total_return": equity / state.config.initial_capital - 1,
            "drawdown": drawdown,
            "short_carry_cost": short_carry_cost,
            "funding_paid": state.funding_paid,
            "margin_used": _margin_used(state, close_price),
            "leverage": state.position_leverage,
            "realized_pnl": state.realized_pnl,
            "daily_return": daily_return,
            "consecutive_losses": state.consecutive_losses,
            "risk_halted": state.risk_halted,
            "skipped_bars": 0,
        },
    )
    append_csv_row(
        paths.positions_csv,
        POSITION_COLUMNS,
        {
            "timestamp": timestamp,
            "symbol": state.symbol,
            "quantity": state.quantity,
            "entry_price": state.entry_price,
            "market_price": close_price,
            "market_value": position_value,
            "unrealized_pnl": (
                (close_price - state.entry_price) * state.quantity
                if state.entry_price is not None
                else 0.0
            ),
            "stop_loss": state.stop_loss,
            "take_profit": state.take_profit,
            "leverage": state.position_leverage,
        },
    )
    return executed_signal, policy_signal, target_fraction
