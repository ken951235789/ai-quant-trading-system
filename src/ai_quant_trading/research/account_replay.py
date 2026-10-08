"""固定候選成交路徑的逐棒權益診斷，不假裝已驗證交易所清算。"""

import numpy as np
import pandas as pd


def replay_mark_to_market(frame, trades, *, initial_capital=1000., allocation=.1,
                         fee_bps=5., funding_reserve_bps=1.):
    """費用只扣一次；槓桿禁止，以 OHLC 估計浮虧並分開列出棒內最差界線。"""
    if not np.isfinite([initial_capital, allocation, fee_bps, funding_reserve_bps]).all() or initial_capital <= 0 or not 0 < allocation <= 1 or min(fee_bps, funding_reserve_bps) < 0:
        raise ValueError("逐棒帳戶參數不合法；此版本不允許槓桿")
    trades = trades.sort_values("entry_endpoint").reset_index(drop=True)
    if len(trades) > 1 and (trades.entry_endpoint.to_numpy()[1:] <= trades.exit_endpoint.to_numpy()[:-1]).any():
        raise ValueError("逐棒帳戶只接受不重疊的固定成交")
    if not trades.empty and ((trades.entry_endpoint <= trades.endpoint).any() or
        (trades.exit_endpoint < trades.entry_endpoint).any() or
        not trades.side.isin([-1, 1]).all() or
        (trades.exit_endpoint >= len(frame)).any()):
        raise ValueError("成交時間或方向不合法")
    opening = {int(r.entry_endpoint): r for r in trades.itertuples()}
    wallet, active, peak = float(initial_capital), None, float(initial_capital)
    rows, settled = [], []
    stamps = pd.to_datetime(frame.timestamp, utc=True)
    for point, bar in enumerate(frame.itertuples()):
        if point in opening:
            if active is not None:
                raise ValueError("固定路徑重疊")
            trade = opening[point]
            notional = wallet * allocation
            active = {"trade": trade, "quantity": notional / trade.entry_price, "notional": notional,
                      "wallet_before": wallet, "funding": 0.}
            wallet -= notional * fee_bps / 10000
        worst, equity = wallet, wallet
        if active:
            trade, qty = active["trade"], active["quantity"]
            if point > trade.entry_endpoint:
                settlements = int(stamps.iloc[point].value // (8*3600*10**9) - stamps.iloc[point-1].value // (8*3600*10**9))
                funding = active["notional"] * funding_reserve_bps / 10000 * settlements
                wallet -= funding
                active["funding"] += funding
            closed = point == trade.exit_endpoint
            adverse = bar.low if trade.side == 1 else bar.high
            # regime/time 在開盤退出，不能把該棒之後的高低價計入持倉浮虧。
            if closed and int(trade.reason_code) in {3, 4}:
                adverse = trade.exit_price
            worst = wallet + qty * trade.side * (adverse - trade.entry_price)
            if closed:
                wallet += qty * trade.side * (trade.exit_price - trade.entry_price)
                wallet -= qty * trade.exit_price * fee_bps / 10000
                equity = wallet
                expected = active["notional"] * trade.net_return
                pnl = wallet - active["wallet_before"]
                if not np.isclose(expected, pnl, rtol=1e-9, atol=1e-8):
                    raise ValueError("逐棒帳戶與既有候選成本帳務不一致")
                settled.append({"endpoint": int(trade.endpoint), "net_pnl": pnl, "equity": wallet})
                active = None
            else:
                equity = wallet + qty * trade.side * (bar.close - trade.entry_price)
        peak = max(peak, equity)
        rows.append({"timestamp": stamps.iloc[point], "wallet": wallet, "equity_close": equity,
            "adverse_equity_bound": min(equity, worst), "close_drawdown": 1-equity/peak,
            "adverse_drawdown_bound": max(0., 1-min(equity, worst)/peak)})
    path = pd.DataFrame(rows)
    summary = {"trades": len(settled), "initial_capital": initial_capital,
        "total_return": wallet/initial_capital-1,
        "close_max_drawdown": float(path.close_drawdown.max()) if len(path) else 0.,
        "adverse_drawdown_bound": float(path.adverse_drawdown_bound.max()) if len(path) else 0.,
        "margin_or_bankruptcy_review_required": bool(len(path) and path.adverse_equity_bound.min() <= 0),
        "scope": "固定成交路徑逐棒 MTM，退出棒內最差價僅上界，不代表實際曾持倉到該價",
        "funding": "reserve_not_actual_history", "liquidation_validated": False,
        "margin_tiers_validated": False, "tax_validated": False, "live_eligible": False}
    return path, pd.DataFrame(settled), summary
