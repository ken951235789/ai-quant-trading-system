"""固定成交路徑成本歸因與保守路徑界線；輸出不可作為模型輸入。"""

import numpy as np
import pandas as pd


def attribute_events(frame, outcomes, contract, source_sha256, post_stop_bars=8):
    if len(source_sha256) != 64 or any(c not in "0123456789abcdef" for c in source_sha256):
        raise ValueError("需要實際來源的 SHA256")
    if post_stop_bars < 1:
        raise ValueError("停損後窗口必須為正整數")
    result = outcomes.copy()
    side = result.side.to_numpy()
    points = result.endpoint.to_numpy(dtype=int)
    entries = result.entry_endpoint.to_numpy(dtype=int)
    exits = result.exit_endpoint.to_numpy(dtype=int)
    entry_reference = frame.open.to_numpy()[entries]
    reference_ratio = result.exit_reference.to_numpy() / entry_reference
    slip, half_spread = contract.slippage_bps / 10000, contract.spread_bps / 20000

    def price_return(friction):
        return side * (reference_ratio * (1 - side * friction) / (1 + side * friction) - 1)

    gross, spread_only, filled = (price_return(f) for f in (0., half_spread, half_spread + slip))
    result["gross_same_path_return"] = gross
    result["spread_return"] = gross - spread_only
    result["slippage_return"] = spread_only - filled
    result["reconciliation_error"] = (gross - result.spread_return - result.slippage_return
        - result.fee_return - result.funding_return - result.net_return)
    if not np.allclose(result.reconciliation_error, 0., atol=1e-12, rtol=0):
        raise ValueError("成本歸因未與淨收益對帳")
    result["contract_sha256"] = contract.digest
    result["source_sha256"] = source_sha256
    result["family"] = contract.family
    result["signal_at"] = (frame.timestamp.iloc[points] + pd.Timedelta(minutes=15)).to_numpy()
    result["entry_at"] = frame.timestamp.iloc[entries].to_numpy()
    result["label_end_at"] = (frame.timestamp.iloc[exits] + pd.Timedelta(minutes=15)).to_numpy()
    result["event_id"] = [f"{source_sha256}:{contract.digest}:{p}:{s}" for p, s in zip(points, side)]
    result["entry_atr_fraction"] = frame.event_atr.to_numpy()[points] / frame.close.to_numpy()[points]
    result["entry_regime"] = frame.event_regime.to_numpy()[points]
    result["entry_utc_hour"] = (frame.timestamp.iloc[points] + pd.Timedelta(minutes=15)).dt.hour.to_numpy()
    result["entry_session"] = np.select([result.entry_utc_hour < 8, result.entry_utc_hour < 16],
                                         ["UTC_00_08", "UTC_08_16"], "UTC_16_24")
    result["entry_volatility_bucket"] = np.select(
        [result.entry_atr_fraction < .002, result.entry_atr_fraction < .005], ["low", "medium"], "high")
    records = []
    for row in result.itertuples():
        start, end, s = row.entry_endpoint, row.exit_endpoint, row.side
        entry = float(frame.open.iloc[start])
        # 退出棒之前的棒完整觀察；退出棒高低點可能發生在平倉後，僅能作為界線。
        known_prices, known_points = [entry, row.exit_reference], [start, end]
        for j in range(start, end):
            known_prices.extend([frame.high.iloc[j], frame.low.iloc[j]])
            known_points.extend([j, j])
        known = s * (np.asarray(known_prices) / entry - 1)
        intrabar = row.exit_reason in {"stop", "target"}
        possible = np.r_[known, s * (frame.loc[end, ["high", "low"]].to_numpy(dtype=float) / entry - 1)] if intrabar else known
        record = {"mfe_confirmed_lower": max(0., known.max()), "mfe_possible_upper": max(0., possible.max()),
            "mae_confirmed_lower": max(0., -known.min()), "mae_possible_upper": max(0., -possible.min()),
            "mfe_confirmed_bar": known_points[int(known.argmax())],
            "mae_confirmed_bar": known_points[int(known.argmin())],
            "holding_minutes_lower": (end - start) * 15,
            "holding_minutes_upper": (end - start + int(intrabar)) * 15,
            "exit_bar_intrabar_order_unknown": intrabar,
            "post_stop_start_endpoint": None, "post_stop_end_endpoint": None,
            "post_stop_best_return_from_exit": None, "post_stop_worst_return_from_exit": None}
        if row.exit_reason == "stop" and end + post_stop_bars < len(frame):
            after = frame.iloc[end + 1:end + post_stop_bars + 1]
            movement = s * (after[["high", "low"]].to_numpy().ravel() / row.exit_reference - 1)
            record.update(post_stop_start_endpoint=end + 1, post_stop_end_endpoint=end + post_stop_bars,
                post_stop_best_return_from_exit=float(movement.max()),
                post_stop_worst_return_from_exit=float(movement.min()))
        records.append(record)
    if records:
        result = pd.concat([result.reset_index(drop=True), pd.DataFrame(records)], axis=1)
    return result


def attribution_summary(trades):
    """描述性集中度，不能把刪去贏家當成選模或部署條件。"""
    values = trades.net_return.to_numpy(dtype=float)
    if not len(values):
        return {"trades": 0, "mean": None, "std": None, "worst": None, "groups": []}
    wins = values[values > 0]
    groups = []
    for column in ("side", "exit_reason", "entry_session", "entry_regime", "entry_volatility_bucket"):
        for bucket, rows in trades.groupby(column):
            groups.append({"dimension": column, "bucket": str(bucket), "n": len(rows),
                           "mean": float(rows.net_return.mean())})
    return {"trades": len(values), "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else None, "worst": float(values.min()),
        "without_best_5_mean": float(np.sort(values)[:-5].mean()) if len(values) > 5 else None,
        "top_5_share_of_positive_returns": float(np.sort(wins)[-5:].sum() / wins.sum()) if len(wins) else None,
        "cost_means": {c: float(trades[c].mean()) for c in ("gross_same_path_return", "fee_return",
            "spread_return", "slippage_return", "funding_return")}, "groups": groups,
        "scope": "不重疊交易，名目報酬；固定路徑成本歸因，非零成本重新撮合；未驗證稅後及完整帳戶風險"}
