"""成本後訊號排序、固定持有回放與研究診斷；不代替 SAC 的完整風控回測。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def score_signals(predictions: pd.DataFrame, horizon: int, penalty: float = 0.10) -> pd.DataFrame:
    """僅使用預測決定方向與分數；未來標籤只在評估時另外合併。"""
    long = predictions[f"predicted_long_edge_{horizon}"].to_numpy(dtype=float)
    short = predictions[f"predicted_short_edge_{horizon}"].to_numpy(dtype=float)
    down = predictions.get(
        f"predicted_downside_excursion_{horizon}", pd.Series(0.0, index=predictions.index)
    )
    up = predictions.get(
        f"predicted_upside_excursion_{horizon}", pd.Series(0.0, index=predictions.index)
    )
    long_score = long - penalty * np.maximum(np.asarray(down, dtype=float), 0.0)
    short_score = short - penalty * np.maximum(np.asarray(up, dtype=float), 0.0)
    choose_long = long_score >= short_score
    return pd.DataFrame(
        {
            "long": choose_long,
            "score": np.where(choose_long, long_score, short_score),
            "predicted_edge": np.where(choose_long, long, short),
        },
        index=predictions.index,
    )


def _profit_factor(values: np.ndarray) -> float:
    gains = float(values[values > 0].sum())
    losses = float(-values[values < 0].sum())
    return gains / losses if losses > 1e-12 else (1_000_000.0 if gains else 0.0)


def block_confidence_interval(values: np.ndarray, repetitions: int = 200) -> tuple[float, float]:
    """循環區塊 bootstrap 保留相鄰交易關聯；結果只屬研究不確定性估計。"""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return (float(values[0]), float(values[0])) if len(values) else (0.0, 0.0)
    length = max(2, int(np.sqrt(len(values))))
    random = np.random.default_rng(20260923)
    starts = random.integers(0, len(values), (repetitions, int(np.ceil(len(values) / length))))
    indices = (starts[..., None] + np.arange(length)) % len(values)
    means = values[indices.reshape(repetitions, -1)[:, : len(values)]].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def fixed_hold_evaluation(
    predictions: pd.DataFrame,
    horizon: int,
    *,
    minimum_edge_bps: float = 2.0,
    downside_penalty: float = 0.10,
    minimum_trades: int = 30,
    extra_cost_bps: float = 0.0,
) -> tuple[dict[str, float], pd.DataFrame]:
    """下一根開盤進場、指定 horizon 收盤退出；同一市場不得重疊持倉。

    已包含標籤中的費用與 funding 估計。回撤只計已平倉權益，不能作為實盤閘門。
    多個市場採等權固定子帳戶；此回放不包含交易所撮合及 SAC 動態風控。
    """
    scores = score_signals(predictions, horizon, downside_penalty)
    frame = predictions.copy()
    frame["score"] = scores["score"]
    frame["side"] = np.where(scores["long"], "long", "short")
    frame["net_return"] = (
        np.where(
            scores["long"],
            frame[f"actual_long_edge_{horizon}"],
            frame[f"actual_short_edge_{horizon}"],
        )
        - extra_cost_bps / 10_000
    )
    if "series_index" not in frame or "endpoint" not in frame:
        raise ValueError("固定持有回放需要來源及 K 線終點索引")
    selected: list[int] = []
    curves: list[float] = []
    drawdowns: list[float] = []
    for _, market in frame.groupby("series_index", sort=True):
        occupied_until = -1
        equity, peak, drawdown = 1.0, 1.0, 0.0
        for index, row in market.sort_values("endpoint").iterrows():
            endpoint = int(row["endpoint"])
            if endpoint < occupied_until or not np.isfinite(row["score"]):
                continue
            if row["score"] <= minimum_edge_bps / 10_000:
                continue
            if not np.isfinite(row["net_return"]):
                continue
            selected.append(index)
            occupied_until = endpoint + horizon
            equity *= max(0.0, 1.0 + float(row["net_return"]))
            peak = max(peak, equity)
            drawdown = max(drawdown, 1.0 - equity / peak)
            if equity <= 0.0:
                break
        curves.append(equity - 1.0)
        drawdowns.append(drawdown)
    trades = frame.loc[selected].copy()
    values = trades["net_return"].to_numpy(dtype=float)
    lower, upper = block_confidence_interval(values)
    max_drawdown = max(drawdowns, default=0.0)
    score = lower - 0.01 * max_drawdown if len(values) >= minimum_trades else -1.0
    return {
        "fixed_hold_trades": float(len(values)),
        "fixed_hold_expectancy": float(values.mean()) if len(values) else 0.0,
        "fixed_hold_profit_factor": _profit_factor(values),
        "fixed_hold_win_rate": float((values > 0).mean()) if len(values) else 0.0,
        "fixed_hold_total_return": float(np.mean(curves)) if curves else 0.0,
        "fixed_hold_closed_equity_drawdown": max_drawdown,
        "fixed_hold_expectancy_ci_low": lower,
        "fixed_hold_expectancy_ci_high": upper,
        "economic_selection_score": score,
    }, trades


def signal_diagnostics(predictions: pd.DataFrame, horizon: int, penalty: float) -> pd.DataFrame:
    """信心分位只供事後診斷，不作為測試期間可預先知道的下單門檻。"""
    frame = score_signals(predictions, horizon, penalty)
    frame["actual_edge"] = np.where(
        frame["long"],
        predictions[f"actual_long_edge_{horizon}"],
        predictions[f"actual_short_edge_{horizon}"],
    )
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return pd.DataFrame()
    frame["score_decile"] = pd.qcut(frame["score"].rank(method="first"), 10, labels=False)
    frame["side"] = np.where(frame["long"], "long", "short")
    if "predicted_regime" in predictions:
        frame["predicted_regime"] = predictions.loc[frame.index, "predicted_regime"]
    if "timestamp_ns" in predictions:
        dates = pd.to_datetime(predictions.loc[frame.index, "timestamp_ns"], utc=True)
        frame["utc_hour"] = dates.dt.hour
        frame["month"] = dates.dt.strftime("%Y-%m")
    rows = []
    for group in ("score_decile", "side", "predicted_regime", "utc_hour", "month"):
        if group not in frame:
            continue
        for value, block in frame.groupby(group, sort=True):
            edges = block["actual_edge"].to_numpy()
            rows.append(
                {
                    "group": group,
                    "bucket": str(value),
                    "samples": len(block),
                    "mean_score": float(block["score"].mean()),
                    "mean_net_edge": float(edges.mean()),
                    "profit_factor": _profit_factor(edges),
                }
            )
    return pd.DataFrame(rows)
