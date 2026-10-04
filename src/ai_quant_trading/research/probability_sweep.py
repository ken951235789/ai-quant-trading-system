"""研究用機率門檻敏感度；重用成交排程，不更動正式模型或部署門檻。"""

import hashlib
import json

import numpy as np
import pandas as pd

from ai_quant_trading.transformer.economics import block_confidence_interval
from ai_quant_trading.transformer.strategy_events import select_event_trades


class ThresholdReplay:
    """先驗證所有候選，再過濾；0% 仍保留淨收益門檻，不等於無 AI。"""

    def __init__(self, predictions, horizon, config, minimum_edge_bps):
        self.predictions = predictions.copy()
        self.horizon = horizon
        self.config = config
        self.edge = minimum_edge_bps
        self.baseline, _ = select_event_trades(
            self.predictions, horizon, config, minimum_edge_bps, filtered=False)
        self._cache = {}

    def select(self, threshold):
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("研究門檻必須介於 0 與 1，且為有限值")
        p = self.predictions
        mask = ((p[f"tradeability_probability_{self.horizon}"] >= threshold)
                & (p[f"predicted_return_{self.horizon}"] > self.edge / 10000))
        key = mask.to_numpy().tobytes()
        if key not in self._cache:
            selected, _ = select_event_trades(
                p.loc[mask], self.horizon, self.config, self.edge, filtered=False)
            self._cache[key] = selected
        return self._cache[key].copy()


def schedule_id(trades):
    """相同交易路徑只保存一次；同一排程的多個門檻不算獨立證據。"""
    columns = ["endpoint", "exit_endpoint", "side", "net_return"]
    payload = trades[columns].to_numpy(dtype=float).tolist()
    return hashlib.sha256(json.dumps(payload, allow_nan=False).encode()).hexdigest()


def assert_trade_parity(actual, expected):
    """零成交 CSV 欄位會被讀成 object；仍以明確數值型別逐筆比對。"""
    for column in ("endpoint", "exit_endpoint"):
        np.testing.assert_array_equal(actual[column].to_numpy(dtype=float), expected[column].to_numpy(dtype=float))
    np.testing.assert_allclose(actual.net_return.to_numpy(dtype=float),
                               expected.net_return.to_numpy(dtype=float), atol=1e-12, rtol=1e-10)


def trade_metrics(trades):
    """名目本金報酬及固定 10% 配置的結算敏感度；不宣稱完整帳戶回測。"""
    fields = ["gross_same_path_return", "spread_return", "slippage_return",
              "fee_return", "funding_return", "net_return"]
    values = trades[fields].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("交易成本或收益包含非有限值")
    np.testing.assert_allclose(values[:, 0] - values[:, 1:5].sum(axis=1),
                               values[:, 5], atol=1e-12, rtol=1e-10)
    names = ["mean_net", "std_net", "worst_trade", "actual_win_rate", "success_rate_2bps",
             "average_win", "average_loss", "payoff_ratio", "breakeven_win_rate",
             "profit_factor", "ci_low", "ci_high", "without_best_5_mean",
             "settled_return_10pct", "settled_drawdown_10pct", "mean_holding_bars"]
    result = {name: None for name in names}
    result.update(trades=len(trades), evidence="INSUFFICIENT_EVIDENCE",
                  account_equity_validated=False, ci_status="insufficient_sample")
    result.update({"mean_" + name: None for name in fields[:-1]})
    if not len(trades):
        return result
    net = values[:, 5]
    gains, losses = net[net > 0], -net[net < 0]
    avg_win = float(gains.mean()) if len(gains) else None
    avg_loss = float(losses.mean()) if len(losses) else None
    # 兩筆資料的循環區塊抽樣會退化成固定均值，不能呈現成零寬度信賴區間。
    # 八筆僅是此描述工具的最低計算條件，絕不表示統計充分或取得交易資格。
    ci = block_confidence_interval(net) if len(net) >= 8 else (None, None)
    # 僅在出場時計入損益；不模擬保證金、清算、每日停機或棒內權益。
    if np.any(1 + .1 * net <= 0):
        raise ValueError("固定配置的結算權益無法使用對數複利計算")
    equity = np.exp(np.r_[0., np.cumsum(np.log1p(.1 * net))])
    result.update(mean_net=float(net.mean()), std_net=float(net.std(ddof=1)) if len(net) > 1 else None,
        worst_trade=float(net.min()), actual_win_rate=float((net > 0).mean()),
        success_rate_2bps=float((net > .0002).mean()), average_win=avg_win, average_loss=avg_loss,
        payoff_ratio=avg_win / avg_loss if avg_win is not None and avg_loss is not None else None,
        breakeven_win_rate=avg_loss / (avg_win + avg_loss)
            if avg_win is not None and avg_loss is not None else None,
        profit_factor=float(gains.sum() / losses.sum()) if losses.sum() > 0 else None,
        ci_low=ci[0], ci_high=ci[1], ci_status="descriptive_unadjusted" if len(net) >= 8 else "insufficient_sample",
        without_best_5_mean=float(np.sort(net)[:-5].mean()) if len(net) > 5 else None,
        settled_return_10pct=float(equity[-1] - 1),
        settled_drawdown_10pct=float((equity / np.maximum.accumulate(equity) - 1).min()),
        mean_holding_bars=float((trades.exit_endpoint - trades.entry_endpoint + 1).mean()))
    result.update({"mean_" + name: float(trades[name].mean()) for name in fields[:-1]})
    return result


def aggregate_seeds(rows):
    """以模型為單位呈現平均與範圍；同一歷史的 seeds 不合併成獨立成交。"""
    frame = pd.DataFrame(rows)
    keys = ["family", "variant", "split", "scenario", "threshold_pct"]
    records = []
    for key, group in frame.groupby(keys, sort=True):
        record = dict(zip(keys, key))
        record.update(seeds=len(group), active_seeds=int((group.trades > 0).sum()))
        for metric in ("trades", "mean_net", "actual_win_rate", "profit_factor",
                       "settled_return_10pct", "settled_drawdown_10pct"):
            values = pd.to_numeric(group[metric]).dropna()
            for stat in ("mean", "min", "max"):
                record[f"{metric}_{stat}"] = float(getattr(values, stat)()) if len(values) else None
        records.append(record)
    return pd.DataFrame(records)
