"""將同期間、同成本的策略比較畫成研究圖，不表示可部署排名。"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS = {"sac": "SAC selected", "sac_final": "SAC final (100k)", "cash": "Cash", "long_risk_managed": "Long + risk limits",
          "ema_trend": "EMA trend", "donchian_breakout": "Donchian 20/10",
          "rsi_reversion": "RSI reversion", "bollinger_reversion": "Bollinger reversion",
          "macd_trend": "MACD trend", "vwap_reversion": "VWAP reversion",
          "hourly_trend": "1h EMA trend", "transformer_return_only": "Transformer return only"}


def plot(root: Path) -> Path:
    data = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
    supplemental = root / "final_checkpoint_results.json"
    if supplemental.exists():
        data["results"].extend(json.loads(supplemental.read_text(encoding="utf-8"))["results"])
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    rows = [row for row in data["results"] if row["split"] == "test" and row["scenario"] == "base"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 10), gridspec_kw={"height_ratios": [1.4, 1]}, layout="constrained")
    for row in rows:
        if row["policy"] == "cash":
            continue
        frame = pd.read_csv(root / row["path"])
        times = pd.to_datetime(frame.timestamp, utc=True) + pd.Timedelta(minutes=15)
        style = {"color": "#333333", "linestyle": "--"} if row["policy"] == "sac" else {}
        axes[0].plot(times, (frame.equity / plan["environment_configs"]["base"]["initial_capital"] - 1) * 100,
                     label=LABELS[row["policy"]], linewidth=2 if row["policy"].startswith("sac") else 1.1, **style)
    axes[0].set(title="BTCUSDT 15m | Same-engine Test comparison", ylabel="Account return (%)")
    axes[0].legend(ncol=3, fontsize=8, loc="lower left")
    axes[0].grid(alpha=.2)
    names = [row["policy"] for row in rows]
    x = np.arange(len(names))
    for offset, scenario, color in ((-.2, "base", "#267f89"), (.2, "stress", "#b3465a")):
        values = [next(row["metrics"]["total_return"] * 100 for row in data["results"]
                       if row["split"] == "test" and row["scenario"] == scenario and row["policy"] == name)
                  for name in names]
        axes[1].bar(x + offset, values, width=.38, color=color, label=scenario)
    axes[1].set_xticks(x, [LABELS[name] for name in names], rotation=25, ha="right", fontsize=8)
    axes[1].axhline(0, color="#555555", linewidth=.7)
    axes[1].set(ylabel="Test return (%)", title="Cost sensitivity | No strategy selected from this chart")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=.2)
    fig.suptitle("~11 days, one SAC seed: diagnostic only; not evidence of long-term profitability", fontsize=11)
    target = root / "test_comparison.png"
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    args = parser.parse_args()
    print(plot(args.study))
