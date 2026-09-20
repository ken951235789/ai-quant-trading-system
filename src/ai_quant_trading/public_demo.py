"""不需要 API Key 或模型權重的公開研究流程展示。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import webbrowser

import numpy as np
import pandas as pd

from ai_quant_trading.features import FEATURE_COLUMNS, build_feature_dataset
from ai_quant_trading.risk import govern_target_position


DEMO_DISCLOSURE = (
    "本結果使用固定 seed 的合成行情與透明因果代理訊號，只用來驗證資料、特徵、"
    "決策、成本與風控流程；不是 Transformer／SAC 訓練績效，也不能推論真實獲利。"
)


@dataclass(frozen=True, slots=True)
class PublicDemoArtifact:
    """公開 Demo 的輸出位置與摘要。"""

    report_path: Path
    html_path: Path
    report: dict[str, object]


def generate_synthetic_btc_ohlcv(*, bars: int = 960, seed: int = 20260917) -> pd.DataFrame:
    """建立具有趨勢、震盪與波動切換的可重現合成 BTC 15 分鐘資料。"""
    if bars < 320:
        raise ValueError("公開 Demo 至少需要 320 根 K 線，才能完成 MA200 暖機")

    rng = np.random.default_rng(seed)
    index = np.arange(bars, dtype=float)
    regime = np.select(
        [index < bars * 0.30, index < bars * 0.58, index < bars * 0.80],
        [0.00012, -0.00009, 0.00016],
        default=-0.00003,
    )
    cyclical = 0.00018 * np.sin(index / 21.0) + 0.00010 * np.sin(index / 73.0)
    volatility = np.where((index > bars * 0.52) & (index < bars * 0.72), 0.0032, 0.0018)
    returns = regime + cyclical + rng.normal(0.0, volatility, bars)

    close = 62_000.0 * np.exp(np.cumsum(returns))
    open_price = np.r_[close[0], close[:-1]]
    candle_range = np.abs(rng.normal(0.0017, 0.00065, bars)).clip(0.0004, 0.006)
    high = np.maximum(open_price, close) * (1 + candle_range)
    low = np.minimum(open_price, close) * (1 - candle_range)
    activity = 1 + 2.5 * np.abs(returns) / np.maximum(volatility, 1e-9)
    volume = rng.lognormal(mean=3.9, sigma=0.38, size=bars) * activity
    taker_ratio = np.clip(0.5 + 38 * returns + rng.normal(0, 0.055, bars), 0.12, 0.88)
    spread_bps = np.clip(1.2 + 220 * np.abs(returns) + rng.normal(0, 0.22, bars), 0.5, 5.0)
    funding = np.clip(0.00002 + pd.Series(returns).rolling(32, min_periods=1).mean() * 0.12, -0.0002, 0.0002)
    open_interest = 8.2e9 * np.exp(np.cumsum(rng.normal(0.0, 0.0016, bars)))

    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-01-01", periods=bars, freq="15min", tz="UTC"
            ),
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "quote_asset_volume": volume * close,
            "number_of_trades": np.maximum((volume * 8 + rng.normal(0, 30, bars)).astype(int), 1),
            "taker_buy_base_volume": volume * taker_ratio,
            "funding_rate": funding.to_numpy(dtype=float),
            "open_interest": open_interest,
            "spread_bps": spread_bps,
            "exchange": "synthetic_binance_futures",
            "symbol": "BTC/USDT",
            "interval": "15m",
        }
    )


def _proxy_model_decisions(features: pd.DataFrame) -> pd.DataFrame:
    """建立透明的因果代理訊號，示範正式模型輸出的資料契約。"""
    result = features.copy()
    trend = np.tanh(pd.to_numeric(result["ema_20_50_atr"], errors="coerce") / 2.5)
    momentum = np.tanh(pd.to_numeric(result["return_5"], errors="coerce") / 0.008)
    order_flow = pd.to_numeric(result["volume_delta_ratio"], errors="coerce").clip(-1, 1)
    structure = pd.to_numeric(
        result["supertrend_direction_10_3"], errors="coerce"
    ).clip(-1, 1)
    score = (0.38 * trend + 0.27 * momentum + 0.20 * order_flow + 0.15 * structure).clip(
        -1, 1
    )

    choppiness = pd.to_numeric(result["choppiness_14"], errors="coerce").clip(0, 1)
    uncertainty = (0.20 + 0.55 * choppiness + 0.20 * (1 - score.abs())).clip(0.05, 0.95)
    expected_return = 0.0032 * score * (1 - 0.45 * uncertainty)
    estimated_round_trip_cost = (
        2 * (0.0004 + 0.0002)
        + pd.to_numeric(result["spread_fraction"], errors="coerce").fillna(0.0)
    )
    edge = expected_return.abs() - estimated_round_trip_cost
    tradeable = (uncertainty <= 0.70) & (edge > 0.00010)
    proposed_target = np.select(
        [tradeable & (expected_return > 0), tradeable & (expected_return < 0)],
        [0.55, -0.40],
        default=0.0,
    )
    result["demo_expected_return"] = expected_return
    result["demo_uncertainty"] = uncertainty
    result["demo_edge_after_cost"] = edge
    result["demo_proposed_target"] = proposed_target
    return result


def _simulate_risk_managed_execution(
    frame: pd.DataFrame,
    *,
    initial_capital: float = 1_000.0,
) -> pd.DataFrame:
    """用下一根開盤到收盤的報酬與顯式成本模擬代理目標曝險。"""
    equity = initial_capital
    peak = initial_capital
    current_position = 0.0
    holding_bars = 0
    rows: list[dict[str, object]] = []

    for index in range(len(frame) - 1):
        row = frame.iloc[index]
        next_row = frame.iloc[index + 1]
        drawdown = equity / peak - 1.0
        decision = govern_target_position(
            proposed_target=float(row["demo_proposed_target"]),
            current_position=current_position,
            drawdown=drawdown,
            holding_bars=holding_bars,
            max_position_fraction=0.55,
            allow_short=True,
            max_short_fraction=0.40,
            hard_drawdown_limit=0.08,
            soft_drawdown_limit=0.04,
            soft_drawdown_multiplier=0.50,
            drawdown_curve_exponent=1.0,
            rebalance_deadband=0.08,
            minimum_holding_bars=2,
        )
        target = decision.approved_target
        turnover = abs(target - current_position)
        market_return = float(next_row["close"] / next_row["open"] - 1.0)
        spread = float(row.get("spread_fraction", 0.0))
        execution_cost = turnover * (0.0004 + 0.0002 + spread / 2)
        timestamp = pd.Timestamp(next_row["timestamp"])
        funding_cost = 0.0
        if timestamp.hour % 8 == 0 and timestamp.minute == 0:
            funding_cost = target * float(row.get("funding_rate", 0.0))
        net_return = target * market_return - execution_cost - funding_cost
        equity *= max(1 + net_return, 0.01)
        peak = max(peak, equity)

        changed = abs(target - current_position) > 1e-12
        if abs(target) <= 1e-12:
            holding_bars = 0
        elif changed:
            holding_bars = 1
        else:
            holding_bars += 1
        current_position = target
        rows.append(
            {
                "timestamp": timestamp,
                "close": float(next_row["close"]),
                "expected_return": float(row["demo_expected_return"]),
                "uncertainty": float(row["demo_uncertainty"]),
                "proposed_target": float(row["demo_proposed_target"]),
                "approved_target": target,
                "turnover": turnover,
                "execution_cost": execution_cost,
                "funding_cost": funding_cost,
                "equity": equity,
                "drawdown": equity / peak - 1.0,
                "risk_reasons": ",".join(decision.reasons),
            }
        )
    return pd.DataFrame(rows)


def _maximum_drawdown(equity: pd.Series) -> float:
    drawdown = equity / equity.cummax() - 1.0
    return abs(float(drawdown.min()))


def _svg_polyline(values: pd.Series, *, width: int = 920, height: int = 280) -> str:
    numeric = pd.to_numeric(values, errors="coerce").ffill().bfill().to_numpy(dtype=float)
    if len(numeric) == 0:
        return ""
    low = float(np.min(numeric))
    span = max(float(np.max(numeric) - low), 1e-12)
    x_values = np.linspace(16, width - 16, len(numeric))
    y_values = height - 18 - (numeric - low) / span * (height - 36)
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(x_values, y_values, strict=True))


def _render_html(evaluation: pd.DataFrame, report: dict[str, object]) -> str:
    recent = evaluation.tail(180).reset_index(drop=True)
    price_points = _svg_polyline(recent["close"])
    equity_points = _svg_polyline(recent["equity"])
    metrics = dict(report["metrics"])
    stages = "".join(
        f"<li><span>{index}</span>{escape(str(stage))}</li>"
        for index, stage in enumerate(report["pipeline_stages"], start=1)
    )
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>BTC AI Quant Research Demo</title>
  <style>
    :root {{ color-scheme: dark; --bg:#0b1118; --panel:#121b25; --line:#233243;
      --text:#edf4fb; --muted:#9fb0c1; --cyan:#36c5d8; --green:#38d996; --red:#ff6b72; }}
    * {{ box-sizing:border-box; }} body {{ margin:0; font:15px/1.6 system-ui,sans-serif;
      background:var(--bg); color:var(--text); }} main {{ width:min(1120px,92vw); margin:36px auto 64px; }}
    .eyebrow {{ color:var(--cyan); font-weight:700; letter-spacing:.08em; }}
    h1 {{ margin:6px 0; font-size:clamp(28px,5vw,54px); line-height:1.08; }}
    .lead {{ max-width:850px; color:var(--muted); font-size:18px; }}
    .notice {{ border-left:4px solid #e9b949; background:#211d12; padding:12px 16px; margin:24px 0; }}
    .metrics {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:24px 0; }}
    .card,.chart,.pipeline {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; }}
    .card {{ padding:16px; }} .card b {{ display:block; font-size:24px; }} .card span {{ color:var(--muted); }}
    .chart {{ padding:18px; margin:12px 0; }} .chart h2 {{ margin:0 0 10px; font-size:18px; }}
    svg {{ width:100%; height:auto; display:block; background:#0e1620; border-radius:6px; }}
    .pipeline {{ padding:18px; margin-top:20px; }} ol {{ display:grid; grid-template-columns:repeat(5,1fr);
      list-style:none; padding:0; gap:8px; }} li {{ color:var(--muted); }} li span {{ display:block; color:var(--cyan);
      font-weight:800; }} code {{ color:#d7e8f5; }} footer {{ color:var(--muted); margin-top:28px; }}
    @media (max-width:760px) {{ .metrics {{ grid-template-columns:repeat(2,1fr); }}
      ol {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body><main>
  <div class="eyebrow">OFFLINE · DETERMINISTIC · NO API KEY</div>
  <h1>BTC AI Quant Research Demo</h1>
  <p class="lead">以合成 BTC 15 分鐘行情走過因果特徵、Transformer 輸出契約、
  SAC 目標曝險契約、獨立風控與含成本執行層。</p>
  <div class="notice">{escape(DEMO_DISCLOSURE)}</div>
  <section class="metrics">
    <div class="card"><span>可用特徵</span><b>{int(metrics['feature_count'])}</b></div>
    <div class="card"><span>決策樣本</span><b>{int(metrics['decision_samples'])}</b></div>
    <div class="card"><span>代理淨報酬</span><b>{float(metrics['net_return']):+.2%}</b></div>
    <div class="card"><span>最大回撤</span><b>{float(metrics['max_drawdown']):.2%}</b></div>
  </section>
  <section class="chart"><h2>最近 180 根合成 BTC 價格</h2>
    <svg viewBox="0 0 920 280" role="img" aria-label="合成 BTC 價格">
      <polyline points="{price_points}" fill="none" stroke="#36c5d8" stroke-width="3"/>
    </svg></section>
  <section class="chart"><h2>含手續費、滑價、Spread 與 Funding 的代理資產曲線</h2>
    <svg viewBox="0 0 920 280" role="img" aria-label="代理資產曲線">
      <polyline points="{equity_points}" fill="none" stroke="#38d996" stroke-width="3"/>
    </svg></section>
  <section class="pipeline"><h2>本次實際執行的流程</h2><ol>{stages}</ol></section>
  <footer>Seed: <code>{int(report['seed'])}</code> · 產生時間：
    <code>{escape(str(report['generated_at_utc']))}</code></footer>
</main></body></html>"""


def run_public_demo(
    output_dir: str | Path = "outputs/public_demo",
    *,
    bars: int = 960,
    seed: int = 20260917,
) -> PublicDemoArtifact:
    """執行公開展示並輸出 JSON 與可直接開啟的 HTML。"""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw = generate_synthetic_btc_ohlcv(bars=bars, seed=seed)
    features = build_feature_dataset(
        raw,
        target_horizon=5,
        target_threshold=0.0,
        annualization_periods=365 * 24 * 4,
        drop_na=True,
    )
    decisions = _proxy_model_decisions(features)
    evaluation = _simulate_risk_managed_execution(decisions)
    if evaluation.empty:
        raise RuntimeError("公開 Demo 沒有產生任何可評估決策")

    initial_equity = 1_000.0
    final_equity = float(evaluation["equity"].iloc[-1])
    active_changes = int((evaluation["turnover"] > 1e-12).sum())
    report: dict[str, object] = {
        "schema_version": 1,
        "kind": "synthetic_pipeline_demo",
        "model_status": "causal_proxy_not_trained_transformer_or_sac",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "disclosure": DEMO_DISCLOSURE,
        "pipeline_stages": [
            "合成已收盤 BTC 15m OHLCV",
            "因果特徵工程",
            "Transformer 輸出契約代理",
            "SAC 目標曝險契約代理",
            "獨立風控與含成本執行",
        ],
        "proxy_inputs": [
            "ema_20_50_atr",
            "return_5",
            "volume_delta_ratio",
            "supertrend_direction_10_3",
            "choppiness_14",
            "spread_fraction",
        ],
        "metrics": {
            "raw_bars": len(raw),
            "feature_rows": len(features),
            "feature_count": len(FEATURE_COLUMNS),
            "decision_samples": len(evaluation),
            "position_changes": active_changes,
            "initial_equity": initial_equity,
            "final_equity": final_equity,
            "net_return": final_equity / initial_equity - 1.0,
            "max_drawdown": _maximum_drawdown(evaluation["equity"]),
            "total_execution_cost": float(evaluation["execution_cost"].sum()),
            "total_funding_cost": float(evaluation["funding_cost"].sum()),
        },
        "controls": {
            "maximum_long_fraction": 0.55,
            "maximum_short_fraction": 0.40,
            "soft_drawdown_limit": 0.04,
            "hard_drawdown_limit": 0.08,
            "fee_rate": 0.0004,
            "slippage_rate": 0.0002,
        },
    }
    report_path = output / "report.json"
    html_path = output / "index.html"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    html_path.write_text(_render_html(evaluation, report), encoding="utf-8")
    return PublicDemoArtifact(report_path=report_path, html_path=html_path, report=report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="執行無網路、無 API Key 的 BTC 研究流程 Demo")
    parser.add_argument("--output", default="outputs/public_demo", help="輸出資料夾")
    parser.add_argument("--bars", type=int, default=960, help="合成 15 分鐘 K 線數")
    parser.add_argument("--seed", type=int, default=20260917, help="可重現亂數 seed")
    parser.add_argument("--open", action="store_true", help="完成後開啟 HTML 報告")
    args = parser.parse_args(argv)
    artifact = run_public_demo(args.output, bars=args.bars, seed=args.seed)
    metrics = dict(artifact.report["metrics"])
    print("公開 Demo 完成")
    print(f"特徵數：{int(metrics['feature_count'])}")
    print(f"決策樣本：{int(metrics['decision_samples'])}")
    print(f"HTML：{artifact.html_path.resolve()}")
    print(f"JSON：{artifact.report_path.resolve()}")
    print(DEMO_DISCLOSURE)
    if args.open:
        webbrowser.open(artifact.html_path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
