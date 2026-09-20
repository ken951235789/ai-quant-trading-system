# BTC Multi-Timeframe AI Quant Trading Research System

[![quality](https://github.com/ken951235789/ai-quant-trading-system/actions/workflows/quality.yml/badge.svg)](https://github.com/ken951235789/ai-quant-trading-system/actions/workflows/quality.yml)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB)
![Status](https://img.shields.io/badge/status-research_only-E9B949)
![Live](https://img.shields.io/badge/live-locked-FF6B72)

[繁體中文](README.md) · [Five-minute demo](docs/PUBLIC_DEMO.md) · [Research results](docs/RESEARCH_RESULTS.md) · [Architecture](docs/ARCHITECTURE.md)

![BTC AI Quant Research System](docs/assets/social-preview.png)

This repository is a research-first Python platform for **BTC/USDT USD-M perpetual
futures**. It connects closed-bar, causal multi-timeframe features to a Transformer V3
market analyst, a Soft Actor-Critic target-exposure trader, an independent risk engine,
cost-aware backtesting, persistent paper trading, Testnet and a deliberately locked live
gateway.

> This is research software, not investment advice. It does not promise profitability.
> Current candidates have not passed the real-capital quality gate.

## Research Questions

1. Can multi-timeframe price, trend, momentum, volatility, volume, derivatives and
   microstructure features produce stable out-of-sample forecasts?
2. Can Transformer context help SAC produce positive expectancy after fees, spread,
   slippage and funding?
3. Can explicit data contracts, walk-forward evaluation, a sealed final holdout and
   operational risk gates reduce leakage and backtest overfitting?

## Architecture

```text
Binance Futures REST / WebSocket
              |
              v
Closed multi-timeframe bars + derivatives / microstructure context
              |
              v
Causal alignment, data-quality gates and normalized features
              |
              v
Transformer V3: return / direction / volatility / regime / uncertainty
              |
              v
SAC: continuous long / short / close / hold target exposure
              |
              v
Independent risk: leverage / stops / daily loss / drawdown / kill switch
              |
              v
Walk-forward -> paper trading -> Testnet -> locked live gateway
```

## Five-Minute Offline Demo

The public demo needs no API key, network access, GPU, market file or model checkpoint.
It runs the real feature and risk modules on deterministic synthetic BTC 15-minute data,
then writes an HTML view and a machine-readable JSON audit report.

```powershell
git clone https://github.com/ken951235789/ai-quant-trading-system.git
cd ai-quant-trading-system
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
ai-quant-demo --open
```

On Linux or macOS, activate with `source .venv/bin/activate`. Outputs are written to
`outputs/public_demo/`.

The demo uses explicitly labelled causal proxy signals. It does **not** claim to run a
trained Transformer or SAC checkpoint and is not evidence of trading performance.

## Honest Current Result

The Transformer showed its clearest directional edge at the longer 20-bar horizon. The
five-bar short-horizon edge remained unstable. Four out-of-sample SAC evaluations were
negative after costs, so no candidate was promoted to Champion and the final holdout
remained sealed. See [the research results](docs/RESEARCH_RESULTS.md) for the numbers and
limitations.

That negative result is retained intentionally: the goal is a trustworthy research and
engineering process, not a selected profitable chart.

## Engineering Highlights

- Causal alignment across `1m / 3m / 5m / 15m / 30m / 1h / 4h / 12h / 1d` bars.
- Train-only fitting for scalers, imputers and feature selection.
- Transformer-to-SAC metadata contracts for features, horizons and provenance.
- Cross-fit out-of-sample Transformer predictions for downstream RL training.
- Shared SAC action semantics across training, backtesting and runtime.
- Fee, spread, slippage, funding, margin and liquidation-aware evaluation.
- Persistent paper accounts, event replay, fault injection and operational watchdogs.
- Live trading locked behind Champion, data, reconciliation and human approval gates.

## Development

Python 3.11 or newer is required. GPU dependencies are optional.

```powershell
python -m pip install -e ".[dev,dashboard,ai,rl,database]"
python -m ruff check src tests scripts
python -m pytest -q
```

Real datasets, checkpoints, credentials, logs, databases and trade records are excluded
from the public repository. See [public release scope](docs/PUBLIC_RELEASE_SCOPE.md).

## Contributing and Security

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Report security
issues privately according to [SECURITY.md](SECURITY.md).

The repository currently grants no redistribution or commercial-use license. It is
published primarily for reproducible research and technical discussion.
