"""模擬交易命令列參數測試。"""

from ai_quant_trading.paper_trading.cli import (
    _validate_realtime_arguments,
    build_parser,
)


def test_rl_auto_accepts_market_and_threshold_arguments() -> None:
    arguments = build_parser().parse_args(
        [
            "rl-auto",
            "--account",
            "rl_aapl",
            "--model-dir",
            "training/ppo",
            "--download-latest",
            "--exchange",
            "yahoo_finance",
            "--symbol",
            "AAPL",
            "--interval",
            "1d",
            "--entry-threshold",
            "0.1",
            "--exit-threshold",
            "0.02",
            "--allow-short",
            "--max-short-fraction",
            "0.05",
            "--short-borrow-rate-annual",
            "0.10",
        ]
    )

    assert arguments.command == "rl-auto"
    assert arguments.symbol == "AAPL"
    assert arguments.entry_threshold == 0.1
    assert arguments.exit_threshold == 0.02
    assert arguments.allow_short
    assert arguments.max_short_fraction == 0.05
    assert arguments.short_borrow_rate_annual == 0.10


def test_rl_auto_accepts_websocket_transformer_arguments() -> None:
    arguments = build_parser().parse_args(
        [
            "rl-auto",
            "--account",
            "btc_5m",
            "--model-dir",
            "training/ppo",
            "--download-latest",
            "--exchange",
            "binance",
            "--symbol",
            "BTC/USDT",
            "--interval",
            "5m",
            "--transport",
            "websocket",
            "--transformer-checkpoint",
            "transformer/best_model.pt",
        ]
    )

    assert arguments.transport == "websocket"
    assert arguments.transformer_checkpoint == "transformer/best_model.pt"
    assert arguments.max_spread_bps == 5.0


def test_rl_auto_accepts_binance_futures_15m_websocket_arguments() -> None:
    arguments = build_parser().parse_args(
        [
            "rl-auto",
            "--account",
            "btc_15m",
            "--model-dir",
            "training/sac",
            "--download-latest",
            "--exchange",
            "binance_futures",
            "--symbol",
            "BTC/USDT",
            "--interval",
            "15m",
            "--transport",
            "websocket",
            "--transformer-checkpoint",
            "transformer/best_model.pt",
        ]
    )

    assert arguments.exchange == "binance_futures"
    assert arguments.interval == "15m"
    assert arguments.transport == "websocket"
    _validate_realtime_arguments(arguments)
