"""PPO／SAC 模擬交易命令列介面。"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

from ai_quant_trading.automation import (
    AutomationConfig,
    automation_paths,
    read_automation_status,
    request_automation_stop,
    run_automation,
)
from ai_quant_trading.market_clock import BTC_MULTITIMEFRAME_INTERVALS
from ai_quant_trading.paper_trading.config import PaperTradingConfig
from ai_quant_trading.paper_trading.state import PaperAccountState
from ai_quant_trading.paper_trading.storage import account_paths, load_account_state
from ai_quant_trading.risk import RiskConfig


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """加入 PPO 單次與自動模式共用參數。"""
    parser.add_argument("--account", required=True, help="模擬帳戶名稱")
    parser.add_argument("--model-dir", required=True, help="PPO／SAC 訓練成品資料夾")
    parser.add_argument("--input", help="市場 OHLCV CSV")
    parser.add_argument("--download-latest", action="store_true")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--paper-dir", default="data/paper_trading")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--exchange")
    parser.add_argument("--symbol")
    parser.add_argument("--interval")
    parser.add_argument("--initial-capital", type=float, default=1000.0)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--slippage-rate", type=float, default=0.0005)
    parser.add_argument("--position-fraction", type=float, default=1.0)
    parser.add_argument("--entry-threshold", type=float, default=0.10)
    parser.add_argument("--exit-threshold", type=float, default=0.02)
    parser.add_argument(
        "--allow-short",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--max-short-fraction", type=float)
    parser.add_argument("--short-borrow-rate-annual", type=float)
    parser.add_argument("--max-risk-per-trade", type=float, default=0.01)
    parser.add_argument("--stop-loss-mode", choices=["fixed", "atr"], default="fixed")
    parser.add_argument("--fixed-stop-loss-pct", type=float, default=0.03)
    parser.add_argument("--atr-multiplier", type=float, default=2.0)
    parser.add_argument("--take-profit-pct", type=float, default=0.06)
    parser.add_argument("--disable-take-profit", action="store_true")
    parser.add_argument("--max-drawdown-limit", type=float, default=0.20)


def build_parser() -> argparse.ArgumentParser:
    """建立單次、自動、停止與狀態子命令。"""
    parser = argparse.ArgumentParser(description="執行 PPO／SAC 模擬交易")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in [
        ("run", "用 PPO／SAC 推進一次模擬帳戶"),
        ("rl-run", "用 PPO／SAC 推進一次模擬帳戶"),
    ]:
        run = subparsers.add_parser(command, help=help_text)
        _add_run_arguments(run)
    for command, help_text in [
        ("auto", "常駐執行 PPO／SAC 模擬交易"),
        ("rl-auto", "常駐執行 PPO／SAC 模擬交易"),
    ]:
        auto = subparsers.add_parser(command, help=help_text)
        _add_run_arguments(auto)
        auto.add_argument("--poll-seconds", type=float, default=60.0)
        auto.add_argument("--max-cycles", type=int, default=0)
        auto.add_argument("--max-consecutive-errors", type=int, default=5)
        auto.add_argument(
            "--transport",
            choices=["rest", "websocket"],
            default="rest",
            help="BTC 多週期即時模式使用 websocket；備援模式使用 REST 輪詢",
        )
        auto.add_argument("--transformer-checkpoint")
        auto.add_argument("--transformer-device", choices=["cpu", "cuda"], default="cpu")
        auto.add_argument("--transformer-min-probability", type=float, default=0.45)
        auto.add_argument("--transformer-max-uncertainty", type=float, default=0.62)
        auto.add_argument("--max-spread-bps", type=float, default=5.0)
        auto.add_argument("--stream-settle-seconds", type=float, default=2.0)
        auto.add_argument("--rolling-storage-bars", type=int, default=2_000)
        auto.add_argument(
            "--persist-rolling-data",
            action=argparse.BooleanOptionalAction,
            default=True,
        )
    for command, help_text in [
        ("status", "顯示模擬帳戶目前狀態"),
        ("auto-status", "顯示無人值守工作狀態"),
        ("stop", "要求無人值守工作安全停止"),
    ]:
        status = subparsers.add_parser(command, help=help_text)
        status.add_argument("--account", required=True)
        status.add_argument("--paper-dir", default="data/paper_trading")
    return parser


def _print_status(state: PaperAccountState) -> None:
    print(f"帳戶：{state.account_id}")
    print(f"標的：{state.symbol}（{state.exchange} / {state.interval}）")
    print(f"現金：{state.cash:.4f}")
    print(f"持倉數量：{state.quantity:.8f}")
    print(f"待執行訊號：{state.pending_signal}（{state.pending_reason}）")
    print(f"最後市場時間：{state.last_processed_timestamp or '-'}")
    print(f"最大回撤停機：{'是' if state.risk_halted else '否'}")


def _run_once(args: argparse.Namespace, policy=None):
    """執行一輪 RL 模擬交易，供手動與常駐模式共用。"""
    from ai_quant_trading.paper_trading.rl_engine import (
        download_latest_rl_paper_market_data,
        prepare_rl_paper_market_frame,
        run_rl_paper_cycle,
    )
    from ai_quant_trading.reinforcement_learning import load_rl_policy

    if not args.download_latest and not args.input:
        raise ValueError("請提供 --input，或使用 --download-latest")
    policy = policy or load_rl_policy(args.model_dir, device="cpu")
    allow_short = (
        policy.env_config.allow_short
        if args.allow_short is None
        else bool(args.allow_short)
    )
    max_short_fraction = (
        policy.env_config.max_short_fraction
        if args.max_short_fraction is None
        else float(args.max_short_fraction)
    )
    short_borrow_rate = (
        policy.env_config.short_borrow_rate_annual
        if args.short_borrow_rate_annual is None
        else float(args.short_borrow_rate_annual)
    )
    if not allow_short:
        max_short_fraction = 0.0
    paths = account_paths(args.paper_dir, args.account)
    market_path = (
        download_latest_rl_paper_market_data(
            policy,
            args.raw_dir,
            exchange=args.exchange,
            symbol=args.symbol,
            interval=args.interval,
            limit=args.limit,
            runtime_feature_dir=paths.account_dir / "market_cache",
        )
        if args.download_latest
        else Path(args.input)
    )
    since_timestamp = None
    if paths.account_json.exists():
        since_timestamp = load_account_state(paths).last_processed_timestamp
    frame = prepare_rl_paper_market_frame(
        market_path,
        policy,
        since_timestamp=since_timestamp,
    )
    return run_rl_paper_cycle(
        frame,
        policy,
        model_dir=args.model_dir,
        root_dir=args.paper_dir,
        account_id=args.account,
        exchange=args.exchange,
        symbol=args.symbol,
        interval=args.interval,
        config=PaperTradingConfig(
            initial_capital=args.initial_capital,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            position_fraction=args.position_fraction,
            allow_short=allow_short,
            max_short_fraction=max_short_fraction,
            short_borrow_rate_annual=short_borrow_rate,
        ),
        risk_config=RiskConfig(
            max_risk_per_trade=args.max_risk_per_trade,
            stop_loss_mode=args.stop_loss_mode,
            fixed_stop_loss_pct=args.fixed_stop_loss_pct,
            atr_multiplier=args.atr_multiplier,
            take_profit_pct=None if args.disable_take_profit else args.take_profit_pct,
            max_drawdown_limit=args.max_drawdown_limit,
            max_position_fraction=args.position_fraction,
        ),
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
    )


def _runner_paths(args: argparse.Namespace):
    return automation_paths(
        account_paths(args.paper_dir, args.account).account_dir / "automation"
    )


def _validate_realtime_arguments(args: argparse.Namespace) -> None:
    """先驗證 WebSocket 市場契約，避免背景程序無聲結束。"""
    if not args.transformer_checkpoint:
        raise ValueError("WebSocket 模式必須選擇 Transformer checkpoint")
    if str(args.exchange or "").lower() not in {"binance", "binance_futures"}:
        raise ValueError("WebSocket 模式目前只支援 Binance BTC 永續合約")
    if str(args.symbol or "").upper() != "BTC/USDT":
        raise ValueError("WebSocket 模式目前只支援 BTC/USDT")
    if str(args.interval or "").lower() not in BTC_MULTITIMEFRAME_INTERVALS:
        supported = "、".join(BTC_MULTITIMEFRAME_INTERVALS)
        raise ValueError(f"WebSocket 模型決策週期必須是：{supported}")


def _run_realtime(args: argparse.Namespace, policy):
    """啟動 WebSocket Transformer＋SAC/PPO 模擬工作。"""
    from ai_quant_trading.paper_trading.realtime import (
        RealtimePaperConfig,
        RealtimePaperTradingSession,
        TransformerTrendGateConfig,
    )

    _validate_realtime_arguments(args)

    allow_short = (
        policy.env_config.allow_short
        if args.allow_short is None
        else bool(args.allow_short)
    )
    max_short_fraction = (
        policy.env_config.max_short_fraction
        if args.max_short_fraction is None
        else float(args.max_short_fraction)
    )
    short_borrow_rate = (
        policy.env_config.short_borrow_rate_annual
        if args.short_borrow_rate_annual is None
        else float(args.short_borrow_rate_annual)
    )
    if not allow_short:
        max_short_fraction = 0.0
    session = RealtimePaperTradingSession(
        policy,
        RealtimePaperConfig(
            account_id=args.account,
            model_dir=Path(args.model_dir),
            raw_dir=Path(args.raw_dir),
            paper_dir=Path(args.paper_dir),
            transformer_checkpoint=Path(args.transformer_checkpoint),
            symbol=args.symbol,
            exchange=args.exchange,
            decision_interval=args.interval,
            history_bars=max(260, int(args.limit)),
            rolling_storage_bars=max(
                int(args.rolling_storage_bars),
                max(260, int(args.limit)),
            ),
            persist_rolling_data=bool(args.persist_rolling_data),
            settle_seconds=float(args.stream_settle_seconds),
            max_spread_bps=float(args.max_spread_bps),
            transformer_device=args.transformer_device,
        ),
        paper_config=PaperTradingConfig(
            initial_capital=args.initial_capital,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
            position_fraction=args.position_fraction,
            allow_short=allow_short,
            max_short_fraction=max_short_fraction,
            short_borrow_rate_annual=short_borrow_rate,
        ),
        risk_config=RiskConfig(
            max_risk_per_trade=args.max_risk_per_trade,
            stop_loss_mode=args.stop_loss_mode,
            fixed_stop_loss_pct=args.fixed_stop_loss_pct,
            atr_multiplier=args.atr_multiplier,
            take_profit_pct=None if args.disable_take_profit else args.take_profit_pct,
            max_drawdown_limit=args.max_drawdown_limit,
            max_position_fraction=args.position_fraction,
        ),
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        trend_gate=TransformerTrendGateConfig(
            minimum_probability=float(args.transformer_min_probability),
            maximum_uncertainty=float(args.transformer_max_uncertainty),
        ),
    )
    return session.run(
        _runner_paths(args),
        max_consecutive_errors=args.max_consecutive_errors,
    )


def _print_auto_status(args: argparse.Namespace) -> None:
    status = read_automation_status(_runner_paths(args))
    print(f"自動模式：{status.state}｜PID：{status.pid or '-'}")
    print(f"輪詢：{status.attempted_cycles}｜成功：{status.successful_cycles}")
    print(f"最後心跳：{status.heartbeat_at or '-'}")
    print(f"訊息：{status.last_message}")
    if status.last_error:
        print(f"錯誤：{status.last_error}")


def main(argv: list[str] | None = None) -> int:
    """執行一次、常駐推進或查詢 PPO 模擬帳戶。"""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "status":
            _print_status(load_account_state(account_paths(args.paper_dir, args.account)))
            return 0
        if args.command == "auto-status":
            _print_auto_status(args)
            return 0
        if args.command == "stop":
            request_automation_stop(_runner_paths(args))
            print("已送出停止請求。")
            return 0
        if args.command in {"auto", "rl-auto"}:
            if not args.download_latest:
                raise ValueError("無人值守模式必須使用 --download-latest")
            from ai_quant_trading.reinforcement_learning import load_rl_policy

            policy = load_rl_policy(args.model_dir, device="cpu")
            if args.transport == "websocket":
                final_status = _run_realtime(args, policy)
                _print_auto_status(args)
                return 1 if final_status.state == "failed" else 0
            final_status = run_automation(
                partial(_run_once, args, policy),
                _runner_paths(args),
                AutomationConfig(
                    poll_seconds=args.poll_seconds,
                    max_cycles=args.max_cycles,
                    max_consecutive_errors=args.max_consecutive_errors,
                ),
            )
            _print_auto_status(args)
            return 1 if final_status.state == "failed" else 0
        result = _run_once(args)
        print(result.message)
        _print_status(result.state)
        print(f"帳戶資料夾：{result.paths.account_dir}")
        return 0
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        print(f"模擬交易失敗：{exc}")
        return 1
