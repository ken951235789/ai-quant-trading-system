"""PPO／SAC Binance 實盤交易命令列入口。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

from ai_quant_trading.automation import (
    AutomationConfig,
    automation_paths,
    read_automation_status,
    request_automation_stop,
    run_automation,
)
from ai_quant_trading.live_trading.config import LiveTradingConfig
from ai_quant_trading.live_trading.factory import build_live_gateway
from ai_quant_trading.live_trading.notification import configure_live_logger, notify_event
from ai_quant_trading.live_trading.operations import OperationalRiskLimits
from ai_quant_trading.live_trading.service import record_order_submission
from ai_quant_trading.live_trading.storage import activate_emergency_halt, live_trading_paths
from ai_quant_trading.risk import DynamicLeverageConfig, RiskConfig
from ai_quant_trading.runtime_secrets import runtime_env_path


DEFAULT_ROOT = (
    Path(sys.executable).resolve().parent
    if bool(getattr(sys, "frozen", False))
    else Path(__file__).resolve().parents[3]
)
DEFAULT_ENV_FILE = runtime_env_path(DEFAULT_ROOT)


def _add_connection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--environment", choices=["testnet", "demo", "live"], default="testnet")
    parser.add_argument(
        "--market-type",
        choices=["usd_m_futures", "spot"],
        default="usd_m_futures",
    )
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    parser.add_argument("--allowed-symbols", nargs="+", default=["BTC/USDT"])
    parser.add_argument("--max-order-quote", type=float, default=100.0)
    parser.add_argument("--max-balance-fraction", type=float, default=0.25)
    parser.add_argument("--leverage", type=int, choices=[1, 2, 3], default=2)


def _build_config(args: argparse.Namespace, *, enabled: bool = False) -> LiveTradingConfig:
    return LiveTradingConfig(
        environment=args.environment,
        market_type=args.market_type,
        trading_enabled=enabled,
        allowed_symbols=tuple(args.allowed_symbols),
        max_order_quote=args.max_order_quote,
        max_balance_fraction=args.max_balance_fraction,
        exchange_protection_enabled=not bool(
            getattr(args, "disable_exchange_protection", False)
        ),
        leverage=args.leverage,
    )


def _add_rl_cycle_arguments(parser: argparse.ArgumentParser) -> None:
    _add_connection_arguments(parser)
    parser.add_argument("--training-dir", required=True)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--input")
    data_group.add_argument("--download-latest", action="store_true")
    parser.add_argument("--symbol")
    parser.add_argument("--interval")
    parser.add_argument("--download-limit", type=int, default=500)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--entry-threshold", type=float, default=0.10)
    parser.add_argument("--exit-threshold", type=float, default=0.02)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--max-risk-per-trade", type=float, default=0.01)
    parser.add_argument("--stop-loss-mode", choices=["fixed", "atr"], default="fixed")
    parser.add_argument("--fixed-stop-loss-pct", type=float, default=0.01)
    parser.add_argument("--atr-multiplier", type=float, default=2.0)
    parser.add_argument("--take-profit-pct", type=float, default=0.015)
    parser.add_argument("--max-position-fraction", type=float, default=0.25)
    parser.add_argument("--maximum-daily-loss", type=float, default=0.01)
    parser.add_argument("--max-drawdown-limit", type=float, default=0.08)
    parser.add_argument("--disable-exchange-protection", action="store_true")
    parser.add_argument(
        "--dynamic-leverage",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--dynamic-uncalibrated-max", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--dynamic-probability-2", type=float, default=0.58)
    parser.add_argument("--dynamic-probability-3", type=float, default=0.65)
    parser.add_argument("--dynamic-risk-reward-2", type=float, default=1.25)
    parser.add_argument("--dynamic-risk-reward-3", type=float, default=1.75)
    parser.add_argument("--dynamic-uncertainty-2", type=float, default=0.45)
    parser.add_argument("--dynamic-uncertainty-3", type=float, default=0.30)
    parser.add_argument("--dynamic-drawdown-2", type=float, default=0.05)
    parser.add_argument("--dynamic-drawdown-3", type=float, default=0.025)
    parser.add_argument("--dynamic-spread-2", type=float, default=8.0)
    parser.add_argument("--dynamic-spread-3", type=float, default=4.0)
    parser.add_argument("--dynamic-volatility-2", type=float, default=0.03)
    parser.add_argument("--dynamic-volatility-3", type=float, default=0.02)
    parser.add_argument(
        "--dynamic-positive-expectancy",
        action=argparse.BooleanOptionalAction,
        default=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PPO／SAC Binance 交易工具")
    parser.add_argument(
        "--storage-root",
        default=str(DEFAULT_ROOT / "data" / "live_trading"),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="測試憑證並查看非零餘額")
    _add_connection_arguments(status)

    order = subparsers.add_parser("order", help="驗證或送出單一市場單")
    _add_connection_arguments(order)
    order.add_argument("--symbol", required=True)
    order.add_argument("--side", choices=["BUY", "SELL"], required=True)
    order.add_argument("--quote-amount", type=float)
    order.add_argument("--quantity", type=float)
    order.add_argument("--full-exit", action="store_true")
    order.add_argument("--reduce-only", action="store_true")
    order.add_argument("--execute", action="store_true")
    order.add_argument("--confirm", default="")

    rl_cycle = subparsers.add_parser("rl-cycle", help="執行 PPO／SAC 目標持倉輪次")
    _add_rl_cycle_arguments(rl_cycle)
    rl_auto = subparsers.add_parser("rl-auto", help="常駐執行 PPO／SAC 交易輪次")
    _add_rl_cycle_arguments(rl_auto)
    rl_auto.add_argument("--poll-seconds", type=float, default=60.0)
    rl_auto.add_argument("--max-cycles", type=int, default=0)
    rl_auto.add_argument("--max-consecutive-errors", type=int, default=5)

    for command, help_text in [
        ("auto-status", "查看常駐工作心跳"),
        ("stop", "要求常駐工作安全停止"),
    ]:
        item = subparsers.add_parser(command, help=help_text)
        item.add_argument(
            "--environment",
            choices=["testnet", "demo", "live"],
            default="testnet",
        )
    return parser


def _status(args: argparse.Namespace) -> int:
    gateway = build_live_gateway(_build_config(args), env_path=args.env_file)
    account = gateway.account_status()
    assets = (
        account.get("assets", [])
        if args.market_type == "usd_m_futures"
        else account.get("balances", [])
    )
    balances = [
        item
        for item in assets
        if float(item.get("walletBalance", item.get("free", 0))) != 0
        or float(item.get("availableBalance", item.get("locked", 0))) != 0
    ]
    print(f"環境：{args.environment}｜可交易：{bool(account.get('canTrade', False))}")
    for item in balances:
        if args.market_type == "usd_m_futures":
            print(
                f"{item['asset']}: wallet={item.get('walletBalance')} "
                f"available={item.get('availableBalance')}"
            )
        else:
            print(f"{item['asset']}: free={item['free']} locked={item['locked']}")
    return 0


def _order(args: argparse.Namespace) -> int:
    config = _build_config(args, enabled=args.execute)
    gateway = build_live_gateway(config, env_path=args.env_file)
    order_options = {
        "quote_amount": args.quote_amount,
        "quantity": args.quantity,
        "full_exit": args.full_exit,
    }
    if config.market_type == "usd_m_futures":
        order_options["reduce_only"] = args.reduce_only
    preview = gateway.preview_market_order(args.symbol, args.side, **order_options)
    submission = (
        gateway.submit_order(preview, args.confirm)
        if args.execute
        else gateway.validate_order(preview)
    )
    paths = live_trading_paths(args.storage_root, config.environment)
    record_order_submission(paths, config.environment, submission, None, None)
    print(
        "訂單已送出。" if args.execute else "驗證通過；未送入撮合引擎。",
        f"client_order_id={submission.client_order_id}",
    )
    return 0


def _rl_cycle_once(args: argparse.Namespace, policy=None, gateway=None):
    """下載或讀取資料後執行一個 RL 目標持倉輪次。"""
    import pandas as pd

    from ai_quant_trading.live_trading.rl_service import (
        download_latest_rl_market_data,
        run_live_rl_cycle,
    )
    from ai_quant_trading.reinforcement_learning import load_rl_policy

    config = _build_config(args, enabled=args.execute)
    if args.execute and config.environment == "live" and not args.download_latest:
        raise ValueError("正式實盤只允許 --download-latest，禁止以手動舊檔送單")
    gateway = gateway or build_live_gateway(config, env_path=args.env_file)
    policy = policy or load_rl_policy(args.training_dir, device=args.device)
    market_path = (
        download_latest_rl_market_data(
            policy,
            DEFAULT_ROOT / "data" / "raw",
            symbol=args.symbol,
            interval=args.interval,
            limit=args.download_limit,
        )
        if args.download_latest
        else Path(args.input)
    )
    risk = RiskConfig(
        max_risk_per_trade=args.max_risk_per_trade,
        stop_loss_mode=args.stop_loss_mode,
        fixed_stop_loss_pct=args.fixed_stop_loss_pct,
        atr_multiplier=args.atr_multiplier,
        take_profit_pct=args.take_profit_pct,
        max_drawdown_limit=args.max_drawdown_limit,
        max_position_fraction=args.max_position_fraction,
        dynamic_leverage=DynamicLeverageConfig(
            enabled=args.dynamic_leverage,
            max_leverage=args.leverage,
            uncalibrated_max_leverage=min(
                args.dynamic_uncalibrated_max,
                args.leverage,
            ),
            leverage_2_min_probability=args.dynamic_probability_2,
            leverage_3_min_probability=args.dynamic_probability_3,
            leverage_2_min_risk_reward=args.dynamic_risk_reward_2,
            leverage_3_min_risk_reward=args.dynamic_risk_reward_3,
            leverage_2_max_uncertainty=args.dynamic_uncertainty_2,
            leverage_3_max_uncertainty=args.dynamic_uncertainty_3,
            leverage_2_max_drawdown=args.dynamic_drawdown_2,
            leverage_3_max_drawdown=args.dynamic_drawdown_3,
            leverage_2_max_spread_bps=args.dynamic_spread_2,
            leverage_3_max_spread_bps=args.dynamic_spread_3,
            leverage_2_max_volatility=args.dynamic_volatility_2,
            leverage_3_max_volatility=args.dynamic_volatility_3,
            require_positive_expectancy=args.dynamic_positive_expectancy,
        ),
    )
    return run_live_rl_cycle(
        pd.read_csv(market_path),
        policy,
        gateway=gateway,
        storage_root=args.storage_root,
        risk_config=risk,
        execute=args.execute,
        confirmation=args.confirm,
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        operational_limits=OperationalRiskLimits(
            maximum_daily_loss=args.maximum_daily_loss,
            maximum_drawdown=args.max_drawdown_limit,
        ),
    )


def _auto_paths(args: argparse.Namespace):
    environment_dir = live_trading_paths(
        args.storage_root,
        args.environment,
    ).environment_dir
    return automation_paths(environment_dir / "automation")


def _print_auto_status(args: argparse.Namespace) -> None:
    status = read_automation_status(_auto_paths(args))
    print(f"自動模式：{status.state}｜PID：{status.pid or '-'}")
    print(f"輪詢：{status.attempted_cycles}｜成功：{status.successful_cycles}")
    print(f"最後心跳：{status.heartbeat_at or '-'}")
    print(f"訊息：{status.last_message}")
    if status.last_error:
        print(f"錯誤：{status.last_error}")


def _rl_auto(args: argparse.Namespace) -> int:
    from threading import Thread
    import time

    from ai_quant_trading.database.config import DatabaseSettings
    from ai_quant_trading.database.repository import repository_from_settings
    from ai_quant_trading.live_trading.futures_gateway import FuturesTradingGateway
    from ai_quant_trading.live_trading.reconciliation import reconcile_futures_state
    from ai_quant_trading.live_trading.user_stream import (
        BinanceFuturesUserStream,
        UserStreamConfig,
    )
    from ai_quant_trading.reinforcement_learning import load_rl_policy
    from ai_quant_trading.startup_refresh import refresh_live_news_sentiment

    if not args.download_latest:
        raise ValueError("RL 無人值守模式必須使用 --download-latest")
    if args.execute and args.disable_exchange_protection:
        raise ValueError("RL 無人值守實際下單必須啟用交易所停損停利保護")
    gateway = build_live_gateway(
        _build_config(args, enabled=args.execute),
        env_path=args.env_file,
    )
    if args.execute:
        from ai_quant_trading.operations.integrity import verify_artifact_manifest

        # SB3 模型含序列化內容，必須在反序列化前先驗證本機成品完整性。
        verify_artifact_manifest(args.training_dir, required=True)
    policy = load_rl_policy(args.training_dir, device=args.device)
    paths = live_trading_paths(
        args.storage_root,
        args.environment,
    )
    user_stream = None
    stream_thread = None
    reconcile = None
    if args.execute and isinstance(gateway, FuturesTradingGateway):
        settings = DatabaseSettings.from_environment(args.env_file)
        if not settings.enabled:
            raise ValueError("無人實盤必須啟用 PostgreSQL，才能保存與重播私有事件")
        repository = repository_from_settings(settings)
        def reconcile():
            return reconcile_futures_state(gateway, repository, paths)

        initial = reconcile()
        if not initial.consistent:
            raise RuntimeError(f"啟動對帳失敗：{initial.reason}")
        user_stream = BinanceFuturesUserStream(
            gateway.client,
            repository,
            UserStreamConfig(args.environment),
            reconcile=reconcile,
        )
        stream_thread = Thread(target=user_stream.run, name="binance-user-stream", daemon=True)
        stream_thread.start()
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            checkpoint = repository.stream_checkpoint(args.environment, "usd_m_user_data")
            if checkpoint and checkpoint["state"] == "connected":
                break
            time.sleep(0.2)
        else:
            user_stream.stop()
            raise RuntimeError("Binance 私有事件流在 20 秒內未完成連線")
    def guarded_cycle():
        if reconcile is not None:
            checkpoint = repository.stream_checkpoint(args.environment, "usd_m_user_data")
            heartbeat = checkpoint.get("heartbeat_at") if checkpoint else None
            if isinstance(heartbeat, str):
                heartbeat = datetime.fromisoformat(heartbeat.replace("Z", "+00:00"))
            if heartbeat is not None and heartbeat.tzinfo is None:
                heartbeat = heartbeat.replace(tzinfo=timezone.utc)
            heartbeat_age = (
                (datetime.now(timezone.utc) - heartbeat).total_seconds()
                if heartbeat is not None
                else float("inf")
            )
            if (
                not checkpoint
                or checkpoint.get("state") != "connected"
                or heartbeat_age > 120
            ):
                message = "Binance 私有事件流不健康，已禁止新的實盤風險"
                activate_emergency_halt(paths, message)
                raise RuntimeError(message)
            result = reconcile()
            if not result.consistent:
                message = f"週期性交易所對帳失敗：{result.reason}"
                activate_emergency_halt(paths, message)
                raise RuntimeError(message)
        ai_context = dict(policy.environment_metadata.get("ai_context", {}))
        if bool(ai_context.get("finbert_enabled", False)):
            news_status = refresh_live_news_sentiment(DEFAULT_ROOT)
            if news_status.get("status") == "failed":
                notify_event(
                    paths.notifications_log,
                    "warning",
                    f"FinBERT 新聞更新失敗，暫用既有資料：{news_status.get('message', '')}",
                )
        return _rl_cycle_once(args, policy, gateway)

    try:
        final_status = run_automation(
            guarded_cycle,
            _auto_paths(args),
            AutomationConfig(
                poll_seconds=args.poll_seconds,
                max_cycles=args.max_cycles,
                max_consecutive_errors=args.max_consecutive_errors,
            ),
            event_handler=lambda level, message: notify_event(
                paths.notifications_log,
                level,
                f"RL AUTO：{message}",
            ),
        )
    finally:
        if user_stream is not None:
            user_stream.stop()
        if stream_thread is not None:
            stream_thread.join(timeout=10)
    _print_auto_status(args)
    return 1 if final_status.state == "failed" else 0


def main(arguments: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(arguments)
    logger = configure_live_logger(Path(args.storage_root).parent.parent / "logs")
    try:
        if args.command == "status":
            return _status(args)
        if args.command == "order":
            return _order(args)
        if args.command == "rl-cycle":
            result = _rl_cycle_once(args)
            print(
                f"{result.signal.timestamp}｜{result.signal.symbol}｜"
                f"目標持倉 {result.signal.target_fraction:.2%}｜"
                f"訊號 {result.signal.action_signal}"
            )
            print(result.message)
            return 0
        if args.command == "rl-auto":
            return _rl_auto(args)
        if args.command == "stop":
            request_automation_stop(_auto_paths(args))
            print("已送出停止請求。")
            return 0
        _print_auto_status(args)
        return 0
    except (ValueError, OSError, PermissionError, RuntimeError) as exc:
        logger.exception("實盤交易命令失敗：%s", exc)
        try:
            paths = live_trading_paths(args.storage_root, args.environment)
            notify_event(
                paths.notifications_log,
                "error",
                f"實盤交易命令失敗：{type(exc).__name__}：{exc}",
                env_path=getattr(args, "env_file", DEFAULT_ENV_FILE),
            )
        except (ValueError, OSError, PermissionError, RuntimeError):
            logger.exception("實盤錯誤通知保存或傳送失敗")
        print(f"錯誤：{exc}")
        return 1
