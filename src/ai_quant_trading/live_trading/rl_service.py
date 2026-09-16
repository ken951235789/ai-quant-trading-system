"""把 PPO／SAC 目標持倉接到既有 Binance 交易與風控流程。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from ai_quant_trading.data_collection import (
    collect_binance_data,
    collect_binance_futures_data,
    collect_binance_futures_timeframe_bundle,
)
from ai_quant_trading.data_collection.assets import normalize_crypto_symbol
from ai_quant_trading.data_collection.binance import INTERVAL_TO_MS
from ai_quant_trading.live_trading.gateway import LiveTradingGateway, PortfolioSnapshot
from ai_quant_trading.live_trading.audit import append_audit_event
from ai_quant_trading.live_trading.futures_gateway import (
    FuturesPortfolioSnapshot,
    FuturesTradingGateway,
)
from ai_quant_trading.live_trading.futures_service import run_futures_live_target_cycle
from ai_quant_trading.live_trading.capital_flows import flow_adjusted_equity
from ai_quant_trading.live_trading.operations import OperationalRiskLimits
from ai_quant_trading.live_trading.service import (
    LiveCycleResult,
    LiveSignal,
    run_live_signal_cycle,
)
from ai_quant_trading.live_trading.storage import (
    CAPITAL_FLOW_COLUMNS,
    RL_CONTEXT_COLUMNS,
    append_csv_row,
    live_trading_paths,
    load_positions,
    read_live_csv,
)
from ai_quant_trading.reinforcement_learning import (
    LoadedRLPolicy,
    ensure_registered_champion,
    ensure_rl_execution_eligible,
    latest_rl_target,
    prepare_rl_policy_market_frame,
)
from ai_quant_trading.risk import (
    PortfolioExposure,
    RiskConfig,
    calculate_period_returns,
    calculate_stop_loss,
    expert_portfolio_risk,
    govern_portfolio_target,
    govern_target_position,
    select_dynamic_leverage,
)
from ai_quant_trading.trading import (
    MarketDataGuard,
    MarketDataGuardConfig,
    ModelForecast,
    apply_target_thresholds,
    govern_model_target,
)
from ai_quant_trading.features import (
    BTC_MULTITIMEFRAME_INTERVALS,
    build_features_from_csv_batch,
    build_multitimeframe_feature_datasets,
)


@dataclass(frozen=True, slots=True)
class LiveRLPortfolioContext:
    """訓練、模擬與實盤共用語意的完整帳戶 observation。"""

    snapshot: PortfolioSnapshot | FuturesPortfolioSnapshot
    cash_ratio: float
    position_ratio: float
    drawdown: float
    unrealized_return: float
    holding_bars: int
    managed_equity: float
    equity_ratio: float
    realized_pnl_ratio: float
    daily_return: float
    consecutive_losses: int
    margin_ratio: float
    stop_distance: float
    liquidation_distance: float
    entry_price_distance: float
    take_profit_distance: float


def ensure_rl_market_execution_eligible(
    policy: LoadedRLPolicy,
    symbol: str,
    market_type: str | None = None,
) -> None:
    """Live 只允許 RL 環境實際看過的 Binance 標的。"""
    selected = normalize_crypto_symbol(symbol)
    expected_exchanges = (
        {"binance_futures"}
        if market_type == "usd_m_futures"
        else {"binance"}
        if market_type == "spot"
        else {"binance", "binance_futures"}
    )
    if policy.universal:
        markets = dict(policy.environment_metadata.get("markets", {})).values()
        eligible = {
            normalize_crypto_symbol(str(values.get("symbol", "")))
            for values in markets
            if str(values.get("exchange", "")).lower() in expected_exchanges
            and values.get("symbol")
        }
    else:
        source = dict(policy.environment_metadata.get("source", {}))
        eligible = (
            {normalize_crypto_symbol(str(source.get("symbol", "")))}
            if str(source.get("exchange", "")).lower() in expected_exchanges
            and source.get("symbol")
            else set()
        )
    if selected not in eligible:
        raise ValueError(
            f"RL 環境未使用相同 Binance 市場的 {selected} 訓練，禁止直接送到 Live"
        )


def _policy_multitimeframe_intervals(policy: LoadedRLPolicy) -> tuple[str, ...]:
    intervals = tuple(
        interval
        for interval in BTC_MULTITIMEFRAME_INTERVALS
        if any(column.startswith(f"mtf_{interval}_") for column in policy.feature_columns)
    )
    if any(column.startswith("mtf_") for column in policy.feature_columns) and len(intervals) < 2:
        raise ValueError("RL 模型包含無法辨識的多週期特徵欄位")
    return intervals


def download_latest_rl_market_data(
    policy: LoadedRLPolicy,
    raw_dir: str | Path,
    *,
    symbol: str | None = None,
    interval: str | None = None,
    limit: int = 500,
) -> Path:
    """依 RL 環境來源下載最新 Binance K 線；通用模型可指定任一白名單標的。"""
    source = dict(policy.environment_metadata.get("source", {}))
    if policy.universal and (not symbol or not interval):
        raise ValueError("通用 RL 模型必須明確指定 symbol 與 interval")
    if not policy.universal and (symbol is not None or interval is not None):
        raise ValueError("單一市場 RL 模型不可改用其他 symbol 或 interval")
    selected_symbol = str(symbol or source.get("symbol", ""))
    selected_interval = str(interval or source.get("interval", ""))
    exchange = str(source.get("exchange", "")).lower()
    if policy.universal:
        markets = dict(policy.environment_metadata.get("markets", {})).values()
        selected_market = next(
            (
                item
                for item in markets
                if normalize_crypto_symbol(str(item.get("symbol", "")))
                == normalize_crypto_symbol(selected_symbol)
                and str(item.get("interval", "")) == selected_interval
            ),
            None,
        )
        exchange = str((selected_market or {}).get("exchange", "")).lower()
    if exchange not in {"binance", "binance_futures"}:
        raise ValueError("RL 自動交易只支援 Binance BTC 現貨或 USD-M 永續資料")
    if not selected_symbol or not selected_interval:
        raise ValueError("RL 環境缺少 symbol 或 interval")
    required_intervals = _policy_multitimeframe_intervals(policy)
    if exchange == "binance_futures" and required_intervals:
        result = collect_binance_futures_timeframe_bundle(
            [selected_symbol],
            raw_dir,
            intervals=required_intervals,
            limit=max(limit, 250),
            include_market_data=True,
            include_derivatives_context=True,
        )
        if len(result.ohlcv_files) != len(required_intervals):
            raise ValueError("多週期 Futures 資料下載不完整")
        feature_dir = Path(raw_dir).resolve().parent / "live_runtime_features"
        artifacts = build_features_from_csv_batch(
            tuple(result.ohlcv_files),
            output_dir=feature_dir,
            target_horizon=1,
            drop_na=True,
        )
        fused = build_multitimeframe_feature_datasets(
            tuple(item.output_path for item in artifacts),
            selected_interval,
            feature_dir,
        )
        selected = next(
            (
                item.output_path
                for item in fused
                if item.symbol.lower() == selected_symbol.lower()
            ),
            None,
        )
        if selected is None:
            raise ValueError("多週期 Futures 資料無法建立 Live 模型輸入")
        return selected
    if exchange == "binance_futures":
        result = collect_binance_futures_data(
            [selected_symbol],
            selected_interval,
            raw_dir,
            limit=limit,
            include_market_data=True,
            include_derivatives_context=True,
        )
    else:
        result = collect_binance_data(
            [selected_symbol],
            selected_interval,
            raw_dir,
            limit=limit,
            include_market_data=False,
            include_derivatives_context=(
                "u_derivatives_context_available" in policy.feature_columns
            ),
        )
    if not result.ohlcv_files:
        raise ValueError("下載完成但沒有 RL OHLCV 檔案")
    return result.ohlcv_files[0]


def _portfolio_context(
    gateway: LiveTradingGateway | FuturesTradingGateway,
    symbol: str,
    storage_root: str | Path,
    interval: str,
) -> LiveRLPortfolioContext:
    """只把本系統管理的部位放進 RL observation，避免誤用其他資產。"""
    snapshot = gateway.portfolio(symbol)
    paths = live_trading_paths(storage_root, gateway.config.environment)
    position_key = (
        f"usd_m_futures:{snapshot.symbol}"
        if isinstance(snapshot, FuturesPortfolioSnapshot)
        else snapshot.symbol
    )
    position = load_positions(paths).get(position_key)
    if isinstance(snapshot, FuturesPortfolioSnapshot):
        managed_equity = max(float(snapshot.margin_balance), 1e-12)
        cash_ratio = float(snapshot.available_balance) / managed_equity
        position_ratio = snapshot.position_fraction
    else:
        position_value = 0.0 if position is None else position.quantity * float(snapshot.price)
        managed_equity = max(float(snapshot.quote_free) + position_value, 1e-12)
        cash_ratio = float(snapshot.quote_free) / managed_equity
        position_ratio = position_value / managed_equity
    contexts = read_live_csv(paths.rl_context_csv, RL_CONTEXT_COLUMNS)
    same_symbol_contexts = (
        contexts.loc[contexts["symbol"].astype(str).eq(snapshot.symbol)].copy()
        if not contexts.empty
        else contexts
    )
    now = pd.Timestamp.now(tz="UTC")
    capital_flows = read_live_csv(paths.capital_flows_csv, CAPITAL_FLOW_COLUMNS)
    flow_equity = flow_adjusted_equity(
        same_symbol_contexts,
        capital_flows,
        current_timestamp=now,
        current_equity=managed_equity,
    )
    drawdown = flow_equity.drawdown
    context_row = {
        "timestamp": now.isoformat(),
        "symbol": snapshot.symbol,
        "managed_equity": managed_equity,
        "cash_ratio": cash_ratio,
        "position_ratio": position_ratio,
        "drawdown": drawdown,
        "flow_adjusted_index": flow_equity.index_value,
        "net_external_flow": flow_equity.net_external_flow,
    }
    append_csv_row(paths.rl_context_csv, RL_CONTEXT_COLUMNS, context_row)
    if position is None or position.entry_price <= 0:
        unrealized = 0.0
    else:
        direction = -1.0 if position.side.upper() == "SHORT" else 1.0
        unrealized = direction * (float(snapshot.price) / position.entry_price - 1.0)
    holding_bars = 0
    if position is not None:
        opened = pd.Timestamp(position.opened_at)
        if opened.tzinfo is None:
            opened = opened.tz_localize("UTC")
        interval_seconds = INTERVAL_TO_MS.get(interval, 86_400_000) / 1000
        holding_bars = max(int((now - opened).total_seconds() // interval_seconds), 0)
    current_context = pd.DataFrame([context_row], columns=RL_CONTEXT_COLUMNS)
    same_symbol_contexts = (
        current_context
        if same_symbol_contexts.empty
        else pd.concat([same_symbol_contexts, current_context], ignore_index=True)
    )
    daily_return, _, _ = calculate_period_returns(
        same_symbol_contexts,
        current_timestamp=now,
        current_value=flow_equity.index_value,
        value_column="flow_adjusted_index",
    )

    # 連續虧損只在一段受管理部位回到空手時結算，避免每根 K 線重複計數。
    consecutive_losses = 0
    if not same_symbol_contexts.empty:
        episode = same_symbol_contexts[["position_ratio", "flow_adjusted_index"]].copy()
        episode["position_ratio"] = pd.to_numeric(episode["position_ratio"], errors="coerce")
        episode["flow_adjusted_index"] = pd.to_numeric(
            episode["flow_adjusted_index"], errors="coerce"
        )
        episode = episode.dropna()
        in_position = episode["position_ratio"].abs() > 1e-9
        entries = episode.index[in_position & ~in_position.shift(fill_value=False)]
        exits = episode.index[~in_position & in_position.shift(fill_value=False)]
        closed_returns: list[float] = []
        for entry_index in entries:
            later_exits = exits[exits > entry_index]
            if len(later_exits):
                start_value = float(episode.loc[entry_index, "flow_adjusted_index"])
                end_value = float(episode.loc[later_exits[0], "flow_adjusted_index"])
                if start_value > 0:
                    closed_returns.append(end_value / start_value - 1.0)
        for value in reversed(closed_returns):
            if value >= 0:
                break
            consecutive_losses += 1

    base_equity = managed_equity / max(flow_equity.index_value, 1e-12)
    unrealized_pnl = (
        float(snapshot.unrealized_pnl)
        if isinstance(snapshot, FuturesPortfolioSnapshot)
        else position_ratio * managed_equity * unrealized
    )
    realized_pnl_ratio = flow_equity.index_value - 1.0 - unrealized_pnl / max(
        base_equity, 1e-12
    )
    price = float(snapshot.price)
    stop_distance = (
        abs(price / position.stop_loss - 1.0)
        if position is not None and position.stop_loss > 0
        else 0.0
    )
    take_profit_distance = (
        abs(price / position.take_profit - 1.0)
        if position is not None and position.take_profit not in {None, 0.0}
        else 0.0
    )
    entry_price_distance = (
        price / position.entry_price - 1.0
        if position is not None and position.entry_price > 0
        else 0.0
    )
    margin_ratio = 0.0
    liquidation_distance = 0.0
    if isinstance(snapshot, FuturesPortfolioSnapshot):
        margin_ratio = float(snapshot.initial_margin) / managed_equity
        if abs(float(snapshot.position_quantity)) > 1e-12 and float(snapshot.liquidation_price) > 0:
            liquidation_distance = abs(price / float(snapshot.liquidation_price) - 1.0)

    return LiveRLPortfolioContext(
        snapshot=snapshot,
        cash_ratio=cash_ratio,
        position_ratio=position_ratio,
        drawdown=drawdown,
        unrealized_return=unrealized,
        holding_bars=holding_bars,
        managed_equity=managed_equity,
        equity_ratio=flow_equity.index_value,
        realized_pnl_ratio=realized_pnl_ratio,
        daily_return=daily_return,
        consecutive_losses=consecutive_losses,
        margin_ratio=margin_ratio,
        stop_distance=stop_distance,
        liquidation_distance=liquidation_distance,
        entry_price_distance=entry_price_distance,
        take_profit_distance=take_profit_distance,
    )


def run_live_rl_cycle(
    market_frame: pd.DataFrame,
    policy: LoadedRLPolicy,
    *,
    gateway: LiveTradingGateway | FuturesTradingGateway,
    storage_root: str | Path,
    risk_config: RiskConfig,
    execute: bool = False,
    confirmation: str = "",
    entry_threshold: float = 0.10,
    exit_threshold: float = 0.02,
    operational_limits: OperationalRiskLimits | None = None,
) -> LiveCycleResult:
    """執行一次 RL 輪次；動作代表目標持倉比例，不代表直接市價方向。"""
    if not 0 <= exit_threshold < entry_threshold <= 1:
        raise ValueError("RL 門檻必須滿足 0 <= exit < entry <= 1")
    source = dict(policy.environment_metadata.get("source", {}))
    symbol = str(source.get("symbol", ""))
    interval = str(source.get("interval", "1d"))
    if policy.universal:
        completed = market_frame.dropna(subset=["symbol"]) if "symbol" in market_frame else market_frame
        symbols = completed["symbol"].astype(str).dropna().unique() if "symbol" in completed else []
        symbol = str(symbols[-1]) if len(symbols) else symbol
        intervals = (
            completed["interval"].astype(str).dropna().unique()
            if "interval" in completed
            else []
        )
        interval = str(intervals[-1]) if len(intervals) else interval
    if not symbol:
        raise ValueError("RL 市場資料缺少 symbol")
    gateway_market_type = (
        "usd_m_futures" if isinstance(gateway, FuturesTradingGateway) else "spot"
    )
    # Testnet 與只驗證模式也必須同市場，避免把 Spot 訓練結果誤送到 Futures。
    ensure_rl_market_execution_eligible(policy, symbol, gateway_market_type)
    account = _portfolio_context(gateway, symbol, storage_root, interval)
    snapshot = account.snapshot
    cash = account.cash_ratio
    position = account.position_ratio
    drawdown = account.drawdown
    holding = account.holding_bars
    managed_equity = account.managed_equity
    prepared_market = prepare_rl_policy_market_frame(market_frame, policy)
    data_health = None
    if execute and gateway.config.environment == "live":
        data_health = MarketDataGuard(
            MarketDataGuardConfig(
                required_intervals=(interval,),
                minimum_contiguous_bars=20,
                settle_seconds=0.0,
                require_spread=False,
            )
        ).ensure_healthy({interval: prepared_market})
    target = latest_rl_target(
        prepared_market,
        policy,
        cash_ratio=cash,
        position_ratio=position,
        drawdown=drawdown,
        unrealized_return=account.unrealized_return,
        holding_bars=holding,
        equity_ratio=account.equity_ratio,
        realized_pnl_ratio=account.realized_pnl_ratio,
        daily_return=account.daily_return,
        consecutive_losses=account.consecutive_losses,
        margin_ratio=account.margin_ratio,
        stop_distance=account.stop_distance,
        liquidation_distance=account.liquidation_distance,
        entry_price_distance=account.entry_price_distance,
        take_profit_distance=account.take_profit_distance,
    )
    current = prepared_market.iloc[-1]
    governed_target, governance = govern_model_target(
        proposed_target=target.target_fraction,
        current_position=position,
        current=current,
        decision_id=f"{symbol}-{interval}-{target.timestamp}",
        timestamp=target.timestamp,
        symbol=symbol,
        drawdown=drawdown,
        max_long_fraction=min(
            policy.env_config.max_position_fraction,
            risk_config.max_position_fraction,
        ),
        max_short_fraction=(
            min(policy.env_config.max_short_fraction, risk_config.max_position_fraction)
            if policy.env_config.allow_short and gateway.config.market_type == "usd_m_futures"
            else 0.0
        ),
        hard_drawdown_limit=risk_config.max_drawdown_limit,
        soft_drawdown_limit=policy.env_config.soft_drawdown_limit,
    )
    thresholded_target = apply_target_thresholds(
        governed_target,
        position,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
    )
    # Futures 使用正負帶方向曝險；舊 Spot 相容路徑仍會由 Gateway 限制為 long-only。
    decision = govern_target_position(
        proposed_target=thresholded_target,
        current_position=position,
        drawdown=drawdown,
        holding_bars=holding,
        max_position_fraction=min(
            policy.env_config.max_position_fraction,
            risk_config.max_position_fraction,
        ),
        allow_short=(
            policy.env_config.allow_short
            and gateway.config.market_type == "usd_m_futures"
        ),
        max_short_fraction=min(
            policy.env_config.max_short_fraction,
            risk_config.max_position_fraction,
        ),
        hard_drawdown_limit=risk_config.max_drawdown_limit,
        soft_drawdown_limit=policy.env_config.soft_drawdown_limit,
        soft_drawdown_multiplier=policy.env_config.soft_drawdown_multiplier,
        drawdown_curve_exponent=policy.env_config.drawdown_curve_exponent,
        rebalance_deadband=policy.env_config.rebalance_deadband,
        minimum_holding_bars=policy.env_config.minimum_holding_bars,
    )
    approved_target = decision.approved_target
    paths = live_trading_paths(storage_root, gateway.config.environment)
    contexts = read_live_csv(paths.rl_context_csv, RL_CONTEXT_COLUMNS)
    symbol_contexts = (
        contexts.loc[contexts["symbol"].astype(str).eq(symbol)]
        if not contexts.empty
        else contexts
    )
    daily_return, weekly_return, monthly_return = calculate_period_returns(
        symbol_contexts,
        current_timestamp=symbol_contexts.iloc[-1]["timestamp"],
        current_value=float(symbol_contexts.iloc[-1]["flow_adjusted_index"]),
        value_column="flow_adjusted_index",
    )
    portfolio_decision = govern_portfolio_target(
        account_id=f"{gateway.config.environment}:{symbol}",
        strategy=policy.env_config.expert_kind,
        asset_group=symbol,
        account_equity=managed_equity,
        current_target=position,
        proposed_target=approved_target,
        exposures=[
            PortfolioExposure(
                account_id=f"{gateway.config.environment}:{symbol}",
                strategy=policy.env_config.expert_kind,
                asset_group=symbol,
                equity=managed_equity,
                market_value=position * managed_equity,
            )
        ],
        config=expert_portfolio_risk(policy.env_config.expert_kind),
        daily_return=daily_return,
        weekly_return=weekly_return,
        monthly_return=monthly_return,
    )
    approved_target = portfolio_decision.approved_target
    leverage_decision = None
    effective_gateway = gateway
    if isinstance(gateway, FuturesTradingGateway):
        dynamic_settings = risk_config.dynamic_leverage
        if dynamic_settings.max_leverage > gateway.config.leverage:
            dynamic_settings = replace(
                dynamic_settings,
                max_leverage=gateway.config.leverage,
                uncalibrated_max_leverage=min(
                    dynamic_settings.uncalibrated_max_leverage,
                    gateway.config.leverage,
                ),
            )
        forecast = ModelForecast.from_mapping(current)
        target_direction = (
            1.0
            if approved_target > 0
            else -1.0
            if approved_target < 0
            else 1.0 if position >= 0 else -1.0
        )
        stop_price = calculate_stop_loss(
            float(target.close),
            risk_config,
            target.atr,
            side="long" if target_direction > 0 else "short",
        )
        stop_fraction = abs(float(target.close) - stop_price) / float(target.close)
        take_profit_fraction = risk_config.take_profit_pct
        if take_profit_fraction is None:
            take_profit_fraction = max(
                target_direction * forecast.return_5,
                target_direction * forecast.return_20,
                0.0,
            )
        configured_leverage = (
            int(snapshot.leverage)
            if risk_config.dynamic_leverage.enabled
            else gateway.config.leverage
        )
        leverage_decision = select_dynamic_leverage(
            proposed_target=approved_target,
            current_position=position,
            current_leverage=configured_leverage,
            max_margin_fraction=gateway.config.max_balance_fraction,
            stop_distance_fraction=stop_fraction,
            take_profit_fraction=float(take_profit_fraction),
            fee_rate=gateway.config.estimated_fee_rate,
            slippage_rate=gateway.config.estimated_slippage_rate,
            forecast=forecast,
            drawdown=abs(min(drawdown, 0.0)),
            spread_bps=float(
                current.get("spread_bps", current.get("bid_ask_spread_bps", 0.0))
                or 0.0
            ),
            probability_calibrated=forecast.probability_calibrated,
            config=dynamic_settings,
        )
        approved_target = leverage_decision.approved_target
        effective_gateway = FuturesTradingGateway(
            gateway.client,
            replace(gateway.config, leverage=leverage_decision.selected_leverage),
        )
    append_audit_event(
        paths.audit_jsonl,
        "rl_role_decision",
        {
            "symbol": symbol,
            "signal_time": target.timestamp,
            "roles": {
                "data_engineer": data_health.to_dict() if data_health is not None else {
                    "ready": True,
                    "mode": "paper_or_testnet",
                },
                "model_monitor": target.model_input_health or {},
                "finbert_analyst": {
                    "available": float(current.get("finbert_available", 0.0)),
                    "sentiment": float(current.get("finbert_sentiment", 0.0)),
                    "confidence": float(current.get("finbert_confidence", 0.0)),
                },
                "transformer_analyst": governance,
                "sac_trader": {"proposed_target": target.target_fraction},
                "fixed_risk_manager": {
                    "approved_target": decision.approved_target,
                    "reasons": list(decision.reasons),
                },
                "portfolio_manager": {
                    "approved_target": portfolio_decision.approved_target,
                    "halted": portfolio_decision.halted,
                    "reasons": list(portfolio_decision.reasons),
                    "daily_return": daily_return,
                    "weekly_return": weekly_return,
                    "monthly_return": monthly_return,
                },
                "dynamic_leverage_manager": (
                    leverage_decision.to_dict()
                    if leverage_decision is not None
                    else {"enabled": False, "reason": "現貨市場不使用槓桿"}
                ),
            },
        },
    )
    delta = approved_target - position
    signal_value = 1 if delta > 1e-9 else -1 if delta < -1e-9 else 0
    signal = LiveSignal(
        target.timestamp,
        target.symbol or symbol,
        approved_target,
        signal_value,
        target.close,
        target.atr,
        leverage=(
            leverage_decision.selected_leverage
            if leverage_decision is not None
            else None
        ),
        required_leverage=(
            leverage_decision.required_leverage
            if leverage_decision is not None
            else None
        ),
        leverage_evidence_cap=(
            leverage_decision.evidence_cap
            if leverage_decision is not None
            else None
        ),
        leverage_win_probability=(
            leverage_decision.win_probability
            if leverage_decision is not None
            else None
        ),
        leverage_risk_reward=(
            leverage_decision.risk_reward_ratio
            if leverage_decision is not None
            else None
        ),
        leverage_net_expectancy=(
            leverage_decision.net_expectancy
            if leverage_decision is not None
            else None
        ),
        leverage_reason=(
            leverage_decision.reason_text
            if leverage_decision is not None
            else "現貨市場不使用槓桿"
        ),
    )
    target_cap = max(approved_target, min(entry_threshold, 0.01))
    effective_risk = replace(
        risk_config,
        max_position_fraction=min(risk_config.max_position_fraction, target_cap),
    )

    def ensure_execution() -> None:
        ensure_rl_execution_eligible(policy.training_metadata)
        ai_context = dict(policy.environment_metadata.get("ai_context", {}))
        if bool(ai_context.get("transformer_enabled", False)) and not ai_context.get(
            "transformer_provenance"
        ):
            raise ValueError("RL 環境缺少 Transformer 時間隔離來源證明，禁止新增實盤風險")
        ensure_rl_market_execution_eligible(
            policy,
            signal.symbol,
            gateway_market_type,
        )
        if gateway.config.environment == "live":
            registry_path = policy.environment_dir.parent.parent / "model_registry.json"
            ensure_registered_champion(
                registry_path,
                policy.training_dir,
                policy.env_config.expert_kind,
            )

    if isinstance(gateway, FuturesTradingGateway):
        return run_futures_live_target_cycle(
            signal,
            model_dir=policy.training_dir,
            gateway=effective_gateway,
            storage_root=storage_root,
            risk_config=risk_config,
            execute=execute,
            confirmation=confirmation,
            execution_eligibility=ensure_execution,
            portfolio_override=snapshot,
            operational_limits=operational_limits,
        )

    return run_live_signal_cycle(
        signal,
        model_dir=policy.training_dir,
        gateway=gateway,
        storage_root=storage_root,
        risk_config=effective_risk,
        execute=execute,
        confirmation=confirmation,
        execution_eligibility=ensure_execution,
        portfolio_override=snapshot,
        sizing_equity_override=managed_equity,
        operational_limits=operational_limits,
    )
