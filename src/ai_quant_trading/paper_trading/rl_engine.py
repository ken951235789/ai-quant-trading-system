"""使用 PPO／SAC 目標持倉推進模擬交易帳戶。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable

import pandas as pd

from ai_quant_trading.data_collection import (
    collect_binance_data,
    collect_binance_timeframe_bundle,
    collect_binance_futures_data,
    collect_binance_futures_timeframe_bundle,
    collect_yahoo_finance_data,
)
from ai_quant_trading.features import (
    build_features_from_csv_batch,
    build_multitimeframe_feature_datasets,
)
from ai_quant_trading.market_clock import BTC_MULTITIMEFRAME_INTERVALS
from ai_quant_trading.paper_trading.config import PaperTradingConfig
from ai_quant_trading.paper_trading.engine import (
    PaperCycleResult,
    _account_equity,
    _margin_used,
    _prepare_market_frame,
    _process_bar,
    _validate_account_source,
)
from ai_quant_trading.paper_trading.state import PaperAccountState
from ai_quant_trading.paper_trading.storage import (
    account_paths,
    list_paper_accounts,
    load_account_state,
    read_account_csv,
    save_account_state,
)
from ai_quant_trading.reinforcement_learning import (
    LoadedRLPolicy,
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
    ModelForecast,
    apply_target_thresholds,
    govern_model_target,
)


RLTargetAdjuster = Callable[
    [float, float, pd.Series],
    tuple[float, dict[str, object]],
]


def _effective_soft_drawdown_limit(
    policy_limit: float | None,
    hard_limit: float,
) -> float | None:
    """使用者若設定更嚴格硬停機，將模型軟門檻同步提前。"""
    if policy_limit is None:
        return None
    return min(float(policy_limit), float(hard_limit) * 0.75)


def _portable_model_reference(model_dir: str | Path) -> str:
    """專案內模型保存相對路徑，讓整個資料夾搬到其他電腦後仍可解析。"""
    path = Path(model_dir)
    if not path.is_absolute():
        return str(path)
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def _policy_market(
    policy: LoadedRLPolicy,
    exchange: str | None,
    symbol: str | None,
    interval: str | None,
) -> tuple[str, str, str]:
    """解析並驗證單一或通用 RL 模型允許的市場。"""
    source = dict(policy.environment_metadata.get("source", {}))
    if policy.universal:
        if not exchange or not symbol or not interval:
            raise ValueError("通用 RL 模型必須選擇市場、標的與 K 線週期")
        candidates = [
            dict(values) for values in dict(policy.environment_metadata.get("markets", {})).values()
        ]
        selected = next(
            (
                market
                for market in candidates
                if str(market.get("exchange", "")).lower() == exchange.lower()
                and str(market.get("symbol", "")).lower() == symbol.lower()
                and str(market.get("interval", "")).lower() == interval.lower()
            ),
            None,
        )
        if selected is None:
            raise ValueError("選擇的市場不在此通用 RL 模型訓練清單內")
        return exchange.lower(), symbol, interval

    expected = (
        str(source.get("exchange", "")).lower(),
        str(source.get("symbol", "")),
        str(source.get("interval", "")),
    )
    supplied = (exchange, symbol, interval)
    for actual, wanted in zip(supplied, expected, strict=True):
        if actual is not None and str(actual).lower() != wanted.lower():
            raise ValueError("單一市場 RL 模型不可改用其他市場或週期")
    if not all(expected):
        raise ValueError("RL environment.json 缺少市場來源")
    return expected


def _policy_multitimeframe_intervals(policy: LoadedRLPolicy) -> tuple[str, ...]:
    """由模型欄位還原必須下載的 K 線週期。"""
    intervals = tuple(
        interval
        for interval in BTC_MULTITIMEFRAME_INTERVALS
        if any(column.startswith(f"mtf_{interval}_") for column in policy.feature_columns)
    )
    has_multitimeframe_features = any(
        column.startswith("mtf_") for column in policy.feature_columns
    )
    if has_multitimeframe_features and len(intervals) < 2:
        raise ValueError("RL 模型包含無法辨識的多週期特徵欄位")
    return intervals


def _apply_rl_target_thresholds(
    proposed_target: float,
    current_position: float,
    *,
    entry_threshold: float,
    exit_threshold: float,
) -> float:
    """套用進出場遲滯，避免小訊號進場或低信心直接反手。"""
    return apply_target_thresholds(
        proposed_target,
        current_position,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
    )


def _apply_default_transformer_governance(
    proposed_target: float,
    current_position: float,
    current: pd.Series,
    account: PaperAccountState,
    *,
    drawdown: float,
) -> tuple[float, dict[str, object]]:
    """歷史回放沒有即時 Adjuster 時，仍套用相同的 Regime 與資金管理。"""
    timestamp = pd.Timestamp(current["timestamp"]).isoformat()
    return govern_model_target(
        proposed_target=proposed_target,
        current_position=current_position,
        current=current,
        decision_id=f"{account.symbol}-{account.interval}-{timestamp}",
        timestamp=timestamp,
        symbol=account.symbol,
        drawdown=drawdown,
        max_long_fraction=min(
            account.config.position_fraction,
            account.risk_config.max_position_fraction,
        ),
        max_short_fraction=(
            account.config.max_short_fraction if account.config.allow_short else 0.0
        ),
        hard_drawdown_limit=account.risk_config.max_drawdown_limit,
    )


def download_latest_rl_paper_market_data(
    policy: LoadedRLPolicy,
    raw_dir: str | Path,
    *,
    exchange: str | None = None,
    symbol: str | None = None,
    interval: str | None = None,
    limit: int = 500,
    runtime_feature_dir: str | Path | None = None,
) -> Path:
    """依 RL 訓練市場下載最新永續合約 K 線。"""
    selected_exchange, selected_symbol, selected_interval = _policy_market(
        policy, exchange, symbol, interval
    )
    required_intervals = _policy_multitimeframe_intervals(policy)
    if selected_exchange == "binance_futures" and required_intervals:
        result = collect_binance_futures_timeframe_bundle(
            [selected_symbol],
            raw_dir,
            intervals=required_intervals,
            limit=max(limit, 250),
            include_market_data=True,
            include_derivatives_context=True,
        )
        if len(result.ohlcv_files) != len(required_intervals):
            raise ValueError("多週期資料下載不完整，禁止使用缺少週期的模型輸入")
        output_dir = Path(
            runtime_feature_dir or (Path(raw_dir).resolve().parent / "paper_runtime_features")
        )
        feature_artifacts = build_features_from_csv_batch(
            tuple(result.ohlcv_files),
            output_dir=output_dir,
            target_horizon=1,
            drop_na=True,
        )
        fused_artifacts = build_multitimeframe_feature_datasets(
            tuple(artifact.output_path for artifact in feature_artifacts),
            selected_interval,
            output_dir,
        )
        selected = next(
            (
                artifact
                for artifact in fused_artifacts
                if artifact.symbol.lower() == selected_symbol.lower()
            ),
            None,
        )
        if selected is None:
            raise ValueError("多週期資料已下載，但沒有建立出指定 BTC 模型資料集")
        return selected.output_path
    if selected_exchange == "binance" and required_intervals:
        result = collect_binance_timeframe_bundle(
            [selected_symbol],
            raw_dir,
            intervals=required_intervals,
            limit=max(limit, 250),
            include_market_data=False,
            include_derivatives_context=False,
        )
        if len(result.ohlcv_files) != len(required_intervals):
            raise ValueError("舊版多週期資料下載不完整")
        output_dir = Path(
            runtime_feature_dir or (Path(raw_dir).resolve().parent / "paper_runtime_features")
        )
        feature_artifacts = build_features_from_csv_batch(
            tuple(result.ohlcv_files),
            output_dir=output_dir,
            target_horizon=1,
            drop_na=True,
        )
        fused_artifacts = build_multitimeframe_feature_datasets(
            tuple(artifact.output_path for artifact in feature_artifacts),
            selected_interval,
            output_dir,
        )
        selected = next(
            (
                artifact
                for artifact in fused_artifacts
                if artifact.symbol.lower() == selected_symbol.lower()
            ),
            None,
        )
        if selected is None:
            raise ValueError("舊版多週期資料無法建立模型資料集")
        return selected.output_path
    if selected_exchange == "binance_futures":
        result = collect_binance_futures_data(
            [selected_symbol],
            selected_interval,
            raw_dir,
            limit=limit,
            include_market_data=True,
            include_derivatives_context=True,
        )
    # 舊模型保留唯讀相容；新建環境只會使用 binance_futures。
    elif selected_exchange == "binance":
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
    elif selected_exchange in {"yahoo", "yahoo_finance"}:
        if required_intervals:
            raise ValueError("多週期模擬交易目前只支援 Binance BTC 資料")
        result = collect_yahoo_finance_data(
            [selected_symbol],
            selected_interval,
            raw_dir,
            limit=limit,
        )
    else:
        raise ValueError(f"RL 模擬交易尚未支援資料源：{selected_exchange}")
    if not result.ohlcv_files:
        raise ValueError("下載完成但沒有 RL OHLCV 檔案")
    return result.ohlcv_files[0]


def prepare_rl_paper_market_frame(
    input_path: str | Path,
    policy: LoadedRLPolicy,
    *,
    since_timestamp: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """讀取 OHLCV 並重建 RL 模型需要的特徵。"""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到 RL 市場資料：{path}")
    return prepare_rl_policy_market_frame(
        pd.read_csv(path),
        policy,
        since_timestamp=since_timestamp,
    )


def _portfolio_context(
    state: PaperAccountState,
    close_price: float,
) -> tuple[
    float,
    float,
    float,
    float,
    int,
    float,
    float,
    float,
    int,
    float,
    float,
    float,
    float,
    float,
]:
    """把模擬帳戶轉成訓練時使用的持倉 Observation。"""
    position_value = state.quantity * close_price
    equity = max(_account_equity(state, close_price), 1e-12)
    peak = max(state.equity_peak, equity)
    margin = _margin_used(state, close_price)
    available_balance = (
        max(equity - margin, 0.0) if state.config.execution_mode == "perpetual" else state.cash
    )
    cash_ratio = available_balance / equity
    position_ratio = position_value / equity
    drawdown = equity / peak - 1.0
    unrealized = (
        0.0
        if state.entry_price is None or state.entry_price <= 0
        else (1.0 if state.quantity >= 0 else -1.0) * (close_price / state.entry_price - 1.0)
    )
    daily_return = equity / max(state.day_start_equity or equity, 1e-12) - 1
    stop_distance = (
        abs(close_price / state.stop_loss - 1) if state.stop_loss not in {None, 0.0} else 0.0
    )
    liquidation_distance = (
        max(1 / max(state.position_leverage, 1) - 0.005, 0.0)
        if state.config.execution_mode == "perpetual" and abs(state.quantity) > 1e-12
        else 0.0
    )
    entry_price_distance = (
        close_price / state.entry_price - 1.0
        if state.entry_price not in {None, 0.0} and abs(state.quantity) > 1e-12
        else 0.0
    )
    take_profit_distance = (
        abs(close_price / state.take_profit - 1.0) if state.take_profit not in {None, 0.0} else 0.0
    )
    return (
        cash_ratio,
        position_ratio,
        drawdown,
        unrealized,
        state.holding_periods,
        equity / state.config.initial_capital,
        state.realized_pnl / state.config.initial_capital,
        daily_return,
        state.consecutive_losses,
        margin / equity,
        stop_distance,
        liquidation_distance,
        entry_price_distance,
        take_profit_distance,
    )


def _paper_portfolio_exposures(
    root_dir: str | Path,
    current_state: PaperAccountState,
    current_close: float,
) -> list[PortfolioExposure]:
    """讀取所有模擬帳戶的最後權益，供跨策略總曝險風控使用。"""
    exposures: list[PortfolioExposure] = []
    for paths in list_paper_accounts(root_dir):
        try:
            state = (
                current_state
                if paths.account_dir.name == current_state.account_id
                else load_account_state(paths)
            )
            if state.account_id == current_state.account_id:
                close = current_close
                market_value = state.quantity * close
                equity = _account_equity(state, close)
            else:
                performance = read_account_csv(paths.performance_csv)
                if performance.empty:
                    close = state.entry_price or 0.0
                    market_value = state.quantity * close
                    equity = _account_equity(state, close)
                else:
                    latest = performance.iloc[-1]
                    close = float(latest.get("close", state.entry_price or 0.0))
                    market_value = state.quantity * close
                    equity = float(latest.get("equity", state.cash + market_value))
            if equity <= 0:
                continue
            exposures.append(
                PortfolioExposure(
                    account_id=state.account_id,
                    strategy=state.expert_kind,
                    asset_group=state.symbol,
                    equity=equity,
                    market_value=market_value,
                )
            )
        except (OSError, ValueError, TypeError):
            # 單一損壞帳戶不應中斷其他模擬帳戶，但也不把它計入可用資金。
            continue
    return exposures


def run_rl_paper_cycle(
    frame: pd.DataFrame,
    policy: LoadedRLPolicy,
    *,
    model_dir: str | Path,
    root_dir: str | Path,
    account_id: str,
    exchange: str | None = None,
    symbol: str | None = None,
    interval: str | None = None,
    config: PaperTradingConfig | None = None,
    risk_config: RiskConfig | None = None,
    entry_threshold: float = 0.10,
    exit_threshold: float = 0.02,
    target_adjuster: RLTargetAdjuster | None = None,
) -> PaperCycleResult:
    """逐根處理新 K 線，將 RL 目標比例轉為下一根開盤訊號。"""
    if not 0 <= exit_threshold < entry_threshold <= 1:
        raise ValueError("RL 門檻必須滿足 0 <= 離場 < 進場 <= 1")
    selected_exchange, selected_symbol, selected_interval = _policy_market(
        policy, exchange, symbol, interval
    )
    paths = account_paths(root_dir, account_id)
    initialized = not paths.account_json.exists()
    if initialized:
        account_config = config or PaperTradingConfig(
            initial_capital=policy.env_config.initial_capital,
            fee_rate=policy.env_config.fee_rate,
            slippage_rate=policy.env_config.slippage_rate,
            position_fraction=policy.env_config.max_position_fraction,
            allow_short=policy.env_config.allow_short,
            max_short_fraction=(
                policy.env_config.max_short_fraction if policy.env_config.allow_short else 0.0
            ),
            short_borrow_rate_annual=policy.env_config.short_borrow_rate_annual,
            execution_mode=policy.env_config.execution_mode,
            leverage=policy.env_config.leverage,
            max_margin_fraction=policy.env_config.max_margin_fraction,
            daily_loss_limit=policy.env_config.daily_loss_limit,
            max_consecutive_losses=policy.env_config.max_consecutive_losses,
        )
        if config is not None and policy.env_config.execution_mode == "perpetual":
            account_config = replace(
                config,
                execution_mode="perpetual",
                leverage=policy.env_config.leverage,
                max_margin_fraction=policy.env_config.max_margin_fraction,
                daily_loss_limit=policy.env_config.daily_loss_limit,
                max_consecutive_losses=policy.env_config.max_consecutive_losses,
            )
        state = PaperAccountState.create(
            account_id=account_id,
            model_dir=_portable_model_reference(model_dir),
            exchange=selected_exchange,
            symbol=selected_symbol,
            interval=selected_interval,
            config=account_config,
            risk_config=risk_config or RiskConfig(),
            model_kind="rl",
            expert_kind=policy.env_config.expert_kind,
            rl_entry_threshold=entry_threshold,
            rl_exit_threshold=exit_threshold,
        )
    else:
        state = load_account_state(paths)
        if state.model_kind != "rl":
            raise ValueError("既有帳戶不是強化學習帳戶，請建立新的 RL 模擬帳戶")
        if Path(state.model_dir).name != Path(model_dir).name:
            raise ValueError("既有帳戶不可更換 RL 模型；請建立新帳戶")
        entry_threshold = state.rl_entry_threshold
        exit_threshold = state.rl_exit_threshold

    # 先一次完成技術、FinBERT 與 Transformer 特徵，確保 SAC 與後續治理角色
    # 讀取的是同一份時間點資料；也避免每個角色各自重建造成結果不一致。
    policy_market = prepare_rl_policy_market_frame(
        frame,
        policy,
        since_timestamp=state.last_processed_timestamp,
    )
    market = _prepare_market_frame(policy_market, state.risk_config)
    if state.last_processed_timestamp is not None:
        newer = market.loc[market["timestamp"] > pd.Timestamp(state.last_processed_timestamp)]
        if newer.empty:
            return PaperCycleResult(
                state,
                paths,
                False,
                False,
                0,
                state.pending_signal,
                state.pending_target_fraction,
                "最新 K 線已處理，沒有重複下單。",
            )
    else:
        newer = market.tail(1)

    for _, row in newer.iterrows():
        _validate_account_source(state, row)

    executed_signal = 0
    policy_signal = 0
    target_fraction: float | None = None
    for sequence, (index, row) in enumerate(newer.iterrows()):
        history = market.loc[:index]

        def prediction_provider(account: PaperAccountState, current: pd.Series) -> pd.Series:
            (
                cash,
                position,
                drawdown,
                unrealized,
                holding,
                equity_ratio,
                realized_pnl_ratio,
                daily_return,
                consecutive_losses,
                margin_ratio,
                stop_distance,
                liquidation_distance,
                entry_price_distance,
                take_profit_distance,
            ) = _portfolio_context(account, float(current["close"]))
            target = latest_rl_target(
                history,
                policy,
                cash_ratio=cash,
                position_ratio=position,
                drawdown=drawdown,
                unrealized_return=unrealized,
                holding_bars=holding,
                equity_ratio=equity_ratio,
                realized_pnl_ratio=realized_pnl_ratio,
                daily_return=daily_return,
                consecutive_losses=consecutive_losses,
                margin_ratio=margin_ratio,
                stop_distance=stop_distance,
                liquidation_distance=liquidation_distance,
                entry_price_distance=entry_price_distance,
                take_profit_distance=take_profit_distance,
            )
            model_health = target.model_input_health or {}
            adjusted_target = target.target_fraction
            adjustment: dict[str, object] = {}
            if target_adjuster is not None:
                adjusted_target, adjustment = target_adjuster(
                    target.target_fraction,
                    position,
                    current,
                )
            else:
                adjusted_target, adjustment = _apply_default_transformer_governance(
                    target.target_fraction,
                    position,
                    current,
                    account,
                    drawdown=drawdown,
                )
            thresholded_target = _apply_rl_target_thresholds(
                adjusted_target,
                position,
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
            )
            decision = govern_target_position(
                proposed_target=thresholded_target,
                current_position=position,
                drawdown=drawdown,
                holding_bars=holding,
                max_position_fraction=min(
                    policy.env_config.max_position_fraction,
                    account.config.position_fraction,
                    account.risk_config.max_position_fraction,
                ),
                allow_short=(policy.env_config.allow_short and account.config.allow_short),
                max_short_fraction=min(
                    policy.env_config.max_short_fraction,
                    account.config.max_short_fraction,
                ),
                hard_drawdown_limit=account.risk_config.max_drawdown_limit,
                soft_drawdown_limit=_effective_soft_drawdown_limit(
                    policy.env_config.soft_drawdown_limit,
                    account.risk_config.max_drawdown_limit,
                ),
                soft_drawdown_multiplier=policy.env_config.soft_drawdown_multiplier,
                drawdown_curve_exponent=policy.env_config.drawdown_curve_exponent,
                rebalance_deadband=policy.env_config.rebalance_deadband,
                minimum_holding_bars=policy.env_config.minimum_holding_bars,
            )
            account_equity = max(
                _account_equity(account, float(current["close"])),
                1e-12,
            )
            daily_return, weekly_return, monthly_return = calculate_period_returns(
                read_account_csv(paths.performance_csv),
                current_timestamp=current["timestamp"],
                current_value=account_equity,
            )
            portfolio_decision = govern_portfolio_target(
                account_id=account.account_id,
                strategy=account.expert_kind,
                asset_group=account.symbol,
                account_equity=account_equity,
                current_target=position,
                proposed_target=decision.approved_target,
                exposures=_paper_portfolio_exposures(
                    root_dir,
                    account,
                    float(current["close"]),
                ),
                config=expert_portfolio_risk(account.expert_kind),
                daily_return=daily_return,
                weekly_return=weekly_return,
                monthly_return=monthly_return,
            )
            approved = portfolio_decision.approved_target
            leverage_details: dict[str, object] = {}
            if account.config.execution_mode == "perpetual":
                forecast_values = dict(current)
                forecast_values.update(adjustment)
                forecast = ModelForecast.from_mapping(forecast_values)
                direction = 1.0 if approved >= 0 else -1.0
                close_price = float(current["close"])
                stop_price = calculate_stop_loss(
                    close_price,
                    account.risk_config,
                    float(target.atr) if target.atr is not None else None,
                    side="long" if direction > 0 else "short",
                )
                stop_fraction = abs(close_price - stop_price) / close_price
                take_profit_fraction = account.risk_config.take_profit_pct
                if take_profit_fraction is None:
                    directional_returns = (
                        direction * forecast.return_5,
                        direction * forecast.return_20,
                    )
                    take_profit_fraction = max(*directional_returns, 0.0)
                leverage_decision = select_dynamic_leverage(
                    proposed_target=approved,
                    current_position=position,
                    current_leverage=account.position_leverage,
                    max_margin_fraction=account.config.max_margin_fraction,
                    stop_distance_fraction=stop_fraction,
                    take_profit_fraction=float(take_profit_fraction),
                    fee_rate=account.config.fee_rate,
                    slippage_rate=account.config.slippage_rate,
                    forecast=forecast,
                    drawdown=abs(min(drawdown, 0.0)),
                    spread_bps=float(
                        current.get(
                            "spread_bps",
                            current.get("bid_ask_spread_bps", 0.0),
                        )
                        or 0.0
                    ),
                    probability_calibrated=forecast.probability_calibrated,
                    config=account.risk_config.dynamic_leverage,
                )
                approved = leverage_decision.approved_target
                leverage_details = {
                    "dynamic_leverage_enabled": leverage_decision.enabled,
                    "selected_leverage": leverage_decision.selected_leverage,
                    "required_leverage": leverage_decision.required_leverage,
                    "leverage_evidence_cap": leverage_decision.evidence_cap,
                    "leverage_win_probability": leverage_decision.win_probability,
                    "leverage_risk_reward": leverage_decision.risk_reward_ratio,
                    "leverage_net_expectancy": leverage_decision.net_expectancy,
                    "leverage_reason": leverage_decision.reason_text,
                }
            guard_reasons = [str(adjustment.get("decision_guard_reason", "")).strip()]
            guard_reasons.extend(decision.reasons)
            guard_reasons.extend(portfolio_decision.reasons)
            if leverage_details:
                guard_reasons.append(str(leverage_details["leverage_reason"]))
            adjustment["decision_guard_reason"] = "；".join(
                value for value in dict.fromkeys(guard_reasons) if value
            ) or "RL policy"
            signal = 1 if approved > position + 1e-9 else 0
            if approved < position - 1e-9:
                signal = -1
            return pd.Series(
                {
                    "action_signal": signal,
                    "target_fraction": approved,
                    "finbert_sentiment": current.get("finbert_sentiment", 0.0),
                    "finbert_confidence": current.get("finbert_confidence", 0.0),
                    "finbert_available": current.get("finbert_available", 0.0),
                    "model_target_fraction": target.target_fraction,
                    "transformer_target_fraction": adjusted_target,
                    "threshold_target_fraction": thresholded_target,
                    "atr_14": target.atr,
                    "model_input_drifted": model_health.get("drifted", False),
                    "model_input_severe_drift": model_health.get("severe_drift", False),
                    "model_input_extreme_fraction": model_health.get(
                        "extreme_value_fraction", 0.0
                    ),
                    "model_input_drift_score": model_health.get("drift_score", 0.0),
                    "model_input_drift_multiplier": model_health.get(
                        "risk_multiplier", 1.0
                    ),
                    "target_risk_multiplier": decision.risk_multiplier,
                    "portfolio_gross_exposure": portfolio_decision.gross_exposure,
                    "portfolio_net_exposure": portfolio_decision.net_exposure,
                    "portfolio_halted": portfolio_decision.halted,
                    **leverage_details,
                    **adjustment,
                }
            )

        executed_signal, policy_signal, target_fraction = _process_bar(
            state,
            row,
            None,
            paths,
            execute_pending=not initialized or sequence > 0,
            prediction_provider=prediction_provider,
        )
    save_account_state(paths, state)
    message = "RL 模擬帳戶已初始化，目標持倉會在下一根 K 線開盤執行。"
    if not initialized:
        message = f"已依序處理 {len(newer)} 根新 K 線並保存 RL 模擬帳戶。"
    return PaperCycleResult(
        state,
        paths,
        True,
        initialized,
        executed_signal,
        policy_signal,
        target_fraction,
        message,
    )
