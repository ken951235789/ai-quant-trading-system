"""版本化候選契約：研究、標籤與回測共用既有訊號及成交引擎。"""

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
import pandas as pd

from ai_quant_trading.backtesting.complex_strategies import (
    ComplexParams, complex_frame, prepare_complex_cache,
)
from ai_quant_trading.backtesting.grid_execution import REASONS, batch_events, holding_limit
from ai_quant_trading.transformer.strategy_events import StrategyEventConfig, prepare_event_frame
from ai_quant_trading.research.candidate_features import FEATURE_SETS, apply_feature_set


FAMILIES = ("original", "trend_pullback", "vwap_reversion")


@dataclass(frozen=True)
class CandidateContract:
    """一個模型只研究一種固定策略，禁止偷偷合併不同交易標籤。"""

    family: str = "original"
    version: str = "btc_candidate_v1"
    timeframe: str = "15m"
    strict: bool = False
    stop_atr: float = 2.0
    reward_r: float = 2.0
    holding_bars: int = 32
    cooldown_bars: int = 4
    regime_exit: str = "loss"
    feature_profile: str = "context_v2"
    feature_ablation: str = "none"
    fee_bps: float = 5.0
    slippage_bps: float = 2.0
    spread_bps: float = 1.0
    funding_reserve_bps: float = 1.0
    feature_set: str = "existing"

    def __post_init__(self):
        if self.version != "btc_candidate_v1" or self.family not in FAMILIES:
            raise ValueError("未登記的候選契約或策略")
        if self.timeframe != "15m" or not isinstance(self.strict, bool):
            raise ValueError("此研究版本只接受 15m 決策與明確的 strict 布林值")
        if self.feature_profile not in {"legacy_v1", "context_v2"}:
            raise ValueError("候選特徵版本不支援")
        if self.feature_ablation not in {"none", "without_context", "without_cost"}:
            raise ValueError("特徵群消融不支援")
        if self.feature_set not in FEATURE_SETS:
            raise ValueError("未登記的候選特徵集合")
        if self.feature_set != "existing" and (self.feature_profile != "context_v2" or self.feature_ablation != "none"):
            raise ValueError("精簡特徵版本必須使用 context_v2，不能混用舊消融")
        costs = [self.fee_bps, self.slippage_bps, self.spread_bps, self.funding_reserve_bps]
        if not np.isfinite(costs).all() or min(costs) < 0 or max(costs) >= 1000:
            raise ValueError("候選成本必須有限且介於 0 至 1000 bps（不含）")
        self.params().execution()

    def params(self):
        return ComplexParams(self.family, self.timeframe, self.strict, self.stop_atr,
                             self.reward_r, self.holding_bars, self.cooldown_bars, self.regime_exit)

    def event_config(self):
        return StrategyEventConfig(max_holding_bars=self.holding_bars,
            cooldown_bars=self.cooldown_bars, stop_atr=self.stop_atr,
            target_atr=self.stop_atr * self.reward_r, spread_bps=self.spread_bps,
            funding_reserve_bps_per_settlement=self.funding_reserve_bps,
            feature_profile=self.feature_profile)

    def costs(self):
        return {"fee": self.fee_bps, "slippage": self.slippage_bps,
                "spread": self.spread_bps, "funding": self.funding_reserve_bps}

    def to_dict(self):
        value = asdict(self)
        # 預設舊路徑省略新欄位，維持已保存 checkpoint 的契約雜湊。
        if self.feature_set == "existing":
            value.pop("feature_set")
        return value

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    def metadata(self):
        value = {"parameters": self.to_dict(), "contract_sha256": self.digest,
            "research_only": True, "model_scope": "one_strategy_per_model",
            "market": "binance_usdm:BTCUSDT:perpetual", "decision_interval": "15m",
            "timestamp": "UTC_bar_open; decision=timestamp+15min",
            "signal_version": "complex_strategies_20261002",
            "execution": "next_open_market; regime_then_stop_then_target; stop_first_same_bar",
            "cost_version": "assumed_fee_slippage_spread_reserve_v1", "tax": "unverified",
            "funding": "positive_reserve_per_crossed_UTC_00_08_16_not_actual_funding",
            "rvol": "volume/currently_known_previous_20_bars_mean_excluding_current",
            "indicators": "ATR_ADX_shared_intraday; EMA_adjust_false; BB_ddof0; UTC_open_day_VWAP",
            "common_tail_purge": 65, "label_end": "exit_bar_close_conservative_upper_bound",
            "sample_weight": "train_only_average_inverse_concurrency_v1"}
        if self.feature_set != "existing":
            value["feature_contract"] = {"version": "candidate_features_v1", "set": self.feature_set,
                "flow": "closed_bar_taker_base_imbalance; previous20_activity_reference; strict_source",
                "setup": "known_prior20_prior96_previousUTCday_levels; ages_capped96",
                "selection": "predeclared_groups_then_train_only_pruning"}
        return value


def validate_training_contract(contract, training):
    """設定重複出現時必須一致，不允許訓練與回測各自解讀。"""
    if training.trading_target_mode != "strategy_event":
        raise ValueError("候選契約僅供事件研究")
    if training.strategy_event_config != contract.event_config().to_dict():
        raise ValueError("候選契約與 strategy_event_config 不一致")
    if training.fee_bps_per_side != contract.fee_bps or training.slippage_bps_per_side != contract.slippage_bps:
        raise ValueError("候選契約與訓練成本不一致")


def prepare_candidate_frame(raw, contract):
    cache = prepare_complex_cache(raw)
    signal = complex_frame(cache, contract.params())
    frame = prepare_event_frame(raw, contract.event_config())
    for column in ("event_atr", "event_regime", "candidate_side"):
        frame[column] = signal[column].to_numpy()
    # 舊特徵集合也用明確的新量能定義，特徵消融不混入指標公式的差異。
    for tf in ("15m", "1h", "4h"):
        bars = cache[tf]["bars"]
        volume = bars.volume / bars.volume.shift(1).rolling(20).mean().replace(0, np.nan)
        frame.drop(columns=[f"mtf_{tf}_relative_volume"], inplace=True)
        frame[f"mtf_{tf}_relative_volume_prev20"] = volume.reindex(cache["decisions"], method="ffill").to_numpy()
    atr_fraction = frame.event_atr / frame.close
    if contract.feature_ablation != "without_cost":
        frame["mtf_15m_cost_roundtrip_over_atr"] = (
            2 * (contract.fee_bps + contract.slippage_bps) + contract.spread_bps
        ) / 10000 / atr_fraction.replace(0, np.nan)
        minutes = cache["decisions"].hour * 60 + cache["decisions"].minute
        frame["mtf_15m_cost_minutes_to_settlement"] = (480 - minutes % 480) / 480
    if contract.feature_ablation == "without_context":
        frame.drop(columns=[c for c in frame if "_context_" in c], inplace=True)
    frame = apply_feature_set(frame, raw, contract.feature_set)
    frame.attrs["candidate_contract_sha256"] = contract.digest
    return frame


def candidate_outcomes(frame, contract):
    """所有策略標籤直接使用已驗證的批次引擎，不實作另一套撮合。"""
    result = holding_limit(batch_events(frame, contract.params().execution(), contract.costs()),
                           frame, contract.holding_bars, contract.costs())
    outcomes = pd.DataFrame(result)
    outcomes["exit_reason"] = outcomes.reason_code.map(REASONS)
    outcomes["label_end_endpoint"] = outcomes.exit_endpoint + 1
    return outcomes


def uniqueness_weights(rows, points, exits):
    """只傳入 Train 事件，平均反重疊權重不得受後續區段標籤影響。"""
    points, exits = np.asarray(points, dtype=int), np.asarray(exits, dtype=int)
    if len(points) != len(exits) or (points < 0).any() or (exits <= points).any() or (exits >= rows).any():
        raise ValueError("事件重疊區間不合法")
    delta = np.zeros(rows + 1)
    np.add.at(delta, points + 1, 1)
    np.add.at(delta, exits + 1, -1)
    concurrency = np.cumsum(delta[:-1])
    inverse = np.divide(1., concurrency, out=np.zeros(rows), where=concurrency > 0)
    prefix = np.r_[0., np.cumsum(inverse)]
    weights = np.ones(rows)
    weights[points] = (prefix[exits + 1] - prefix[points + 1]) / (exits - points)
    return weights
