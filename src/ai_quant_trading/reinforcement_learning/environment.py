"""Gymnasium 相容的單一標的投資組合環境。"""

from __future__ import annotations

from math import log
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from ai_quant_trading.features.builder import infer_annualization_periods
from ai_quant_trading.market_clock import interval_duration
from ai_quant_trading.reinforcement_learning.config import PortfolioEnvConfig
from ai_quant_trading.reinforcement_learning.actions import map_continuous_action
from ai_quant_trading.risk import govern_target_position


class PortfolioTradingEnv(gym.Env[np.ndarray, np.ndarray]):
    """以帶方向的目標持倉比例為動作，於下一根開盤成交。"""

    metadata = {"render_modes": []}

    def __init__(
        self,
        frame: pd.DataFrame,
        feature_columns: list[str],
        config: PortfolioEnvConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or PortfolioEnvConfig()
        self.feature_columns = list(feature_columns)
        required = ["timestamp", "open", "high", "low", "close", *self.feature_columns]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"Portfolio Environment 缺少欄位：{missing}")
        if len(frame) < 2:
            raise ValueError("Portfolio Environment 至少需要 2 根 K 線")
        if not self.feature_columns:
            raise ValueError("Portfolio Environment 至少需要一個市場特徵")

        self.frame = frame.reset_index(drop=True).copy()
        self.frame["timestamp"] = pd.to_datetime(
            self.frame["timestamp"], utc=True, errors="coerce", format="mixed"
        )
        execution_columns = [
            column
            for column in (
                "funding_rate",
                "spread_bps",
                "expected_return",
                "event_blackout",
            )
            if column in self.frame
        ]
        numeric = ["open", "high", "low", "close", *self.feature_columns, *execution_columns]
        self.frame[numeric] = self.frame[numeric].apply(pd.to_numeric, errors="coerce")
        required_numeric = ["open", "high", "low", "close", *self.feature_columns]
        if (
            self.frame["timestamp"].isna().any()
            or not np.isfinite(self.frame[required_numeric].to_numpy(dtype=float)).all()
        ):
            raise ValueError("Portfolio Environment 含有無效數值")

        previous_close = self.frame["close"].shift(1)
        true_range = pd.concat(
            [
                self.frame["high"] - self.frame["low"],
                (self.frame["high"] - previous_close).abs(),
                (self.frame["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=1).mean()
        self._atr_fraction = (
            (atr / self.frame["close"])
            .replace([np.inf, -np.inf], np.nan)
            .fillna(self.config.minimum_stop_distance)
        )

        feature_low = np.full(len(self.feature_columns), -10.0, dtype=np.float32)
        feature_high = np.full(len(self.feature_columns), 10.0, dtype=np.float32)
        short_limit = self.config.max_short_fraction if self.config.allow_short else 0.0
        portfolio_low = [0.0, -short_limit, -1.0]
        portfolio_high = [1.0 + short_limit, self.config.max_position_fraction, 0.0]
        if self.config.include_position_context:
            portfolio_low.extend([-1.0, 0.0])
            portfolio_high.extend([1.0, 1.0])
        if self.config.include_risk_context:
            portfolio_low.extend([0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0])
            portfolio_high.extend([10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        if self.config.include_trade_plan_context:
            portfolio_low.extend([-1.0, 0.0])
            portfolio_high.extend([1.0, 1.0])
        self.observation_space = spaces.Box(
            low=np.concatenate([feature_low, np.array(portfolio_low, dtype=np.float32)]),
            high=np.concatenate([feature_high, np.array(portfolio_high, dtype=np.float32)]),
            dtype=np.float32,
        )
        if self.config.normalized_action_space:
            self.action_space = spaces.Box(
                low=np.array([-1.0 if self.config.allow_short else 0.0], dtype=np.float32),
                high=np.array([1.0], dtype=np.float32),
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.Box(
                low=np.array([-short_limit], dtype=np.float32),
                high=np.array([self.config.max_position_fraction], dtype=np.float32),
                dtype=np.float32,
            )
        self._periods_per_year = max(infer_annualization_periods(self.frame), 1)
        self._index = 0
        self._episode_end = len(self.frame) - 1
        self._episode_initial_capital = self.config.initial_capital
        self._episode_slippage_rate = self.config.slippage_rate
        self._cash = self._episode_initial_capital
        self._quantity = 0.0
        self._equity_peak = self._episode_initial_capital
        self._drawdown = 0.0
        self._average_entry_price = 0.0
        self._holding_bars = 0
        self._steps = 0
        self._done = False
        self._realized_pnl = 0.0
        self._day_start_equity = self._episode_initial_capital
        self._current_day: object | None = None
        self._consecutive_losses = 0
        self._trade_entry_equity: float | None = None
        self._stop_price: float | None = None
        self._take_profit_price: float | None = None
        self._liquidation_price: float | None = None
        self._liquidation_count = 0
        self._total_funding = 0.0
        self._total_liquidation_fees = 0.0
        self._last_trade_closed = False
        self._last_closed_trade_pnl = 0.0

    def _equity(self, price: float) -> float:
        if self.config.execution_mode == "perpetual":
            unrealized = (
                self._quantity * (price - self._average_entry_price)
                if self._average_entry_price > 0
                else 0.0
            )
            return max(self._cash + unrealized, 0.0)
        return max(self._cash + self._quantity * price, 0.0)

    def _margin_used(self, price: float) -> float:
        if self.config.execution_mode != "perpetual":
            return max(abs(self._quantity) * price, 0.0)
        return abs(self._quantity) * price / self.config.leverage

    def _available_balance(self, price: float) -> float:
        if self.config.execution_mode != "perpetual":
            return max(self._cash, 0.0)
        return max(self._equity(price) - self._margin_used(price), 0.0)

    def _position_fraction(self, price: float) -> float:
        equity = max(self._equity(price), 1e-12)
        return self._quantity * price / equity

    @property
    def current_position_fraction(self) -> float:
        """提供 PPO 離散動作轉接器目前的帶方向曝險。"""
        return self._position_fraction(float(self.frame.iloc[self._index]["close"]))

    def _unrealized_return(self, price: float) -> float:
        """以實際平均成交價計算目前持倉的未實現報酬。"""
        if abs(self._quantity) <= 1e-12 or self._average_entry_price <= 0:
            return 0.0
        direction = 1.0 if self._quantity > 0 else -1.0
        return direction * (price / self._average_entry_price - 1)

    def _stop_distance(self, index: int) -> float:
        atr_distance = (
            float(self._atr_fraction.iloc[index]) * self.config.disaster_stop_atr_multiplier
        )
        return float(
            np.clip(
                atr_distance,
                self.config.minimum_stop_distance,
                self.config.maximum_stop_distance,
            )
        )

    def _liquidation_distance(self) -> float:
        return max(1 / self.config.leverage - self.config.maintenance_margin_rate, 0.0)

    def _update_protective_prices(self, index: int) -> None:
        if abs(self._quantity) <= 1e-12 or self._average_entry_price <= 0:
            self._stop_price = None
            self._take_profit_price = None
            self._liquidation_price = None
            return
        direction = 1.0 if self._quantity > 0 else -1.0
        self._stop_price = self._average_entry_price * (1 - direction * self._stop_distance(index))
        self._take_profit_price = (
            self._average_entry_price * (1 + direction * self.config.take_profit_distance)
            if self.config.take_profit_distance is not None
            else None
        )
        self._liquidation_price = self._average_entry_price * (
            1 - direction * self._liquidation_distance()
        )

    def _map_action_to_target(
        self,
        action_value: float,
        current_position: float = 0.0,
    ) -> float:
        """將 SAC 連續動作轉成目標曝險，並保留舊模型相容性。"""
        return map_continuous_action(
            action_value,
            self.config,
            current_position=current_position,
        )[0]

    def _action_intent(self, action_value: float, current_position: float) -> str:
        """提供診斷用的動作語意，不讓「續抱」再被誤認成「平倉」。"""
        return map_continuous_action(
            action_value,
            self.config,
            current_position=current_position,
        )[1]

    @staticmethod
    def _increases_risk(target: float, current: float) -> bool:
        if abs(target) <= abs(current) and target * current >= 0:
            return False
        return abs(target) > 1e-12

    def _spread_rate(self, row: pd.Series) -> float:
        spread_bps = pd.to_numeric(row.get("spread_bps"), errors="coerce")
        if pd.notna(spread_bps) and float(spread_bps) >= 0:
            return float(spread_bps) / 10_000
        return self.config.spread_rate

    def _data_gap_detected(self, index: int) -> bool:
        if index <= 0:
            return False
        interval = str(self.frame.iloc[index].get("interval", ""))
        duration = interval_duration(interval)
        if duration is None:
            timestamps = self.frame["timestamp"].diff().dropna()
            duration = timestamps.median() if not timestamps.empty else None
        if duration is None or duration <= pd.Timedelta(0):
            return False
        gap = self.frame.iloc[index]["timestamp"] - self.frame.iloc[index - 1]["timestamp"]
        return bool(gap > duration * 1.5)

    def _risk_cap_target(self, target: float, index: int) -> float:
        if self.config.execution_mode != "perpetual":
            return target
        direction_limit = (
            self.config.max_position_fraction if target >= 0 else self.config.max_short_fraction
        )
        margin_cap = self.config.max_margin_fraction * self.config.leverage
        stop_cap = self.config.risk_per_trade / max(self._stop_distance(index), 1e-12)
        allowed = min(direction_limit, margin_cap, stop_cap)
        return float(np.clip(target, -allowed, allowed))

    def _perpetual_rebalance(
        self,
        desired_quantity: float,
        reference_price: float,
        spread_rate: float,
    ) -> tuple[str, float, float, float, float]:
        previous_quantity = self._quantity
        quantity_delta = desired_quantity - previous_quantity
        if abs(quantity_delta) <= 1e-12:
            return "HOLD", reference_price, 0.0, 0.0, 0.0
        side = "BUY" if quantity_delta > 0 else "SELL"
        direction = 1.0 if quantity_delta > 0 else -1.0
        execution_rate = self._episode_slippage_rate + spread_rate / 2
        fill_price = reference_price * (1 + direction * execution_rate)
        trade_notional = abs(quantity_delta) * fill_price
        fee = trade_notional * self.config.fee_rate
        slippage_cost = abs(quantity_delta) * abs(fill_price - reference_price)

        closed_quantity = 0.0
        if previous_quantity * quantity_delta < 0:
            closed_quantity = min(abs(previous_quantity), abs(quantity_delta))
        realized = 0.0
        if closed_quantity > 0 and self._average_entry_price > 0:
            previous_direction = 1.0 if previous_quantity > 0 else -1.0
            realized = (
                closed_quantity * (fill_price - self._average_entry_price) * previous_direction
            )
        self._cash += realized - fee
        self._realized_pnl += realized - fee
        new_quantity = desired_quantity

        completed_trade = abs(previous_quantity) > 1e-12 and (
            abs(new_quantity) <= 1e-12 or previous_quantity * new_quantity < 0
        )
        if completed_trade and self._trade_entry_equity is not None:
            closing_equity = self._cash
            self._last_trade_closed = True
            self._last_closed_trade_pnl = closing_equity - self._trade_entry_equity
            if closing_equity < self._trade_entry_equity - 1e-9:
                self._consecutive_losses += 1
            else:
                self._consecutive_losses = 0
            self._trade_entry_equity = None

        self._average_entry_price = self._next_average_entry(
            previous_quantity,
            self._average_entry_price,
            quantity_delta,
            fill_price,
        )
        self._quantity = new_quantity
        if abs(previous_quantity) <= 1e-12 and abs(new_quantity) > 1e-12:
            self._trade_entry_equity = self._cash
        elif previous_quantity * new_quantity < 0:
            self._trade_entry_equity = self._cash
        return side, fill_price, trade_notional, fee, slippage_cost

    def _force_perpetual_exit(
        self,
        fill_price: float,
        *,
        liquidation: bool,
    ) -> tuple[float, float]:
        if abs(self._quantity) <= 1e-12:
            return 0.0, 0.0
        quantity = self._quantity
        notional = abs(quantity) * fill_price
        realized = quantity * (fill_price - self._average_entry_price)
        fee_rate = self.config.fee_rate + (self.config.liquidation_fee_rate if liquidation else 0.0)
        fee = notional * fee_rate
        self._cash += realized - fee
        self._realized_pnl += realized - fee
        if self._trade_entry_equity is not None:
            self._last_trade_closed = True
            self._last_closed_trade_pnl = self._cash - self._trade_entry_equity
            if self._cash < self._trade_entry_equity - 1e-9:
                self._consecutive_losses += 1
            else:
                self._consecutive_losses = 0
        if liquidation:
            self._liquidation_count += 1
            self._total_liquidation_fees += notional * self.config.liquidation_fee_rate
        self._quantity = 0.0
        self._average_entry_price = 0.0
        self._trade_entry_equity = None
        self._holding_bars = 0
        self._stop_price = None
        self._take_profit_price = None
        self._liquidation_price = None
        return notional, fee

    @staticmethod
    def _next_average_entry(
        previous_quantity: float,
        previous_average: float,
        quantity_delta: float,
        fill_price: float,
    ) -> float:
        """調倉後保留同向成本，反手時以新方向成交價重設成本。"""
        new_quantity = previous_quantity + quantity_delta
        if abs(new_quantity) <= 1e-12:
            return 0.0
        if abs(previous_quantity) <= 1e-12 or previous_quantity * new_quantity < 0:
            return fill_price
        if previous_quantity * quantity_delta > 0:
            previous_cost = abs(previous_quantity) * previous_average
            added_cost = abs(quantity_delta) * fill_price
            return (previous_cost + added_cost) / abs(new_quantity)
        return previous_average

    def _observation(self) -> np.ndarray:
        row = self.frame.iloc[self._index]
        close = float(row["close"])
        equity = max(self._equity(close), 1e-12)
        market = row[self.feature_columns].to_numpy(dtype=np.float32)
        portfolio_values = [
            np.clip(
                self._available_balance(close) / equity,
                0.0,
                1.0 + self.config.max_short_fraction,
            ),
            np.clip(
                self._quantity * close / equity,
                -self.config.max_short_fraction,
                self.config.max_position_fraction,
            ),
            np.clip(self._drawdown, -1.0, 0.0),
        ]
        if self.config.include_position_context:
            portfolio_values.extend(
                [
                    np.clip(self._unrealized_return(close), -1.0, 1.0),
                    np.clip(
                        self._holding_bars / self.config.holding_period_reference,
                        0.0,
                        1.0,
                    ),
                ]
            )
        if self.config.include_risk_context:
            margin_ratio = self._margin_used(close) / equity
            daily_return = equity / max(self._day_start_equity, 1e-12) - 1
            stop_distance = (
                abs(close / self._stop_price - 1) if self._stop_price not in {None, 0.0} else 0.0
            )
            liquidation_distance = (
                abs(close / self._liquidation_price - 1)
                if self._liquidation_price not in {None, 0.0}
                else 0.0
            )
            loss_scale = max(self.config.max_consecutive_losses, 1)
            portfolio_values.extend(
                [
                    np.clip(equity / self._episode_initial_capital, 0.0, 10.0),
                    np.clip(
                        self._realized_pnl / self._episode_initial_capital,
                        -1.0,
                        10.0,
                    ),
                    np.clip(daily_return, -1.0, 1.0),
                    np.clip(self._consecutive_losses / loss_scale, 0.0, 1.0),
                    np.clip(margin_ratio, 0.0, 1.0),
                    np.clip(stop_distance, 0.0, 1.0),
                    np.clip(liquidation_distance, 0.0, 1.0),
                ]
            )
        if self.config.include_trade_plan_context:
            entry_distance = (
                close / self._average_entry_price - 1
                if self._average_entry_price > 0 and abs(self._quantity) > 1e-12
                else 0.0
            )
            take_profit_distance = (
                abs(close / self._take_profit_price - 1)
                if self._take_profit_price not in {None, 0.0}
                else 0.0
            )
            portfolio_values.extend(
                [
                    np.clip(entry_distance, -1.0, 1.0),
                    np.clip(take_profit_distance, 0.0, 1.0),
                ]
            )
        portfolio = np.array(portfolio_values, dtype=np.float32)
        return np.concatenate([market, portfolio]).astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """重設資金與持倉，可指定 episode 起點供 Walk-forward 使用。"""
        super().reset(seed=seed)
        options = options or {}
        episode_length = self.config.episode_length or (len(self.frame) - 1)
        max_start = max(len(self.frame) - episode_length - 1, 0)
        if "start_index" in options:
            start_index = int(options["start_index"])
            if not 0 <= start_index < len(self.frame) - 1:
                raise ValueError("start_index 必須落在可執行交易的資料範圍")
        elif self.config.random_start and max_start > 0:
            start_index = int(self.np_random.integers(0, max_start + 1))
        else:
            start_index = 0

        capital_range = self.config.initial_capital_randomization
        capital_multiplier = (
            float(self.np_random.uniform(1 - capital_range, 1 + capital_range))
            if capital_range > 0
            else 1.0
        )
        self._episode_initial_capital = self.config.initial_capital * capital_multiplier
        slippage_range = self.config.slippage_randomization
        self._episode_slippage_rate = (
            max(
                self.config.slippage_rate
                + float(self.np_random.uniform(-slippage_range, slippage_range)),
                0.0,
            )
            if slippage_range > 0
            else self.config.slippage_rate
        )

        self._index = start_index
        self._episode_end = min(start_index + episode_length, len(self.frame) - 1)
        self._cash = self._episode_initial_capital
        self._quantity = 0.0
        self._equity_peak = self._episode_initial_capital
        self._drawdown = 0.0
        self._average_entry_price = 0.0
        self._holding_bars = 0
        self._steps = 0
        self._done = False
        self._realized_pnl = 0.0
        self._current_day = pd.Timestamp(self.frame.iloc[start_index]["timestamp"]).date()
        self._day_start_equity = self._episode_initial_capital
        self._consecutive_losses = 0
        self._trade_entry_equity = None
        self._stop_price = None
        self._take_profit_price = None
        self._liquidation_price = None
        self._liquidation_count = 0
        self._total_funding = 0.0
        self._total_liquidation_fees = 0.0
        self._last_trade_closed = False
        self._last_closed_trade_pnl = 0.0
        info = {
            "timestamp": pd.Timestamp(self.frame.iloc[self._index]["timestamp"]).isoformat(),
            "equity": self._episode_initial_capital,
            "target_fraction": 0.0,
            "episode_initial_capital": self._episode_initial_capital,
            "episode_slippage_rate": self._episode_slippage_rate,
        }
        return self._observation(), info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """依本期狀態決策，在下一根開盤調倉並以收盤資產計算 Reward。"""
        if self._done:
            raise RuntimeError("Episode 已結束，請先呼叫 reset()")
        self._last_trade_closed = False
        self._last_closed_trade_pnl = 0.0
        action_value = float(np.asarray(action, dtype=float).reshape(-1)[0])
        invalid_action = not np.isfinite(action_value)
        current_row = self.frame.iloc[self._index]
        next_index = self._index + 1
        next_row = self.frame.iloc[next_index]
        previous_equity = max(self._equity(float(current_row["close"])), 1e-12)
        current_day = pd.Timestamp(current_row["timestamp"]).date()
        if current_day != self._current_day:
            self._current_day = current_day
            self._day_start_equity = previous_equity
            self._consecutive_losses = 0
        next_open = float(next_row["open"])
        equity_at_open = max(self._equity(next_open), 1e-12)
        current_fraction_at_open = self._quantity * next_open / equity_at_open
        proposed_target_fraction = self._map_action_to_target(
            0.0 if invalid_action else action_value,
            current_fraction_at_open,
        )
        action_intent = (
            "INVALID_FLATTEN"
            if invalid_action
            else self._action_intent(action_value, current_fraction_at_open)
        )
        risk_decision = govern_target_position(
            proposed_target=proposed_target_fraction,
            current_position=current_fraction_at_open,
            drawdown=self._drawdown,
            holding_bars=self._holding_bars,
            max_position_fraction=self.config.max_position_fraction,
            allow_short=self.config.allow_short,
            max_short_fraction=self.config.max_short_fraction,
            hard_drawdown_limit=self.config.max_drawdown_limit,
            soft_drawdown_limit=self.config.soft_drawdown_limit,
            soft_drawdown_multiplier=self.config.soft_drawdown_multiplier,
            drawdown_curve_exponent=self.config.drawdown_curve_exponent,
            rebalance_deadband=self.config.rebalance_deadband,
            minimum_holding_bars=self.config.minimum_holding_bars,
        )
        governed_target = risk_decision.approved_target
        target_fraction = self._risk_cap_target(governed_target, self._index)
        hard_reasons = list(risk_decision.reasons)
        # 續抱動作不應因點差、資金費率或浮點誤差，每根產生極小的反向調倉。
        # 若部位實質超出風控上限，差距仍會大於容差並正常減倉。
        hold_tolerance = max(self.config.rebalance_deadband, 1e-3)
        if (
            action_intent == "HOLD_POSITION"
            and not risk_decision.halted
            and not risk_decision.reasons
            and abs(target_fraction - current_fraction_at_open) <= hold_tolerance
        ):
            target_fraction = current_fraction_at_open
        if not np.isclose(target_fraction, governed_target):
            hard_reasons.append("risk_position_cap")
        risk_increase = self._increases_risk(target_fraction, current_fraction_at_open)
        daily_return = previous_equity / max(self._day_start_equity, 1e-12) - 1
        if invalid_action:
            target_fraction = 0.0
            hard_reasons.append("invalid_action_flatten")
        elif daily_return <= -self.config.daily_loss_limit:
            target_fraction = 0.0
            hard_reasons.append("daily_loss_limit")
        elif (
            self.config.max_consecutive_losses > 0
            and self._consecutive_losses >= self.config.max_consecutive_losses
        ):
            target_fraction = 0.0
            hard_reasons.append("consecutive_loss_limit")
        elif self._data_gap_detected(self._index) and risk_increase:
            target_fraction = current_fraction_at_open
            hard_reasons.append("market_data_gap")
        spread_rate = self._spread_rate(current_row)
        if (
            self.config.max_spread_bps is not None
            and spread_rate * 10_000 > self.config.max_spread_bps
            and risk_increase
        ):
            target_fraction = current_fraction_at_open
            hard_reasons.append("spread_limit")
        if (
            self.config.max_slippage_bps is not None
            and self._episode_slippage_rate * 10_000 > self.config.max_slippage_bps
            and risk_increase
        ):
            target_fraction = current_fraction_at_open
            hard_reasons.append("slippage_limit")
        event_blackout = pd.to_numeric(current_row.get("event_blackout"), errors="coerce")
        if pd.notna(event_blackout) and float(event_blackout) >= 0.5 and risk_increase:
            target_fraction = current_fraction_at_open
            hard_reasons.append("event_blackout")
        if (
            self.config.max_atr_fraction is not None
            and float(self._atr_fraction.iloc[self._index]) > self.config.max_atr_fraction
            and risk_increase
        ):
            target_fraction = current_fraction_at_open
            hard_reasons.append("atr_volatility_limit")
        expected_return = pd.to_numeric(current_row.get("expected_return"), errors="coerce")
        if (
            self.config.minimum_gross_target_cost_multiple > 0
            and pd.notna(expected_return)
            and risk_increase
        ):
            estimated_round_trip_cost = (
                2 * self.config.fee_rate + spread_rate + 2 * self._episode_slippage_rate
            )
            required_gross_return = (
                self.config.minimum_gross_target_cost_multiple * estimated_round_trip_cost
            )
            direction = 1.0 if target_fraction >= 0 else -1.0
            if float(expected_return) * direction < required_gross_return:
                target_fraction = current_fraction_at_open
                hard_reasons.append("insufficient_gross_target")
        if self.config.minimum_net_risk_reward > 0 and risk_increase:
            if self.config.take_profit_distance is not None:
                gross_target = self.config.take_profit_distance
            elif pd.notna(expected_return):
                direction = 1.0 if target_fraction >= 0 else -1.0
                gross_target = float(expected_return) * direction
            else:
                gross_target = 0.0
            estimated_round_trip_cost = (
                2 * self.config.fee_rate + spread_rate + 2 * self._episode_slippage_rate
            )
            stop_distance = self._stop_distance(self._index)
            net_risk_reward = (gross_target - estimated_round_trip_cost) / (
                stop_distance + estimated_round_trip_cost
            )
            if net_risk_reward < self.config.minimum_net_risk_reward:
                target_fraction = current_fraction_at_open
                hard_reasons.append("insufficient_net_risk_reward")
        # 反手必須分兩次決策：本根先平倉，下一根模型仍維持反向才開新倉。
        if current_fraction_at_open * target_fraction < 0:
            target_fraction = 0.0
            hard_reasons.append("reverse_flatten_first")
        target_value = equity_at_open * target_fraction
        desired_quantity = target_value / next_open
        desired_quantity_delta = desired_quantity - self._quantity
        trade_notional = 0.0
        fee = 0.0
        slippage_cost = 0.0
        side = "HOLD"
        previous_quantity = self._quantity

        quantity_delta = 0.0
        fill_price = next_open
        if self.config.execution_mode == "perpetual":
            side, fill_price, trade_notional, fee, slippage_cost = self._perpetual_rebalance(
                desired_quantity, next_open, spread_rate
            )
            quantity_delta = self._quantity - previous_quantity
        elif desired_quantity_delta > 1e-12:
            side = "BUY"
            fill_price = next_open * (1 + self._episode_slippage_rate)
            affordable_quantity = self._cash / (fill_price * (1 + self.config.fee_rate))
            quantity_delta = min(desired_quantity_delta, affordable_quantity)
            trade_notional = quantity_delta * fill_price
            fee = trade_notional * self.config.fee_rate
            slippage_cost = quantity_delta * (fill_price - next_open)
            self._cash = max(self._cash - trade_notional - fee, 0.0)
        elif desired_quantity_delta < -1e-12:
            side = "SELL"
            fill_price = next_open * (1 - self._episode_slippage_rate)
            quantity_delta = desired_quantity_delta
            trade_notional = abs(quantity_delta) * fill_price
            fee = trade_notional * self.config.fee_rate
            slippage_cost = abs(quantity_delta) * (next_open - fill_price)
            self._cash += trade_notional - fee

        if self.config.execution_mode != "perpetual" and abs(quantity_delta) > 1e-12:
            self._average_entry_price = self._next_average_entry(
                previous_quantity,
                self._average_entry_price,
                quantity_delta,
                fill_price,
            )
            self._quantity += quantity_delta

        if abs(self._quantity) <= 1e-12:
            self._quantity = 0.0
            self._average_entry_price = 0.0
            self._holding_bars = 0
        elif abs(previous_quantity) <= 1e-12 or previous_quantity * self._quantity < 0:
            self._holding_bars = 1
        else:
            self._holding_bars += 1

        # 進場時只能使用訊號 K 線已知的 ATR，不能拿下一根尚未走完的高低價決定停損。
        self._update_protective_prices(self._index)

        next_close = float(next_row["close"])
        short_carry_cost = 0.0
        funding_cost = 0.0
        liquidation_fee = 0.0
        liquidated = False
        stop_triggered = False
        take_profit_triggered = False
        if self.config.execution_mode == "perpetual" and abs(self._quantity) > 1e-12:
            direction = 1.0 if self._quantity > 0 else -1.0
            liquidation_hit = bool(
                self._liquidation_price is not None
                and (
                    float(next_row["low"]) <= self._liquidation_price
                    if direction > 0
                    else float(next_row["high"]) >= self._liquidation_price
                )
            )
            stop_hit = bool(
                self._stop_price is not None
                and (
                    float(next_row["low"]) <= self._stop_price
                    if direction > 0
                    else float(next_row["high"]) >= self._stop_price
                )
            )
            take_profit_hit = bool(
                self._take_profit_price is not None
                and (
                    float(next_row["high"]) >= self._take_profit_price
                    if direction > 0
                    else float(next_row["low"]) <= self._take_profit_price
                )
            )
            # 災難停損理論上永遠早於強平；只有跳空穿越強平價才計為強平。
            gap_liquidation = bool(
                self._liquidation_price is not None
                and (
                    next_open <= self._liquidation_price
                    if direction > 0
                    else next_open >= self._liquidation_price
                )
            )
            if liquidation_hit and gap_liquidation:
                exit_price = next_open
                _, liquidation_fee = self._force_perpetual_exit(exit_price, liquidation=True)
                liquidated = True
                side = "LIQUIDATION"
            elif stop_hit:
                if self._stop_price is None:
                    raise RuntimeError("停損已觸發但環境缺少停損價格")
                exit_price = (
                    min(next_open, self._stop_price)
                    if direction > 0
                    else max(next_open, self._stop_price)
                )
                stop_notional, stop_fee = self._force_perpetual_exit(exit_price, liquidation=False)
                trade_notional += stop_notional
                fee += stop_fee
                stop_triggered = True
                side = "STOP"
            elif take_profit_hit:
                if self._take_profit_price is None:
                    raise RuntimeError("停利已觸發但環境缺少停利價格")
                exit_price = (
                    max(next_open, self._take_profit_price)
                    if direction > 0
                    else min(next_open, self._take_profit_price)
                )
                take_profit_notional, take_profit_fee = self._force_perpetual_exit(
                    exit_price, liquidation=False
                )
                trade_notional += take_profit_notional
                fee += take_profit_fee
                take_profit_triggered = True
                side = "TAKE_PROFIT"

        if self.config.execution_mode == "perpetual" and abs(self._quantity) > 1e-12:
            funding_value = pd.to_numeric(next_row.get("funding_rate"), errors="coerce")
            if pd.notna(funding_value):
                elapsed_hours = max(
                    (
                        pd.Timestamp(next_row["timestamp"]) - pd.Timestamp(current_row["timestamp"])
                    ).total_seconds()
                    / 3_600,
                    0.0,
                )
                funding_cost = (
                    abs(self._quantity)
                    * next_close
                    * float(funding_value)
                    * (1.0 if self._quantity > 0 else -1.0)
                    * elapsed_hours
                    / self.config.funding_interval_hours
                )
                self._cash -= funding_cost
                self._realized_pnl -= funding_cost
                self._total_funding += funding_cost
        elif self._quantity < -1e-12 and self.config.short_borrow_rate_annual > 0:
            short_carry_cost = (
                abs(self._quantity)
                * next_close
                * self.config.short_borrow_rate_annual
                / self._periods_per_year
            )
            self._cash -= short_carry_cost
        equity = max(self._equity(next_close), 1e-12)
        previous_drawdown_depth = abs(self._drawdown)
        self._equity_peak = max(self._equity_peak, equity)
        self._drawdown = equity / self._equity_peak - 1
        drawdown_increase = max(abs(self._drawdown) - previous_drawdown_depth, 0.0)
        turnover = trade_notional / equity_at_open
        log_return = log(equity / previous_equity)
        position_fraction = self._quantity * next_close / equity
        downside_penalty = self.config.downside_penalty * max(-log_return, 0.0)
        direction_limit = (
            self.config.max_position_fraction
            if position_fraction >= 0
            else self.config.max_short_fraction
        )
        concentration_excess = max(abs(position_fraction) - direction_limit * 0.5, 0.0)
        concentration_penalty = self.config.concentration_penalty * concentration_excess**2
        risk_end = self._drawdown <= -self.config.max_drawdown_limit or liquidated
        risk_termination_penalty = self.config.risk_termination_penalty if risk_end else 0.0
        reward = self.config.reward_scale * (
            log_return
            - self.config.drawdown_penalty * drawdown_increase
            - self.config.turnover_penalty * turnover
            - downside_penalty
            - concentration_penalty
            - risk_termination_penalty
        )

        self._index = next_index
        self._steps += 1
        market_end = self._index >= len(self.frame) - 1
        episode_end = self._index >= self._episode_end
        terminated = bool(market_end or risk_end)
        truncated = bool(episode_end and not terminated)
        self._done = terminated or truncated
        unrealized_return = self._unrealized_return(next_close)
        info = {
            "timestamp": pd.Timestamp(next_row["timestamp"]).isoformat(),
            "side": side,
            "action_intent": action_intent,
            "raw_action": action_value,
            "proposed_target_fraction": proposed_target_fraction,
            "target_fraction": target_fraction,
            "risk_multiplier": risk_decision.risk_multiplier,
            "risk_reasons": ",".join(dict.fromkeys(hard_reasons)),
            "position_fraction": float(
                np.clip(
                    position_fraction,
                    -self.config.max_short_fraction,
                    self.config.max_position_fraction,
                )
            ),
            "trade_notional": trade_notional,
            "fee": fee,
            "slippage_cost": slippage_cost,
            "short_carry_cost": short_carry_cost,
            "funding_cost": funding_cost,
            "liquidation_fee": liquidation_fee,
            "liquidated": liquidated,
            "stop_triggered": stop_triggered,
            "take_profit_triggered": take_profit_triggered,
            "cash": self._cash,
            "available_balance": self._available_balance(next_close),
            "quantity": self._quantity,
            "average_entry_price": self._average_entry_price,
            "unrealized_return": unrealized_return,
            "holding_bars": self._holding_bars,
            "realized_pnl": self._realized_pnl,
            "daily_return": equity / max(self._day_start_equity, 1e-12) - 1,
            "consecutive_losses": self._consecutive_losses,
            "trade_closed": self._last_trade_closed,
            "closed_trade_pnl": self._last_closed_trade_pnl,
            "margin_used": self._margin_used(next_close),
            "effective_leverage": abs(position_fraction),
            "configured_leverage": self.config.leverage,
            "stop_price": self._stop_price,
            "take_profit_price": self._take_profit_price,
            "liquidation_price": self._liquidation_price,
            "episode_initial_capital": self._episode_initial_capital,
            "episode_slippage_rate": self._episode_slippage_rate,
            "liquidation_count": self._liquidation_count,
            "total_funding": self._total_funding,
            "total_liquidation_fees": self._total_liquidation_fees,
            "equity": equity,
            "drawdown": self._drawdown,
            "log_return": log_return,
            "drawdown_penalty": self.config.drawdown_penalty * drawdown_increase,
            "turnover_penalty": self.config.turnover_penalty * turnover,
            "downside_penalty": downside_penalty,
            "concentration_penalty": concentration_penalty,
            "risk_termination_penalty": risk_termination_penalty,
            "risk_terminated": risk_end,
        }
        return self._observation(), float(reward), terminated, truncated, info


class DiscretePortfolioActionWrapper(gym.ActionWrapper):
    """把 PPO 的四個離散動作轉為維持、多、空與平倉目標。"""

    def __init__(
        self,
        env: PortfolioTradingEnv | UniversalPortfolioTradingEnv,
    ) -> None:
        super().__init__(env)
        self.action_space = spaces.Discrete(4)

    def action(self, action: int | np.ndarray) -> np.ndarray:
        value = int(np.asarray(action).reshape(-1)[0])
        if value not in {0, 1, 2, 3}:
            raise ValueError("PPO 離散動作必須是 0、1、2 或 3")
        config = self.env.config
        current = self.env.current_position_fraction
        if value == 0:
            target = current
        elif value == 1:
            target = config.max_position_fraction
        elif value == 2:
            target = -config.max_short_fraction if config.allow_short else 0.0
        else:
            target = 0.0
        if config.normalized_action_space:
            if target >= 0:
                target = target / max(config.max_position_fraction, 1e-12)
            else:
                target = target / max(config.max_short_fraction, 1e-12)
        return np.array([target], dtype=np.float32)


class UniversalPortfolioTradingEnv(gym.Env[np.ndarray, np.ndarray]):
    """每個 Episode 選擇一個市場，共用同一個連續持倉 Policy。"""

    metadata = {"render_modes": []}

    def __init__(
        self,
        market_frames: dict[str, pd.DataFrame],
        feature_columns: list[str],
        config: PortfolioEnvConfig | None = None,
        *,
        selection_mode: str = "random",
    ) -> None:
        super().__init__()
        if len(market_frames) < 2:
            raise ValueError("通用 Portfolio Environment 至少需要兩個市場")
        if selection_mode not in {"random", "cycle"}:
            raise ValueError("selection_mode 只支援 random 或 cycle")
        self.market_frames = {
            str(name): frame.reset_index(drop=True).copy() for name, frame in market_frames.items()
        }
        self.market_names = list(self.market_frames)
        self.feature_columns = list(feature_columns)
        self.config = config or PortfolioEnvConfig()
        self.selection_mode = selection_mode
        sample = PortfolioTradingEnv(
            self.market_frames[self.market_names[0]],
            self.feature_columns,
            self.config,
        )
        self.observation_space = sample.observation_space
        self.action_space = sample.action_space
        self._cycle_index = -1
        self._active_market: str | None = None
        self._active_env: PortfolioTradingEnv | None = None

    @property
    def current_position_fraction(self) -> float:
        if self._active_env is None:
            return 0.0
        return self._active_env.current_position_fraction

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """選擇市場後重設獨立帳戶，市場之間不會產生價格跳接。"""
        super().reset(seed=seed)
        options = dict(options or {})
        requested = options.pop("market", None)
        if requested is not None:
            market = str(requested)
            if market not in self.market_frames:
                raise ValueError(f"通用環境沒有市場：{market}")
        elif self.selection_mode == "random":
            market = self.market_names[int(self.np_random.integers(0, len(self.market_names)))]
        else:
            self._cycle_index = (self._cycle_index + 1) % len(self.market_names)
            market = self.market_names[self._cycle_index]
        self._active_market = market
        self._active_env = PortfolioTradingEnv(
            self.market_frames[market],
            self.feature_columns,
            self.config,
        )
        observation, info = self._active_env.reset(seed=seed, options=options)
        return observation, {**info, "market": market}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._active_env is None or self._active_market is None:
            raise RuntimeError("通用 Episode 尚未 reset")
        observation, reward, terminated, truncated, info = self._active_env.step(action)
        return (
            observation,
            reward,
            terminated,
            truncated,
            {**info, "market": self._active_market},
        )


def run_environment_diagnostic(
    env: PortfolioTradingEnv | UniversalPortfolioTradingEnv,
    max_steps: int = 12,
) -> pd.DataFrame:
    """用固定動作序列檢查環境交易、成本、Reward 與終止條件。"""
    env.reset(seed=42)
    if env.config.normalized_action_space:
        allocations = [0.0, -0.5, 0.0, 0.5, 1.0, -1.0, 0.0]
    elif env.config.allow_short:
        allocations = [0.0, -0.05, 0.0, 0.05, 0.10, -0.03, 0.0]
    else:
        allocations = [0.0, 0.25, 0.5, 0.75, 1.0, 0.5, 0.0]
    rows: list[dict[str, Any]] = []
    for index in range(max_steps):
        action = np.array([allocations[index % len(allocations)]], dtype=np.float32)
        _, reward, terminated, truncated, info = env.step(action)
        rows.append({**info, "reward": reward})
        if terminated or truncated:
            break
    return pd.DataFrame(rows)
