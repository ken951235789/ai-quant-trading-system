"""外部成交配對與有限研究資料契約；不提供交易訊號或強化學習 reward。"""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai_quant_trading.features.indicators import (
    add_trend_features,
    add_momentum_features,
    add_volatility_features,
)
from ai_quant_trading.research.external_traders import number
from ai_quant_trading.operations.integrity import verify_artifact_manifest

ZERO = Decimal(0)
SEQUENCE_LENGTH = 64
FEATURES = [
    "return_1",
    "return_4",
    "return_24",
    "ema20_gap",
    "ema60_gap",
    "rsi14_scaled",
    "macd_hist_scaled",
    "atr14_fraction",
    "bb_width",
    "volume_relative_prior20",
    "range_fraction",
    "body_fraction",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
]
SAMPLE_COLUMNS = [
    "episode_id",
    "alias",
    "side",
    "split",
    "entry_ms",
    "label_end_ms",
    "feature_available_ms",
    "feature_start_ms",
    "target_return_before_funding",
    "target_profitable_before_funding",
    "funding_status",
    "full_cost_training_eligible",
    "overlap_group",
    "uniqueness_weight",
]


def order_fills(rows):
    """逐毫秒用數量鏈還原，遇到歧義或缺口保留已知前綴並隔離餘下資料。"""
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        key = (row["order_id"], row["trade_id"])
        if key in seen:
            raise ValueError("輸入成交重複")
        seen.add(key)
        if (
            row["coin"] != "BTC"
            or row["fee_token"] != "USDC"
            or row["execution_side"] not in {"A", "B"}
        ):
            raise ValueError("成交契約不符")
        start, size, end, px = (
            number(row[k]) for k in ("start_position", "size", "end_position", "price")
        )
        if size <= 0 or px <= 0 or end != start + (size if row["execution_side"] == "B" else -size):
            raise ValueError("成交部位不守恆")
        groups[int(row["time_ms"])].append(dict(row))
    ordered, position, issue = [], None, None
    for timestamp in sorted(groups):
        remaining = groups[timestamp].copy()
        if position is None:
            # 左邊截斷時只接受唯一鏈起點；不以成交檔原始排列猜先後。
            ends = {number(r["end_position"]) for r in remaining}
            starts = {
                number(r["start_position"])
                for r in remaining
                if number(r["start_position"]) not in ends
            }
            if len(starts) != 1:
                issue = "ambiguous_initial_chain"
                break
            position = starts.pop()
        while remaining:
            matches = [r for r in remaining if number(r["start_position"]) == position]
            if len(matches) != 1:
                issue = "ambiguous_chain" if len(matches) > 1 else "position_gap"
                break
            row = matches[0]
            ordered.append(row)
            position = number(row["end_position"])
            remaining.remove(row)
        if issue:
            break
    return ordered, dict(
        issue=issue,
        input_fills=len(rows),
        ordered_fills=len(ordered),
        quarantined_fills=len(rows) - len(ordered),
    )


def pair_episodes(rows, alias):
    ordered, audit = order_fills(rows)
    episodes, legs, active = [], [], None
    for row in ordered:
        start, size, end, px, fee, pnl = (
            number(row[k])
            for k in ("start_position", "size", "end_position", "price", "fee", "closed_pnl")
        )
        sign = 1 if row["execution_side"] == "B" else -1
        close_size = min(abs(start), size) if start * sign < 0 else ZERO
        open_size = size - close_size
        pieces = []
        if close_size:
            pieces.append(("reduce", close_size, fee * close_size / size, pnl))
        if open_size:
            pieces.append(
                (
                    "increase",
                    open_size,
                    fee - (fee * close_size / size),
                    ZERO if close_size else pnl,
                )
            )
        position = start
        for action, qty, allocated_fee, allocated_pnl in pieces:
            if active is None:
                active = dict(
                    episode_id=f"{alias}_E{len(episodes) + 1:03d}",
                    alias=alias,
                    entry_ms=int(row["time_ms"]),
                    side=1 if (position or sign) > 0 else -1,
                    left_censored=position != 0,
                    cash=ZERO,
                    fee=ZERO,
                    api_pnl=ZERO,
                    opening_notional=ZERO,
                    opening_size=ZERO,
                    closing_notional=ZERO,
                    closing_size=ZERO,
                    max_size=abs(position),
                    legs=0,
                )
            position += sign * qty
            leg = dict(
                episode_id=active["episode_id"],
                alias=alias,
                time_ms=int(row["time_ms"]),
                order_id=row["order_id"],
                trade_id=row["trade_id"],
                action=action,
                signed_size=str(sign * qty),
                size=str(qty),
                price=str(px),
                fee=str(allocated_fee),
                api_closed_pnl=str(allocated_pnl),
                position_after=str(position),
            )
            legs.append(leg)
            active["cash"] -= sign * qty * px
            active["fee"] += allocated_fee
            active["api_pnl"] += allocated_pnl
            active["max_size"] = max(active["max_size"], abs(position))
            active["legs"] += 1
            active["last_ms"] = int(row["time_ms"])
            if action == "increase":
                active["opening_size"] += qty
                active["opening_notional"] += qty * px
            else:
                active["closing_size"] += qty
                active["closing_notional"] += qty * px
            if position == 0:
                episodes.append(finish(active, False))
                active = None
        if position != end:
            raise ValueError("反向成交拆分後部位不符")
    if active is not None:
        episodes.append(finish(active, True))
    # 每筆費用含反向拆分後完全守恆；不從來源偷刪難配對交易。
    if sum((number(r["fee"]) for r in ordered), ZERO) != sum(
        (number(r["fee"]) for r in legs), ZERO
    ):
        raise ValueError("配對費用不守恆")
    audit["ordered_fee"] = str(sum((number(r["fee"]) for r in ordered), ZERO))
    return episodes, legs, audit


def finish(active, right_censored):
    a = active
    complete = not a["left_censored"] and not right_censored
    error = a["cash"] - a["api_pnl"] if complete else None
    tolerance = Decimal("0.00001") + Decimal("0.000001") * a["legs"]
    reconciled = bool(complete and abs(error) <= tolerance)
    result = dict(
        episode_id=a["episode_id"],
        alias=a["alias"],
        side=a["side"],
        entry_ms=a["entry_ms"],
        exit_ms=a["last_ms"] if complete else None,
        last_ms=a["last_ms"],
        holding_hours=(a["last_ms"] - a["entry_ms"]) / 3600000 if complete else None,
        left_censored=a["left_censored"],
        right_censored=right_censored,
        complete=complete,
        accounting_reconciled=reconciled,
        cashflow_pnl_usdc=float(a["cash"]) if complete else None,
        api_closed_pnl_usdc=float(a["api_pnl"]),
        fee_usdc=float(a["fee"]),
        accounting_error_usdc=float(error) if error is not None else None,
        opening_notional_usdc=float(a["opening_notional"]),
        max_btc_size=float(a["max_size"]),
        entry_vwap=float(a["opening_notional"] / a["opening_size"]) if a["opening_size"] else None,
        exit_vwap=float(a["closing_notional"] / a["closing_size"]) if a["closing_size"] else None,
        pnl_before_funding_usdc=float(a["cash"] - a["fee"]) if complete else None,
        return_before_funding=float((a["cash"] - a["fee"]) / a["opening_notional"])
        if reconciled and a["opening_notional"]
        else None,
        funding_usdc=None,
        net_pnl_usdc=None,
        planned_rr=None,
        initial_stop=None,
        exit_reason="unknown",
        slippage_separate_usdc=None,
        spread_separate_usdc=None,
        tax_usdc=None,
        full_cost_training_eligible=False,
        legs=a["legs"],
    )
    return result


def funding_evidence(episodes, rows):
    aggregated = sum(r["delta"].get("nSamples") is not None for r in rows)
    for episode in episodes:
        # nSamples 彙總區間未知，不擅自按日期、持有時長或倉位分攤。
        episode["funding_status"] = (
            "aggregated_account_history" if aggregated else "coverage_unverified"
        )
    return dict(
        records=len(rows),
        aggregated_records=aggregated,
        observed_cash_usdc=str(sum((number(r["delta"]["usdc"]) for r in rows), ZERO)),
        attributed_cash_usdc=None,
        attribution_complete=False,
    )


def market_features(bars):
    """沿用既有指標定義，只輸出無尺度的因果市場特徵，不產生未來標籤。"""
    frame = bars.copy()
    frame["timestamp"] = pd.to_datetime(frame["open_time_utc"], utc=True)
    frame = add_volatility_features(add_momentum_features(add_trend_features(frame)), 365 * 96)
    close = frame.close
    available = frame.time_ms.to_numpy(dtype=np.int64) + 900000
    t = pd.to_datetime(available, unit="ms", utc=True)
    values = pd.DataFrame(
        dict(
            return_1=frame.return_1,
            return_4=close.pct_change(4),
            return_24=close.pct_change(24),
            ema20_gap=close / frame.ema_20 - 1,
            ema60_gap=close / frame.ema_60 - 1,
            rsi14_scaled=frame.rsi_14 / 100,
            macd_hist_scaled=frame.macd_histogram / close,
            atr14_fraction=frame.atr_14 / close,
            bb_width=frame.bb_width_20,
            volume_relative_prior20=frame.volume
            / frame.volume.shift(1).rolling(20).mean().replace(0, np.nan),
            range_fraction=(frame.high - frame.low) / close,
            body_fraction=(close - frame.open) / frame.open,
            hour_sin=np.sin(2 * np.pi * (t.hour + t.minute / 60) / 24),
            hour_cos=np.cos(2 * np.pi * (t.hour + t.minute / 60) / 24),
            weekday_sin=np.sin(2 * np.pi * t.dayofweek / 7),
            weekday_cos=np.cos(2 * np.pi * t.dayofweek / 7),
        )
    )
    return values[FEATURES].to_numpy(dtype=np.float32), available


def split_for(start, end, bounds):
    if end is None:
        return "excluded"
    for name, lo, hi in zip(("train", "validation", "research_test"), bounds[:-1], bounds[1:]):
        if lo <= start < hi:
            return name if end < hi else "purged_boundary"
    return "excluded"


def overlap_metadata(episodes):
    """相交標籤共享群組，權重為持有期間平均 inverse concurrency。"""
    good = sorted((e for e in episodes if e["dataset_eligible"]), key=lambda e: e["entry_ms"])
    group, right = 0, -1
    for e in good:
        if e["entry_ms"] > right:
            group += 1
        right = max(right, e["exit_ms"])
        e["overlap_group"] = f"G{group:03d}"
        start, end = e["entry_ms"], e["exit_ms"]
        boundaries = sorted(
            {
                start,
                end,
                *[max(start, min(end, x[k])) for x in good for k in ("entry_ms", "exit_ms")],
            }
        )
        weighted = 0.0
        for a, b in zip(boundaries, boundaries[1:]):
            count = sum(x["entry_ms"] < b and x["exit_ms"] > a for x in good)
            weighted += (b - a) / max(1, count)
        e["uniqueness_weight"] = weighted / (end - start) if end > start else 1.0


def prepare_training(episodes, bars, start, end, seq_len=SEQUENCE_LENGTH):
    x, available = market_features(bars)
    bounds = [start, start + int((end - start) * 0.6), start + int((end - start) * 0.8), end]
    sequences, metadata = [], []
    for e in episodes:
        split = split_for(e["entry_ms"], e["exit_ms"], bounds)
        endpoint = int(np.searchsorted(available, e["entry_ms"] - 1, side="right") - 1)
        begin = endpoint - seq_len + 1
        reasons = []
        if not e["accounting_reconciled"]:
            reasons.append("incomplete_or_accounting_mismatch")
        if split not in {"train", "validation", "research_test"}:
            reasons.append(split)
        if (
            begin < 0
            or endpoint >= len(x)
            or not np.isfinite(x[max(0, begin) : endpoint + 1]).all()
        ):
            reasons.append("feature_warmup")
        elif np.any(np.diff(available[begin : endpoint + 1]) != 900000):
            reasons.append("feature_gap")
        e.update(
            dataset_eligible=not reasons,
            split=split,
            exclusion_reason=";".join(reasons),
            overlap_group=None,
            uniqueness_weight=None,
        )
        if not reasons:
            sequences.append(x[begin : endpoint + 1])
            metadata.append(
                dict(
                    episode_id=e["episode_id"],
                    alias=e["alias"],
                    side=e["side"],
                    split=split,
                    entry_ms=e["entry_ms"],
                    label_end_ms=e["exit_ms"],
                    feature_available_ms=int(available[endpoint]),
                    feature_start_ms=int(available[begin]),
                    target_return_before_funding=e["return_before_funding"],
                    target_profitable_before_funding=int(e["return_before_funding"] > 0),
                    funding_status=e["funding_status"],
                    full_cost_training_eligible=False,
                )
            )
    overlap_metadata(episodes)
    lookup = {e["episode_id"]: e for e in episodes}
    for row in metadata:
        e = lookup[row["episode_id"]]
        row.update(overlap_group=e["overlap_group"], uniqueness_weight=e["uniqueness_weight"])
    return (
        np.stack(sequences)
        if sequences
        else np.empty((0, seq_len, len(FEATURES)), dtype=np.float32),
        metadata,
        bounds,
    )


def mark_to_market(episode, legs, bars):
    """只在已完成 episode 還原持倉路徑；首末事件及棒收盤，非帳戶權益。"""
    if not episode["complete"]:
        return []
    records = [r for r in legs if r["episode_id"] == episode["episode_id"]]
    marks = [
        (int(r.time_ms) + 900000, float(r.close))
        for r in bars.itertuples()
        if episode["entry_ms"] < int(r.time_ms) + 900000 < episode["exit_ms"]
    ]
    marks += [(int(r["time_ms"]), float(r["price"])) for r in records]
    answer = []
    # 同毫秒以已還原鏈的最後成交作代理 mark，不畫出不存在的多個時間順序。
    for timestamp, mark in sorted(dict(marks).items()):
        done = [r for r in records if int(r["time_ms"]) <= timestamp]
        position = sum(float(r["signed_size"]) for r in done)
        cash = -sum(float(r["signed_size"]) * float(r["price"]) for r in done)
        fee = sum(float(r["fee"]) for r in done)
        answer.append(
            dict(
                episode_id=episode["episode_id"],
                time_ms=timestamp,
                price=mark,
                position=position,
                pnl_before_funding_usdc=cash + position * mark - fee,
            )
        )
    return answer


def load_episode_dataset(folder, *, target="net_return", allow_missing_funding=False):
    """明確允許未含資金費目標才載入；禁止默默替代既有淨收益訓練目標。"""
    folder = Path(folder)
    verify_artifact_manifest(folder)
    contract = json.loads((folder / "dataset_contract.json").read_text(encoding="utf-8"))
    if target != "return_before_funding" or not allow_missing_funding:
        raise ValueError("完整淨成本尚未驗證；須明確選擇 return_before_funding 研究目標")
    if contract["feature_columns"] != FEATURES or contract["sequence_length"] != SEQUENCE_LENGTH:
        raise ValueError("資料特徵契約不相容")
    metadata = pd.read_csv(folder / "samples.csv")
    with np.load(folder / "sequences.npz", allow_pickle=False) as archive:
        x = archive["x"].copy()
    if x.shape != (len(metadata), SEQUENCE_LENGTH, len(FEATURES)) or not np.isfinite(x).all():
        raise ValueError("序列形狀或數值不合法")
    if not (metadata.feature_available_ms < metadata.entry_ms).all():
        raise ValueError("特徵含有未來資訊")
    bounds = contract["split_boundaries_ms"]
    for row in metadata.itertuples():
        if row.split != split_for(row.entry_ms, row.label_end_ms, bounds):
            raise ValueError("標籤跨越允許的切分範圍")
        if row.split not in {"train", "validation", "research_test"}:
            raise ValueError("含已排除樣本")
    if not np.isfinite(metadata.target_return_before_funding.to_numpy(dtype=float)).all():
        raise ValueError("標籤非有限值")
    return dict(
        x=x,
        side=metadata.side.to_numpy(dtype=np.float32),
        y=metadata.target_return_before_funding.to_numpy(dtype=np.float32),
        weights=metadata.uniqueness_weight.to_numpy(dtype=np.float32),
        metadata=metadata,
    )
