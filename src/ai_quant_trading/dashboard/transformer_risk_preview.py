"""在監控頁執行不送單的 Transformer 單次風控預測。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ai_quant_trading.data_collection.csv_storage import canonical_ohlcv_path
from ai_quant_trading.market_clock import BTC_MULTITIMEFRAME_INTERVALS
from ai_quant_trading.paper_trading.realtime import (
    TransformerTrendGateConfig,
    build_realtime_multitimeframe_frame,
    classify_transformer_trend,
)
from ai_quant_trading.transformer import infer_latest_transformer_context


@dataclass(frozen=True, slots=True)
class TransformerRiskPreview:
    """最新已收盤 K 線的 Transformer 方向與風控摘要。"""

    timestamp: str
    close: float
    trend: str
    risk_conclusion: str
    expected_return_1: float
    expected_return_5: float
    expected_return_20: float
    expected_volatility: float
    bull_probability: float
    bear_probability: float
    uncertainty: float
    checkpoint_name: str


def _risk_conclusion(trend: str) -> str:
    """沿用交易機器人的保守門檻，把方向轉成可讀的風控結論。"""
    conclusions = {
        "偏多": "可供強化學習模型確認多頭；空頭只允許減倉或平倉",
        "偏空": "可供強化學習模型確認空頭；多頭只允許減倉或平倉",
        "中性": "不建立或增加部位，只允許減倉或平倉",
        "不可用": "預測不可用，不增加任何市場風險",
    }
    return conclusions.get(trend, "預測狀態不明，不增加任何市場風險")


def summarize_transformer_risk(
    row: pd.Series,
    *,
    checkpoint_name: str,
    gate_config: TransformerTrendGateConfig | None = None,
) -> TransformerRiskPreview:
    """將 Transformer 最新輸出整理成不會觸發交易的風控預覽。"""
    trend, details = classify_transformer_trend(
        row,
        gate_config or TransformerTrendGateConfig(),
    )
    timestamp = pd.to_datetime(row.get("timestamp"), utc=True, errors="coerce")
    timestamp_text = "-" if pd.isna(timestamp) else timestamp.isoformat()
    return TransformerRiskPreview(
        timestamp=timestamp_text,
        close=float(pd.to_numeric(row.get("close"), errors="coerce")),
        trend=trend,
        risk_conclusion=_risk_conclusion(trend),
        expected_return_1=float(details["transformer_return_1"]),
        expected_return_5=float(details["transformer_return_5"]),
        expected_return_20=float(details["transformer_return_20"]),
        expected_volatility=float(details["transformer_volatility"]),
        bull_probability=float(details["transformer_bull_probability"]),
        bear_probability=float(details["transformer_bear_probability"]),
        uncertainty=float(details["transformer_uncertainty"]),
        checkpoint_name=checkpoint_name,
    )


def run_transformer_risk_preview(
    *,
    raw_dir: str | Path,
    checkpoint_path: str | Path,
    symbol: str,
    decision_interval: str,
    device: str = "cpu",
    history_bars: int = 512,
    gate_config: TransformerTrendGateConfig | None = None,
) -> TransformerRiskPreview:
    """用最新本機多週期資料推論一次；不載入 RL，也不寫入交易帳戶。"""
    raw_root = Path(raw_dir)
    checkpoint = Path(checkpoint_path)
    raw_paths = tuple(
        canonical_ohlcv_path(
            raw_root,
            "binance_futures",
            symbol,
            interval,
        )
        for interval in BTC_MULTITIMEFRAME_INTERVALS
    )
    missing = [path.name for path in raw_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少多週期 K 線：" + "、".join(missing))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"找不到 Transformer 模型：{checkpoint}")

    features = build_realtime_multitimeframe_frame(
        raw_paths,
        decision_interval=decision_interval,
        history_bars=history_bars,
    )
    inferred = infer_latest_transformer_context(
        checkpoint,
        features,
        device=device,
        mixed_precision=device != "cpu",
    )
    if inferred.empty:
        raise ValueError("Transformer 沒有產生有效預測")
    return summarize_transformer_risk(
        inferred.iloc[-1],
        checkpoint_name=checkpoint.parent.name,
        gate_config=gate_config,
    )
