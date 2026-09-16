"""將舊市場 CSV 搬到新的資產分類與易讀檔名。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ai_quant_trading.data_collection.assets import infer_asset_class, market_file_symbol_slug
from ai_quant_trading.data_collection.csv_storage import (
    market_data_output_path,
    ohlcv_output_path,
)


@dataclass(frozen=True, slots=True)
class OrganizationAction:
    """一個不覆寫原檔的 CSV 搬移計畫。"""

    source: Path
    target: Path
    data_type: str


def _first_value(frame: pd.DataFrame, column: str) -> str:
    if frame.empty or column not in frame.columns or pd.isna(frame.iloc[0][column]):
        raise ValueError(f"CSV 缺少可用的 {column} 欄位")
    return str(frame.iloc[0][column])


def _feature_output_path(frame: pd.DataFrame, processed_dir: Path, source_name: str) -> Path:
    exchange = _first_value(frame, "exchange").lower()
    symbol = _first_value(frame, "symbol")
    interval = _first_value(frame, "interval")
    asset_class = infer_asset_class(exchange, symbol)
    symbol_slug = market_file_symbol_slug(symbol, asset_class)
    horizon = "h1"
    for part in Path(source_name).stem.split("_"):
        if part.startswith("h") and part[1:].isdigit():
            horizon = part
    name = f"features_{asset_class}_{exchange}_{symbol_slug}_{interval}_{horizon}.csv"
    return processed_dir / name


def plan_market_data_organization(
    raw_dir: str | Path,
    processed_dir: str | Path | None = None,
) -> list[OrganizationAction]:
    """依 CSV 內容建立搬移計畫，無法辨識的檔案保持原位。"""
    raw_root = Path(raw_dir).resolve()
    actions: list[OrganizationAction] = []
    for source in sorted(raw_root.rglob("*.csv")):
        frame = pd.read_csv(source)
        if frame.empty:
            continue
        try:
            exchange = _first_value(frame, "exchange").lower()
            symbol = _first_value(frame, "symbol")
            if {"open", "high", "low", "close", "volume"}.issubset(frame.columns):
                target = ohlcv_output_path(
                    frame,
                    raw_root,
                    exchange,
                    symbol,
                    _first_value(frame, "interval"),
                )
                data_type = "ohlcv"
            elif "price_change" in frame.columns or "last_price" in frame.columns:
                target = market_data_output_path(frame, raw_root, exchange, symbol)
                data_type = "market_data"
            else:
                continue
        except (KeyError, ValueError):
            continue
        if source.resolve() != target.resolve():
            actions.append(OrganizationAction(source.resolve(), target.resolve(), data_type))

    if processed_dir is not None:
        processed_root = Path(processed_dir).resolve()
        for source in sorted(processed_root.glob("features_*.csv")):
            frame = pd.read_csv(source)
            if frame.empty:
                continue
            try:
                target = _feature_output_path(frame, processed_root, source.name)
            except (KeyError, ValueError):
                continue
            if source.resolve() != target.resolve():
                actions.append(OrganizationAction(source.resolve(), target.resolve(), "features"))
    return actions


def apply_organization(
    actions: list[OrganizationAction],
    allowed_roots: list[str | Path],
) -> list[OrganizationAction]:
    """安全套用搬移計畫；遇到同名檔案時停止，不覆寫任何資料。"""
    roots = [Path(root).resolve() for root in allowed_roots]
    completed: list[OrganizationAction] = []
    for action in actions:
        source = action.source.resolve()
        target = action.target.resolve()
        if not any(source.is_relative_to(root) and target.is_relative_to(root) for root in roots):
            raise ValueError(f"搬移路徑超出允許範圍：{source} -> {target}")
        if not source.exists():
            raise FileNotFoundError(f"來源檔案不存在：{source}")
        if target.exists():
            raise FileExistsError(f"目標檔案已存在，未覆寫：{target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        source.replace(target)
        completed.append(action)
    return completed
