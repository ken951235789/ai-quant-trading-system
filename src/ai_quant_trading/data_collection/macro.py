"""FRED 宏觀市場資料收集器。"""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ai_quant_trading.data_collection.csv_storage import save_canonical_context_csv
from ai_quant_trading.data_collection.http_client import create_secure_session, require_session
from ai_quant_trading.data_collection.time_utils import utc_now_iso


class FredDataError(RuntimeError):
    """FRED 資料下載或格式解析失敗。"""


@dataclass(frozen=True, slots=True)
class FredSeriesSpec:
    """宏觀序列名稱與保守公開延遲。"""

    series_id: str
    label: str
    publication_lag_days: int


FRED_SERIES: dict[str, FredSeriesSpec] = {
    "DGS2": FredSeriesSpec("DGS2", "美國 2 年期公債殖利率", 1),
    "DGS10": FredSeriesSpec("DGS10", "美國 10 年期公債殖利率", 1),
    "DFF": FredSeriesSpec("DFF", "聯邦基金有效利率", 1),
    "CPIAUCSL": FredSeriesSpec("CPIAUCSL", "美國消費者物價指數", 45),
    "UNRATE": FredSeriesSpec("UNRATE", "美國失業率", 10),
    "VIXCLS": FredSeriesSpec("VIXCLS", "VIX 波動率指數", 1),
    "DTWEXBGS": FredSeriesSpec("DTWEXBGS", "美元廣義指數", 2),
    "BAMLH0A0HYM2": FredSeriesSpec("BAMLH0A0HYM2", "美國高收益債利差", 1),
}


@dataclass(slots=True)
class FredPublicClient:
    """使用 FRED 官方 CSV 圖表端點，不需要 API key。"""

    base_url: str = "https://fred.stlouisfed.org"
    timeout: int = 45
    session: requests.Session | None = None

    def __post_init__(self) -> None:
        if self.session is None:
            self.session = create_secure_session()

    def fetch_series(
        self,
        series_id: str,
        *,
        start: str | None = None,
        end: str | None = None,
        publication_lag_days: int | None = None,
    ) -> pd.DataFrame:
        """下載單一序列，將保守公開日設為模型可使用時間。"""
        normalized = series_id.strip().upper()
        spec = FRED_SERIES.get(normalized)
        lag_days = (
            publication_lag_days
            if publication_lag_days is not None
            else (spec.publication_lag_days if spec else 1)
        )
        session = require_session(self.session)
        params = {"id": normalized, "cosd": start, "coed": end}
        params = {key: value for key, value in params.items() if value}
        try:
            response = session.get(
                f"{self.base_url}/graph/fredgraph.csv",
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise FredDataError(f"FRED 下載失敗：{exc}") from exc
        try:
            source = pd.read_csv(StringIO(response.text))
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            raise FredDataError("FRED 回傳內容不是有效 CSV") from exc
        date_column = next(
            (
                column
                for column in ("observation_date", "DATE", "date")
                if column in source.columns
            ),
            None,
        )
        if date_column is None or normalized not in source:
            raise FredDataError(f"FRED {normalized} 回傳欄位不完整")
        observation = pd.to_datetime(source[date_column], utc=True, errors="coerce")
        values = pd.to_numeric(source[normalized].replace(".", np.nan), errors="coerce")
        result = pd.DataFrame(
            {
                "observation_date": observation,
                "value": values,
            }
        ).dropna(subset=["observation_date", "value"])
        result = result.sort_values("observation_date").drop_duplicates(
            "observation_date", keep="last"
        )
        result["timestamp"] = result["observation_date"] + pd.to_timedelta(
            lag_days, unit="D"
        )
        result["series_id"] = normalized
        result["label"] = spec.label if spec else normalized
        result["publication_lag_days"] = lag_days
        result["change"] = result["value"].diff()
        result["pct_change"] = result["value"].pct_change(fill_method=None)
        rolling_mean = result["value"].rolling(60, min_periods=12).mean()
        rolling_std = result["value"].rolling(60, min_periods=12).std(ddof=0)
        result["zscore_60"] = (
            (result["value"] - rolling_mean) / rolling_std.where(rolling_std != 0)
        )
        result["collected_at"] = utc_now_iso()
        result["observation_date"] = result["observation_date"].map(
            lambda value: value.isoformat()
        )
        result["timestamp"] = result["timestamp"].map(lambda value: value.isoformat())
        return result[
            [
                "timestamp",
                "observation_date",
                "series_id",
                "label",
                "publication_lag_days",
                "value",
                "change",
                "pct_change",
                "zscore_60",
                "collected_at",
            ]
        ].reset_index(drop=True)


def fred_series_path(raw_dir: str | Path, series_id: str) -> Path:
    """回傳單一 FRED 序列的固定檔名。"""
    normalized = series_id.strip().upper()
    return Path(raw_dir) / "macro" / "fred" / f"macro_fred_{normalized}_latest.csv"


def collect_fred_series(
    series_ids: list[str] | tuple[str, ...],
    raw_dir: str | Path,
    *,
    start: str | None = None,
    end: str | None = None,
    client: FredPublicClient | None = None,
) -> tuple[Path, ...]:
    """批次下載宏觀序列，每個序列只保留一份可更新 CSV。"""
    if not series_ids:
        raise ValueError("至少選擇一個 FRED 宏觀序列")
    service = client or FredPublicClient()
    paths: list[Path] = []
    for series_id in dict.fromkeys(series_ids):
        frame = service.fetch_series(series_id, start=start, end=end)
        paths.append(
            save_canonical_context_csv(
                frame,
                fred_series_path(raw_dir, series_id),
                key_columns=("timestamp", "series_id"),
                merge_existing=True,
            )
        )
    return tuple(paths)
