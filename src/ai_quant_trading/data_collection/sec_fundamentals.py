"""SEC EDGAR 公司基本面資料收集器。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from ai_quant_trading.data_collection.csv_storage import save_canonical_context_csv
from ai_quant_trading.data_collection.http_client import create_secure_session, require_session
from ai_quant_trading.data_collection.time_utils import utc_now_iso


class SecDataError(RuntimeError):
    """SEC 資料下載或欄位解析失敗。"""


SEC_CONCEPTS: dict[str, tuple[str, ...]] = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ),
    "net_income": ("NetIncomeLoss",),
    "operating_income": ("OperatingIncomeLoss",),
    "assets": ("Assets",),
    "liabilities": ("Liabilities",),
    "equity": (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ),
    "diluted_eps": ("EarningsPerShareDiluted",),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
    "capital_expenditure": (
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ),
}


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.where(denominator.abs() > 1e-12)


@dataclass(slots=True)
class SecCompanyFactsClient:
    """讀取 SEC Company Facts，使用申報日而非財報期末做時間對齊。"""

    user_agent: str
    base_url: str = "https://data.sec.gov"
    files_url: str = "https://www.sec.gov"
    timeout: int = 45
    session: requests.Session | None = None

    def __post_init__(self) -> None:
        if "@" not in self.user_agent or len(self.user_agent.strip()) < 8:
            raise ValueError("SEC User-Agent 必須包含可聯絡的電子郵件")
        if self.session is None:
            self.session = create_secure_session()
        session = require_session(self.session)
        session.headers.update(
            {
                "User-Agent": self.user_agent.strip(),
                "Accept-Encoding": "gzip, deflate",
            }
        )

    def _get_json(self, url: str) -> Any:
        session = require_session(self.session)
        try:
            response = session.get(url, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise SecDataError(f"SEC 下載失敗：{exc}") from exc
        try:
            return response.json()
        except ValueError as exc:
            raise SecDataError("SEC 回傳內容不是有效 JSON") from exc

    def fetch_ticker_cik_map(self) -> dict[str, str]:
        """下載 SEC 官方 ticker 與 CIK 對照表。"""
        payload = self._get_json(f"{self.files_url}/files/company_tickers.json")
        result: dict[str, str] = {}
        for item in payload.values():
            ticker = str(item.get("ticker", "")).upper()
            cik = str(item.get("cik_str", "")).zfill(10)
            if ticker and cik:
                result[ticker] = cik
        return result

    @staticmethod
    def _concept_rows(
        facts: dict[str, Any],
        aliases: tuple[str, ...],
        canonical_name: str,
    ) -> pd.DataFrame:
        us_gaap = facts.get("facts", {}).get("us-gaap", {})
        for concept in aliases:
            payload = us_gaap.get(concept)
            if not payload:
                continue
            units = payload.get("units", {})
            preferred_units = (
                ("USD/shares", "USD")
                if canonical_name == "diluted_eps"
                else ("USD", "shares", "pure")
            )
            rows: list[dict[str, Any]] = []
            for unit in preferred_units:
                if unit in units:
                    rows = list(units[unit])
                    break
            if not rows:
                continue
            frame = pd.DataFrame(rows)
            if frame.empty or not {"filed", "val"}.issubset(frame.columns):
                continue
            frame = frame[
                frame.get("form", pd.Series("", index=frame.index))
                .astype(str)
                .isin({"10-Q", "10-Q/A", "10-K", "10-K/A"})
            ].copy()
            if frame.empty:
                continue
            frame["filed"] = pd.to_datetime(frame["filed"], utc=True, errors="coerce")
            frame["end"] = pd.to_datetime(frame.get("end"), utc=True, errors="coerce")
            frame["start"] = pd.to_datetime(frame.get("start"), utc=True, errors="coerce")
            frame["value"] = pd.to_numeric(frame["val"], errors="coerce")
            frame = frame.dropna(subset=["filed", "value"])
            # 保守地從申報日次日才允許模型使用，避免同日盤中偷看晚間申報。
            frame["timestamp"] = frame["filed"].dt.normalize() + pd.Timedelta(days=1)
            frame["duration_days"] = (frame["end"] - frame["start"]).dt.days
            selected_rows: list[pd.Series] = []
            for _, group in frame.groupby("timestamp", sort=True):
                group = group.sort_values("end")
                latest_end = group["end"].max()
                if pd.notna(latest_end):
                    group = group[group["end"] == latest_end]
                form = str(group.iloc[-1].get("form", ""))
                duration = pd.to_numeric(group["duration_days"], errors="coerce")
                if duration.notna().any():
                    index = duration.idxmax() if form.startswith("10-K") else duration.idxmin()
                    selected_rows.append(group.loc[index])
                else:
                    selected_rows.append(group.iloc[-1])
            result = pd.DataFrame(selected_rows)
            result["concept"] = canonical_name
            return result[
                ["timestamp", "filed", "end", "form", "concept", "value"]
            ].reset_index(drop=True)
        return pd.DataFrame(
            columns=["timestamp", "filed", "end", "form", "concept", "value"]
        )

    def fetch_company_fundamentals(
        self,
        ticker: str,
        *,
        cik: str | None = None,
    ) -> pd.DataFrame:
        """取得公司逐次申報後可知的基本面快照與衍生比率。"""
        normalized = ticker.strip().upper()
        if not normalized:
            raise ValueError("美股代號不可為空")
        resolved_cik = cik.zfill(10) if cik else self.fetch_ticker_cik_map().get(normalized)
        if not resolved_cik:
            raise SecDataError(f"SEC 對照表找不到 {normalized}")
        facts = self._get_json(
            f"{self.base_url}/api/xbrl/companyfacts/CIK{resolved_cik}.json"
        )
        long_rows = [
            self._concept_rows(facts, aliases, canonical)
            for canonical, aliases in SEC_CONCEPTS.items()
        ]
        long_rows = [frame for frame in long_rows if not frame.empty]
        if not long_rows:
            raise SecDataError(f"SEC {normalized} 沒有可用的 10-Q／10-K Company Facts")
        combined = pd.concat(long_rows, ignore_index=True)
        pivot = combined.pivot_table(
            index="timestamp",
            columns="concept",
            values="value",
            aggfunc="last",
        ).sort_index()
        pivot = pivot.ffill().reset_index()
        for column in SEC_CONCEPTS:
            if column not in pivot:
                pivot[column] = np.nan
        pivot["fundamental_revenue_growth"] = pivot["revenue"].pct_change(
            4, fill_method=None
        )
        pivot["fundamental_profit_margin"] = _safe_ratio(
            pivot["net_income"], pivot["revenue"]
        )
        pivot["fundamental_operating_margin"] = _safe_ratio(
            pivot["operating_income"], pivot["revenue"]
        )
        pivot["fundamental_debt_to_equity"] = _safe_ratio(
            pivot["liabilities"], pivot["equity"]
        )
        pivot["fundamental_ocf_margin"] = _safe_ratio(
            pivot["operating_cash_flow"], pivot["revenue"]
        )
        pivot.insert(1, "symbol", normalized)
        pivot.insert(2, "cik", resolved_cik)
        pivot["fundamentals_available"] = 1
        pivot["collected_at"] = utc_now_iso()
        pivot["timestamp"] = pd.to_datetime(pivot["timestamp"], utc=True).map(
            lambda value: value.isoformat()
        )
        return pivot.reset_index(drop=True)


def sec_fundamentals_path(raw_dir: str | Path, ticker: str) -> Path:
    """回傳單一美股基本面的固定檔名。"""
    normalized = ticker.strip().upper().replace(".", "-")
    return (
        Path(raw_dir)
        / "us_equity"
        / "sec"
        / "fundamentals"
        / f"fundamentals_SEC_{normalized}_latest.csv"
    )


def collect_sec_fundamentals(
    tickers: list[str] | tuple[str, ...],
    raw_dir: str | Path,
    *,
    user_agent: str,
    client: SecCompanyFactsClient | None = None,
) -> tuple[Path, ...]:
    """批次收集 SEC 基本面，每個 ticker 覆寫同一份成品。"""
    normalized = tuple(dict.fromkeys(ticker.strip().upper() for ticker in tickers if ticker.strip()))
    if not normalized:
        raise ValueError("至少輸入一個美股代號")
    service = client or SecCompanyFactsClient(user_agent=user_agent)
    cik_map = service.fetch_ticker_cik_map()
    paths: list[Path] = []
    for ticker in normalized:
        frame = service.fetch_company_fundamentals(ticker, cik=cik_map.get(ticker))
        paths.append(
            save_canonical_context_csv(
                frame,
                sec_fundamentals_path(raw_dir, ticker),
                key_columns=("timestamp", "symbol"),
                merge_existing=True,
            )
        )
    return tuple(paths)
