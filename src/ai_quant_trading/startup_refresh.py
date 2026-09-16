"""桌面 App 啟動時的一次性行情、新聞、情緒與特徵更新。"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Callable, Iterator
from uuid import uuid4

import pandas as pd

from ai_quant_trading.ai_pipeline import AIPipelineConfig, load_ai_pipeline_config
from ai_quant_trading.data_collection.collectors import (
    collect_binance_data,
    collect_binance_futures_data,
    collect_yahoo_finance_data,
)
from ai_quant_trading.features import (
    build_features_from_csv,
    refresh_multitimeframe_feature_datasets,
)
from ai_quant_trading.market_clock import interval_duration
from ai_quant_trading.sentiment import (
    FinBERTAnalyzer,
    NewsCollectionError,
    aggregate_finbert_features,
    collect_market_news,
    load_news_history,
    load_scored_news,
    news_history_path,
    save_news_history,
    save_scored_news,
)


REFRESH_CONFIG_NAME = "startup_refresh.json"
REFRESH_STATUS_NAME = "startup_refresh_status.json"
REFRESH_LOCK_NAME = "startup_refresh.lock"
SCORED_NEWS_NAME = "finbert_news_latest.csv"
LIVE_NEWS_STATUS_NAME = "live_news_refresh_status.json"
ProgressCallback = Callable[[dict[str, object]], None]


@dataclass(frozen=True, slots=True)
class StartupRefreshConfig:
    """每次開啟桌面 App 時的一次性更新設定。"""

    enabled: bool = True
    refresh_market_data: bool = True
    collect_news: bool = True
    score_finbert: bool = True
    refresh_features: bool = True
    enrich_features_with_finbert: bool = True
    yahoo_news_per_symbol: int = 10
    news_retention_days: int = 3650
    news_max_rows: int = 100_000

    def __post_init__(self) -> None:
        if not 1 <= self.yahoo_news_per_symbol <= 50:
            raise ValueError("每個美股標的新聞數必須介於 1 與 50")
        if self.news_retention_days <= 0 or self.news_max_rows <= 0:
            raise ValueError("新聞保留天數與最大筆數必須大於 0")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MarketRefreshTarget:
    """由既有 OHLCV 檔案辨識出的更新目標。"""

    path: Path
    exchange: str
    symbol: str
    interval: str
    latest_timestamp: pd.Timestamp
    include_derivatives_context: bool = False

    @property
    def key(self) -> tuple[str, str, str]:
        return self.exchange, self.symbol, self.interval


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def refresh_config_path(project_root: str | Path) -> Path:
    return Path(project_root) / "data" / "config" / REFRESH_CONFIG_NAME


def refresh_status_path(project_root: str | Path) -> Path:
    return Path(project_root) / "data" / "config" / REFRESH_STATUS_NAME


def scored_news_path(processed_dir: str | Path) -> Path:
    return Path(processed_dir) / "sentiment" / SCORED_NEWS_NAME


def live_news_status_path(project_root: str | Path) -> Path:
    return Path(project_root) / "data" / "config" / LIVE_NEWS_STATUS_NAME


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_startup_refresh_config(project_root: str | Path) -> StartupRefreshConfig:
    """讀取啟動更新設定；未知舊欄位會被忽略。"""
    path = refresh_config_path(project_root)
    if not path.exists():
        return StartupRefreshConfig()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        allowed = {item.name for item in fields(StartupRefreshConfig)}
        return StartupRefreshConfig(
            **{key: value for key, value in dict(payload).items() if key in allowed}
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return StartupRefreshConfig()


def save_startup_refresh_config(
    project_root: str | Path,
    config: StartupRefreshConfig,
) -> Path:
    """保存啟動更新設定。"""
    path = refresh_config_path(project_root)
    _atomic_json(path, config.to_dict())
    return path


def load_startup_refresh_status(project_root: str | Path) -> dict[str, object]:
    """讀取最近一次更新狀態。"""
    path = refresh_status_path(project_root)
    if not path.exists():
        return {"status": "idle", "message": "尚未執行啟動更新"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {"status": "failed", "message": "更新狀態檔無法讀取"}
    return payload if isinstance(payload, dict) else {"status": "failed"}


def _save_status(project_root: Path, **values: object) -> None:
    current = load_startup_refresh_status(project_root)
    current.update(values)
    current["updated_at"] = _utc_now()
    _atomic_json(refresh_status_path(project_root), current)


def _process_is_running(pid: int) -> bool:
    """確認鎖檔記錄的程序是否仍存活。"""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
            process_query_limited_information,
            False,
            pid,
        )
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


@contextmanager
def _refresh_lock(project_root: Path) -> Iterator[bool]:
    """避免快速重開 App 時同時執行兩份更新。"""
    path = project_root / "data" / "config" / REFRESH_LOCK_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor: int | None = None
    try:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                lock_pid = int(path.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                lock_pid = -1
            age_seconds = time.time() - path.stat().st_mtime
            stale = (
                lock_pid > 0 and not _process_is_running(lock_pid)
            ) or (lock_pid <= 0 and age_seconds > 30)
            if stale:
                path.unlink(missing_ok=True)
                descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            else:
                yield False
                return
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        yield True
    finally:
        if descriptor is not None:
            os.close(descriptor)
            path.unlink(missing_ok=True)


def discover_market_targets(raw_dir: str | Path) -> tuple[MarketRefreshTarget, ...]:
    """從固定 OHLCV 檔案內容辨識所有本機追蹤市場。"""
    targets: dict[tuple[str, str, str], MarketRefreshTarget] = {}
    for path in Path(raw_dir).rglob("ohlcv/*_latest.csv"):
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            continue
        required = {"timestamp", "exchange", "symbol", "interval"}
        if frame.empty or not required.issubset(frame.columns):
            continue
        timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        if timestamps.notna().sum() == 0:
            continue
        exchange = str(frame.iloc[-1]["exchange"]).strip().lower()
        symbol = str(frame.iloc[-1]["symbol"]).strip().upper()
        interval = str(frame.iloc[-1]["interval"]).strip().lower()
        target = MarketRefreshTarget(
            path=path,
            exchange=exchange,
            symbol=symbol,
            interval=interval,
            latest_timestamp=timestamps.max(),
            include_derivatives_context=bool(
                {"funding_rate", "open_interest"}.intersection(frame.columns)
            ),
        )
        previous = targets.get(target.key)
        if previous is None or path.stat().st_mtime > previous.path.stat().st_mtime:
            targets[target.key] = target
    return tuple(sorted(targets.values(), key=lambda item: item.key))


def _overlap_start(target: MarketRefreshTarget) -> str:
    duration = interval_duration(target.interval) or pd.Timedelta(days=1)
    return (target.latest_timestamp - duration * 2).isoformat()


def refresh_market_target(target: MarketRefreshTarget, raw_dir: str | Path) -> Path:
    """增量下載單一市場並合併回原本的固定 CSV。"""
    if target.exchange == "binance_futures":
        result = collect_binance_futures_data(
            [target.symbol],
            target.interval,
            raw_dir,
            start=_overlap_start(target),
            limit=1500,
            include_market_data=True,
            include_derivatives_context=target.include_derivatives_context,
        )
    elif target.exchange == "binance":
        result = collect_binance_data(
            [target.symbol],
            target.interval,
            raw_dir,
            start=_overlap_start(target),
            limit=1000,
            include_market_data=True,
            include_derivatives_context=target.include_derivatives_context,
        )
    elif target.exchange in {"yahoo", "yahoo_finance"}:
        result = collect_yahoo_finance_data(
            [target.symbol],
            target.interval,
            raw_dir,
            start=_overlap_start(target),
            limit=None,
        )
    else:
        raise ValueError(f"尚未支援自動更新資料源：{target.exchange}")
    if not result.ohlcv_files:
        raise RuntimeError(f"{target.symbol} 沒有更新任何 OHLCV")
    return result.ohlcv_files[0]


def _feature_horizon(path: Path) -> int | None:
    matched = re.search(r"_h(\d+)\.csv$", path.name)
    return int(matched.group(1)) if matched else None


def _frame_market_key(frame: pd.DataFrame) -> tuple[str, str, str] | None:
    required = {"exchange", "symbol", "interval"}
    if frame.empty or not required.issubset(frame.columns):
        return None
    return (
        str(frame.iloc[0]["exchange"]).strip().lower(),
        str(frame.iloc[0]["symbol"]).strip().upper(),
        str(frame.iloc[0]["interval"]).strip().lower(),
    )


def _preserve_extra_feature_columns(
    previous: pd.DataFrame,
    refreshed: pd.DataFrame,
) -> pd.DataFrame:
    """重建技術特徵後保留既有 Transformer 等外部推論欄位。"""
    if "timestamp" not in previous or "timestamp" not in refreshed:
        return refreshed
    extra_columns = [
        column
        for column in previous.columns
        if column not in refreshed.columns and not column.startswith("Unnamed:")
    ]
    if not extra_columns:
        return refreshed
    left = refreshed.copy()
    right = previous[["timestamp", *extra_columns]].copy()
    left["timestamp"] = pd.to_datetime(left["timestamp"], utc=True, errors="coerce")
    right["timestamp"] = pd.to_datetime(right["timestamp"], utc=True, errors="coerce")
    right = right.dropna(subset=["timestamp"]).drop_duplicates("timestamp", keep="last")
    result = left.merge(right, on="timestamp", how="left")
    for column in extra_columns:
        if column.endswith("_available"):
            result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)
    result["timestamp"] = result["timestamp"].map(
        lambda value: pd.Timestamp(value).isoformat() if pd.notna(value) else ""
    )
    return result


def refresh_existing_features(
    targets: tuple[MarketRefreshTarget, ...],
    processed_dir: str | Path,
    *,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[Path], list[str]]:
    """只重建已存在的特徵檔，不擅自新增其他標的。"""
    target_map = {target.key: target for target in targets}
    feature_paths = sorted(Path(processed_dir).glob("features_*_h*.csv"))
    outputs: list[Path] = []
    errors: list[str] = []
    total = len(feature_paths)
    for index, feature_path in enumerate(feature_paths, start=1):
        try:
            previous = pd.read_csv(feature_path)
            key = _frame_market_key(previous)
            horizon = _feature_horizon(feature_path)
            if key is None or horizon is None or key not in target_map:
                continue
            refreshed, output_path = build_features_from_csv(
                target_map[key].path,
                output_dir=processed_dir,
                target_horizon=horizon,
            )
            restored = _preserve_extra_feature_columns(previous, refreshed)
            _atomic_csv(output_path, restored)
            outputs.append(output_path)
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append(f"{feature_path.name}：{exc}")
        if progress_callback:
            progress_callback(
                {
                    "phase": "features",
                    "completed": index,
                    "total": total,
                    "message": f"更新特徵 {index}/{total}",
                }
            )
    return outputs, errors


def _load_pipeline_config(project_root: Path) -> AIPipelineConfig:
    path = project_root / "data" / "config" / "ai_pipeline.json"
    if not path.exists():
        return AIPipelineConfig()
    try:
        return load_ai_pipeline_config(path)
    except (OSError, ValueError, TypeError):
        return AIPipelineConfig()


def score_pending_news(
    project_root: Path,
    raw_news_path: Path,
    output_path: Path,
    *,
    retention_days: int,
    max_rows: int,
) -> tuple[int, pd.DataFrame]:
    """只評分尚未處理的新文章，避免每次啟動重跑全部 FinBERT。"""
    pipeline = _load_pipeline_config(project_root)
    existing = load_scored_news(output_path)
    if not pipeline.finbert_enabled:
        return 0, existing
    raw_news = load_news_history(raw_news_path)
    known_ids = set(existing.get("news_id", pd.Series(dtype=str)).astype(str))
    pending = raw_news[~raw_news["news_id"].astype(str).isin(known_ids)].copy()
    if pending.empty:
        return 0, existing
    cache_dir = Path(pipeline.finbert.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = project_root / cache_dir
    config = replace(pipeline.finbert, cache_dir=str(cache_dir))
    scored = FinBERTAnalyzer(config).score_news(pending)
    save_scored_news(
        scored,
        output_path,
        retention_days=retention_days,
        max_rows=max_rows,
    )
    return len(scored), load_scored_news(output_path)


def refresh_live_news_sentiment(
    project_root: str | Path,
    *,
    crypto_symbols: tuple[str, ...] = ("BTC/USDT",),
    minimum_interval_seconds: float = 900.0,
) -> dict[str, object]:
    """常駐交易期間增量收集並評分新聞；冷卻時間內直接沿用最新結果。"""
    if minimum_interval_seconds < 0:
        raise ValueError("minimum_interval_seconds 不可小於 0")
    root = Path(project_root).resolve()
    status_path = live_news_status_path(root)
    now = datetime.now(timezone.utc)
    if status_path.exists():
        try:
            previous = json.loads(status_path.read_text(encoding="utf-8"))
            attempted = pd.to_datetime(previous.get("attempted_at"), utc=True, errors="coerce")
            if pd.notna(attempted) and (now - attempted.to_pydatetime()).total_seconds() < minimum_interval_seconds:
                return {**dict(previous), "cached": True}
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    pipeline = _load_pipeline_config(root)
    if not pipeline.finbert_enabled:
        result = {
            "status": "disabled",
            "attempted_at": now.isoformat(),
            "message": "AI 管線未啟用 FinBERT",
            "cached": False,
        }
        _atomic_json(status_path, result)
        return result

    config = load_startup_refresh_config(root)
    with _refresh_lock(root) as acquired:
        if not acquired:
            return {
                "status": "busy",
                "attempted_at": now.isoformat(),
                "message": "另一個資料更新工作正在執行",
                "cached": True,
            }
        try:
            collected = collect_market_news(
                crypto_symbols=list(crypto_symbols),
                equity_symbols=[],
                yahoo_limit_per_symbol=config.yahoo_news_per_symbol,
            )
            raw_path = news_history_path(root / "data" / "raw")
            if not collected.frame.empty:
                save_news_history(
                    collected.frame,
                    raw_path,
                    retention_days=config.news_retention_days,
                    max_rows=config.news_max_rows,
                )
            scored_count = 0
            if raw_path.exists():
                scored_count, _ = score_pending_news(
                    root,
                    raw_path,
                    scored_news_path(root / "data" / "processed"),
                    retention_days=config.news_retention_days,
                    max_rows=config.news_max_rows,
                )
            result = {
                "status": "partial" if collected.errors else "completed",
                "attempted_at": now.isoformat(),
                "finished_at": _utc_now(),
                "news_collected": len(collected.frame),
                "news_scored": scored_count,
                "errors": list(collected.errors),
                "cached": False,
            }
        except Exception as exc:
            result = {
                "status": "failed",
                "attempted_at": now.isoformat(),
                "finished_at": _utc_now(),
                "message": str(exc),
                "errors": [str(exc)],
                "cached": False,
            }
        _atomic_json(status_path, result)
        return result


def enrich_feature_files(
    processed_dir: str | Path,
    scored_news: pd.DataFrame,
    project_root: Path,
) -> tuple[int, list[str]]:
    """把最新 FinBERT 因果特徵覆寫回所有現有特徵檔。"""
    if scored_news.empty:
        return 0, []
    config = _load_pipeline_config(project_root).finbert
    updated = 0
    errors: list[str] = []
    for path in sorted(Path(processed_dir).glob("features_*_h*.csv")):
        try:
            frame = pd.read_csv(path)
            enriched = aggregate_finbert_features(frame, scored_news, config)
            _atomic_csv(path, enriched)
            updated += 1
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append(f"{path.name}：{exc}")
    return updated, errors


def run_startup_refresh(
    project_root: str | Path,
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, object]:
    """執行一次完整更新；任何單一市場失敗都不會中止其他市場。"""
    root = Path(project_root).resolve()
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    config = load_startup_refresh_config(root)
    if not config.enabled:
        result = {"status": "disabled", "message": "啟動自動更新已停用"}
        _save_status(root, **result)
        return result

    with _refresh_lock(root) as acquired:
        if not acquired:
            return load_startup_refresh_status(root)
        started_at = _utc_now()
        errors: list[str] = []
        summary: dict[str, object] = {
            "market_targets": 0,
            "market_updated": 0,
            "news_collected": 0,
            "news_scored": 0,
            "features_updated": 0,
            "features_enriched": 0,
            "multitimeframe_updated": 0,
        }
        _save_status(
            root,
            status="running",
            phase="discover",
            message="正在盤點本機市場資料",
            started_at=started_at,
            finished_at=None,
            completed=0,
            total=0,
            errors=[],
            summary=summary,
        )
        try:
            targets = discover_market_targets(raw_dir)
            summary["market_targets"] = len(targets)
            if config.refresh_market_data:
                for index, target in enumerate(targets, start=1):
                    message = f"更新 {target.symbol} {target.interval}"
                    _save_status(
                        root,
                        phase="market",
                        message=message,
                        completed=index - 1,
                        total=len(targets),
                        summary=summary,
                    )
                    if progress_callback:
                        progress_callback(
                            {
                                "phase": "market",
                                "message": message,
                                "completed": index - 1,
                                "total": len(targets),
                            }
                        )
                    try:
                        refresh_market_target(target, raw_dir)
                        summary["market_updated"] = int(summary["market_updated"]) + 1
                    except Exception as exc:
                        errors.append(f"{target.symbol} {target.interval}：{exc}")

            raw_news_path = news_history_path(raw_dir)
            if config.collect_news:
                _save_status(root, phase="news", message="正在收集市場新聞", summary=summary)
                crypto_symbols = [
                    target.symbol
                    for target in targets
                    if target.exchange in {"binance", "binance_futures", "bybit"}
                ]
                equity_symbols = [
                    target.symbol
                    for target in targets
                    if target.exchange in {"yahoo", "yahoo_finance"}
                ]
                news_result = collect_market_news(
                    crypto_symbols=crypto_symbols,
                    equity_symbols=equity_symbols,
                    yahoo_limit_per_symbol=config.yahoo_news_per_symbol,
                )
                errors.extend(news_result.errors)
                if not news_result.frame.empty:
                    save_news_history(
                        news_result.frame,
                        raw_news_path,
                        retention_days=config.news_retention_days,
                        max_rows=config.news_max_rows,
                    )
                summary["news_collected"] = len(news_result.frame)

            scored = load_scored_news(scored_news_path(processed_dir))
            if config.score_finbert and raw_news_path.exists():
                _save_status(root, phase="finbert", message="正在評分新文章", summary=summary)
                try:
                    count, scored = score_pending_news(
                        root,
                        raw_news_path,
                        scored_news_path(processed_dir),
                        retention_days=config.news_retention_days,
                        max_rows=config.news_max_rows,
                    )
                    summary["news_scored"] = count
                except (OSError, ValueError, RuntimeError, NewsCollectionError) as exc:
                    errors.append(f"FinBERT：{exc}")

            if config.refresh_features:
                def feature_progress(values: dict[str, object]) -> None:
                    _save_status(root, status="running", summary=summary, **values)
                    if progress_callback:
                        progress_callback(values)

                outputs, feature_errors = refresh_existing_features(
                    targets,
                    processed_dir,
                    progress_callback=feature_progress,
                )
                errors.extend(feature_errors)
                summary["features_updated"] = len(outputs)

            if config.enrich_features_with_finbert and not scored.empty:
                _save_status(
                    root,
                    phase="enrich",
                    message="正在附加新聞情緒特徵",
                    summary=summary,
                )
                enriched, enrich_errors = enrich_feature_files(processed_dir, scored, root)
                errors.extend(enrich_errors)
                summary["features_enriched"] = enriched

            if config.refresh_features:
                _save_status(
                    root,
                    phase="multitimeframe",
                    message="正在重建多週期特徵資料",
                    summary=summary,
                )
                mtf_artifacts, mtf_errors = refresh_multitimeframe_feature_datasets(
                    processed_dir
                )
                errors.extend(mtf_errors)
                summary["multitimeframe_updated"] = len(mtf_artifacts)

            status = "partial" if errors else "completed"
            message = "資料更新完成" if not errors else f"資料更新完成，{len(errors)} 項失敗"
            result = {
                "status": status,
                "phase": "complete",
                "message": message,
                "started_at": started_at,
                "finished_at": _utc_now(),
                "completed": 1,
                "total": 1,
                "errors": errors[-50:],
                "summary": summary,
            }
            _save_status(root, **result)
            return result
        except Exception as exc:
            result = {
                "status": "failed",
                "phase": "failed",
                "message": f"啟動更新失敗：{exc}",
                "started_at": started_at,
                "finished_at": _utc_now(),
                "errors": [*errors, str(exc)][-50:],
                "summary": summary,
            }
            _save_status(root, **result)
            return result
