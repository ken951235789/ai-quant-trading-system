"""產生包含資料、程式、環境與故障結果的可重現研究報告。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import platform
from pathlib import Path
import subprocess
import sys

import pandas as pd

from ai_quant_trading.operations.integrity import build_artifact_manifest
from ai_quant_trading.persistence import write_json_atomic
from ai_quant_trading.research.faults import run_fault_injection_experiments
from ai_quant_trading.research.replay import (
    HistoricalMarketEventSource,
    MarketEventReplayer,
    MarketFrameAccumulator,
)
from ai_quant_trading.trading.market_events import MarketEventBus


TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "SQLAlchemy",
    "psycopg",
    "websocket-client",
    "torch",
    "transformers",
    "stable-baselines3",
)


def _git_value(project_root: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def _package_versions() -> dict[str, str]:
    values: dict[str, str] = {}
    for package in TRACKED_PACKAGES:
        try:
            values[package] = version(package)
        except PackageNotFoundError:
            values[package] = "not-installed"
    return values


def _result_digest(records: list[dict[str, object]]) -> str:
    payload = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class ReplayResearchArtifacts:
    experiment_dir: Path
    report_path: Path
    manifest_path: Path
    results_path: Path
    artifact_manifest_path: Path
    passed: bool


def _knowledge_table() -> str:
    return """| 領域 | 必須掌握 | 本專案可展示的證據 |
|---|---|---|
| 機率統計 | 分布、信賴區間、Bootstrap、多重檢定 | 後續策略結果不可只報單一次報酬 |
| 時間序列 | 非平穩、自相關、異方差、Walk-forward | K 線依時間排序且禁止未來資料 |
| 市場微結構 | Spread、Slippage、部分成交、Funding | WebSocket、BookTicker、成交與對帳 |
| 強化學習 | MDP、PPO/SAC、Reward、POMDP | SAC/PPO 只提出目標曝險，風控可否決 |
| 事件驅動架構 | Event sourcing、冪等、at-least-once | MarketEvent、事件 ID、checkpoint 重播 |
| 資料庫 | ACID、唯一鍵、索引、交易隔離 | PostgreSQL 訂單、成交、事件投影 |
| 可靠性工程 | RPO/RTO、心跳、熔斷、故障注入 | 中斷續傳、亂序與缺 K 偵測、Watchdog |
| 資安 | 最小權限、秘密管理、供應鏈安全 | API 白名單、ACL、依賴掃描、模型雜湊 |
| 科學方法 | 假說、基線、控制變因、可重現性 | 輸入 SHA、Git commit、參數與結果摘要 |
"""


def _report_text(manifest: dict[str, object], results: pd.DataFrame) -> str:
    source = dict(manifest["source"])
    environment = dict(manifest["environment"])
    result_rows = results.to_dict(orient="records")
    passed = int(results["passed"].sum())
    rows = [
        "# 歷史／即時統一事件重播與故障注入研究報告",
        "",
        f"產生時間：{manifest['created_at']}",
        "",
        "## 研究問題",
        "",
        "歷史 K 線與即時 WebSocket 若使用不同資料路徑，回測通過的程式仍可能在即時環境產生",
        "重複處理、順序倒退或斷線遺失。本實驗驗證兩種來源是否可共用同一 MarketEvent 合約、",
        "事件匯流排、consumer 與 checkpoint 語意。",
        "",
        "## 可反駁假說",
        "",
        "在不連接交易 API 的條件下，系統應能阻擋重複及損壞事件、偵測亂序與缺 K，並在傳輸或",
        "consumer 中斷後從最後成功位置恢復，最終保存的唯一事件數必須等於輸入事件數。",
        "",
        "## 輸入與環境",
        "",
        f"- 輸入：`{source.get('source_path')}`",
        f"- SHA-256：`{source.get('sha256')}`",
        f"- 事件數：{source.get('rows')}",
        f"- 時間範圍：{source.get('first_event_time')} 至 {source.get('last_event_time')}",
        f"- Python：{environment.get('python')}",
        f"- Git commit：`{environment.get('git_commit') or 'unavailable'}`",
        f"- Git working tree dirty：{environment.get('git_dirty')}",
        f"- 結果摘要 SHA-256：`{manifest['result_sha256']}`",
        "",
        "## 故障注入結果",
        "",
        f"通過 {passed}/{len(results)} 個情境。",
        "",
        "| 情境 | 通過 | 預期 | 實際觀察 |",
        "|---|---:|---|---|",
    ]
    for row in result_rows:
        rows.append(
            f"| {row['scenario']} | {'是' if row['passed'] else '否'} | "
            f"{row['expected']} | {row['observed']} |"
        )
    rows.extend(
        [
            "",
            "## 重現方式",
            "",
            "```powershell",
            str(manifest["reproduce_command"]),
            "```",
            "",
            "重新執行後，`result_sha256` 應相同；產生時間、執行秒數與輸出資料夾可以不同。",
            "",
            "## 已知限制",
            "",
            "- 這是離線可靠性實驗，不證明策略能獲利。",
            "- 本實驗沒有模擬作業系統斷電、磁碟毀損或 Binance 全區故障。",
            "- Event Bus 保證單程序內去重；跨程序交易事件仍由 PostgreSQL 唯一鍵負責。",
            "- Consumer 必須以 event_id 冪等，因為分散式系統無法只靠傳輸保證 exactly-once。",
            "- 工作目錄若為 dirty，報告可重現性低於已提交且有版本標籤的 commit。",
            "",
            "## 推甄知識地圖",
            "",
            _knowledge_table(),
        ]
    )
    return "\n".join(rows).rstrip() + "\n"


def run_replay_research(
    input_path: str | Path,
    *,
    output_root: str | Path,
    limit: int = 500,
    project_root: str | Path | None = None,
    default_exchange: str = "binance_futures",
    default_symbol: str = "BTC/USDT",
    default_interval: str = "15m",
) -> ReplayResearchArtifacts:
    """執行基線重播、故障注入並產生完整研究成品。"""
    root = Path(project_root or Path.cwd()).resolve()
    source = HistoricalMarketEventSource.from_csv(
        input_path,
        limit=limit,
        default_exchange=default_exchange,
        default_symbol=default_symbol,
        default_interval=default_interval,
    )
    stream_keys = {event.stream_key for event in source.events}
    if len(stream_keys) != 1:
        raise ValueError("單次故障實驗必須使用同一標的與同一週期")
    created_at = datetime.now(timezone.utc)
    run_name = created_at.strftime("%Y%m%dT%H%M%SZ") + "_" + source.content_hash[:12]
    experiment_dir = Path(output_root).resolve() / run_name
    suffix = 1
    while experiment_dir.exists():
        experiment_dir = Path(output_root).resolve() / f"{run_name}_{suffix:02d}"
        suffix += 1
    experiment_dir.mkdir(parents=True)

    accumulator = MarketFrameAccumulator(max_rows=limit)
    baseline_bus = MarketEventBus([accumulator], strict_continuity=True)
    baseline = MarketEventReplayer(source.events, source_id=source.source_id).run(baseline_bus)
    exchange, symbol, _kind, interval = next(iter(stream_keys))
    reconstructed = accumulator.frame(
        exchange=exchange,
        symbol=symbol,
        interval=interval,
    )
    if len(reconstructed) != len(source.events):
        raise RuntimeError("基線重播後 DataFrame 筆數不一致")

    fault_results = run_fault_injection_experiments(
        source.events,
        source_id=source.source_id,
        output_dir=experiment_dir / "checkpoints",
    )
    records = [result.to_dict() for result in fault_results]
    results = pd.DataFrame.from_records(records)
    results_path = experiment_dir / "fault_results.csv"
    results.to_csv(results_path, index=False, encoding="utf-8")
    reconstructed_path = experiment_dir / "reconstructed_events.csv"
    reconstructed.to_csv(reconstructed_path, index=False, encoding="utf-8")

    command = (
        "python scripts\\run_replay_research.py "
        f"--input \"{Path(input_path).resolve()}\" --limit {limit} "
        f"--output-root \"{Path(output_root).resolve()}\""
    )
    status = _git_value(root, "status", "--porcelain")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "experiment_type": "unified_market_event_replay_fault_injection",
        "created_at": created_at.isoformat(),
        "passed": bool(results["passed"].all()),
        "source": source.metadata,
        "baseline_replay": baseline.to_dict(),
        "event_bus_metrics": baseline_bus.metrics.to_dict(),
        "fault_scenarios": len(results),
        "result_sha256": _result_digest(records),
        "parameters": {
            "limit": limit,
            "default_exchange": default_exchange,
            "default_symbol": default_symbol,
            "default_interval": default_interval,
        },
        "environment": {
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "git_commit": _git_value(root, "rev-parse", "HEAD"),
            "git_dirty": bool(status) if status is not None else None,
            "packages": _package_versions(),
        },
        "reproduce_command": command,
    }
    manifest_path = experiment_dir / "research_manifest.json"
    write_json_atomic(manifest_path, manifest)
    report_path = experiment_dir / "research_report.md"
    report_path.write_text(_report_text(manifest, results), encoding="utf-8")
    artifact_manifest_path = build_artifact_manifest(
        experiment_dir,
        files=(manifest_path, results_path, reconstructed_path, report_path),
    )
    return ReplayResearchArtifacts(
        experiment_dir=experiment_dir,
        report_path=report_path,
        manifest_path=manifest_path,
        results_path=results_path,
        artifact_manifest_path=artifact_manifest_path,
        passed=bool(manifest["passed"]),
    )
