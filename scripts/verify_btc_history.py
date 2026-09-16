"""獨立重讀九份歷史 CSV，核對品質、覆蓋、雜湊並輸出標準資料報告。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_quant_trading.data_collection.binance import INTERVAL_TO_MS  # noqa: E402
from ai_quant_trading.data_collection.history import validate_history_chunk  # noqa: E402
from ai_quant_trading.market_clock import BTC_SUPPORTED_MULTITIMEFRAME_INTERVALS  # noqa: E402
from ai_quant_trading.persistence import write_json_atomic  # noqa: E402


def audit_dataset(item: dict[str, object]) -> dict[str, object]:
    path = Path(str(item["path"]))
    interval = str(item["interval"])
    lower = int(pd.Timestamp(str(item["requested_start_utc"])).timestamp() * 1000)
    upper = int(pd.Timestamp(str(item["requested_end_exclusive_utc"])).timestamp() * 1000)
    duration = INTERVAL_TO_MS[interval]
    rows = covered = 0
    previous = None
    first = last = None
    expected = lower
    gaps = []
    # 每次讀 25,000 根；跨批次也檢查排序及重複，不能只檢查單批內部。
    for chunk in pd.read_csv(path, chunksize=25_000):
        validate_history_chunk(chunk, interval)
        opened = pd.to_numeric(chunk["open_time_ms"])
        if previous is not None and int(opened.iloc[0]) <= previous:
            raise ValueError(f"{interval} CSV 跨批次逆序或重複")
        previous = int(opened.iloc[-1])
        first = int(opened.iloc[0]) if first is None else first
        last = previous
        rows += len(chunk)
        active = opened.loc[opened.between(lower, upper - 1)]
        covered += len(active)
        for value in active:
            value = int(value)
            if value > expected:
                gaps.append((expected, value))
            expected = value + duration
    if expected < upper:
        gaps.append((expected, upper))
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    if rows != item["rows"] or covered != item["requested_rows"] or checksum.hexdigest() != item["sha256"]:
        raise ValueError(f"{interval} CSV 與下載清單不一致，可能已被更新；先重跑下載建立快照")
    missing = sum((right - left) // duration for left, right in gaps)
    if missing != item["missing_bars"]:
        raise ValueError(f"{interval} CSV 缺口與清單不一致")
    if last is None or first is None:
        raise ValueError(f"{interval} CSV 為空")
    return {
        "interval": interval, "rows": rows, "requested_rows": covered, "missing_bars": missing,
        "duplicates": 0, "sha256": checksum.hexdigest(), "bytes": path.stat().st_size,
        "first_open_utc": pd.Timestamp(first, unit="ms", tz="UTC").isoformat(),
        "last_open_utc": pd.Timestamp(last, unit="ms", tz="UTC").isoformat(),
        "status": "PASS" if missing == 0 else "WARN",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "reports" / pd.Timestamp.now(tz="UTC").strftime("%Y%m%d"))
    args = parser.parse_args()
    manifest_path = PROJECT_ROOT / "data/raw/crypto/binance_futures/ohlcv/btc_futures_history_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] not in {"PASS", "WARN"}:
        raise ValueError("歷史下載尚未完成")
    items = manifest["datasets"]
    if [item["interval"] for item in items] != list(BTC_SUPPORTED_MULTITIMEFRAME_INTERVALS):
        raise ValueError("驗收要求完整九週期清單，先用預設下載命令補齊")
    audits = []
    for item in items:
        audit = audit_dataset(item)
        audits.append(audit)
        print(f"{audit['interval']}: {audit['rows']:,} 根，缺 {audit['missing_bars']} 根，CSV 與 SHA-256 驗證通過", flush=True)
    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "btc_nine_timeframe_history.json"
    helper_path = PROJECT_ROOT / ".agents/skills/ai-quant-engineering/scripts/new_report.py"
    spec = importlib.util.spec_from_file_location("quant_new_report", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("找不到標準報告工具")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    report = helper.build_report(
        report_type="data_integrity", project="AIQuantTradingSystem", revision=revision + " + working tree",
        scope="BTC USD-M 永續合約五年九週期原始資料下載與獨立 CSV 驗收",
    )
    report["project"]["environment"] = "Windows / CPU / public market data only"
    report["overall_status"] = "PASS" if all(item["status"] == "PASS" for item in audits) else "WARN"
    report["summary"] = "九週期原始資料已下載及獨立重讀驗收；未建新版特徵、未訓練模型、未連接私人交易帳戶或下單。"
    report["changes"] = [
        "新增官方月檔 SHA-256 校驗、分批 SQLite 續傳及固定 CSV 原子發布。",
        "九週期歷史集合獨立於既有五週期交易預設。",
        "修正新舊 UTC 時間格式混合解析並加入回歸測試。",
    ]
    report["checks"] = [
        {"id": "CSV-" + item["interval"], "name": "CSV 身分、有限值、OHLC、時間邊界、排序、去重、覆蓋及 SHA-256",
         "status": item["status"], "evidence": [str(source["path"]), f"rows={item['rows']}; missing_bars={item['missing_bars']}; sha256={item['sha256']}"]}
        for item, source in zip(audits, items, strict=True)
    ]
    report["metrics"] = {
        "training_start_utc": manifest["training_start_utc"], "snapshot_cutoff_utc": manifest["snapshot_cutoff_utc"],
        "warmup_bars": manifest["warmup_bars"], "datasets": audits,
        "total_rows": sum(item["rows"] for item in audits), "total_bytes": sum(item["bytes"] for item in audits),
        "verified_archive_count": sum(len(item["archive_sources"]) for item in items),
    }
    report["artifacts"] = [{"path": str(manifest_path), "role": "來源、下載區間及官方月檔雜湊清單"}] + [
        {"path": str(item["path"]), "role": item["interval"] + " 原始歷史 CSV", "sha256": str(item["sha256"])} for item in items
    ]
    report["residual_risks"] = [
        "下載完成時的固定快照，不是永久即時資料。交易所歷史可能日後修訂。",
        "九週期原始 K 線不代表具備五年 OI、訂單簿、新聞或 FinBERT 情緒。",
        "既有五週期特徵、模型、EXE 及 Kaggle 匯出包未更新；本次未使用 final holdout 做模型評估。",
    ]
    report["next_actions"] = ["批次建立九週期特徵及收盤時間因果對齊資料，再準備新版 Transformer V3 與 SAC 訓練包。"]
    write_json_atomic(report_path, report)
    lines = [
        "# BTC 五年九週期歷史資料驗收", "", f"- Report ID：{report['report_id']}",
        f"- Overall Status：{report['overall_status']}", f"- Snapshot UTC：{manifest['snapshot_cutoff_utc']}",
        f"- 研究起點 UTC：{manifest['training_start_utc']}；各週期另外保留 {manifest['warmup_bars']} 根暖機資料。",
        "", "## 執行摘要", report["summary"], "", "## 資料", "| 週期 | 根數 | 起點 UTC | 最後開盤 UTC | 缺漏 | MiB |",
        "|---|---:|---|---|---:|---:|",
    ]
    lines += [f"| {item['interval']} | {item['rows']:,} | {item['first_open_utc']} | {item['last_open_utc']} | {item['missing_bars']} | {item['bytes'] / 1024**2:.1f} |" for item in audits]
    lines += ["", f"合計 {report['metrics']['total_rows']:,} 根、{report['metrics']['total_bytes'] / 1024**2:.1f} MiB。",
              "", "## 驗證", "- 命令：`python scripts/verify_btc_history.py`。",
              "- 獨立重讀全部九份 CSV；跨批次檢查排序及重複，核對清單列數、覆蓋與 SHA-256。",
              "- 月檔逐一驗證官方 CHECKSUM，來源記錄於下載清單。", "", "## 剩餘限制"]
    lines += ["- " + value for value in report["residual_risks"]]
    lines += ["", "## 下一步"] + ["- " + value for value in report["next_actions"]]
    report_path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    validator = helper_path.with_name("validate_report.py")
    subprocess.run([sys.executable, str(validator), str(report_path)], check=True)
    print(f"驗收報告：{report_path.with_suffix('.md')}")
    return 0 if report["overall_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
