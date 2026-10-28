"""離線核對已下載資料，避免以下載成功取代資料內容驗證。"""

import argparse
import csv
from decimal import Decimal
import gzip
import json
from pathlib import Path
import re
import subprocess

from ai_quant_trading.operations.integrity import verify_artifact_manifest, sha256_file
from ai_quant_trading.persistence import write_json_atomic
from ai_quant_trading.research.external_traders import digest
from external_report_helpers import load_helper

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "驗證輸出已存在，請用新檔名")
    data, report = args.data.resolve(), args.report.resolve()
    require(
        not args.output.resolve().is_relative_to(data)
        and not args.output.resolve().is_relative_to(report),
        "驗證輸出不可改動已封存資料或報告",
    )
    verified = len(verify_artifact_manifest(data)) + len(verify_artifact_manifest(report))
    raw = []
    for path in (data / "raw").glob("*.json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            entry = json.load(stream)
        require(entry["data_sha256"] == digest(entry["data"]), "原始回應內容雜湊不符")
        raw.append(entry)
    selection = json.loads((data / "selected_accounts.json").read_text(encoding="utf-8"))
    summary = json.loads((data / "collection_summary.json").read_text(encoding="utf-8"))
    total_fills = total_btc = funding_count = 0
    for account in selection["accounts"]:
        alias, address = account["alias"], account["address"]
        rows = {}
        funding = {}
        for entry in raw:
            req = entry["request"]
            if req.get("user") != address:
                continue
            for row in entry["data"] if isinstance(entry["data"], list) else []:
                if req["type"] == "userFillsByTime":
                    rows[(row["coin"], str(row["tid"]), str(row["oid"]))] = row
                elif req["type"] == "userFunding" and row["delta"]["coin"] == "BTC":
                    funding[(row["time"], row["hash"], row["delta"]["coin"])] = row
        exported = read_csv(data / "accounts" / alias / "fills_all.csv")
        require(len(exported) == len(rows), "原始與整理後成交筆數不符")
        for row in exported:
            # CSV 對 @ 開頭現貨代號加單引號，還原只供鍵值核對。
            coin = row["coin"].removeprefix("'")
            original = rows[(coin, row["trade_id"], row["order_id"])]
            for new, old in (
                ("fee", "fee"),
                ("price", "px"),
                ("size", "sz"),
                ("closed_pnl", "closedPnl"),
                ("start_position", "startPosition"),
            ):
                require(Decimal(row[new]) == Decimal(original[old]), "成交數值或費用對帳失敗")
            require(row["training_eligible"] == "False", "資料不可取得訓練資格")
        btc = read_csv(data / "accounts" / alias / "fills_btc.csv")
        require(len(btc) == sum(r["coin"] == "BTC" for r in rows.values()), "BTC 篩選筆數不符")
        f = read_csv(data / "accounts" / alias / "funding_btc.csv")
        require(len(f) == len(funding), "資金費記錄數不符")
        require(
            sum((Decimal(r["signed_cashflow_usdc"]) for r in f), Decimal(0))
            == sum((Decimal(r["delta"]["usdc"]) for r in funding.values()), Decimal(0)),
            "資金費對帳不符",
        )
        total_fills += len(exported)
        total_btc += len(btc)
        funding_count += len(f)
    candles = read_csv(data / "normalized/btc_15m.csv")
    times = [int(row["time_ms"]) for row in candles]
    plan = summary["plan"]
    require(times == list(range(plan["start_ms"], plan["end_ms"], 900000)), "K線缺口或時間範圍不符")
    schema_errors = load_helper("validate_report").validate_report(
        json.loads((report / "report.json").read_text(encoding="utf-8"))
    )
    require(not schema_errors, "報告 schema 不符")
    for path in report.iterdir():
        if path.suffix in {".html", ".md", ".csv", ".json"}:
            require(
                not re.search(r"\b0x[0-9a-fA-F]{40}\b", path.read_text(encoding="utf-8-sig")),
                "分享摘要含完整地址",
            )
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", str(data / "selected_accounts.json")], cwd=ROOT
    )
    require(ignored.returncode == 0, "公開地址原始檔未受 Git 排除規則保護")
    output = dict(
        status="PASS",
        manifest_verified_files=verified,
        raw_responses=len(raw),
        all_fills=total_fills,
        btc_fills=total_btc,
        btc_funding_records=funding_count,
        candles=len(candles),
        raw_data_bytes=sum(p.stat().st_size for p in data.rglob("*") if p.is_file()),
        report_schema="PASS",
        report_address_redaction="PASS",
        git_ignore="PASS",
        data_manifest_sha256=sha256_file(data / "artifact_manifest.json"),
        report_manifest_sha256=sha256_file(report / "artifact_manifest.json"),
        scope="原始回應、整理表格、費用與資金費一致；非完整交易績效驗證",
    )
    write_json_atomic(args.output, output)
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
