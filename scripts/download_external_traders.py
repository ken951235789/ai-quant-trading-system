"""下載有限公開交易樣本；相同目錄續跑不改選樣本、不重新抓取已保存回應。"""

# ruff: noqa: E402
# 必須先限制數值運算執行緒，再匯入研究工具。
import os

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"

import argparse
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import html
import json
from pathlib import Path
import shutil
import subprocess

from ai_quant_trading.operations.integrity import (
    build_artifact_manifest,
    verify_artifact_manifest,
    sha256_file,
)
from ai_quant_trading.persistence import write_json_atomic
from ai_quant_trading.research.external_traders import (
    DOCUMENTS,
    LEADERBOARD,
    PublicClient,
    digest,
    history,
    normalize_candles,
    normalize_fills,
    normalize_funding,
    number,
    position_checks,
    save_progress,
    select_accounts,
    utc,
    write_csv,
)
from external_report_helpers import load_helper

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/research/external_traders/20261028_pilot_v1"
REPORT = ROOT / "artifacts/reports/20261028/external_traders_v1"
PROTOCOL = ROOT / "docs/EXTERNAL_TRADER_DATA_PROTOCOL_20261028.md"
SOURCES = [
    dict(
        source="Hyperliquid",
        url="https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint",
        data="公開成交、資金費、帳務、持倉、BTC K線",
        status="本次有限下載",
        limitation="最近10000筆成交限制；非完整交易生涯；不同於 Binance",
    ),
    dict(
        source="Hyperliquid 排行發現",
        url=LEADERBOARD,
        data="公開帳戶抽樣框",
        status="只保存六個樣本及快照雜湊",
        limitation="非穩定文件化 API；有排行榜與存活偏誤",
    ),
    dict(
        source="Binance Copy Trading",
        url="https://www.binance.com/en/support/faq/detail/607c124c574a417bbf734f9035ee9d28",
        data="公開績效指標與交易資訊",
        status="來源目錄，未下載帳戶交易明細",
        limitation="未確認可匿名批次匯出完整歷史；不登入或繞過限制",
    ),
    dict(
        source="Myfxbook",
        url="https://www.myfxbook.com/help/knowledge-base/verification/",
        data="公開帳戶驗證與績效",
        status="來源目錄，未下載帳戶交易明細",
        limitation="需個別核對公開權限與驗證；非 BTC 同場所樣本",
    ),
    dict(
        source="MQL5 / MetaTrader",
        url="https://www.metatrader5.com/en/terminal/help/signals/signal_monitoring",
        data="公開訊號監測與歷史",
        status="來源目錄，未下載帳戶交易明細",
        limitation="監測區間不等於完整生涯；商品與券商成本不同",
    ),
    dict(
        source="Hyperliquid S3",
        url="https://hyperliquid.gitbook.io/hyperliquid-docs/historical-data",
        data="較長期歷史封存",
        status="未下載",
        limitation="requester-pays，未授權付費；不啟用",
    ),
]


def freeze(folder):
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "plan.json"
    hashes = {
        p.relative_to(ROOT).as_posix(): sha256_file(p)
        for p in (
            PROTOCOL,
            Path(__file__),
            ROOT / "src/ai_quant_trading/research/external_traders.py",
        )
    }
    if target.exists():
        plan = json.loads(target.read_text(encoding="utf-8"))
        if plan["implementation_hashes"] != hashes:
            raise ValueError("程式或計畫已改變；請使用新資料及報告目錄，保留既有資料")
        return plan
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    plan = dict(
        version="external_traders_v1",
        frozen_at_utc=utc(),
        start_ms=int((end - timedelta(days=30)).timestamp() * 1000),
        end_ms=int(end.timestamp() * 1000),
        implementation_hashes=hashes,
        per_stratum=3,
        max_pages=8,
        request_budget=180,
        training_eligible=False,
        live_eligible=False,
        seed="external_traders_v1",
    )
    write_json_atomic(target, plan)
    shutil.copy2(PROTOCOL, folder / "protocol.md")
    save_progress(folder, "PLAN_FROZEN")
    return plan


def collect(folder, plan):
    client = PublicClient(folder / "raw")
    selected_path = folder / "selected_accounts.json"
    if selected_path.exists():
        selection = json.loads(selected_path.read_text(encoding="utf-8"))
        if selection["accounts_sha256"] != digest(selection["accounts"]):
            raise ValueError("抽樣清單完整性失敗")
    else:
        body, meta = client.fetch(LEADERBOARD, limit=80 * 1024 * 1024)
        snapshot = json.loads(body)
        accounts, strata = select_accounts(snapshot, plan["per_stratum"])
        if not accounts:
            raise ValueError("公開抽樣框無有效樣本")
        selection = dict(
            accounts=accounts,
            accounts_sha256=digest(accounts),
            population_rows=len(snapshot["leaderboardRows"]),
            strata=strata,
            snapshot=meta,
            full_snapshot_retained=False,
        )
        write_json_atomic(selected_path, selection)
        # 不持久保存未選取的地址或名稱，減少不必要的外部個資聚集。
        del snapshot, body
    save_progress(folder, "SAMPLE_FROZEN", accounts=len(selection["accounts"]))
    docs = folder / "source_documents"
    docs.mkdir(exist_ok=True)
    documents, errors = [], []
    for name, url in DOCUMENTS.items():
        path = docs / f"{name}.md"
        try:
            if not path.exists():
                body, meta = client.fetch(url, limit=2 * 1024 * 1024)
                with path.open("xb") as stream:
                    stream.write(body)
                write_json_atomic(docs / f"{name}.json", meta)
            meta = json.loads((docs / f"{name}.json").read_text(encoding="utf-8"))
            if sha256_file(path) != meta["sha256"]:
                raise ValueError("來源文件雜湊不符")
            documents.append(name)
        except Exception as exc:
            errors.append(dict(component=f"document_{name}", error_type=type(exc).__name__))
    start, end = plan["start_ms"], plan["end_ms"]
    candle_audit = dict(status="FAIL")
    try:
        raw = client.info(
            dict(
                type="candleSnapshot",
                req=dict(coin="BTC", interval="15m", startTime=start, endTime=end),
            )
        )
        candles, candle_audit = normalize_candles(raw, start, end)
        candle_audit["status"] = "PASS" if candle_audit["missing_bars"] == 0 else "WARN"
        write_csv(folder / "normalized/btc_15m.csv", candles)
    except Exception as exc:
        errors.append(dict(component="btc_candles", error_type=type(exc).__name__))
    summaries = []
    for account in selection["accounts"]:
        alias, address = account["alias"], account["address"]
        target = folder / "accounts" / alias
        target.mkdir(parents=True, exist_ok=True)
        record = dict(
            alias=alias,
            stratum=account["stratum"],
            status="PASS",
            fills_all=0,
            fills_btc=0,
            distinct_btc_orders=0,
            btc_funding_records=0,
            ledger_records=0,
            training_eligible=False,
            live_eligible=False,
            lifetime_complete=False,
        )
        metadata = {}
        for kind in ("userFillsByTime", "userFunding", "userNonFundingLedgerUpdates"):
            try:
                rows, audit = history(client, kind, address, start, end, plan["max_pages"])
                metadata[kind] = audit
                if audit["warnings"]:
                    record["status"] = "WARN"
                if kind == "userFillsByTime":
                    fills = normalize_fills(rows, alias)
                    btc = [r for r in fills if r["coin"] == "BTC"]
                    write_csv(target / "fills_all.csv", fills)
                    write_csv(target / "fills_btc.csv", btc)
                    record.update(
                        fills_all=len(fills),
                        fills_btc=len(btc),
                        distinct_btc_orders=len({r["order_id"] for r in btc}),
                        **position_checks(btc),
                    )
                    # 這些只是觀察片段的帳務總和，絕非完整交易收益或帳戶 ROI。
                    record["observed_btc_closed_pnl"] = (
                        str(sum((number(r["closed_pnl"]) for r in btc), Decimal(0)))
                        if btc
                        else None
                    )
                    record["observed_btc_fee_usdc"] = (
                        str(sum((number(r["fee"]) for r in btc), Decimal(0)))
                        if btc and all(r["fee_token"] == "USDC" for r in btc)
                        else None
                    )
                    record["first_btc_fill_utc"] = btc[0]["timestamp_utc"] if btc else None
                    record["last_btc_fill_utc"] = btc[-1]["timestamp_utc"] if btc else None
                elif kind == "userFunding":
                    funding = normalize_funding(rows, alias)
                    write_csv(target / "funding_all.csv", funding)
                    btc = [r for r in funding if r["coin"] == "BTC"]
                    write_csv(target / "funding_btc.csv", btc)
                    record["btc_funding_records"] = len(btc)
                    record["observed_btc_funding_usdc"] = (
                        str(sum((number(r["signed_cashflow_usdc"]) for r in btc), Decimal(0)))
                        if btc
                        else None
                    )
                else:
                    record["ledger_records"] = len(rows)
                    # 帳務對手地址只保存在隔離原始回應，不放入報告。
                    write_json_atomic(
                        target / "ledger_event_types.json",
                        sorted({r["delta"]["type"] for r in rows}),
                    )
            except Exception as exc:
                record["status"] = "FAIL"
                errors.append(dict(component=f"{alias}_{kind}", error_type=type(exc).__name__))
        try:
            state = client.info(dict(type="clearinghouseState", user=address))
            if not isinstance(state, dict) or "assetPositions" not in state:
                raise ValueError("持倉快照格式不符")
            record["snapshot_received"] = True
        except Exception as exc:
            record["snapshot_received"] = False
            record["status"] = "FAIL"
            errors.append(dict(component=f"{alias}_state", error_type=type(exc).__name__))
        write_json_atomic(target / "integrity.json", dict(summary=record, endpoints=metadata))
        summaries.append(record)
        save_progress(
            folder, "ACCOUNT_COLLECTION", completed=[r["alias"] for r in summaries], errors=errors
        )
        print(
            f"{alias}: {record['status']} fills={record['fills_all']} BTC={record['fills_btc']}",
            flush=True,
        )
    result = dict(
        plan=plan,
        accounts=summaries,
        candles=candle_audit,
        documents=documents,
        errors=errors,
        network_attempts_this_run=client.attempts,
        selection_population=selection["population_rows"],
        engineering_status="PASS" if not errors else "PARTIAL",
        research_status="INSUFFICIENT",
        live_status="BLOCKED",
    )
    write_json_atomic(folder / "collection_summary.json", result)
    save_progress(
        folder, "COLLECTION_COMPLETE" if not errors else "COLLECTION_PARTIAL", errors=errors
    )
    return result


def report(folder, out, result):
    out.mkdir(parents=True, exist_ok=True)
    rows = result["accounts"]
    columns = list(dict.fromkeys(k for row in rows for k in row))
    write_csv(out / "accounts_summary.csv", rows, columns)
    write_csv(out / "sources.csv", SOURCES)
    write_json_atomic(out / "sources.json", SOURCES)
    plan = result["plan"]
    text = [
        "# 公開交易紀錄下載與完整性報告",
        "",
        f"取得區間 UTC：{utc(plan['start_ms'])} 至 {utc(plan['end_ms'])}（終點不含）。",
        "",
        f"工程下載狀態：{result['engineering_status']}；研究：證據不足；實盤：BLOCKED。",
        "",
        "本次資料為 Hyperliquid 公開帳戶，不是 Binance。六個樣本按固定雜湊抽樣，月 PnL 正負各三個；並非六個獲利策略。",
        "",
        "| 代號 | 抽樣組 | 全商品成交片段 | BTC成交片段 | BTC訂單ID數 | BTC資金費記錄 | 下載檢查 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        text.append(
            f"| {row['alias']} | {row['stratum']} | {row['fills_all']} | {row['fills_btc']} | {row['distinct_btc_orders']} | {row['btc_funding_records']} | {row['status']} |"
        )
    text += [
        "",
        "## 保存內容",
        f"- 本機隔離資料：`{folder}`",
        f"- BTC 15m K線檢查：`{json.dumps(result['candles'], ensure_ascii=False)}`",
        f"- 已保存官方文件：{', '.join(result['documents'])}",
        "- 原始 API 回應以 gzip JSON 保存；包含請求、取得時間及雜湊。CSV 是整理後的觀察資料。",
        "- 部分成交不等於一次完整交易；訂單 ID 數也不等於來回交易數。",
        "",
        "## 不能直接得出的結論",
        "- API 最近10000筆限制、排行榜選擇偏誤及未知其他帳戶對沖，無法確認完整人生績效。",
        "- 起始或結束仍有持倉、持倉銜接缺口、同毫秒順序不明，都可能妨礙還原進出場。",
        "- fee 已含 builderFee，不能再扣一次；資金費獨立保存。未知滑價、價差、稅務、未實現損益不能補零。",
        "- 不產生收益率排名、勝率或可供交易的訊號；本批資料不進入模型訓練。",
        "- 排行發現清單只保存選中帳戶，未保存全榜；可重現選中資料處理，但不能從本機重算全榜抽樣。",
        "",
        "## 其他來源",
        *[f"- [{s['source']}]({s['url']})：{s['status']}。{s['limitation']}。" for s in SOURCES],
        "",
        "## 未完成與下一步",
        "先檢查可對帳的 BTC 持倉片段，再決定是否做持有時間及市場情境分析；尚未完成交易策略還原或獲利驗證。",
        f"下載或格式錯誤：`{json.dumps(result['errors'], ensure_ascii=False)}`。空資料不代表沒有交易生涯。",
        "",
        "## 續跑",
        "相同命令會沿用固定樣本與時間範圍，使用驗證後的快取；已完成則只驗證清單。新研究請指定新的 data/report 路徑。",
    ]
    (out / "report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    # HTML 只放匿名摘要，所有外部文字跳脫，不顯示原始公開地址。
    headers = ["代號", "抽樣組", "所有成交片段", "BTC成交片段", "BTC訂單ID數", "資金費記錄", "檢查"]
    fields = [
        "alias",
        "stratum",
        "fills_all",
        "fills_btc",
        "distinct_btc_orders",
        "btc_funding_records",
        "status",
    ]
    table = (
        "<table><thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead><tbody>"
    )
    table += (
        "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(row[k]))}</td>" for k in fields) + "</tr>"
            for row in rows
        )
        + "</tbody></table>"
    )
    page = '<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>公開交易資料</title><style>body{font-family:system-ui;margin:32px;max-width:1120px;color:#222;background:#fff}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ccc;padding:12px;text-align:left}a{color:#0964ab}.scroll{overflow:auto}p{line-height:1.8}</style><h1>公開交易紀錄研究素材</h1>'
    page += f"<p>資料期間（UTC）：{utc(plan['start_ms'])} 至 {utc(plan['end_ms'])}<br>研究證據不足，未接入模型或交易。</p><div class=scroll>{table}</div>"
    page += '<p>成交片段不是完整來回交易。排行榜抽樣不代表可複製的獲利策略。</p><p><a href="report.md">完整中文報告</a> · <a href="accounts_summary.csv">帳戶資料摘要 CSV</a> · <a href="sources.csv">來源目錄 CSV</a></p></html>'
    (out / "index.html").write_text(page, encoding="utf-8")
    revision = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        + "+working-tree"
    )
    r = load_helper("new_report").build_report(
        report_type="data_integrity",
        project="AIQuantTradingSystem",
        revision=revision,
        scope="公開交易資料隔離下載",
    )
    r.update(
        overall_status="WARN",
        summary="已保存有限公開樣本；非完整歷史，禁止用作獲利排名或直接訓練。",
        changes=["新增唯讀下載器、抽樣及隔離驗證、來源目錄與中文報告"],
        checks=[
            dict(
                id="EXT-COLLECT",
                name="公開資料下載",
                status="PASS" if not result["errors"] else "WARN",
                evidence=[str(folder / "collection_summary.json")],
            ),
            dict(
                id="EXT-CANDLES",
                name="同場所已收盤K線完整性",
                status=result["candles"]["status"],
                evidence=[str(folder / "normalized/btc_15m.csv")],
            ),
        ],
        metrics={
            "accounts": len(rows),
            "fills": sum(r["fills_all"] for r in rows),
            "btc_fills": sum(r["fills_btc"] for r in rows),
            "candles": result["candles"],
            "errors": result["errors"],
            "training_eligible": False,
            "live_eligible": False,
        },
        residual_risks=[
            "非完整歷史、抽樣與存活偏誤",
            "未完成完整持倉及現金流對帳",
            "Binance、Myfxbook、MQL5 僅目錄，未下載交易明細",
        ],
        next_actions=["只分析可對帳的 BTC 片段；不能還原持倉與成本則停止策略績效推論"],
    )
    r["findings"] = [
        dict(
            id="EXT-001",
            severity="MEDIUM",
            status="OPEN",
            category="data_completeness",
            component="external_traders",
            title="公開資料不等於完整且可模仿的策略",
            evidence=[str(folder / "collection_summary.json")],
            impact="不能直接當作正式訓練或績效證據",
            recommendation="保持隔離並先做持倉路徑與成本對帳",
        )
    ]
    r["artifacts"] = [
        dict(path=str(out / name), role=name, sha256=sha256_file(out / name))
        for name in ("report.md", "index.html", "accounts_summary.csv", "sources.csv")
    ]
    errors = load_helper("validate_report").validate_report(r)
    if errors:
        raise ValueError(errors)
    write_json_atomic(out / "report.json", r)
    build_artifact_manifest(out, files=[p for p in out.rglob("*") if p.is_file()])
    verify_artifact_manifest(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    folder, out = args.data.resolve(), args.report.resolve()
    if not folder.is_relative_to(ROOT / "data/research/external_traders"):
        raise ValueError("原始資料必須放在專用隔離資料夾")
    if not out.is_relative_to(ROOT / "artifacts/reports"):
        raise ValueError("報告必須放在研究報告資料夾")
    if (folder / "artifact_manifest.json").exists():
        verify_artifact_manifest(folder)
        verify_artifact_manifest(out)
        print("已完成，雜湊驗證通過；未重抓或覆寫。")
        return
    plan = freeze(folder)
    try:
        result = collect(folder, plan)
        report(folder, out, result)
        if not result["errors"]:
            save_progress(folder, "COMPLETE", report=str(out / "index.html"))
            build_artifact_manifest(folder, files=[p for p in folder.rglob("*") if p.is_file()])
            verify_artifact_manifest(folder)
        print(
            json.dumps(
                dict(
                    status=result["engineering_status"],
                    report=str(out / "index.html"),
                    errors=result["errors"],
                ),
                ensure_ascii=False,
            )
        )
    except Exception as exc:
        save_progress(
            folder,
            "INTERRUPTED",
            error_type=type(exc).__name__,
            resume="原命令續跑；不變更固定樣本與區間",
        )
        # 錯誤類型即可定位；不將可能含地址或服務回應的例外全文寫入畫面。
        print(f"下載中斷：{type(exc).__name__}；已保存進度。")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
