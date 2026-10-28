"""以封存公開成交建立完整持倉圖表與明確限制的離線研究資料。"""

# ruff: noqa: E402
# 必須先限制數值運算執行緒，再匯入 numpy 等套件。
import os

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"

import argparse
import csv
import gzip
import html
import json
from pathlib import Path
import shutil
import subprocess
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ai_quant_trading.operations.integrity import (
    build_artifact_manifest,
    verify_artifact_manifest,
    sha256_file,
)
from ai_quant_trading.persistence import write_json_atomic
from ai_quant_trading.research.external_traders import digest, write_csv, utc
from ai_quant_trading.research.external_episodes import (
    FEATURES,
    SEQUENCE_LENGTH,
    SAMPLE_COLUMNS,
    pair_episodes,
    funding_evidence,
    prepare_training,
    mark_to_market,
    load_episode_dataset,
)
from external_report_helpers import load_helper

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/research/external_traders/20261028_pilot_v2"
DATA = ROOT / "data/research/external_traders/20261028_episodes_v2"
REPORT = ROOT / "artifacts/reports/20261028/external_episodes_v2"
PROTOCOL = ROOT / "docs/EXTERNAL_EPISODES_PROTOCOL_20261028.md"


def rows(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def progress(out, stage, **extra):
    write_json_atomic(out / "progress.json", dict(stage=stage, updated_at_utc=utc(), **extra))


def freeze(out):
    verify_artifact_manifest(SOURCE)
    inputs = [
        SOURCE / "artifact_manifest.json",
        PROTOCOL,
        Path(__file__),
        ROOT / "src/ai_quant_trading/research/external_episodes.py",
        ROOT / "src/ai_quant_trading/features/indicators.py",
    ]
    plan = dict(
        version="external_episodes_v1",
        source=str(SOURCE),
        hashes={str(p.relative_to(ROOT)): sha256_file(p) for p in inputs},
        sequence_length=SEQUENCE_LENGTH,
        features=FEATURES,
        split_ratios=[0.6, 0.2, 0.2],
        target="return_before_funding",
        full_cost_training_eligible=False,
        deploy_eligible=False,
        formal_training_executed=False,
    )
    path = out / "plan.json"
    out.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != plan:
            raise ValueError("研究設定或程式變更，請指定新目錄")
    else:
        write_json_atomic(path, plan)
        shutil.copy2(PROTOCOL, out / "protocol.md")
    return plan


def raw_funding():
    selection = json.loads((SOURCE / "selected_accounts.json").read_text(encoding="utf-8"))
    alias_for = {a["address"]: a["alias"] for a in selection["accounts"]}
    result = {alias: {} for alias in alias_for.values()}
    for path in (SOURCE / "raw").glob("*.json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            entry = json.load(stream)
        if digest(entry["data"]) != entry["data_sha256"]:
            raise ValueError("原始快取遭修改")
        if entry["request"]["type"] != "userFunding":
            continue
        alias = alias_for[entry["request"]["user"]]
        for r in entry["data"]:
            if r["delta"]["coin"] == "BTC":
                key = (r["time"], r["hash"], r["delta"]["coin"])
                result[alias][key] = r
    return {alias: list(records.values()) for alias, records in result.items()}


def chart(bars, episode, legs, marks, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import Rectangle
    import matplotlib.dates as mdates

    font = Path("C:/Windows/Fonts/msjh.ttc")
    if font.exists():
        plt.rcParams["font.family"] = FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    start, end = episode["entry_ms"], episode["last_ms"]
    if episode["right_censored"]:
        end = int(bars.time_ms.iloc[-1]) + 900000
    selected = bars[
        (bars.time_ms >= start - 900000 * 24) & (bars.time_ms <= end + 900000 * 16)
    ].copy()
    selected.index = pd.to_datetime(selected.time_ms, unit="ms", utc=True)
    interval = 15
    if len(selected) > 400:
        selected = (
            selected.resample("1h")
            .agg(dict(open="first", high="max", low="min", close="last", volume="sum"))
            .dropna()
        )
        interval = 60
    fig, (ax, curve) = plt.subplots(
        2, 1, figsize=(14, 8.2), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    for timestamp, r in selected.iterrows():
        x = mdates.date2num(timestamp.to_pydatetime())
        color = "#158568" if r.close >= r.open else "#c7424e"
        ax.vlines(x, r.low, r.high, color=color, linewidth=0.7)
        ax.add_patch(
            Rectangle(
                (x - interval / 1440 * 0.32, min(r.open, r.close)),
                interval / 1440 * 0.64,
                max(abs(r.close - r.open), 0.01),
                facecolor=color,
                edgecolor=color,
                linewidth=0.5,
            )
        )
    own = [r for r in legs if r["episode_id"] == episode["episode_id"]]
    for action, color, marker, label in [
        ("increase", "#176db3", "^", "開倉／加碼"),
        ("reduce", "#a12b81", "v", "減碼／平倉"),
    ]:
        r = [leg for leg in own if leg["action"] == action]
        ax.scatter(
            [
                mdates.date2num(pd.Timestamp(leg["time_ms"], unit="ms", tz="UTC").to_pydatetime())
                for leg in r
            ],
            [float(leg["price"]) for leg in r],
            s=35,
            marker=marker,
            c=color,
            label=label,
            zorder=4,
        )
    first, last = own[0], own[-1]
    ax.axvline(pd.Timestamp(start, unit="ms", tz="UTC"), color="#176db3", linestyle="--", alpha=0.5)
    ax.annotate(
        "首次進場",
        (pd.Timestamp(start, unit="ms", tz="UTC"), float(first["price"])),
        xytext=(15, 25),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->"},
    )
    if episode["complete"]:
        ax.axvline(
            pd.Timestamp(episode["exit_ms"], unit="ms", tz="UTC"),
            color="#a12b81",
            linestyle="--",
            alpha=0.5,
        )
        ax.annotate(
            "全部平倉",
            (pd.Timestamp(episode["exit_ms"], unit="ms", tz="UTC"), float(last["price"])),
            xytext=(15, -35),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->"},
        )
    trace = [m for m in marks if m["episode_id"] == episode["episode_id"]]
    if trace:
        curve.plot(
            pd.to_datetime([m["time_ms"] for m in trace], unit="ms", utc=True),
            [m["pnl_before_funding_usdc"] for m in trace],
            color="#176db3",
            linewidth=1.2,
        )
    else:
        curve.text(0.03, 0.5, "未完整平倉，不計算完整交易標籤", transform=curve.transAxes)
    curve.axhline(0, color="#777", linewidth=0.7)
    curve.set_ylabel("持倉損益 USDC\n已扣手續費，未含資金費")
    locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
    curve.xaxis.set_major_locator(locator)
    curve.xaxis.set_major_formatter(
        mdates.DateFormatter("%m/%d\n%H:%M", tz=ZoneInfo("Asia/Taipei"))
    )
    curve.set_xlabel(f"台北時間 UTC+8；圖示 {interval} 分 K；標記為實際成交，不是預測")
    ax.set_ylabel("BTC 成交價格 / USDC")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.15)
    curve.grid(alpha=0.15)
    direction = "做多" if episode["side"] == 1 else "做空"
    fig.suptitle(
        f"{episode['episode_id']} | {direction} | {'完整持倉' if episode['complete'] else '區間截斷'}",
        fontsize=15,
        x=0.08,
        ha="left",
    )

    def money(v):
        return "未知" if v is None else f"{v:,.4f}"

    ax.set_title(
        f"成交現金流損益 {money(episode['cashflow_pnl_usdc'])} - 手續費 {money(episode['fee_usdc'])} = 未含資金費 {money(episode['pnl_before_funding_usdc'])} USDC",
        fontsize=10,
        loc="left",
    )
    fig.text(
        0.08,
        0.02,
        "資金費無法精確分攤；原始停損／停利與預定盈虧比未知。下方不是完整帳戶權益。",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def make_report(out, data, episodes, checks, x, metadata, funding, audit):
    out.mkdir(parents=True, exist_ok=True)
    metrics = dict(
        episodes=len(episodes),
        complete=sum(e["complete"] for e in episodes),
        reconciled=sum(e["accounting_reconciled"] for e in episodes),
        open_or_censored=sum(not e["complete"] for e in episodes),
        accounting_mismatch=sum(e["complete"] and not e["accounting_reconciled"] for e in episodes),
        training_rows=len(metadata),
        sequence_shape=list(x.shape),
        split_counts={
            name: sum(r["split"] == name for r in metadata)
            for name in ("train", "validation", "research_test")
        },
        full_cost_training_rows=0,
        live_eligible=False,
        formal_training_ready=False,
        aggregated_funding_records=sum(f["aggregated_records"] for f in funding.values()),
    )
    table = [
        "| 持倉 | 方向 | 完整 | 對帳 | 研究資料 | 未含資金費損益 USDC | 持有小時 |",
        "|---|---|---|---|---|---:|---:|",
    ]

    def money(v):
        return "未知" if v is None else f"{v:.4f}"

    for e in episodes:
        table.append(
            f"| {e['episode_id']} | {'多' if e['side'] == 1 else '空'} | {e['complete']} | {e['accounting_reconciled']} | {e['dataset_eligible']} | {money(e['pnl_before_funding_usdc'])} | {money(e['holding_hours'])} |"
        )
    text = [
        "# 外部交易完整配對、圖表與訓練資料",
        "",
        "工程：PASS（僅已執行檢查）；研究充分性：證據不足；正式淨收益訓練與實盤：BLOCKED。",
        "",
        "## 實際結果",
        "```json",
        json.dumps(metrics, ensure_ascii=False, indent=2),
        "```",
        "",
        *table,
        "",
        "## 訓練用途",
        "可載入的是經對帳、暖機、時間切分及邊界清除後的 64×16 市場序列。方向是觀察到的條件，不是專家正確答案。",
        "目標 return_before_funding =（買賣現金流 − 實際手續費）／所有開倉名目金額，不是帳戶 ROI。",
        "使用既有 Wilder RSI/ATR、EMA、MACD 與布林通道定義；所有輸入均來自首次成交前已收盤 K 棒。沒有擬合全期 scaler。",
        "不能直接替換 Transformer V3 的完整淨收益標籤；不能當 SAC reward。提供安全載入器，預設拒絕未知成本。",
        "樣本只足以做載入與流程測試，不足正式訓練；本次 validation 為空，不移動切分邊界湊樣本。沒有未交易機會的反事實標籤。稅務未驗證。",
        "",
        "## 發現與限制",
        "1. 原始 funding 有 nSamples 聚合記錄；上一批 CSV 未保留此欄。本次從原始回應補回，舊資料不覆寫。",
        "2. 部分成交現金流與 API closedPnl 有差異，可能涉及供應商成本基礎或四捨五入，但未證實；不調大容差、不把差異偷偷列為費用。",
        "3. 未平倉、對帳失敗、特徵暖機不足、跨切分邊界的持倉保留紀錄，但不納入研究目標樣本。",
        "4. 已扣實際 fee，不再扣 builderFee。價差及滑價已反映於成交價，但缺參考報價，無法各自量化。",
        "5. 無原始停利停損委託，不畫假的保護線與盈虧比；出場原因保持 unknown。",
        "6. 公開榜單是事後抽樣，只有30天、少數帳戶，存在選擇偏誤；切分後仍只是研究測試。",
        "7. 跨帳戶交易可能高度重疊；提供重疊群與唯一性權重，但不是充分性保證。",
        "",
        "## 產物",
        f"- 資料：`{data}`",
        "- episodes.csv：全部持倉、成本、對帳及排除原因。",
        "- legs.csv：逐成交配對、反向拆分與費用；funding_evidence.json：聚合限制與觀察金額。",
        "- training/samples.csv 與 sequences.npz：特徵／目標／時間／權重；dataset_contract.json 記錄契約。",
        "- 每個持倉均有圖，未平倉會標示；mtm.csv 含已扣費、未含資金費的持倉浮損益路徑。",
        "",
        "## 下一步與停止條件",
        "先確認 funding 彙總涵蓋時間及 closedPnl 差異的來源，再補完整淨成本標籤。未能對帳就停止正式訓練，不用缺值補零。",
        "本次未正式訓練模型、未上傳 GitHub、未更新 EXE、未部署或交易。",
    ]
    (out / "report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    write_csv(out / "episodes_summary.csv", episodes)
    h = '<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>公開交易配對研究</title><style>body{font-family:system-ui;max-width:1300px;margin:30px auto;padding:0 20px;color:#222;background:#fff}table{border-collapse:collapse;width:100%}th,td{padding:9px;border-bottom:1px solid #ccc;text-align:right}th:first-child,td:first-child{text-align:left}a{color:#146fb0}.scroll{overflow:auto}p{line-height:1.8}img{width:100%;height:auto}</style><h1>公開交易配對與成本</h1>'
    h += f"<p>{metrics['complete']} 個完整持倉，{metrics['reconciled']} 個損益對帳通過，{metrics['training_rows']} 個未含資金費研究樣本。<br>目前僅適合資料管線測試，樣本不足正式訓練；完整淨成本資格未通過。點擊持倉編號查看圖片。</p>"
    h += '<p><a href="report.md">完整報告</a> · <a href="episodes_summary.csv">配對與成本表</a></p><div class=scroll><table><thead><tr><th>持倉／圖片</th><th>方向</th><th>成交毛損益</th><th>費用</th><th>未含資金費損益</th><th>持有小時</th><th>資料狀態</th></tr></thead><tbody>'
    for e in episodes:
        state = "研究樣本" if e["dataset_eligible"] else e["exclusion_reason"]
        h += f'<tr><td><a href="charts/{e["episode_id"]}.png">{e["episode_id"]}</a></td><td>{"多" if e["side"] == 1 else "空"}</td>'
        h += "".join(
            f"<td>{money(e[k])}</td>"
            for k in ("cashflow_pnl_usdc", "fee_usdc", "pnl_before_funding_usdc", "holding_hours")
        )
        h += f"<td>{html.escape(state)}</td></tr>"
    h += "</tbody></table></div><p>損益單位 USDC。資金費未能精確分攤，不能把此表當成完整帳戶收益或獲利策略排行。</p>"
    if episodes:
        h += f'<h2>第一個持倉（按時間，未挑選贏家）</h2><img src="charts/{episodes[0]["episode_id"]}.png"></html>'
    (out / "index.html").write_text(h, encoding="utf-8")
    revision = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        + "+working-tree"
    )
    r = load_helper("new_report").build_report(
        report_type="data_integrity",
        project="AIQuantTradingSystem",
        revision=revision,
        scope="公開成交配對與隔離研究資料",
    )
    r.update(
        overall_status="WARN",
        summary="完成配對、成本拆解、圖表與未含資金費研究資料；完整淨收益資格仍封鎖。",
        changes=["新增公開成交配對、交易圖表、因果特徵與安全資料載入器"],
        checks=checks,
        metrics=metrics,
        residual_risks=[
            "資金費聚合範圍未驗證",
            "部分 API PnL 與現金流有差異",
            "短區間、選擇偏誤及跨帳戶相關",
            "非 SAC 軌跡或完整淨收益資料",
        ],
        next_actions=["取得 funding 精確歸屬與成本基礎說明；解決前不正式訓練"],
    )
    r["findings"] = [
        dict(
            id="PAIR-001",
            severity="HIGH",
            status="OPEN",
            category="costs",
            component="funding",
            title="完整淨收益標籤不可用",
            evidence=[str(data / "funding_evidence.json")],
            impact="不能以缺少成本樣本替代既有模型契約",
            recommendation="保持隔離，先取得資金費彙總的精確時間語意",
        ),
        dict(
            id="PAIR-002",
            severity="MEDIUM",
            status="OPEN",
            category="accounting",
            component="closedPnl",
            title="部分現金流對帳差異待釐清",
            evidence=[str(data / "episodes.csv")],
            impact="差異樣本排除訓練",
            recommendation="驗證供應商成本基礎，不調大容差強行納入",
        ),
    ]
    r["artifacts"] = [
        dict(path=str(out / name), role=name, sha256=sha256_file(out / name))
        for name in ("report.md", "index.html", "episodes_summary.csv")
    ]
    errors = load_helper("validate_report").validate_report(r)
    if errors:
        raise ValueError(errors)
    write_json_atomic(out / "report.json", r)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    data, out = args.data.resolve(), args.report.resolve()
    if not data.is_relative_to(ROOT / "data/research/external_traders") or not out.is_relative_to(
        ROOT / "artifacts/reports"
    ):
        raise ValueError("產物必須位於專用研究目錄")
    if (data / "artifact_manifest.json").exists():
        verify_artifact_manifest(data)
        verify_artifact_manifest(out)
        print("已完成，既有成品完整性通過；不重跑。")
        return
    freeze(data)
    progress(data, "PLAN_FROZEN")
    source_plan = json.loads((SOURCE / "plan.json").read_text(encoding="utf-8"))
    bars = pd.read_csv(SOURCE / "normalized/btc_15m.csv")
    all_episodes, all_legs, audits, funding_audits = [], [], {}, {}
    raw = raw_funding()
    enriched = []
    for alias, funding in raw.items():
        episodes, legs, audit = pair_episodes(
            rows(SOURCE / "accounts" / alias / "fills_btc.csv"), alias
        )
        funding_audits[alias] = funding_evidence(episodes, funding)
        all_episodes.extend(episodes)
        all_legs.extend(legs)
        audits[alias] = audit
        enriched.extend(
            dict(
                alias=alias,
                time_ms=r["time"],
                cash_usdc=r["delta"]["usdc"],
                n_samples=r["delta"].get("nSamples"),
                aggregated=r["delta"].get("nSamples") is not None,
            )
            for r in funding
        )
    all_episodes.sort(key=lambda e: (e["entry_ms"], e["episode_id"]))
    progress(data, "PAIRING_COMPLETE", episodes=len(all_episodes))
    x, metadata, bounds = prepare_training(
        all_episodes, bars, source_plan["start_ms"], source_plan["end_ms"]
    )
    write_csv(data / "episodes.csv", all_episodes)
    write_csv(data / "legs.csv", all_legs)
    write_csv(data / "funding_enriched.csv", enriched)
    write_json_atomic(data / "pairing_audit.json", audits)
    write_json_atomic(data / "funding_evidence.json", funding_audits)
    training = data / "training"
    training.mkdir(exist_ok=True)
    write_csv(training / "samples.csv", metadata, SAMPLE_COLUMNS)
    np.savez_compressed(training / "sequences.npz", x=x)
    contract = dict(
        version="external_observed_episode_v1",
        source_manifest_sha256=sha256_file(SOURCE / "artifact_manifest.json"),
        feature_columns=FEATURES,
        sequence_length=SEQUENCE_LENGTH,
        split_boundaries_ms=bounds,
        target="return_before_funding",
        label_definition="(execution_cashflow-fees)/total_opening_notional",
        decision_proxy="first_observed_fill_time_minus_1ms",
        live_eligible=False,
        full_cost_training_eligible=False,
        selection_bias=True,
        source_venue="Hyperliquid",
        not_compatible_with="Transformer V3 net target / SAC reward",
        no_scaler_fitted=True,
        side_is_observed_condition=True,
        formal_training_ready=False,
        tax_status="unverified",
        allowed_purpose="offline_data_pipeline_smoke_only",
    )
    write_json_atomic(training / "dataset_contract.json", contract)
    build_artifact_manifest(training, files=[p for p in training.rglob("*") if p.is_file()])
    batch = load_episode_dataset(
        training, target="return_before_funding", allow_missing_funding=True
    )
    if len(batch["y"]) != len(metadata) or not np.isfinite(batch["y"]).all():
        raise ValueError("資料載入煙霧檢查失敗")
    try:
        load_episode_dataset(training)
    except ValueError:
        pass
    else:
        raise ValueError("完整淨成本閘門未擋住未知資金費")
    marks = [row for e in all_episodes for row in mark_to_market(e, all_legs, bars)]
    write_csv(data / "mtm.csv", marks)
    for e in all_episodes:
        trace = [m for m in marks if m["episode_id"] == e["episode_id"]]
        if (
            trace
            and abs(trace[-1]["pnl_before_funding_usdc"] - e["pnl_before_funding_usdc"]) > 1e-5
        ):
            raise ValueError("持倉損益尾端與結算帳務不符")
    progress(data, "DATASET_COMPLETE", shape=list(x.shape))
    charts = out / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    for i, e in enumerate(all_episodes):
        path = charts / f"{e['episode_id']}.png"
        if not path.exists():
            chart(bars, e, all_legs, marks, path)
        print(f"CHART {i + 1}/{len(all_episodes)} {e['episode_id']}", flush=True)
    checks = [
        dict(
            id="PAIR-SOURCE",
            name="封存來源雜湊",
            status="PASS",
            evidence=[str(SOURCE / "artifact_manifest.json")],
        ),
        dict(
            id="PAIR-LOAD",
            name="64棒序列載入及完整成本預設拒絕",
            status="PASS",
            evidence=[str(training / "dataset_contract.json")],
        ),
        dict(
            id="PAIR-MTM",
            name="持倉浮損益結尾對帳",
            status="PASS",
            evidence=[str(data / "mtm.csv")],
        ),
        dict(
            id="PAIR-CASH",
            name="所有持倉現金流對帳",
            status="WARN"
            if any(e["complete"] and not e["accounting_reconciled"] for e in all_episodes)
            else "PASS",
            evidence=[str(data / "episodes.csv")],
        ),
    ]
    metrics = make_report(out, data, all_episodes, checks, x, metadata, funding_audits, audits)
    write_json_atomic(data / "summary.json", metrics)
    progress(data, "COMPLETE", report=str(out / "index.html"))
    build_artifact_manifest(data, files=[p for p in data.rglob("*") if p.is_file()])
    build_artifact_manifest(out, files=[p for p in out.rglob("*") if p.is_file()])
    verify_artifact_manifest(data)
    verify_artifact_manifest(out)
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
