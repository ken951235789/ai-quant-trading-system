"""只讀既有門檻研究，產出完整成交圖與核對報告，不重訓或改部署。"""

# ruff: noqa: E402
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from ai_quant_trading.operations.integrity import build_artifact_manifest, sha256_file
from ai_quant_trading.research.probability_sweep import trade_metrics
from ai_quant_trading.research.trade_review import contained_file, render_example_png, review_trades
from audit_candidate_training_research import verify_trade_table


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", required=True, type=Path)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sweep, study, output = args.sweep.resolve(), args.study.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("請指定新目錄，保留既有研究")
    evidence = []

    def checked(root, relative):
        path = contained_file(root, relative)
        expected = read_json(root / "artifact_manifest.json")["artifacts"][relative]["sha256"]
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"來源雜湊不符：{relative}")
        evidence.append({"path": str(path), "sha256": actual})
        return path

    plan = read_json(checked(sweep, "plan.json"))
    if sha256_file(study / "artifact_manifest.json") != plan["input_manifest_sha256"]:
        raise ValueError("訓練來源清單不符")
    table = pd.read_csv(checked(sweep, "all_thresholds.csv"))
    baseline = pd.read_csv(checked(sweep, "baselines.csv"))
    source = checked(study, "frozen_fold_3.csv")
    source_hash = sha256_file(source)
    bars = pd.read_csv(source, usecols=["timestamp", "open", "high", "low", "close", "volume"])
    chosen = table[(table.family == "original") & (table.variant == "F_existing")
                   & table.threshold_pct.isin([32, 33])].copy()
    examples = chosen[(chosen.seed == 137) & (chosen.threshold_pct == 32)
                      & (chosen.split == "test") & (chosen.scenario == "base")]
    if len(examples) != 1:
        raise ValueError("圖表設定不唯一")
    # 種子不合併為獨立樣本。逐一核對本次比較的所有成交與摘要。
    cache = {}
    for row in chosen.to_dict("records"):
        filename = row["schedule_file"]
        if filename not in cache:
            trades = pd.read_csv(checked(sweep, filename))
            verify_trade_table(trades, 4)
            reviewed = review_trades(bars, trades, source_hash)
            if len(trades) and set(trades.contract_sha256) != {row["contract_sha256"]}:
                raise ValueError("契約雜湊不一致")
            cache[filename] = (trades, reviewed, trade_metrics(trades))
        metrics = cache[filename][2]
        for field in ("trades", "actual_win_rate", "mean_net", "payoff_ratio", "settled_return_10pct"):
            expected, actual = row[field], metrics[field]
            if pd.isna(expected):
                if actual is not None:
                    raise ValueError(f"空值不一致：{field}")
            else:
                np.testing.assert_allclose(expected, actual, atol=1e-11, rtol=1e-9)

    output.mkdir(parents=True)
    chosen.to_csv(output / "comparison_32_33.csv", index=False)
    rows, trades, metrics = cache[examples.iloc[0].schedule_file]
    rows.to_csv(output / "all_10_trades.csv", index=False)
    save_json(output / "trade_annotations.json", trades)
    for number, trade in enumerate(trades, 1):
        render_example_png(bars, trade, output / f"trade_{number:02d}.png",
            context_label="原突破 / Transformer 舊特徵 / 門檻 32% / seed 137",
            trade_label=f"Test 第 {number}/{len(trades)} 筆（完整依時序列出） / 基本成本")

    base_test = chosen[(chosen.threshold_pct == 32) & (chosen.split == "test") & (chosen.scenario == "base")]
    current = table[(table.threshold_pct == 55) & (table.split == "test") & (table.scenario == "base")]
    vwap = baseline[(baseline.seed == 42) & (baseline.variant == "F_existing")
                    & (baseline.family == "vwap_reversion") & (baseline.split == "test")
                    & (baseline.scenario == "base")].iloc[0]
    lines = ["# 預測機率、實際勝率與逐筆 K 線", "",
        "## 結論", "",
        "舊特徵原突破在事後掃描的 32% / 33% 機率門檻，三 seeds 的基本成本 Test 均為正，但只有 5–15 筆成交，證據不足。",
        "這是 Transformer 篩選固定候選規則的回放，並非 SAC 自主進出場，也不代表目前部署機器人的實績。",
        f"現行 55%：此批 {len(current)} 個模型的 Test 合計 {int(current.trades.sum())} 成交；零成交不等於營利。",
        "門檻是模型預測『淨收益 >2 bps』的機率，另保留預測淨收益 >2 bps 條件；不是實際勝率。",
        "", "## 32% 門檻比較", "",
        "| Seed | 成交 | 實際勝率 | 淨盈虧比 | 損益平衡勝率 | 每筆平均淨報酬 | 10%配置結算報酬 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in base_test.itertuples():
        lines.append(f"| {r.seed} | {r.trades} | {r.actual_win_rate:.2%} | {r.payoff_ratio:.3f}:1 | {r.breakeven_win_rate:.2%} | {r.mean_net:+.4%} | {r.settled_return_10pct:+.4%} |")
    lines += ["", "相同盈虧分布下，損益平衡勝率 = 1 / (1 + 淨盈虧比)，不是另一個可直接設定的模型門檻。",
        "32% 基本成本 Selection：seed 42 與 2026 為負；seed 137 在壓力成本 Test 轉負。置信區間未支持穩定正期望。",
        "33% 的 seed 2026 為 14 筆、57.14% 勝率、平均 +0.3482%；其他 seeds 並未同幅改善。不能用反覆查看的 Test 選出正式門檻。",
        f"無 AI VWAP 對照為 {vwap.trades} 筆、勝率 {vwap.actual_win_rate:.2%}、淨盈虧比 {vwap.payoff_ratio:.3f}:1、平均淨報酬 {vwap.mean_net:+.4%}，壓力成本仍轉負。",
        "", "## 出場與成本", "",
        "- 訊號收盤後，下一根 15m 開盤進場；停損固定 2 ATR、停利 4 ATR，預定報酬:風險為 2:1。",
        "- 原突破方向狀態消失時提前退出；最多持有 32 根（8 小時）；出場後冷卻 4 根訊號棒。",
        "- 同棒觸及停損與停利先停損；跳空可能使停損更差。棒內時間未知，不把出場棒標記當成精確成交秒數。",
        "- 基本成本假設：單邊手續費 5 bps、單邊滑價 2 bps、全價差 1 bps；跨結算時點 funding 準備金 1 bps，非歷史實際 funding。",
        "- 價差與滑價已在成交價，不能再重扣。壓力成本滑價為單邊 5 bps，各情境重新撮合，不是固定路徑歸因比較。",
        "- 單筆實現 R = 淨收益 / 初始停損比例。多笔平均盈虧比與原定 2:1 不同，成本及提前退出會改變结果。",
        "", "## 完整成交圖", "",
        "以下固定展示 seed 137 的全部 10 筆，非依最高收益挑選種子或只顯示獲利交易。台北 UTC+8；出場後 K 線是歷史，不是預測。"]
    for number, t in enumerate(trades, 1):
        lines += ["", f"### 第 {number} 筆：{'做多' if t['side'] == 1 else '做空'}，淨收益 {t['net_return']:+.3%}，實現 {t['net_r']:+.3f}R",
                  "", f"![第 {number} 筆](trade_{number:02d}.png)"]
    lines += ["", "## 驗證與限制", "",
        f"本次核對 {len(chosen)} 列摘要與 {len(cache)} 份成交排程的來源、帳務及統計；不重新訓練、不重新撮合。",
        "工程核對 PASS；研究品質：證據不足；實盤資格：BLOCKED。不能把三 seeds 的相同歷史合併為独立證據。",
        "約 2025-09 至 2026-03 的 Test 已被重複研究，不是五年樣本外績效或全新 holdout。",
        "10% 配置報酬僅結算權益，未驗證持倉浮虧、保證金、清算或稅；資金費率只是準備金。",
        "下一步先固定候選門檻並在未看過的時段驗證成本後期望；跨期或壓力成本未通過則停止升級，勿加槓桿掩蓋。",
        "EXE：模型中心 → 模型回測 → 逐筆進出場 → 原突破 / Transformer / 舊特徵 / seed 137 / 32% / Test / 基本成本。",
        "本次未更改部署模型、55% 門檻、交易規則或 EXE，也未上傳 GitHub。"]
    with (output / "report.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    save_json(output / "verification.json", {"sources": evidence, "summary_rows": len(chosen),
        "schedules_checked": len(cache), "charts": len(trades), "account_equity_validated": False,
        "command": [sys.executable, *sys.argv]})
    spec = importlib.util.spec_from_file_location("report_helper", ROOT / "tools/research_reports/new_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.build_report(report_type="model_research", project="AIQuantTradingSystem",
        revision=plan["revision"] + " + working tree", scope="既有 32/33% 門檻結果與逐筆成交視覺核對")
    report.update(overall_status="WARN", summary="部分歷史回測為正，但跨期、成本壓力與樣本不足，不改部署。",
        changes=["新增全部 10 筆已保存成交的 K 線圖、R 與來源核對報告。"],
        checks=[{"id": "CHECK-001", "name": "雜湊、成交時間與成本統計一致", "status": "PASS", "evidence": ["verification.json"]},
                {"id": "LIVE-001", "name": "實盤資格", "status": "BLOCKED", "evidence": ["report.md"]}],
        metrics={"engineering": "PASS", "research_quality": "INSUFFICIENT_EVIDENCE", "live_eligible": False,
                 "example_seed": 137, "threshold_pct": 32, "example_metrics": metrics},
        artifacts=[{"path": p.name, "role": "research_evidence", "sha256": sha256_file(p)} for p in sorted(output.iterdir())],
        residual_risks=["門檻為事後診斷，非正式選擇。", "樣本少且跨期、壓力成本不穩定。", "完整帳戶浮虧、清算、實際 funding、稅未驗證。"],
        next_actions=["固定候選後使用未看過時段與成本壓力驗證，不提高槓桿或更換部署。"])
    save_json(output / "report.json", report)
    build_artifact_manifest(output, files=list(output.iterdir()))
    print(json.dumps({"output": str(output), "charts": len(trades), "schedules_checked": len(cache)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
