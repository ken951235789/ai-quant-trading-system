"""獨立核對逐筆來源守恆、樣本因果時間、交易圖與訓練批次。"""

# ruff: noqa: E402
# 必須先限制數值運算執行緒，再匯入 numpy / PyTorch。
import os

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"

import argparse
import csv
from decimal import Decimal
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image

from ai_quant_trading.operations.integrity import verify_artifact_manifest
from ai_quant_trading.persistence import write_json_atomic
from ai_quant_trading.research.external_episodes import load_episode_dataset
from external_report_helpers import load_helper


def rows(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    require(not a.output.exists(), "不覆寫已保存驗證")
    verified = len(verify_artifact_manifest(a.data)) + len(verify_artifact_manifest(a.report))
    plan = json.loads((a.data / "plan.json").read_text(encoding="utf-8"))
    source = Path(plan["source"])
    verify_artifact_manifest(source)
    episodes = rows(a.data / "episodes.csv")
    legs = rows(a.data / "legs.csv")
    source_rows = []
    for folder in (source / "accounts").iterdir():
        source_rows.extend(rows(folder / "fills_btc.csv"))
    for row in source_rows:
        matched = [
            leg
            for leg in legs
            if (leg["alias"], leg["order_id"], leg["trade_id"])
            == (row["alias"], row["order_id"], row["trade_id"])
        ]
        require(bool(matched), "來源成交未分配")
        require(sum(Decimal(leg["size"]) for leg in matched) == Decimal(row["size"]), "數量不守恆")
        require(sum(Decimal(leg["fee"]) for leg in matched) == Decimal(row["fee"]), "費用不守恆")
        require(
            sum(Decimal(leg["api_closed_pnl"]) for leg in matched) == Decimal(row["closed_pnl"]),
            "來源 PnL 不守恆",
        )
    training = load_episode_dataset(
        a.data / "training", target="return_before_funding", allow_missing_funding=True
    )
    require(training["x"].shape[0] == len(training["y"]), "序列標籤不匹配")
    independent_complete = 0
    for e in episodes:
        own = [leg for leg in legs if leg["episode_id"] == e["episode_id"]]
        if e["complete"] == "True":
            position = sum(Decimal(leg["signed_size"]) for leg in own)
            require(position == 0, "完整交易未平倉")
            pnl = -sum(Decimal(leg["signed_size"]) * Decimal(leg["price"]) for leg in own)
            fee = sum(Decimal(leg["fee"]) for leg in own)
            require(
                abs(float(pnl - fee) - float(e["pnl_before_funding_usdc"])) < 1e-8, "交易損益不符"
            )
            independent_complete += 1
        require(e["funding_usdc"] == "" and e["net_pnl_usdc"] == "", "未知資金費被補值")
        require(e["planned_rr"] == "" and e["initial_stop"] == "", "原始未知風險被捏造")
        path = a.report / "charts" / f"{e['episode_id']}.png"
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"))
            require(
                image.width >= 1000 and image.height >= 600 and pixels.std() > 15,
                "圖表尺寸或內容異常",
            )
    for row in training["metadata"].itertuples():
        episode = next(e for e in episodes if e["episode_id"] == row.episode_id)
        require(episode["accounting_reconciled"] == "True", "未對帳樣本進入訓練")
        require(row.feature_available_ms < row.entry_ms, "未來特徵洩漏")
        require(
            abs(row.target_return_before_funding - float(episode["return_before_funding"])) < 1e-10,
            "訓練標籤不符",
        )
    schema_errors = load_helper("validate_report").validate_report(
        json.loads((a.report / "report.json").read_text(encoding="utf-8"))
    )
    require(not schema_errors, "報告 schema 失敗")
    for path in a.report.iterdir():
        if path.suffix in {".md", ".json", ".csv", ".html"}:
            require(
                not re.search(r"\b0x[0-9a-fA-F]{40}\b", path.read_text(encoding="utf-8-sig")),
                "報告含公開完整地址",
            )
    # 只測試張量與批次，沒有優化器、反向傳播或模型更新。
    import torch

    torch.set_num_threads(2)
    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(training["x"]),
        torch.from_numpy(training["side"]),
        torch.from_numpy(training["y"]),
    )
    batches = list(torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False, num_workers=0))
    require(sum(len(b[0]) for b in batches) == len(ds), "訓練批次不完整")
    result = dict(
        status="PASS",
        verified_files=verified,
        source_fills=len(source_rows),
        allocated_legs=len(legs),
        independent_complete_episodes=independent_complete,
        verified_images=len(episodes),
        dataset_shape=list(training["x"].shape),
        torch_batches=len(batches),
        training_executed=False,
        formal_training_ready=False,
        evidence_scope="封存資料、逐來源數量／費用／API PnL 守恆、完整持倉現金流、因果時間、PNG、schema、離線批次",
    )
    write_json_atomic(a.output, result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
