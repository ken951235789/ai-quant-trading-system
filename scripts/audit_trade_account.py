"""重播已保存候選成交的逐棒權益，不改原研究、不送訂單。"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_quant_trading.operations.integrity import build_artifact_manifest, sha256_file  # noqa: E402
from ai_quant_trading.persistence import write_json_atomic  # noqa: E402
from ai_quant_trading.research.account_replay import replay_mark_to_market  # noqa: E402
from ai_quant_trading.research.candidate_contract import CandidateContract  # noqa: E402


def main():
    import pandas as pd
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--trades", type=Path, required=True)
    parser.add_argument("--training-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allocation", type=float, default=.1)
    args = parser.parse_args()
    metadata = json.loads(args.training_json.read_text(encoding="utf-8"))
    expected = metadata["data_diagnostics"]["candidate_contract"]
    if sha256_file(args.source) != expected["source_sha256"]:
        raise ValueError("來源與模型研究快照不同")
    contract = CandidateContract(**expected["parameters"])
    trades = pd.read_csv(args.trades)
    if len(trades) and (not trades.contract_sha256.eq(contract.digest).all() or not trades.source_sha256.eq(expected["source_sha256"]).all()):
        raise ValueError("成本或來源契約不同；請使用同模型 base 情境成交")
    path, settled, summary = replay_mark_to_market(pd.read_csv(args.source), trades,
        allocation=args.allocation, fee_bps=contract.fee_bps, funding_reserve_bps=contract.funding_reserve_bps)
    args.output.mkdir(parents=True, exist_ok=False)
    path.to_csv(args.output / "equity.csv.gz", index=False, compression="gzip")
    settled.to_csv(args.output / "settled.csv", index=False)
    summary.update(source_sha256=sha256_file(args.source), trades_sha256=sha256_file(args.trades), contract_sha256=contract.digest)
    write_json_atomic(args.output / "report.json", summary)
    build_artifact_manifest(args.output)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
