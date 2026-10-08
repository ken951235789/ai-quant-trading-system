"""隔離帳戶重播二十根歷史 K 棒，驗證成交日誌；動作為測試腳本，非模型績效。"""

# ruff: noqa: E402
import argparse
import json
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from ai_quant_trading.operations.integrity import build_artifact_manifest, sha256_file
from ai_quant_trading.paper_trading import PaperTradingConfig, run_rl_paper_cycle
from ai_quant_trading.paper_trading.storage import read_account_csv
from ai_quant_trading.reinforcement_learning import LoadedRLPolicy, PortfolioEnvConfig
from ai_quant_trading.risk import RiskConfig
from ai_quant_trading.trading.journal import TradeJournal


class ScriptedPolicy:
    """僅用來覆蓋買賣、加減碼和重播路徑，沒有預測能力。"""

    class ObservationSpace:
        shape = (6,)

    observation_space = ObservationSpace()

    def predict(self, observation, deterministic=True):
        return np.asarray([observation[0]], dtype=float), None


def replay(frame, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for column, accepted in (("exchange", {"binance_futures", "binance_usdm"}),
                             ("symbol", {"BTC/USDT", "BTCUSDT"}), ("interval", {"15m"})):
        if column in frame and not set(frame[column].dropna().unique()).issubset(accepted):
            raise ValueError(f"測試來源的 {column} 並非 BTC USD-M 15m")
    market = frame[["timestamp", "open", "high", "low", "close", "volume"]].tail(20).copy().reset_index(drop=True)
    if len(market) != 20:
        raise ValueError("測試必須提供至少二十根歷史 K 棒")
    market["timestamp"] = pd.to_datetime(market.timestamp, utc=True)
    if not market.timestamp.diff().dropna().eq(pd.Timedelta(minutes=15)).all():
        raise ValueError("二十根測試 K 棒必須連續且為 15m")
    market["exchange"], market["symbol"], market["interval"] = "binance_futures", "BTC/USDT", "15m"
    market["test_action"] = [.2, .2, .35, .1, 0, 0, -.2, -.2, -.35, -.1, 0, 0, .2, 0, 0, -.2, 0, 0, 0, 0]
    environment = {"environment_kind": "single", "source": {
        "exchange": "binance_futures", "symbol": "BTC/USDT", "interval": "15m"},
        "normalization": {"mean": {"test_action": 0.}, "std": {"test_action": 1.}}}
    policy = LoadedRLPolicy(ScriptedPolicy(), Path("scripted_test_only"), Path("test_environment"),
        {"training_config": {"algorithm": "sac"}, "test_only": True}, environment, ["test_action"],
        PortfolioEnvConfig(max_position_fraction=.5, allow_short=True, max_short_fraction=.5,
                           execution_mode="perpetual", leverage=1., max_leverage=1., max_margin_fraction=.5))
    common = dict(policy=policy, model_dir="scripted_test_only", root_dir=output, account_id="journal_smoke",
        entry_threshold=.05, exit_threshold=.01,
        config=PaperTradingConfig(initial_capital=1000., fee_rate=.0005, slippage_rate=.0002,
            allow_short=True, max_short_fraction=.5, position_fraction=.5),
        risk_config=RiskConfig(max_risk_per_trade=.1, fixed_stop_loss_pct=.02, take_profit_pct=.04,
                               max_position_fraction=.5))
    for i in range(1, 21):
        result = run_rl_paper_cycle(market.iloc[:i].copy(), **common)
    journal = TradeJournal(result.paths.account_dir / "trade_journal")
    journal.render_pending(limit=100)
    records = journal.records()
    trades = read_account_csv(result.paths.trades_csv)
    closed = [r for r in records if r["record"]["outcome"] is not None]
    assert len(closed) == len(trades) > 0
    assert all(r["status"] == "complete" for r in records)
    assert {r["record"]["entry"]["side"] for r in closed} == {-1, 1}
    np.testing.assert_allclose(sum(r["record"]["outcome"]["net_pnl"] for r in closed), trades.net_pnl.sum())
    count = len(read_account_csv(result.paths.orders_csv))
    repeated = run_rl_paper_cycle(market.copy(), **common)
    assert not repeated.processed
    assert count == len(read_account_csv(result.paths.orders_csv))
    assert len(records) == len(journal.records())
    exported = journal.export_training_review(output / "training_review_only")
    summary = {"engineering": "PASS", "research_quality": "NOT_APPLICABLE", "live_eligible": False,
        "test_only": True, "policy": "scripted_not_trained", "bars": 20,
        "start": str(market.timestamp.iloc[0]), "end": str(market.timestamp.iloc[-1]),
        "orders": count, "closed_trade_rows": len(trades), "journal_records": len(records),
        "png_files": len(list(journal.root.glob("*.png"))), "exported": exported,
        "kinds": sorted({r["kind"] for r in records}), "source_unchanged": True,
        "limitations": ["測試動作不是訓練模型，損益不代表投資效果", "實盘回報、費率換算及整合未驗證"]}
    (output / "smoke.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    market.to_csv(output / "replay_bars.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = sha256_file(args.source)
    summary = replay(pd.read_csv(args.source), args.output)
    assert before == sha256_file(args.source)
    (args.output / "source.json").write_text(json.dumps({"sha256": before, "runner_sha256": sha256_file(Path(__file__))}), encoding="utf-8")
    build_artifact_manifest(args.output, files=[p for p in args.output.rglob("*") if p.is_file()])
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
