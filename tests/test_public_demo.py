from __future__ import annotations

import json

from ai_quant_trading.public_demo import generate_synthetic_btc_ohlcv, run_public_demo


def test_synthetic_btc_data_is_reproducible() -> None:
    first = generate_synthetic_btc_ohlcv(bars=320, seed=7)
    second = generate_synthetic_btc_ohlcv(bars=320, seed=7)

    assert first.equals(second)
    assert first["timestamp"].is_monotonic_increasing
    assert (first["high"] >= first[["open", "close"]].max(axis=1)).all()
    assert (first["low"] <= first[["open", "close"]].min(axis=1)).all()


def test_public_demo_writes_auditable_outputs(tmp_path) -> None:
    artifact = run_public_demo(tmp_path, bars=360, seed=11)

    report = json.loads(artifact.report_path.read_text(encoding="utf-8"))
    html = artifact.html_path.read_text(encoding="utf-8")
    assert report["kind"] == "synthetic_pipeline_demo"
    assert report["model_status"] == "causal_proxy_not_trained_transformer_or_sac"
    assert report["metrics"]["feature_count"] >= 50
    assert report["metrics"]["decision_samples"] > 0
    assert "不是 Transformer／SAC 訓練績效" in report["disclosure"]
    assert "BTC AI Quant Research Demo" in html
    assert "合成 BTC" in html
