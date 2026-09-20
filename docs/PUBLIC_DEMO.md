# 五分鐘公開 Demo

## 目的

公開版本不含真實市場資料、模型權重與憑證。這個 Demo 讓審查者不需準備外部資源，也能
確認專案的核心模組可以共同運作。

Demo 會實際執行：

1. 固定 seed 的合成 BTC/USDT 15 分鐘 OHLCV 與合約市場欄位。
2. 正式的因果技術指標、日內特徵與 SMC 特徵工程。
3. 符合 Transformer 輸出欄位的透明因果代理訊號。
4. 符合 SAC 連續目標曝險語意的多、空與空手決策。
5. 正式的回撤降倉、持有期、調倉死區與多空曝險限制。
6. 手續費、滑價、Spread 與 Funding 成本。
7. HTML 視覺報告與 JSON 稽核摘要。

## 執行

```powershell
python -m pip install -e .
ai-quant-demo --open
```

也可以直接使用腳本：

```powershell
python scripts/run_public_demo.py --bars 960 --seed 20260917 --open
```

輸出：

- `outputs/public_demo/index.html`：價格、資產曲線與流程摘要。
- `outputs/public_demo/report.json`：模式、seed、成本、風控參數與指標。

## 誠信邊界

Demo 的 `model_status` 固定標示為
`causal_proxy_not_trained_transformer_or_sac`。代理訊號不使用未來 target，但也不是正式
checkpoint。它只能證明工程流程可以執行，不能證明模型準確率、策略優勢或真實獲利能力。

正式候選必須另行完成資料雜湊、訓練 provenance、多 seed、Walk-forward、成本壓力測試、
Champion 審查及封存 final holdout。
