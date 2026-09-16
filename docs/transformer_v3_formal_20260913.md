# Transformer V3 正式候選訓練報告（2026-09-13）

## 結論

- 訓練流程：成功。
- 模型檔完整性：成功。
- 本機 CPU 推論：成功。
- 實盤品質閘門：未通過。
- 模型狀態：保留為研究用 Candidate，不升級為 Champion，也不自動提供給交易機器人。

這次結果證明完整 V3 管線可以在 Kaggle T4 上執行與下載，但單一 seed 42 的預測品質尚不足以證明有可交易優勢。

## 凍結資料

| 項目 | 內容 |
|---|---|
| 市場 | Binance Futures BTC/USDT |
| 決策週期 | 15m |
| 多時間框架 | 5m、15m、1h、4h、1d |
| 資料期間 | 2021-09-08 00:00 UTC 至 2026-09-08 06:00 UTC |
| K 線數 | 175,321 |
| 特徵數 | 256 |
| Train / Validation / Test | 60% / 20% / 20% |
| 實際樣本數 | 105,077 / 35,044 / 35,045 |
| 資料 SHA-256 | `c3fb148a67b04f464d32746ba0b89b861fb4dfa2bd89e25f4ca9b11e552e16ff` |

資料依時間先後切分，Scaler 只使用 Train 區段估計。標籤使用下一根 K 線開盤作為可成交進場價，並納入雙邊手續費、滑價、價差與資金費率。禁止的 future、target、label 與舊 Transformer 輸出欄位未進入模型。

## 模型設定

| 項目 | 設定 |
|---|---|
| 架構 | Transformer V3 |
| 序列長度 | 96 根 15m K（24 小時） |
| `d_model` | 96 |
| Attention heads | 4 |
| Encoder layers | 3 |
| Feed-forward | 192 |
| Dropout | 0.10 |
| 預測 horizon | 1、5、20 根 K |
| 最大 epochs | 50 |
| Early stopping | patience 8 |
| Batch size | 128 |
| Optimizer | AdamW，LR 3e-4，weight decay 1e-4 |
| 執行 | Tesla T4、mixed precision、seed 42 |

模型同時學習報酬、成本後多空 edge、方向、波動、波動狀態、價格路徑不利／有利幅度、可交易性與分位數範圍。最佳權重出現在 epoch 2，epoch 10 觸發 early stopping；總訓練與測試時間約 11 分 11 秒。

## 時間外結果

整體測試指標：

| 指標 | 結果 |
|---|---:|
| Test loss | 1.510957 |
| Regime accuracy | 35.36% |
| 成本感知方向 accuracy | 58.67% |
| 一般報酬方向 accuracy | 50.33% |
| Return MAE | 0.3988% |
| Volatility MAE | 0.0707% |
| Cost-aware edge MAE | 0.3976% |
| Tradeability accuracy | 72.00% |
| Tradeability Brier score | 0.1862 |

成本感知方向必須和類別基準一起判讀：

| Horizon | 模型 accuracy | 多數類基準 | Balanced accuracy | 報酬相關係數 |
|---|---:|---:|---:|---:|
| 1 根（15 分） | 81.28% | 81.53% | 36.59% | 0.0002 |
| 5 根（75 分） | 53.21% | 53.88% | 41.41% | -0.0029 |
| 20 根（5 小時） | 41.52% | 34.64% | 41.50% | 0.0178 |

1 根與 5 根結果沒有超過永遠選擇多數類別的基準。20 根雖超過基準，但報酬相關性很弱，不能據此宣稱可營利。

## 品質判定

下列工程檢查已通過：

- 完整資料訓練與純時間切分。
- 最佳模型保存、溫度校準與時間外測試。
- `best_model.pt`、`training.json` SHA-256 完整性驗證。
- 下載後在本機 CPU 載入 checkpoint 並完成最新序列推論。
- Kaggle Kernel 沒有模型執行警告；日誌中的兩個 SyntaxWarning 來自 Kaggle `mistune/nbconvert` 匯出環境，與模型程式無關。

下列效能檢查未通過：

- 1 根與 5 根成本感知方向未超過多數類基準。
- 報酬與 edge 的時間外相關性接近零。
- 僅有單一 seed，尚未驗證結果穩定性。
- 尚未經 SAC、風控、滑價壓力測試與 walk-forward 交易績效驗證。

因此不得把本次模型宣稱為正式可營利模型，也不得自動升級或連接實盤。

## 成品位置

- Kaggle 執行頁：<https://www.kaggle.com/code/ezkenn/ai-quant-btc-transformer-v3-formal-training>
- Kaggle 私人資料集：<https://www.kaggle.com/datasets/ezkenn/ai-quant-btc-transformer-v3-formal-input>
- 本機下載根目錄：`outputs/kaggle_transformer_v3_formal/result/`
- 模型與訓練結果：`outputs/kaggle_transformer_v3_formal/result/transformer_v3_formal/20260913T054824732636Z_btc_15m_mtf_v3_formal_seed42/`
- 本機推論驗證：`outputs/kaggle_transformer_v3_formal/result/latest_inference_smoke.json`
- 各 horizon 診斷：`outputs/kaggle_transformer_v3_formal/result/per_horizon_diagnostics.json`

## 重現方式

建立安全資料包：

```powershell
python scripts\build_kaggle_transformer_v3_formal.py --username ezkenn
```

建立或更新私人 Dataset 後，提交 T4 Kernel：

```powershell
python scripts\kaggle_cli_truststore.py kernels push `
  -p outputs\kaggle_transformer_v3_formal\kernel `
  --accelerator NvidiaTeslaT4
```

本報告已查看本次 Test，因此該 Test 不可再用來反覆挑參數。下一輪應只在 Train／Validation 進行類別平衡、loss 權重與 seed 選擇，並以 2026-09-08 之後的新資料建立未被查看的新 final holdout。
