# Kaggle Transformer V3／SAC 實測報告

## 測試目的

以實際專案程式碼與 BTC 15 分鐘多週期特徵，估算 Kaggle 每週 30 GPU 小時是否足夠完成正式訓練。

本次只量測速度，不用短訓練模型的報酬判斷策略優劣。

## 測試環境

| 項目 | 實測值 |
|---|---:|
| GPU | Tesla T4 × 2 |
| 實際使用 GPU | `cuda:0` 一張 |
| CPU | 4 cores |
| Python | 3.12.13 |
| PyTorch | 2.10.0+cu128 |
| Stable-Baselines3 | 2.7.1 |
| Transformer 測試資料 | 20,000 根 15m K 線 |
| 正式資料 | 175,321 根 15m K 線 |
| Transformer 特徵 | 256 |
| Transformer 參數 | 481,276 |

## 實測結果

| 工作 | 實測 |
|---|---:|
| Transformer V3 一個 epoch（20,000 根） | 18.40 秒 |
| Transformer 完整單次測試（含載入與評估） | 21.44 秒 |
| Transformer 峰值 GPU 記憶體 | 0.048 GB |
| SAC GPU 10,000 steps | 121.02 秒 |
| SAC GPU 速度 | 82.63 steps/s |
| SAC CPU 10,000 steps | 177.93 秒 |
| SAC CPU 速度 | 56.20 steps/s |

在目前環境中，SAC 使用 T4 約比 4 核 CPU 快 47%。程式尚未使用第二張 T4；目前模型很小，強行使用多 GPU 不一定更快。

## 正式訓練外推

| 正式工作 | 線性外推 |
|---|---:|
| Transformer 20 epochs | 約 0.90 小時 |
| Transformer 50 epochs | 約 2.24 小時 |
| SAC 1,000,000 steps，單一 run | 約 3.36 小時 |
| SAC 5 seeds × 3 folds | 約 50.43 小時 |
| Transformer 50 epochs + 完整 15 次 SAC | 約 52.67 小時 |

外推未包含 Kaggle 排隊、偶發 I/O、Checkpoint 上傳與工作重啟成本。Transformer 若 early stopping 提前停止，實際時間會更短。

## 結論

每週 30 GPU 小時足以完成：

- 一次 Transformer V3 正式訓練。
- 一次 SAC 1,000,000 steps。
- 數個 seed／fold 的候選實驗。

每週 30 小時不足以穩定完成 Transformer 50 epochs 加上 `5 seeds × 3 folds` 的全部正式流程。建議分兩週執行，每週安排 6 至 7 個 SAC run，保留額度給失敗重跑與最終評估。

## 建議順序

1. 先正式訓練 Transformer V3，保存最佳 checkpoint。
2. 產生完整、無未來洩漏的 Transformer 樣本外特徵。
3. 先跑 SAC 單一 seed 的 1,000,000 steps，確認 reward、交易頻率與回撤正常。
4. 執行 5 seeds × 3 walk-forward folds，分兩週排程。
5. 只用驗證結果挑選候選模型。
6. 最終 holdout 只開封一次，再決定是否升級 Champion。

## Kaggle 注意事項

- CLI 預設 P100 與目前 Kaggle 的 PyTorch 2.10 不相容；必須指定 `NvidiaTeslaT4`，或另裝支援 P100 的 PyTorch CUDA 12.6 版本。
- Dataset 與 Kernel 均設為私人。
- Benchmark Dataset 不包含 `.env`、交易 API Key、Kaggle Token、既有模型或交易紀錄。
- 本次測試結束後，Kaggle 顯示 GPU 已使用 0.12 小時、剩餘 29.88 小時。
