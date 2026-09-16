# Transformer V3 五 Seeds 正式訓練報告

## 結論

本次已完成 5 個 seeds 的 Transformer V3 正式訓練、模型完整性驗證與 CPU 推論測試。
驗證集選出的候選為 seed 42。它在 20 根 K 線方向預測明顯超過多數類基準，但 1 根與
5 根仍未超過各自基準，因此只能視為研究候選，不能直接升級為模擬或實盤模型。

「大量超過基準」不能靠增加 seed 保證。多 seed 的作用是估計訓練穩定性、降低單次
初始化運氣，並提供事先定義的集成候選；真正的市場訊號仍須由未見資料證明。

## 正式設定

| 項目 | 設定 |
|---|---:|
| 市場 | Binance Futures BTC/USDT |
| 主時間框 | 15m |
| 輸入時間框 | 5m、15m、1h、4h、1d |
| 資料期間 | 2021-09-08 至 2026-09-08 |
| 資料列數 | 175,321 |
| 輸入特徵 | 256 |
| 序列長度 | 192 根，約 48 小時 |
| 預測週期 | 1、5、20 根 |
| 模型 | d_model 96、4 heads、3 layers、FFN 192 |
| Patch | size 4、stride 2 |
| Seeds | 11、23、42、67、101 |
| 最多 Epoch | 每 seed 60，Early Stopping patience 10 |
| Batch size | 128 |
| Optimizer 參數 | lr 1.5e-4、weight decay 5e-4、warmup 10% |
| 類別失衡 | Train-only inverse-frequency power 0.5 |
| Focal gamma | 1.5 |
| 選模 | Validation `direction_skill_score`，Test 不參與 |
| 硬體 | Kaggle Tesla T4、mixed precision |

## 五 Seeds 結果

下表的 Balanced Accuracy 與 Accuracy Lift 均為三個 horizon 的平均；Lift 是模型方向
準確率減去各 horizon 的多數類基準。

| Seed | 最佳 Epoch | Validation Score | Test Balanced Accuracy | Test Accuracy | 基準 | Lift |
|---:|---:|---:|---:|---:|---:|---:|
| 11 | 6 | 0.385386 | 41.68% | 56.54% | 56.68% | -0.14% |
| 23 | 5 | 0.382018 | 42.12% | 56.72% | 56.68% | +0.03% |
| 42 | 5 | **0.388727** | **42.41%** | 56.70% | 56.68% | +0.02% |
| 67 | 5 | 0.383592 | 42.05% | 57.73% | 56.68% | +1.05% |
| 101 | 5 | 0.382447 | 42.39% | 57.57% | 56.68% | +0.88% |

seed 42 是依 Validation 選出。seed 67 的 Test Accuracy 較高，但不能事後改選，否則
會把 Test 變成調參資料並造成選擇偏誤。

## Seed 42 分週期結果

| Horizon | 實際時間 | Accuracy | 多數類基準 | Lift | Balanced Accuracy | Macro F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15 分鐘 | 76.39% | 81.53% | **-5.14%** | 43.38% | 42.36% |
| 5 | 75 分鐘 | 52.26% | 53.88% | **-1.62%** | 42.37% | 40.26% |
| 20 | 5 小時 | 41.45% | 34.64% | **+6.81%** | 41.50% | 41.61% |

20 根 horizon 有可研究的方向訊號；1 根與 5 根尚未證明超過簡單基準。三個 horizon
不可只看平均後就宣稱全部有效。

## 五模型等權集成診斷

等權平均五個 seeds 的方向機率，不用 Test 選權重。此診斷改善整體 Accuracy，但仍未
讓每個 horizon 都勝過基準。

| Horizon | Accuracy | 多數類基準 | Lift | Balanced Accuracy | Macro F1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 79.24% | 81.53% | -2.29% | 41.27% | 42.55% |
| 5 | 52.23% | 53.88% | -1.65% | 43.32% | 43.27% |
| 20 | 41.59% | 34.64% | +6.95% | 41.79% | 41.87% |
| 平均 | 57.69% | 56.68% | +1.00% | 42.13% | 42.56% |

## 本次程式改進

- 方向分類使用僅由 Train 計算的 horizon-specific 類別權重。
- 新增可調整的 Focal Loss，降低大量容易樣本支配梯度的問題。
- 新增每個 horizon 的 Accuracy、Majority Baseline、Balanced Accuracy、Macro F1 與 Lift。
- Checkpoint 改由 Validation Skill Score 選擇，不再只看混合多任務 Loss。
- 五 seed runner 會保留所有模型、測試預測、穩定性摘要與等權集成診斷。
- 區分研究閘門與實盤閘門；研究平均通過不等於可以部署。

## 驗證結果

- 五份 `artifact_manifest.json` 的檔案大小與 SHA-256 全數通過。
- seed 42 使用 CPU、最新 192 根 K 線完成推論，14 個主要預測欄位成功產生。
- 專案完整測試：448 passed，另有 4 個 subtests passed。
- Transformer 測試：13 passed；本次相關測試：26 passed。

模型目錄：

```text
outputs/kaggle_transformer_v3_formal/result_five_seed/transformer_v3_five_seed/
```

摘要與診斷：

```text
outputs/kaggle_transformer_v3_formal/result_five_seed/five_seed_summary.json
outputs/kaggle_transformer_v3_formal/result_five_seed/five_seed_ensemble_diagnostic.json
outputs/kaggle_transformer_v3_formal/result_five_seed/selected_seed42_cpu_inference_smoke.json
```

## 下一個合規步驟

這個 Test 已在本報告中查看，不可再拿來挑 loss、seed、horizon 或集成權重。下一輪若要
改善 1 根與 5 根預測，應只在 Train／Validation 做 walk-forward 實驗，並把
2026-09-08 之後的新資料封存成全新的 final holdout。之後還要完成含手續費、滑價、
資金費率及交易門檻的策略回測，再通過紙上交易，才有資格評估是否升級。
