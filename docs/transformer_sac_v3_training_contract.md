# Transformer V3 與 SAC 部署訓練契約

本文件記錄 BTC 15 分鐘短線研究主線的固定資料與模型語意。它是研究設定，不代表模型已具備獲利能力；任何候選模型仍須通過 walk-forward、final holdout、成本壓力測試與 Testnet。

## Transformer

- 決策週期：15 分鐘。
- 輸入序列：256 根決策 K 線，並包含因果對齊的多時間週期特徵。
- 預測週期：5、20、48 根，約等於 75 分鐘、5 小時與 12 小時。
- 角色：5 根負責執行訊號，20 根負責 setup，48 根提供趨勢與風險背景。
- Checkpoint 權重：0.50、0.35、0.15。
- Checkpoint 指標：`deployment_horizon_skill_score`。
- 特徵清理：只使用訓練區段移除常數特徵與絕對相關係數大於 0.985 的重複特徵。
- 原始 OHLCV 價格水準不可直接進入模型，只使用報酬、比例、距離與標準化資訊。

建議正式架構：

```text
sequence_length = 256
d_model = 128
n_heads = 8
n_layers = 4
feedforward_dim = 384
dropout = 0.15
patch_size = 8
patch_stride = 4
hierarchical_direction = true
horizon_adapter_dim = 64
```

## SAC

- `expected_return` 固定由模型中繼資料指定，短線執行契約目前使用 Transformer 5 根預測。
- SAC observation 同時保留 5、20、48 根的報酬、分位數、可交易性、多空 edge、跨週期共識、分歧與訊號熵。
- 動作語意使用 `continuous_target`，輸出是連續目標曝險，不再用大範圍門檻把動作硬切成空手。
- `rebalance_deadband = 0.01` 只抑制小於 1% 曝險差異的無效調倉。
- Replay buffer 建議 400,000，warm-up 20,000 steps。
- 熵係數建議 `auto_0.01`，常態動作雜訊標準差 0.05。
- 訓練、回測、模擬與實盤共用同一個動作映射函式。

## 模型升級門檻

下列任一條件成立，候選模型不可升級為 Champion：

- final holdout 報酬不為正。
- final holdout Sharpe、Profit Factor 或 Expectancy 未達門檻。
- 調倉次數不足，或策略近乎全程空手。
- 交易成本、最大回撤、動作飽和或強平次數超過限制。
- Transformer 特徵覆蓋不足，或無法重建訓練時的 observation。

正式訓練應先完成 Transformer 多 seed 選模，再使用時間因果正確的 cross-fit/OOS Transformer 訊號訓練 SAC。測試集與 final holdout 不得用於挑 seed、調參或選 checkpoint。
