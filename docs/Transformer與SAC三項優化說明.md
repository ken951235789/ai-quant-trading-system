# Transformer 與 SAC 三項優化說明

## 目的

這一輪不是把回測數字「調漂亮」，而是把三個研究問題拆開：

1. Transformer 單獨交易時，扣除成本後是否真的有價值。
2. SAC 為什麼長時間空手，以及續抱、平倉、換手懲罰和硬閘門各自造成多少影響。
3. 如何讓 SAC 使用更長、但仍然沒有未來資料洩漏的 Transformer 預測。

任何候選參數都只能在 train／validation／walk-forward folds 選擇。Final holdout 只能在候選完全凍結後使用一次，不能看完結果再調參。

## 1. Transformer-only 扣成本基準

`TransformerSignalConfig` 現在多了以下設定：

| 參數 | 意義 | 建議起點 |
|---|---|---:|
| `horizon` | 預測幾根後報酬 | 比較 5 與 20 |
| `round_trip_cost_multiple` | 預測毛利至少為來回成本的幾倍 | 1.0～1.5 |
| `minimum_net_return` | 扣成本後還要求的最低淨優勢 | 0～0.05% |
| `exit_threshold_ratio` | 進場後訊號衰退到多少比例才退出 | 0.5 |
| `minimum_tradeability` | Transformer 可交易機率門檻 | 0.5；未輸出此欄時不限制 |

進場條件已改為：

```text
abs(predicted_return)
>= round_trip_cost_multiple * (雙邊手續費 + 雙邊滑價 + spread)
   + minimum_net_return
```

回測仍採訊號 K 線收盤後決策、下一根開盤成交，並另外輸出：

- `transformer_with_costs`
- `transformer_without_costs`
- `flat_cash`
- `buy_and_hold`
- `predicted_net_edge`
- `cost_drag`

使用 `compare_transformer_horizons(...)` 可在完全相同資料與成本下比較 5 根與 20 根。不要只看總報酬，至少同時檢查最大回撤、Sharpe、Profit Factor、交易數、訊號覆蓋率與成本拖累。

## 2. SAC 動作與 Reward 消融

新版 SAC 仍維持一維連續動作，既有 checkpoint 可以繼續使用。新訓練可使用：

```python
PortfolioEnvConfig(
    normalized_action_space=True,
    action_semantics="hold_close_target",
    hold_action_threshold=0.03,
    close_action_threshold=0.10,
)
```

動作區域定義如下：

| 動作絕對值 | 空倉時 | 持倉時 |
|---|---|---|
| `0～0.03` | 等待 | 續抱，不調倉 |
| `0.03～0.10` | 等待 | 主動平倉 |
| `> 0.10` | 建立多／空目標 | 加減倉或提出反向目標 |

反手仍必須先平倉，下一根再次確認後才開反向倉。風控仍可在超限、回撤或 Kill Switch 時覆寫「續抱」。

`build_sac_environment_ablations(base_config)` 會產生七組預先宣告的實驗：

| 實驗 | 要回答的問題 |
|---|---|
| `legacy_control` | 舊結果可否重現 |
| `hold_close_semantics` | 空手是否來自中性動作被當成平倉 |
| `no_extra_turnover_penalty` | Reward 是否重複計算交易成本 |
| `cost_aligned_turnover_penalty` | 小幅換手正則化是否比 0.01 合理 |
| `pnl_only_reward` | Drawdown／downside／concentration shaping 是否過強 |
| `entry_exit_filters_off` | 硬閘門是否把探索全部封死 |
| `relaxed_entry_exit_filters` | 較寬鬆但仍扣成本的閘門是否改善 OOS |

每組至少跑相同的 3 個 seeds 與相同 walk-forward folds。先排除交易數極低、長期空手、成本後負期望或跨 seed 不穩定的方案，再比較中位數 OOS 報酬與回撤。不要用 final holdout 挑選其中一組。

## 3. Transformer cross-fit OOS

直接拿一顆看過完整五年資料的 Transformer 回頭替五年資料產生預測，會造成資料洩漏。Cross-fit 流程改成：

```text
Fold 1：只看早期資料訓練 -> 預測下一段 OOS
Fold 2：擴展或滾動訓練窗 -> 預測再下一段 OOS
Fold 3：擴展或滾動訓練窗 -> 預測再下一段 OOS
Fold 4：擴展或滾動訓練窗 -> 預測再下一段 OOS
尾端 10%：Final holdout，整個選模期間不可使用
```

正式 GPU 電腦可在專案根目錄執行：

```powershell
python scripts/run_transformer_crossfit.py `
  --source data/processed/你的BTC_15m多週期特徵.csv `
  --output-dir artifacts/transformer/crossfit `
  --folds 4 `
  --mode expanding `
  --epochs 50 `
  --batch-size 96 `
  --seed 11 `
  --device cuda
```

輸出包含：

- `oos_predictions.csv`：只含各 fold 真正沒看過的下一段資料。
- `crossfit.json`：每折 fit/OOS 時間、seed、checkpoint SHA-256 與封存 holdout 邊界。
- `folds/`：每折模型成品。
- `artifact_manifest.json`：成品完整性雜湊。

建立 SAC 資料前請用 `load_transformer_crossfit_predictions(...)` 載入；它會驗證
`oos_predictions.csv` 的 SHA-256、確認每列都有 `transformer_oos=1`，並恢復
`expected_return` 的 5 根中繼資料契約。直接用 `pd.read_csv` 會遺失 DataFrame attrs，
正式環境保存器會因此拒絕把預測週期當成猜測值。

`save_rl_environment(...)` 與 `save_universal_rl_environment(...)` 現在可接收 `transformer_crossfit_summary`。建立 SAC 訓練環境時可只使用 cross-fit OOS 特徵；但模擬盤／實盤仍需要另外訓練一顆部署用 Transformer checkpoint，因此此時 `runtime_ready` 會維持 `false`。

## 建議執行順序

1. 先用現有 OOS 資料跑 5 根與 20 根 Transformer-only 扣成本比較。
2. 若兩者扣成本都沒有正向、穩定的淨優勢，先修 Transformer，不要急著讓 SAC 放大訊號。
3. 使用短版資料對七組 SAC 消融各跑 3 seeds，找出空手的主要來源。
4. 參數凍結後，在 GPU 電腦執行 4 folds expanding cross-fit。
5. 用 cross-fit OOS 訓練 SAC，完成 walk-forward 選模。
6. 最後才開封 final holdout；通過後仍需紙上交易、Testnet、斷線與 Kill Switch 演練。

這套流程能提高研究可信度與找出失敗原因，但不能保證獲利。真正升級 Champion 的條件仍是成本後 OOS 穩健、跨 seed 一致、回撤受控，且未使用 final holdout 反覆調參。
