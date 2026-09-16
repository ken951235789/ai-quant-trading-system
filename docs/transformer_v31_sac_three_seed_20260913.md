# Transformer V3.1 + SAC 三種子研究紀錄

## 研究目的

本次實驗針對 `BTC/USDT` Binance USD-M 永續合約，修正 Transformer 短週期方向
標籤被大量「不交易」樣本主導的問題，並讓 SAC 使用更明確的多週期分析師訊號。
本次成品仍是候選模型，不因單次回測獲利就自動開放實盤。

## 固定資料

- 決策週期：15 分鐘
- 多週期背景：5m、15m、1h、4h、1d
- 資料快照：2021-09-08 至 2026-09-08，共 175,321 根 K 線
- Transformer 輸入：256 欄高優先特徵
- 其他 SAC 情境：funding、OI 變化、spread、帳戶多空比、主動買賣比、basis
- 資料摘要：`SHA-256 76109aa863d21c...51f022b4911`

## Transformer 架構改版

Transformer V3.1 將方向分析拆成三個互相協作的任務：

1. `movement`：預測不含交易成本的下跌、盤整、上漲。
2. `tradeability`：扣除 fee、slippage、spread 與 funding 後判斷是否值得交易。
3. `conditional side`：只在可交易樣本內學習做空或做多。

1、5、20 根 K 線各有獨立 horizon adapter，避免短線執行與較長趨勢使用同一個
輸出表徵。SAC 會收到 20 根趨勢、5 根 setup、1 根執行，以及共識、分歧、信心、
不交易壓力與扣除成本後的多空邊際。

## 選模與資料流程

- Transformer seeds：11、42、101。
- Transformer 只按驗證集 `hierarchical_skill_score` 挑選，不使用 test 挑 seed。
- SAC 每個 seed 跑 2 個 expanding walk-forward fold，共 6 次訓練。
- SAC 總步數：每個 run 200,000 steps。
- 最後 10% 時間資料先封存；只有選模門檻通過才開啟 final holdout 一次。
- Final holdout 不可拿來反覆調參或更換 seed。

## 成本與風控假設

- Taker fee：單邊 4 bps。
- 基礎滑價：單邊 2 bps，訓練時額外隨機化 0 至 1 bp。
- 永續合約 funding 依資料列與持倉方向計入。
- 多空名目曝險上限 75%，逐倉 2 倍，最高設定 3 倍。
- 每筆風險 1%、單日虧損 3%、最大回撤 20%。
- ATR 災難停損、spread／slippage／波動品質閘門均啟用。

## 執行狀態

Kaggle 任務：`ezkenn/ai-quant-btc-transformer-sac-v31-three-seed`

完整任務已於 2026-09-14 完成，總耗時約 7 小時 33 分。

### Transformer 樣本外結果

| Seed | 最佳 epoch | 驗證分數 | 聚合方向準確率 | 多數類基準 | 準確率提升 |
|---:|---:|---:|---:|---:|---:|
| 11 | 6 | 0.3686 | 58.44% | 56.68% | +1.75 個百分點 |
| 42 | 5 | 0.3677 | 58.50% | 56.68% | +1.81 個百分點 |
| 101 | 5 | 0.3646 | 58.70% | 56.68% | +2.02 個百分點 |

依預先固定規則只用驗證集選出 seed 11。三個 seed 的 20 根方向皆明顯高於基準，
5 根只有小幅改善，1 根皆低於基準，因此目前較適合作為中短線情境分析，不應把
1 根輸出視為可靠的獨立進場訊號。

### 第一輪 SAC 結果與根因

第一輪 3 seeds × 2 folds 共 6 個 run 全部維持空倉，報酬、交易數與成本皆為零。
這不是有效策略結果，而是環境設定契約錯誤：已開啟
`minimum_net_risk_reward=1.2`，但資料沒有 `expected_return`，也沒有固定
`take_profit_distance`，所以每次增加風險的動作都被
`insufficient_net_risk_reward` 拒絕。

修正內容：

- 建立 RL 環境時，缺少必要 `expected_return` 會立即失敗，不再靜默產生空倉模型。
- 使用 `transformer_return_5` 作為預期毛報酬，只供交易成本與方向閘門使用。
- 固定停利 1.5%、最大停損 1.0%，保留淨風報比 1.2 門檻。
- 中性動作與再平衡死區由 5% 改為 3%，避免過度壓抑 SAC 的早期探索。
- Final holdout 在第一輪未被開啟，沒有因除錯而污染。

修正版任務：`ezkenn/ai-quant-btc-sac-v31-corrected-three-seed`

### 修正版 200k SAC 篩選結果

任務於 2026-09-14 完成，耗時約 5.62 小時。`expected_return` 覆蓋率為 100%，
但 6 個 run 中只有 2 個測試窗各短暫建立一次曝險，正報酬比例為 0%，final holdout
依規則維持封存，沒有用於本次除錯。

進一步檢查逐根決策後確認，SAC 並非輸出空倉：各測試窗約 3,400 根 K 都提出非零
動作，平均絕對 action 約 0.65 至 0.79，但幾乎全部被
`insufficient_gross_target` 擋掉。這代表使用 Transformer 點預測報酬作 1.5 倍
成本硬閘門仍會封死 SAC 的探索，200k 結果不能解讀成模型沒有交易意願。

處理方式：訓練環境將 `minimum_gross_target_cost_multiple` 設為 0，讓 SAC 透過
實際 fee、slippage、funding 與 turnover penalty 學習成本；固定 1.5% 停利、1.0%
最大停損與淨風報比門檻仍保留。實盤執行期仍可使用獨立的分析師品質閘門。

第一個 1 seed × 1 fold × 100k steps 探索 smoke 已確認交易、成本與 reward 路徑
正常，但測試 5,259 根 K 發生 1,640 次調倉，周轉為本金 52.01 倍，交易成本約
3.09%，報酬 -3.75%、最大回撤 3.75%。模型沒有被風控封死，但行為屬於不可接受
的高換手，因此不能直接放大為正式候選。

第二個 anti-churn smoke 採用 150k steps，將 turnover penalty 提高到 0.01，
neutral action threshold 與 rebalance deadband 提高到 0.10，最短持有期提高到 4 根，
同時把多空曝險、單筆風險、最大停損、單日損失與連續虧損限制收緊。只有確認交易
次數與成本顯著下降且沒有再次空倉，才接續 2 seeds × 2 folds × 500k steps 的配額
節制正式候選。

第二個 smoke 的 turnover 與成本降為 0，但 49 個超過模型中性區的測試訊號仍被
`rebalance_deadband` 全部拒絕。原因是該參數使用實際持倉比例，不是標準化 action：
在最大部位 50% 時，0.10 deadband 實際要求 action 超過 0.20。第三個 smoke 保留
`neutral_action_threshold=0.10`，將實際持倉調整 deadband 校正為 0.02。

第三個 balanced smoke 使用 150k steps：測試調倉由 1,640 次降至 110 次，周轉由
本金 52.01 倍降至 5.79 倍，成本由 3.09% 降至 0.35%。測試報酬仍為 -0.27%、
Sharpe -7.07，且主要採取空方曝險，因此 smoke 本身不可部署；但交易、成本、風控與
持倉路徑已不再封死或失控，符合進入較長訓練的工程門檻。

配額節制正式候選已啟動：seeds 11、42，各 2 個 expanding walk-forward folds，
每個 run 500k steps，共 2M steps。是否可進模擬盤只由四個樣本外 run 與 sealed
final holdout 決定，不使用 smoke 報酬作為部署依據。

### 配額節制正式候選結果

正式候選於 2026-09-15 完成，Kaggle 執行約 8.93 小時。四個 run 的樣本外結果：

| Seed | Fold | 測試報酬 | Sharpe | 最大回撤 | 成本／本金 | Turnover | Profit Factor |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | 1 | -0.22% | -6.69 | 0.23% | 0.27% | 4.47x | 0.55 |
| 11 | 2 | -0.88% | -15.81 | 0.90% | 1.18% | 19.88x | 0.53 |
| 42 | 1 | -0.10% | -5.92 | 0.10% | 0.03% | 0.54x | 0.01 |
| 42 | 2 | -0.41% | -15.54 | 0.42% | 0.41% | 6.86x | 0.28 |

- 樣本外正報酬 run 比例：0%。
- 測試報酬中位數：-0.315%；Sharpe 中位數：-11.12。
- 最差最大回撤：0.90%；成本中位數：約 0.34%。
- 所有 run 的 Profit Factor 均低於 1，勝率約 20% 至 31.6%。
- 模型約 93.3% 至 99.9% 時間空倉，且交易明顯偏空；低回撤主要來自低曝險，
  不能解讀成穩定 alpha。
- 四個 run 都沒有通過單次樣本外品質門檻；使用 2 seeds 也低於正式研究要求的
  至少 3 seeds。
- Final holdout 維持 `sealed_not_evaluated`，沒有拿來選模或反覆調參。
- 模型包包含四份 best model 與四份 final model，封裝完整。

結論：`deployment_allowed=false`。本輪完成工程驗證與失敗假設排除，但沒有建立可
部署的交易優勢；不可升級 Champion，也不應以獲利策略身分接入模擬盤。後續優先
增加 rolling/cross-fitted Transformer 樣本外年份、改善 5 根方向與條件式多空訊號，
再重新設計 SAC 的動作持續性與成本敏感 reward，而不是繼續堆疊同一設定的步數。
