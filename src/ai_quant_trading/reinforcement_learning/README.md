# BTC 15m 強化學習

## 精簡短線輸入

新 BTC 環境預設最多 112 欄，本機五週期資料為 107 欄，排除重複的基礎 15m 指標與整批 latent。
保留 1／5／20 根角色訊號、九個市場情境與訊號熵。風控的 `expected_return` 另由
`environment.json → ai_context.expected_return_contract` 明確指定，新短線預設 5 根。
舊模型無契約時只維持既有 20 根執行語意；新版不得猜測來源，輸入變動必須重建環境與重訓。
詳見 `docs/compact_ai_inputs_20260915.md`。

此模組只服務 `BTC/USDT` Binance USD-M 永續合約研究。SAC 是主要連續部位模型，
PPO 是 Hold、Long、Short、Close 四動作比較模型。正式候選模型必須經過多 seed、
expanding Walk-forward、獨立 final holdout 及模擬倉驗證。

## 時序與資料

1. 第 `t` 根 15m K 線收盤後，讀取當時已完成的 `5m / 15m / 1h / 4h / 1d` 特徵。
2. Transformer 提供報酬、波動、市場狀態、latent，以及趨勢／setup／執行三種角色訊號。
3. SAC 輸出 `-1 ~ 1`，再映射成最多 50% 的多頭或空頭名目曝險。
4. 調倉在第 `t+1` 根開盤成交，並計入 fee、spread、slippage、funding。
5. 使用第 `t+1` 根收盤淨值計算 Reward，不會重複扣除已反映在淨值內的成本。

資料按時間切成 60% 訓練、20% 驗證、20% 測試。Scaler 只使用訓練集估計；
Episode 可隨機起點、初始本金及滑價，但驗證與測試會自動關閉 domain randomization。

使用 Transformer 時，RL 環境還會驗證上游 checkpoint 的市場、原始時間範圍及切分比例，
只保留 Transformer 完成模型選擇與機率校準之後的樣本外輸出，再進行自己的時間切分。
來源證明會寫入 `environment.json`；缺少證明的舊環境不得新增實盤風險，需重新建立環境。

## Observation

- 經 train-only mean/std 標準化的市場與 Transformer 特徵。
- Transformer 的 20 根趨勢、5 根 setup、1 根執行訊號，以及跨週期共識、分歧、信心、
  不交易壓力與扣除成本後的多空淨優勢。
- 可用餘額、方向曝險、淨值、回撤、已實現與未實現損益。
- 平均進場價距離、持倉 K 線數、今日損益與連虧次數。
- 保證金占比，以及距停損、固定停利與強平價格的距離。

舊環境沒有 `include_trade_plan_context` 時維持原 observation 維度；新 BTC 模型會啟用
這兩個交易計畫欄位，舊模型不能在不重訓的情況下假裝取得新狀態。

## Action 與固定風控

- SAC 的 `-0.10 ~ +0.10` 是中性區，視為空手；目標變動不到 10% 不調倉。
- 反向交易分兩次決策，先平倉，下一根仍維持反向才開倉。
- 預設逐倉 2 倍、最高 3 倍，保證金最多 20%，名目曝險最多 50%。
- 動態槓桿只接受「指定停利先於停損」的樣本外校準成功率；偏多／偏空 regime 機率不能
  冒充交易勝率。沒有這項專用機率時仍可減倉，但新增部位最多使用最低槓桿。
- 每筆風險 0.25%，每日虧損 1%、連虧 3 筆或總回撤 10% 後停止增加風險。
- ATR 災難停損、1.5% 保護性停利；同一根同時碰觸時採保守的先停損順序。
- 毛目標需達成本 4 倍，淨風報比至少 1.5；spread、滑價、波動或事件異常不開倉。

## Reward

```text
reward = log(equity_t+1 / equity_t)
         - drawdown_penalty * drawdown_increase
         - turnover_penalty * turnover
         - downside_penalty * downside_return
         - concentration_penalty * concentration_excess^2
         - risk_termination_penalty
```

## 訓練與成品

Dashboard 的「模型研究」提供環境建構、PPO／SAC 訓練及訓練結果。成品位於：

```text
data/processed/rl/environments/{environment_id}/
    environment.json
    train.csv
    validation.csv
    test.csv
    diagnostic.csv
    training/{training_id}/
        training.json
        progress.json
        final_model.zip
        best_model/best_model.zip
        validation_evaluation.csv
        test_evaluation.csv
```

`experiments.py` 會先封存資料尾端 10% 作為 final holdout，再用其餘資料依 seed 與
Walk-forward fold 串行訓練。seed、參數與候選模型只能根據選模 folds 決定；選模門檻
通過後才會開啟 final holdout 評估一次，並保存模型、資料與評估 CSV 的 SHA-256。
Champion 升級會重新驗證完整證據鏈及模擬倉紀錄；舊研究缺少 protocol v2 證據時只能
離線查看，不能升級。正式研究至少使用 5 個 seed；smoke training 只證明管線可執行。

SB3 `zip` 含 Python 序列化內容，所有載入入口都要求訓練時建立的
`artifact_manifest.json`。不要匯入來源不可信的模型，也不要為了載入舊模型手動偽造清單。

參數用途與建議值請看
[`docs/強化學習全參數教學與硬體建議.md`](../../../docs/強化學習全參數教學與硬體建議.md)。
