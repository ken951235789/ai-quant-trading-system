# paper_trading

持久化 PPO／SAC 模擬交易模組。它使用完整風控，並把現金、持倉、待執行目標與績效保存到
`data/paper_trading/{account_id}/`。

## 執行規則

1. 只使用已經收盤的 K 線；交易所仍在形成中的最後一根會被排除。
2. 模型在本期收盤後產生目標持倉；正值做多、負值做空、0 代表空手。
3. 買進、賣出、開空與回補都要等下一根新 K 線開盤才成交，避免 Look Ahead Bias。
4. 成交納入手續費、滑價、Position Sizing、停損、停利、空頭持有成本與最大回撤停機。
5. 同一根 K 線重跑時不會重複預測或下單。
6. 第一次建立帳戶只會記錄最新訊號，不會把歷史訊號假裝成即時成交。

RL Action 是連續目標持倉比例，Risk Governor 會再套用多頭／空頭上限、調倉死區、
最短持有期與回撤限制。舊 long-only 模型的 Action 下限仍為 0；只有以負 Action
環境重新訓練的模型才可開空。

進場與離場門檻會形成遲滯區：空手時未達進場門檻不下單；持倉後低於離場門檻
才平倉；反向訊號未達進場門檻時只平掉原方向，不會低信心直接反手。

Dashboard 固定依模型來源自動下載至少 220 根 K 線，再計算 SMA 200 等長週期特徵，
不需要使用者自己準備 CSV。命令列仍保留 `--input`，供測試固定歷史樣本使用。
如果模型使用 `mtf_*` 多週期欄位，模擬交易會依模型欄位自動下載
`1m、3m、5m、15m、30m、1h、4h、12h、1d` 所需週期，並在帳戶的
`market_cache/` 覆蓋更新單週期與融合特徵，不會退回只使用決策週期。

如果兩次執行之間漏掉多根 K 線，系統會依時間順序逐根補跑。每一根都會執行前期
待處理訊號、檢查當期停損／停利，再建立新的模型訊號，因此不會用最新價格替代
中間本來應發生的成交。

## 每個帳戶的檔案

- `account.json`：現金、持倉、待執行訊號及風控停機狀態
- `orders.csv`：每筆 Buy／Sell 模擬成交
- `trades.csv`：已完成的來回交易與損益
- `predictions.csv`：模型原始目標、門檻後目標、風控核准目標與動作訊號
- `performance.csv`：資產、總報酬、回撤與每期空頭持有成本
- `positions.csv`：每次執行後的持倉快照

永續帳戶啟用動態槓桿後，上述 CSV 也會保存實際槓桿、證據上限、方向勝率、淨損益比、
淨期望值與核准原因。舊帳戶預設維持固定槓桿，可在 App 停止背景機器人後再切換。

## 命令列

使用七市場 PPO 對 AAPL 執行一次模擬：

```powershell
python scripts\run_paper_trading.py rl-run `
  --account rl_aapl `
  --model-dir data\processed\rl\environments\你的環境\training\你的PPO訓練 `
  --exchange yahoo_finance --symbol AAPL --interval 1d `
  --entry-threshold 0.10 --exit-threshold 0.02 `
  --download-latest
```

使用已訓練的短期多空模型時，可明確限制空頭曝險與年化持有成本：

```powershell
python scripts\run_paper_trading.py rl-run `
  --account rl_short_btc `
  --model-dir data\processed\rl\environments\你的多空環境\training\你的PPO訓練 `
  --exchange binance --symbol BTC/USDT --interval 5m `
  --allow-short --max-short-fraction 0.05 `
  --short-borrow-rate-annual 0.10 --download-latest
```

也可以指定已有 CSV：

```powershell
python scripts\run_paper_trading.py rl-run --account rl_btc --model-dir data\processed\rl\environments\你的環境\training\你的訓練 --exchange binance --symbol BTC/USDT --interval 5m --input data\processed\你的九週期特徵檔.csv
```

查看帳戶狀態：

```powershell
python scripts\run_paper_trading.py status --account rl_btc
```

這個模組不含任何真實交易所下單程式碼。

## 無人值守模式

常駐模式會定期下載最新資料並呼叫相同的單次模擬流程；同一根 K 線仍不會重複
預測或成交。每個帳戶的 `automation/` 會保存心跳、單一執行鎖與停止請求。

```powershell
python scripts\run_paper_trading.py rl-auto `
  --account rl_btc `
  --model-dir data\processed\rl\environments\你的環境\training\你的訓練 `
  --exchange binance --symbol BTC/USDT --interval 5m `
  --download-latest `
  --poll-seconds 60
```

```powershell
python scripts\run_paper_trading.py auto-status --account rl_btc
python scripts\run_paper_trading.py stop --account rl_btc
```

Dashboard 可以直接選擇訓練紀錄與 RL 模型曾看過的市場，不必手動輸入路徑。

### BTC 15m WebSocket 模式

BTC 15m 即時模式不再固定重抓整批資料。啟動後會：

1. 以 REST 將 `5m、15m、1h、4h、1d` 暖機資料載入記憶體。
2. 以 Binance WebSocket 持續接收五週期 K 線與最佳 Bid／Ask。
3. 只有 `x=true` 的已收盤 K 線會進入固定長度的記憶體窗口，不讓半根 K 線進入模型。
4. 每根 15m 收盤後用 REST 從記憶體最後時間戳補漏；REST 暫時失敗時會先做資料時效檢查。
5. 模型直接使用記憶體資料，不會在每次決策時重讀五份完整 CSV。
6. 本機只覆寫保存最近 2,000 根作為圖表與重啟備援，訓練用歷史下載不受此上限影響。
7. Transformer 判斷趨勢，SAC／PPO 提出目標持倉；資金管理、Spread 與 Risk Governor 共同限制新增風險。

```powershell
python scripts\run_paper_trading.py rl-auto `
  --account btc_15m_realtime `
  --model-dir data\processed\rl\environments\你的環境\training\你的PPO訓練 `
  --exchange binance_futures --symbol BTC/USDT --interval 15m `
  --transport websocket --download-latest `
  --transformer-checkpoint data\processed\transformer\models\你的模型\best_model.pt `
  --transformer-min-probability 0.45 `
  --transformer-max-uncertainty 0.62 `
  --max-spread-bps 5
```

即時狀態保存在：

```text
data/paper_trading/<帳戶>/automation/realtime_status.json
```

其中會記錄 WebSocket 連線、訊息延遲、各週期目前形成中的 K 線、最新 5m／15m 收盤、
Spread、RL 原始目標、Transformer 牛熊機率、趨勢確認後目標、最終風控目標與拒絕
原因。模擬機器人頁每兩秒更新狀態；機器人監控交易圖每五秒更新。未收盤 K 線只供
畫面顯示，不會送入 Transformer 或 RL 模型。

目前已保存的 Transformer 若只有 smoke 訓練，介面會顯示研究品質警告。即時資料
管線可正常運作不代表模型已有統計優勢；未通過完整 Walk-forward、成本後績效與
多 seed 驗證前，只能使用模擬倉。

Dashboard 可啟動獨立背景程序；桌面視窗關閉後工作仍會繼續，但電腦必須保持
開機且不可睡眠。在「交易中心 → 模擬機器人」開啟「機器人持續運行」一次後，
下次啟動 App 會自動恢復工作；關閉切換會送出安全停止。連續錯誤達上限時 runner
仍會自動熔斷，避免失敗工作無限重啟。

目前真實下單 Gateway 使用 Binance Spot，不支援裸賣空。多空模型的負目標可在
訓練、回測評估與模擬交易使用；送到 Spot Live 時只會平掉既有多單。
