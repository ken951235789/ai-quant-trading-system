# Transformer 固定策略事件研究

## 這次改了什麼

這是新研究分支，不覆寫原模型、不改 SAC 行動、不啟動機器人，也不放寬 Live 閘門。
重點不是加深模型或增加方向準確率，而是讓學習問題和要執行的交易一致。

1. 修正多種子集成評估：直接保存並使用 `actual_direction_*`，不再以固定 12 bps
   重建實際採用波動門檻的訓練答案。核對市場身分、時間、endpoint、標籤及契約雜湊。
2. 保存校準前後機率、Brier、NLL、預測 HOLD 比例；事件模式只報有效的交易成功機率。
3. 新增 `strategy_event`，重用現有 V3 訓練器、標準化、時間切分、校準、早停及模型保存。
4. 53 個候選輸入，再只以 Train 刪除常數／高相關欄位。沒有原始價格或成交量輸入。
5. 標籤與研究評估共用 `transformer/strategy_events.py` 成交結果，不另算另一套停損停利。

## 固定規則 A

| 項目 | 已實作的定義 |
|---|---|
| 市場 | Binance USD-M BTCUSDT 永續，15m 決策，UTC 開盤時間 |
| 趨勢 | 連續兩根已收盤 1h 棒，EMA50 > EMA200 且 ADX14 >=25 偏多；偏空相反 |
| 進場 | 15m 收盤突破前 20 根最高價做多／最低價做空；當根不納入前高低 |
| 成交 | 訊號後下一根開盤，扣滑價及半邊 spread |
| 停損 | 實际成交價上下 2 倍訊號棒 ATR14；進場時固定 |
| 停利 | 實際成交價上下 4 倍訊號棒 ATR14；進場時固定 |
| 時間退出 | 最多持有 32 根完整 15m 棒，於下一根開盤退出；不是固定百分比停利 |
| 制度退出 | 已收盤 1h 趨勢不再符合持倉方向，下一開盤退出 |
| 保護成交 | 同棒同碰先停損，跳空停損採更差開盤價；停利不給正跳空價改善 |
| 持倉 | 不加碼、不重疊；出場當棒不反手，完整等待 4 根後才接受新訊號 |
| 暖機 | 預設 1,000 小時；暖機期間不產生交易候選 |

EMA：`ewm(span=N, adjust=False, min_periods=N)`。
ATR／ADX 共用既有 `features.intraday._directional_movement` 的遞迴定義；ADX 轉成 0–100 比較。
這些定義不等於任何外部平台的預設暖機／初始化方式，沒有宣稱逐根符合外部平台。

候選 B 的均值回歸尚未加入，先單獨測 A，避免策略混合後無法歸因。

## 模型看到與學到什麼

- 時間框：這個精簡模式使用 15m、1h、4h，由同一份 15m 資料建立，不需要九份 CSV。
- 每個時間框 16 個比例特徵：1／4 根報酬、EMA20／50／200 距離、EMA50/200 比例、
  EMA50 斜率、ATR 比例、ADX、DI 差、RSI、布林 Z 值／寬度、相對量、已實現波動及高低價幅度。
- 加上兩個突破距離、當下候選方向、UTC 小時 sin／cos，共 53 個，再按 Train 去重。
- 候選方向是固定策略已知的輸入，成功及失敗事件都保留，不能只取獲利交易。
- 有效輸出只有：候選交易淨收益、淨收益分位數、淨收益超過 2 bps 的機率。
- 淨收益按進場名目本金計算，包含出場費用；不是槓桿帳戶報酬率、也不是價格方向。
- 只啟用收益、分位數及可交易性三個 loss；其他既有頭保留相容性，但不訓練、不使用。
- 不保證三個輸出完全一致；預設須機率 >=0.55 且預測淨收益 >2 bps 才通過研究過濾。

參數是待驗證研究起點，不叫「最佳參數」。使用 32 根是固定策略的最長持有上限，
不是把舊模型的 20 根預測直接改成 32 根。舊模型不能靠修改 metadata 取得新能力。

## 成本與風控邊界

| 成本 | 目前假設 |
|---|---|
| 手續費 | 每邊 5 bps =0.05%，無折扣；出場按出場名目本金計算 |
| 滑價 | 每邊 2 bps =0.02%，所有出場原因都扣 |
| Spread | 全寬 1 bp，每次成交使用半邊 |
| Funding | 每經 UTC 00／08／16 點扣 1 bp 的雙向成本準備金 |
| 稅 | 未驗證，不宣稱稅後收益 |

Funding 是明列的保守假設，不是歷史實際費率，不給空單假想 funding 收入。
在結算邊界進場視為結算後；原持倉於結算邊界退出視為先結算再退出。
15m 棒內退出不多收下一根開盤的結算。正式研究仍需導入真實結算帳單和 mark price。

研究輸出 `notional_return_sum` 只是每筆報酬相加，不是帳戶累積報酬。
這個事件診斷不包含個人帳戶設定，也不會改動既有實盤風控。

## 如何使用

在 PowerShell 執行。程式使用本機原始資料，不下載新資料、不花 Kaggle 額度。

### 只檢查資料及候選數

```powershell
python scripts/run_transformer_strategy_event.py `
  --source data/raw/crypto/binance_futures/ohlcv/ohlcv_crypto_binance_futures_BTC-USDT_15m_latest.csv
```

### 一輪流程測試

```powershell
python scripts/run_transformer_strategy_event.py `
  --source data/raw/crypto/binance_futures/ohlcv/ohlcv_crypto_binance_futures_BTC-USDT_15m_latest.csv `
  --smoke
```

只取最後 16,000 根，CPU、2 threads、1 epoch、小型 V3。顯示百分比、階段及 epoch。
百分比是訓練批次進度，100% 後還要完成校準、測試與存檔，以 `complete` 訊息為準。
重跑會產生新的時間戳資料夾，不覆蓋舊結果。

### 日後完整研究

```powershell
python scripts/run_transformer_strategy_event.py `
  --source data/raw/crypto/binance_futures/ohlcv/ohlcv_crypto_binance_futures_BTC-USDT_15m_latest.csv `
  --train
```

完整研究使用來源全部資料，設定在 `configs/transformer_strategy_event.example.json`。
預設 20 epochs、早停 5、seed42、裝置 auto；有相容 CUDA 才會用 GPU。
2026-09-24 已完成一輪完整資料的單 seed Kaggle 研究，負面結果見
[研究結果](RESEARCH_RESULTS.md)。這個模式由 CLI 操作，沒有加進舊版訓練 UI。

### 建立私人 Kaggle 工作包

```powershell
python scripts/build_kaggle_strategy_event.py `
  --source data/raw/crypto/binance_futures/ohlcv/ohlcv_crypto_binance_futures_BTC-USDT_15m_latest.csv `
  --output outputs/my_strategy_event_run --username YOUR_KAGGLE_USERNAME
```

這個命令只建立本機工作包，不會上傳或花 GPU 額度。檢查包內檔案及
`strategy_event_manifest.json` 後，再使用自己的 Kaggle CLI 提交。
Dataset 必須保持私人；Notebook metadata 預設私人、關閉 Internet。
Kaggle 認證由 CLI 的使用者設定或環境變數提供，不能放入專案、Notebook 或上傳包。
工作包只包含相依原始碼、九欄行情、設定及 SHA-256 清單，沒有舊模型或交易狀態。
市場資料由使用者自行準備；GitHub 不附行情、權重或完整研究輸出。

## 怎麼看結果

預設產出在 `artifacts/strategy_event_research/<UTC時間>_<run名稱>/`：

| 檔案 | 用途 |
|---|---|
| `training.json` | 樣本數、實際特徵、設定、校準前後 Brier、基準與過濾後交易指標 |
| `history.csv` | 訓練／驗證 loss、選模數值、學習率；事件模式不展示未訓練方向頭的準確率 |
| `best_model.pt` | 研究 checkpoint；禁止直接接舊 SAC／模擬盤／實盤 |
| `validation_predictions.csv` | Selection 區段預測，不含拿來擬合校準器的區段 |
| `test_predictions.csv` | 研究 Test 預測、原標籤、候選進出場價及成本、退出原因代碼 |
| `experiment.json` | 來源 SHA-256、設定、是否 smoke、來源是否在執行中改變 |
| 完整性清單 | 由現有 integrity 模組建立的產物雜湊 |

退出代碼：0 停損、1 停利、2 持有上限、3 趨勢制度退出。
預測 CSV 是所有候選，不代表全部成交；對照指標才有套用各自接受條件、持倉及冷卻。
CSV 仍含共享訓練器的輔助輸出欄位；事件模式未訓練的方向、regime、edge 頭不可採用。

## 驗證順序與限制

1. Train 60%，只用此段選欄位及 scaler；Validation 前半校準、後半選 checkpoint。
2. Selection 不按 Test 調整過濾門檻；Test 20% 只產生研究報告。
3. 邊界清除 33 根可能跨界的事件標籤，再留 32 根 embargo。
4. 基準及 AI 過濾都使用相同成本／成交／冷卻；不同進場決定會產生不同的可用時段。
5. `event_sufficient_trades` 只檢查至少 30 筆的研究最低門檻，不代表統計顯著或 Live 通過。
6. 目前看過多次的歷史 Test 不是全新 final holdout。正式部署前仍需未看過資料、
   walk-forward、多種子、成本壓力、完整逐棒權益及模擬交易驗證。

本輪 smoke 只能證明資料、訓練、校準、存檔及評估能跑通，不證明新增模式改善營利。
下一步是先驗證固定策略 A 自己是否有成本後優勢，再比較 AI 過濾是否有穩定增益。
若基準也沒有優勢，不應用更多訓練步數或提高槓桿掩蓋問題。
