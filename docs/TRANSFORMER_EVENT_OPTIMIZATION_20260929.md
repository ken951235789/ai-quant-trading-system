# 最新事件模型分析與優化

## 結論

本次檢查的是 2026-09-24 的 BTC 15m 固定策略事件研究，不是較早的 SAC 成績。
目前證據不足以把它升級為可交易模型。最新一次沒有成交的直接原因是所有候選的成功機率都低於 55%，但降低門檻並不能解決預測品質與策略本身期望值偏弱的問題。

本次已完成學習目標、因果情境特徵、選模與診斷的程式優化；只執行本機一輪小模型測試，未提交 Kaggle、未更換模擬或實盤模型。
「事件研究 V2」是設定版本，底層仍然使用 Transformer V3 架構，並非恢復舊的 Transformer V2。

## 分析對象

- 訓練目錄：`outputs/kaggle_strategy_event_20260924/results/strategy_event_results/20260924T140908271774Z_btc_event_seed42/`。
- Binance USD-M BTCUSDT 15m，共 176,729 根已收盤 K 棒，UTC 2021-09-05 09:30 至 2026-09-20 07:30。
- Train 1,919 個事件；Validation 806 個事件，再切出 Calibration 與 Selection；Selection 397 個事件；Research Test 774 個事件。
- 原始資料 SHA256：`791b3297b07e640ca12c06b1140b9abac3a69b0f31c0c01247936e282927de2b`。
- 本機測試使用當次凍結、清理後的 Kaggle CSV，SHA256：`0dff02b3ec7943d0a4cd650b5717d961ef1793ebfe5f8d7dec0c95795278b5c6`。
- 這些 Test 結果已被檢視，之後不能再把同一區間宣稱為從未使用的 final holdout。

## 最新完整訓練的證據

| 指標 | Selection | Research Test | 解讀 |
|---|---:|---:|---|
| 固定規則、不經 AI 過濾的不重疊交易數 | 156 | 329 | 不是 AI 實際執行的交易 |
| 固定規則每筆平均淨報酬 | -0.2147% | -0.2561% | 已扣設定的費用、滑價、價差、funding 準備金 |
| 每筆期望值區塊 bootstrap 95% 區間 | [-0.3129%, -0.1126%] | [-0.3631%, -0.1568%] | 區間估計仍依賴樣本及區塊抽樣假設 |
| 固定規則勝率 | 32.05% | 30.40% | 不足以單獨判斷獲利能力 |
| AI 過濾後成交數 | 0 | 0 | 無交易不能認定策略有正期望值 |
| 預測成功機率最大值 | 42.32% | 45.66% | 都低於事先固定的 55% |
| 報酬 MSE 相對 Train 常數基準 | 改善 1.61% | 惡化 8.16% | 平均收益估計的泛化能力不足 |
| 機率 Brier loss 相對 Train 先驗 | 改善 0.12% | 惡化 1.05% | 機率品質也未勝過簡單基準 |
| 報酬 Pearson / Spearman | 0.186 / 0.215 | -0.031 / 0.196 | 有部分排序訊息，不等於絕對收益估計可靠 |

Test 的 67.70% 可交易性分類準確率，等於「全部預測不具可交易性」的結果，不能當成 67.70% 的交易勝率。
將 Test 候選依預測收益分成五組，最高分組的平均實際淨收益仍約 -0.1792%；這只是含重疊事件的事後描述，不能拿分位值回頭改交易門檻。

### 成本不是唯一原因

固定規則在原本成交路徑不變的情況下，Test 每筆拆解約為：

| 項目 | 每筆平均，占入場名目金額 |
|---|---:|
| 扣摩擦成本前的同成交路徑收益 | -0.10238% |
| 手續費 | 0.10004% |
| 滑價與價差 | 0.05004% |
| Funding 準備金 | 0.00365% |
| 扣成本後 | -0.25611% |

這是成本歸因，不是把費用設成零後重新產生停利停損及成交。
以上不是帳戶報酬，未處理槓桿、複利、完整資金配置或個人稅後所得；不可加總每筆百分比後直接當成帳戶虧損率。
Funding 使用假設準備金，並非歷史實際付款；稅務仍標示未驗證。

## 已完成的優化

### 1. 平均收益頭與學習目標對齊

新增 `return_loss_kind`。舊模型預設維持 `smooth_l1`；新事件設定使用 `mse`。
平方誤差對應條件平均的估計目標；Smooth L1 是較不敏感於離群值的穩健損失，不能直接當成條件平均的同義詞。
這是待驗證的改善假說，不代表已證明舊模型失敗只因為損失函數。
MSE 也更怕極端值，因此仍保留輸入檢查、Train-only 標準化、梯度裁切、早停與報酬分位數頭。

技術依據：[scikit-learn 線性模型文件](https://github.com/scikit-learn/scikit-learn/blob/main/doc/modules/linear_model.rst)、[PyTorch SmoothL1Loss](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.loss.SmoothL1Loss.html)。

### 2. 新增 20 個可重現情境特徵

| 類型 | 特徵 |
|---|---|
| K 棒形狀 | 實體 / ATR、上下影線 / ATR、收盤在棒內的位置 |
| 前日結構 | 距前日高低的 ATR 距離、前日區間位置、越過前高後收回、跌破前低後收回 |
| 當日情境 | 截至當下的日內區間 / ATR、日內位置、日內 VWAP 的 ATR 距離 |
| 突破品質 | 突破先前 20 棒高低的 ATR 幅度 |
| 跨週期 | 15m / 1h / 4h 趨勢共識、分歧、快慢週期符號乘積，負值表示衝突 |
| 時間 | UTC 星期 sin / cos、週末旗標 |

保留原 53 個候選欄位，加入後為 73 欄；五年份資料以 Train 去除常數與高度相關欄位後保留 71 欄。
價格距離比例化，不把原始 OHLCV 當成模型特徵；OHLCV 仍是生成特徵及成交回放的來源。
前日高低只使用完整的前一 UTC 日，當日高低與 VWAP 使用累計值。沒有用當日最終高低、未收盤高週期或居中窗口。
這些是價格情境，不是聲稱觀察到了真正的停損單、資產同步、CiC 或 PoP；未添加無資料支援的概念。

### 3. 選模要勝過簡單基準

Train 固定保存平均收益與正標籤比例，Selection 上計算：

```text
return_skill = 1 - 模型 MSE / Train 常數 MSE
probability_skill = 1 - 模型 Brier / Train 先驗 Brier
event_prediction_skill_score = (return_skill + probability_skill) / 2
```

基準與模型都在同一個評估區間算損失，但常數只能由 Train 決定，不能使用 Test 平均值。
新設定以 Selection 的此分數選 checkpoint，機率校準仍只用更早的 Calibration。
平均分數僅供選模；研究品質閘門另外要求兩種 skill 都為正、足夠實際交易、淨期望值信賴區間下界大於零。
預設 30 筆只是研究最低檢查值，不是達到 30 筆就具有實盤資格。所有事件研究輸出仍 `live_eligible=false`。

### 4. 零成交與未訓練輸出不再掩蓋問題

- 共用一份不重疊成交挑選器，分開統計持倉／冷卻、只差機率、只差預期報酬、兩者皆不足。
- 輸出 Train 基準、分組診斷、成本拆解及額外 6 bps 成本敏感度。
- 零成交的診斷平均報酬是 `null`，不以 0% 冒充有證據的收益；既有摘要欄位保留相容性並配合 `has_trades`。
- 事件預測 CSV 不再輸出未訓練的方向、行情分類與風險輔助頭；保留有訓練的淨收益、分位數及成功機率。
- 仍禁止舊 SAC／模擬／實盤推論入口載入事件研究模型，避免把不同標籤語意混用。

## 如何執行

在專案根目錄啟動 PowerShell，使用已安裝專案依賴的 Python。
以下 `--source` 使用已存在的凍結 CSV，不會下載資料或連線交易所。

### 資料預檢，不訓練

```powershell
python -X utf8 scripts/run_transformer_strategy_event.py `
  --source outputs/kaggle_strategy_event_20260924/staging/data/btc_15m.csv `
  --config configs/transformer_strategy_event_v2.example.json
```

### 小型流程測試

```powershell
python -X utf8 scripts/run_transformer_strategy_event.py `
  --source outputs/kaggle_strategy_event_20260924/staging/data/btc_15m.csv `
  --config configs/transformer_strategy_event_v2.example.json `
  --smoke
```

`--smoke` 使用最後 16,000 棒、一輪、CPU、縮小的模型；不能拿來評估五年份正式模型的品質。
將 `--smoke` 改為 `--train` 才依新設定啟動研究訓練，並不會自動部署。
未指定 `--config` 的舊指令仍使用舊事件設定，避免既有研究被悄悄改動。桌面 EXE 與舊 SAC 管線未在本次重打包或替換。

### 分析已完成的研究

```powershell
python -X utf8 scripts/diagnose_strategy_event_run.py `
  --run-dir outputs/kaggle_strategy_event_20260924/results/strategy_event_results/20260924T140908271774Z_btc_event_seed42 `
  --output artifacts/event_diagnostics_new.json
```

診斷程式不載入模型權重，不重新訓練，只讀摘要與預測 CSV，記錄輸入雜湊且拒絕覆寫已有輸出。
本次完整數據在 `artifacts/reports/20260929/event_optimization/archived_run_diagnostics.json`。
新訓練另產生 `validation_event_diagnostics.json`、`test_event_diagnostics.json` 和每輪選模分數。

## 下一輪研究順序

1. 先預先登記比較方案，固定候選進出場、成本、資料切分與 seed。不要以這次 Test 的結果挑選新交易門檻。
2. 在相同的新選模指標下，分別比較舊特徵 + Smooth L1、舊特徵 + MSE、新特徵 + Smooth L1、新特徵 + MSE，才知道貢獻來自哪裡。
3. 先以 Train／Calibration／Selection 做時間向前推進的驗證；最後使用全新保留區間或前向模擬。不能用同一 Test 反覆調到好看再宣布成功。
4. 固定規則基準目前為負。如情境特徵仍無法找到可重現的正期望值子集合，應重新研究候選策略及執行成本，不要只擴大 Transformer 或強迫 SAC 交易。
5. 通過後才以合法的 OOS 輸出建立 SAC 訓練資料；再驗證 SAC 是否比固定倉位策略改善成本後報酬與風險，而不是只增加步數。

本次保留舊模型、資料與結果，不保證新增特徵或改變損失函數必然提高未來收益。
