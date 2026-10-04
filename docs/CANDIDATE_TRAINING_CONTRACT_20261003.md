# 候選交易契約與 Transformer 研究流程

## 本次做什麼

新增 `btc_candidate_v1` 研究契約，接通既有複合策略與 Transformer V3。
不啟動 SAC、Kaggle、實盤或部署；舊研究成果不覆寫，舊設定仍走舊事件管線。
程式測試通過與策略有正期望是兩回事，煙霧模型不具實盤資格。

## 固定的三種候選

| 策略 | 訊號來源 | 退出趨勢條件 |
|---|---|---|
| original | 原 H1 EMA50/200、ADX 確認後的 15m 前 20 棒突破 | loss |
| trend_pullback | 已有 H1/H4 同向、15m EMA20 回調、RSI 與量能 | opposite |
| vwap_reversion | 已有低 ADX、低 H4 斜率、布林回歸與 UTC VWAP 距離 | disabled |

三者皆為 normal、15m、2 ATR 停損、2R 停利、32 根持有上限、4 根冷卻。
新增策略是依趨勢／均值回歸機制選定，不引用舊 Forward 最佳參數。
每個模型只有一種固定策略，一份同市場快照，不混合不同交易標籤，也不是三策略組合帳戶。
如果以後需要混合訓練，必須另增策略身分及條件輸入，不可直接拼接這三份模型樣本。

## 共用與差異

- 訊號直接使用 `complex_strategies.py`，標籤使用既有 `batch_events`、`holding_limit`。
- 舊慢速 `replay_event` 保留為參考；新增測試逐筆核對。
- 新契約的相對成交量一律用「當根成交量 / 前 20 根均量」，排除當根，欄位明列 `relative_volume_prev20`。
- 原事件設定仍保留包含當根均量的舊定義，沒有宣稱舊定義前視或錯誤。
- ATR/ADX 沿用 intraday；EMA 使用 adjust=False；布林 ddof=0；高週期只能於完整收盤後使用。
- 複合策略的日內 VWAP 依 UTC 開盤日歸屬，午夜收盤棒不能提前歸到隔日。
- 新增成本／ATR 與距下次假設結算時間兩欄；既有情境特徵不重複新增。
- 標籤與候選共用 65 根尾端保留，延續批次引擎契約；資料切分再保守隔離持有上限加 2 根。

## 成本與路徑

基本假設為單邊手續費 5bp、滑價 2bp、完整價差 1bp；壓力情境單邊滑價加 3bp。
Funding 每跨 UTC 00/08/16 扣 1bp 雙向準備金，不是實際歷史付款；稅後未驗證。
訊號收盤後下一棒開盤進場，趨勢退出只看前一收盤，棒內停損優先於停利。

固定路徑歸因順序為無摩擦毛收益、扣價差、再扣滑價、手續費及 funding。
滑價與價差存在交互作用，這個分配順序明確固定，兩者加總才是完整摩擦成本。
與此分開，zero/base/stress 各自重新撮合、決定退出並排程冷卻；不是直接從報酬扣常數。
目前只模擬市場成交，沒有宣稱 Post-only 成交、掛單排隊或 Maker 費率已成立。

MFE/MAE 使用不含退出棒未知高低點的已知價格建立下界，加入退出棒高低作可能上界。
退出棒高低可能發生在平倉後，因此不提供假的精確棒內先後或秒級時間。
停損後觀察從下一棒開始固定 8 根；不足窗口標示缺值，不假裝為零。
這些欄位只存在歸因輸出，不進特徵白名單；模型 metadata 只保留必要的標籤及因果情境。
分組採事前固定 UTC 八小時區間、方向、已知趨勢及 ATR/close 的 0.2%／0.5% 波動界線。
移除最佳五筆只作敏感度描述，不是要求趨勢策略不能依靠少數大贏家。

## 訓練與驗證

沿用 d_model=64、2 層、96 棒序列的現有 V3。煙霧只改成 CPU、一輪，不加大網路。
Train 的事件重疊以平均反並發度加權抽樣；權重只由 Train 事件建立。
Calibration、Selection、Test 維持原自然分布；已實現退出與保守標籤結束皆需早於切點。
Train-only 中位數、標準化及特徵刪選不得用到未來資料。

正式計畫包含 3 策略 × 3 時段 × 3 seeds × 5 組 = 135 次訓練：
1. 舊特徵 + Smooth L1。
2. 舊特徵 + MSE。
3. 新情境特徵 + Smooth L1。
4. 新情境特徵 + MSE。
5. 新情境特徵 + MSE，移除成本特徵。

上述舊／新指特徵集合；兩者均使用新候選契約與相同 canonical 量能定義。
不會自動選擇一個測試表現最好的模型並部署。三 seeds 是探索，不是三倍獨立市場樣本。
統計濾網只用 Train 的方向 × 已知 H1 趨勢，固定 20 樣本先驗收縮。
三方比較為無 AI、統計濾網、Transformer，皆用同一候選及固定 55%／2bp 淨收益門檻。
預測淨收益已扣成本，篩選時不再扣一輪費用。
分數五組邊界在 Selection 凍結，Test 只套用；機率可靠度用十等寬區間。
記錄 MSE/Brier 相對 Train 常數、校準誤差、分位數覆蓋率、成交數及成本後均值/區間。
候選群組可以重疊，不能加總為帳戶報酬；只有 policies 檔案是不重疊成交。
不輸出未包含持倉浮虧的完整帳戶 Sharpe、回撤或清算資格。

## 使用方式

在專案根目錄 PowerShell 執行；不需要交易所 API Key。

```powershell
$env:PYTHONPATH='src'
# 只寫計畫，不訓練；輸出必須是不存在的新目錄。
python -X utf8 scripts/run_candidate_training_research.py `
  --source outputs/kaggle_strategy_event_20260924/staging/data/btc_15m.csv `
  --output artifacts/candidate_training_plan_new --seeds 42 137 2026

# 三種策略各一輪，本機 CPU 2 執行緒；預設用研究前 90% 的最後 48000 棒。
python -X utf8 scripts/run_candidate_training_research.py `
  --source outputs/kaggle_strategy_event_20260924/staging/data/btc_15m.csv `
  --output artifacts/candidate_training_smoke_new --smoke

# 完整研究命令，須另外確認資源與實驗方案才執行；不是本次自動執行項目。
python -X utf8 scripts/run_candidate_training_research.py `
  --source outputs/kaggle_strategy_event_20260924/staging/data/btc_15m.csv `
  --output artifacts/candidate_training_formal_new --seeds 42 137 2026 --run-research
```

進度顯示於終端與 `progress.json`。每個模型保留 checkpoint、training.json、預測、歸因與三方交易 CSV。
plan.json 保留來源、程式、設定雜湊與完整矩陣；study.json 彙整執行及失敗。
缺樣本時保留失敗，不偷偷降低門檻；契約／因果／帳務错误則停止。
輸出不支援原地覆寫；中斷成品保留供診斷，重新執行使用新目錄。

## 品質與下一步

工程通過只能證明流程及已測契約可用。煙霧結果永遠標示研究證據不足、live_eligible=false。
完整研究亦不自動授予實盤資格，需檢查跨期分布、種子、成本壓力、集中度與多重搜尋影響。
歷史已反覆研究，即使最後 10% 本次不讀，也不宣稱它是全新 final holdout。
未來封存必須晚於整個專案已觀察歷史，不能只取本輪 fold 的結尾；plan 保留完整來源結尾，盤點後再前向收集。
P2/SAC 保持封鎖，待合法樣本外訊號有可信成本後優勢，再比較固定部位、波動率部位與 SAC。
最小下一輪是先批准少量策略的配對跨期研究；若持續不勝過常數/統計基準，不擴大模型或強迫成交。
