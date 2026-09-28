# Transformer 時序模型

## 固定策略事件研究模式

新增明確選用的 `strategy_event`，不是既有模擬盤模型的直接替換品。
只在 1h 趨勢確認後的 15m 突破候選上，學習包含保護出場與成本的交易淨收益、
收益分位數及淨收益超過門檻的機率。沿用本模組的 V3、時間切分、校準與訓練器；
沒有建立另一套訓練主程式，也沒有改動舊模型契約。

使用 `configs/transformer_strategy_event.example.json` 與
`scripts/run_transformer_strategy_event.py`；預設僅資料檢查，`--smoke` 才跑一輪小測試。
操作、規則及限制詳見 [事件研究說明](../../../docs/transformer_strategy_event_20260924.md)。
這類 checkpoint 明確標記 research-only，舊版 SAC／模擬盤／實盤推論入口會拒絕載入。

所有新的一般訓練結果另外保存 `actual_direction_*`、市場身分、標籤契約雜湊、
校準前機率及校準後機率。集成評估只使用原始答案；舊 CSV 缺答案時拒絕推算。

## V3.4 經濟目標研究設定

2026-09-24 的 Kaggle 實驗使用 `trading_target_mode="terminal_net"`：下一根開盤進場、
持有指定 horizon 後收盤，方向、可交易性和淨優勢頭使用相同的成本後報酬定義。
價格移動的 first-touch 頭只作輔助；舊 checkpoint 的 first-touch 契約不會被改寫。

`training_horizon_weights=(0.15, 0.70, 0.15)` 同時作用於多週期訓練任務。
驗證區間前半用於溫度校準、後半用於 checkpoint/seed 選擇，中間保留 purge/embargo。
`checkpoint_metric="economic_selection_score"` 以不重疊固定持倉的淨期望值信賴下界選模，
低於最低交易筆數會被否決。分數只使用當時的預測淨優勢與不利波動；不靠未來報酬選訊號。

新增 `validation_predictions.csv`、`*_signal_diagnostics.csv`、`*_fixed_hold_trades.csv`，
以及 `training.json` 內的 `calibration_protocol` 和 `economic_evaluation`。
這是 1x 固定持倉研究診斷，不是完整逐筆成交回測；回撤只含已結算資產，資金費率是估計值。
實驗與操作詳見 `docs/transformer_v34_economic_training_20260924.md`。
下方 V3.1 的分類選模、高信心排序和 first-touch 主標籤敘述屬舊設定，不是本次 V3.4 契約。

## 精簡特徵契約

新訓練使用「時間週期 × 指標類型」輪流挑選，排除原始 OHLCV 與美元價格指標，
保留比例化資訊與九個跨週期市場情境。推論依 checkpoint 記錄的順序與 scaler 重建輸入。
既有 V3 模型維持原欄位；新特徵需重新訓練，不可只修改 checkpoint 的欄位清單。
完整欄位與操作見 `docs/compact_ai_inputs_20260915.md`。

此模組提供金融市場專用的多任務 Transformer 架構、時間序列 Dataset、訓練器與模型保存。
新訓練預設使用 v3 架構。模型把特徵依時間框分為快速（1m／3m／5m）、中速
（15m／30m／1h）與慢速（4h／12h／1d）三個分支，各自進行輸入閘門、局部卷積、
Patch 投影與 RoPE Transformer 編碼，再以跨時間框注意力融合。缺失遮罩會明確告訴
模型哪些即時欄位不存在，避免把補值誤認成真實的零。主線只接受 V3 checkpoint，
舊架構的 Checkpoint 不會被載入。

## 使用方式

1. 先在「特徵資料」建立 Step 3 特徵 CSV。
2. 開啟「策略研究」→「Transformer 訓練」。
3. 第一次先保持「快速驗證模式」，用 1～2 Epoch 確認資料與硬體可以運作。
4. 正式研究時關閉快速模式，選擇相同 K 線週期的多個市場，再調整模型與訓練參數。
5. 訓練期間可切換其他功能；工作進度會保存在磁碟，回到此頁仍可繼續查看。
6. 訓練完成後切到「批次推論」，選擇模型與市場，將模型輸出寫回特徵 CSV。

多週期模型請先在「資料與 AI → 特徵資料 → 多週期 Transformer／SAC 資料集」建立
`features_mtf_*.csv`。同一次訓練不可混用單週期與多週期資料；正式 v3 建議提供
1m、3m、5m、15m、30m、1h、4h、12h、1d，並至少保留 128 個高價值輸入特徵。

批次推論會使用 `best_model.pt` 內保存的訓練特徵順序與 scaler，不會用完整歷史
重新估計標準化參數。只有序列長度足夠、核心特徵覆蓋達標，且必要多週期群組存在時，
才會設定 `transformer_available=1`。`transformer_input_coverage` 與
`transformer_input_valid` 可供監控判斷降級原因；FinBERT、新聞及總經等選用特徵仍可透過
缺失遮罩安全降級。

訓練會依時間順序切分 Train、Validation、Test，Scaler 只使用 Train 區段估計，
且不同市場的 K 線不會被串成同一條序列。V3.1 可把方向任務拆成三個部分：先判斷
價格本身向上、盤整或向下，再判斷扣除成本後是否值得交易，最後只在可交易樣本上
學習做多或做空。5 根負責進場 timing、20 根是主要交易目標、48 根提供 regime；
Checkpoint 權重固定為 0.15／0.70／0.15，使短期雜訊不會主導選模。
新訓練只用 Train 統計類別權重，再依 Validation 的分層平衡準確率、相對多數類
基準提升及 Loss 組成的 `hierarchical_skill_score` 保存模型；Test 不參與選模。

Train／Validation／Test 邊界會套用 purge 與 48 根 embargo，避免標籤視窗跨越資料切分。
Train 可使用 365 天半衰期的時間衰減抽樣，讓近期市場狀態有較高權重但不抹除舊行情。
訓練結果會輸出類別分布、Jensen-Shannon divergence 與特徵平均漂移；這些診斷是警報，
不是看到 Test 後調參的理由。

正式研究建議至少訓練 5 個固定 seeds，先依 Validation 分數選候選，再一次性查看
Test。多 seed 的等權機率平均只能列為另一個事先定義的候選，不能在看過 Test 後
反覆挑選 seed 或權重。方向準確率、平衡準確率、Macro F1 與多數類基準必須按每個
預測週期分開檢查，平均數通過不代表每個週期都有優勢。部署閘門以主要 20 根目標的
高信心 5%／10%／20% 樣本為主，檢查扣除 fee、slippage、spread、funding 後的
expectancy、win rate 與 profit factor；全體樣本準確率只保留為診斷。

模型輸出：

- 未來 5、20、48 根 K 線的標準化報酬
- 扣除成本門檻後的下跌、中性、上漲機率
- 不含交易成本的價格移動方向機率
- 可交易樣本中的條件式做多／做空機率
- 報酬 q10、q50、q90 分位數與預測區間
- 預估波動度
- 多頭、空頭與盤整行情機率
- 扣除手續費、滑價、價差與資金費率後的多單／空單淨優勢
- 未來窗口的最大不利波動與最大有利波動
- 各預測週期的可交易性機率
- 低、中、高波動狀態機率
- 快、中、慢時間框注意力權重
- 模型不確定度
- 可供 SAC 使用的壓縮時序向量

v3 的交易標籤以「下一根開盤可成交、未來 N 根內先碰到多方或空方成本障礙」建立；
同一根 K 線同時碰到兩側時保守標記為不交易。成本障礙包含手續費、滑價、觀測價差、
資金費率與安全邊際，參數可在訓練頁調整。分類機率可使用 Validation 區段做溫度校準，
Test 區段不參與校準。
這些設計能使研究假設更接近成交現實，但不保證模型具有未來獲利能力。

BTC 15m 低硬體建議先使用 256 個高優先特徵、96 根序列、`d_model=96`、3 層與
batch size 64。特徵上限不是越高越好；資料不足時擴大模型只會增加過度擬合風險。

訓練結果保存於：

```text
data/processed/transformer/models/<訓練時間_名稱>/
    best_model.pt
    history.csv
    test_predictions.csv
    training.json
```

`best_model.pt` 同時保存模型權重、實際特徵順序、Scaler、標籤標準化參數、
機率校準溫度、特徵完整性合約與資料來源。
快速驗證模型只用來確認流程，不應直接用於模擬或實盤交易。

所有輸出都必須帶有 `transformer_available`。只有值為 `1` 時，PPO 才能把它視為有效模型訊號。
訓練完成本身不會自動改動既有 PPO；還需要執行 Transformer 推論建立市場特徵，
重新建立強化學習環境並重新訓練 PPO。
