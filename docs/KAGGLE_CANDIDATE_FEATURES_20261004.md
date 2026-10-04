# 候選新特徵 Kaggle 配對研究

## 結論

18 個模型完成，沒有程序失敗；Transformer 基本成本篩選回放全部零成交。
工程可執行不等於模型有正期望值。研究品質為證據不足，SAC / Champion / Live 保持封鎖。
公開版只包含程式、測試與彙總，不包含私人 Notebook、行情、權重、逐筆預測或帳戶資料。

## 固定設計

- Binance USD-M BTCUSDT 永續，15m 決策，參考已收盤 1h / 4h。
- 原突破、趨勢回調、VWAP 回歸各自訓練，不把不同交易規則混為同一標籤。
- 原有 `F_existing` 對照 `F_compact_combined`，Train 去重後分別 73 / 60 欄。
- seeds 42、137、2026；三策略乘兩特徵乘三 seeds，固定第 3 個研究窗。
- V3 主體：d_model 64、2 層、4 heads、序列 96、horizon 32、MSE 加分位數與成功機率。
- 最多 20 epochs、早停 5、batch 64、lr 0.0002；CPU 數值執行緒 2。
- 固定成本後機率 55% 與淨收益 2bp 門檻，不為增加成交而修改。
- 基本成本：單邊費用 5bp、滑價 2bp、價差全寬 1bp；每個假設 funding 結算扣 1bp 準備金。
- 壓力情境：單邊滑價加 3bp，重新撮合，不把改成本後的交易路徑當成同路徑歸因。

來源 176,729 根 15m 棒，約 2021-09-05 至 2026-09-20。第 3 窗使用來源前 90%，
Train 截至約前 60%、Calibration / Selection 在 60-80%、研究 Test 在 80-90%，
並清除跨界標籤。這不是五年全用於 fitting，也不是五年的 Test；未使用的最後 10%
不能宣稱為整個專案從未看過的封存資料。

Tesla T4 整輪程式時間 1121.96 秒，約 18.7 分鐘，不含排隊和下載。訓練使用的是固定規則
產生的候選事件，不是每根 K 棒都是一筆獨立訓練事件；不能據此估算大型 SAC 耗時。

## 新舊特徵

新組精簡重複價格表示，再加入已收盤主動買賣量不平衡、成交活動、單筆規模、
已知價位距離、回調幅度與訊號年齡。完整欄位定義見
[特徵研究](CANDIDATE_FEATURE_STUDY_20261003.md)。

下表為三 seeds 平均。收益 skill = 1 - 模型 MSE / Train 常數基準 MSE；
機率 skill 類似地以 Brier 衡量。大於 0 表示勝過該常數基準，不是收益率或準確率。

| 策略 | 特徵 | 收益 skill | 機率 skill |
|---|---|---:|---:|
| 原突破 | 原有 | -0.01454 | +0.00192 |
| 原突破 | 精簡合併 | -0.03026 | -0.00255 |
| 趨勢回調 | 原有 | -0.00797 | +0.00272 |
| 趨勢回調 | 精簡合併 | -0.00344 | +0.00807 |
| VWAP 回歸 | 原有 | +0.04226 | -0.06114 |
| VWAP 回歸 | 精簡合併 | +0.02445 | -0.04116 |

解讀：趨勢回調小幅改善，突破下降；VWAP 機率改善但仍劣於基準，收益預測反而下降。
不能只選一個改善數字說新特徵有效。JSON 同時保留每組 seeds 的標準差、最差值及個別數字。
本輪只跑既有/合併兩組，不能拆出 flow 和 setup 各自貢獻；完整五組三時段的 135 組尚未執行。

## 實際回放

| 無 AI 策略 | Test 候選，可重疊 | 不重疊交易 | 平均淨收益 | 標準差 | 最差單筆 | 壓力平均 |
|---|---:|---:|---:|---:|---:|---:|
| 原突破 | 393 | 163 | -0.2017% | 1.1251% | -3.9862% | -0.2798% |
| 趨勢回調 | 502 | 241 | -0.1508% | 0.9871% | -2.6459% | -0.2029% |
| VWAP 回歸 | 109 | 72 | +0.0533% | 0.8010% | -1.1510% | -0.0348% |

上表是每筆名目本金的成本後收益，不是完整帳戶報酬，亦非三策略混合組合。
VWAP 的基本成本描述性區塊區間約 [-0.1257%, +0.1996%]，包含 0，且成本壓力下轉負。
多 seeds 共用同一段行情，不能把上述交易數乘以 3 或 6。

18 組 Transformer 基本情境都沒有成交；所有 Test 候選被機率門檻或機率與收益門檻共同擋下。
這證明目前分數未產生符合既定條件的交易，不證明所有被擋交易都應該不做。
零成交的平均收益應為 null，不是 0% 正期望，也不是零風險證明。
不能根據本次 Test 直接調低門檻再宣稱改善。

## 可重現命令

先依 [候選契約](CANDIDATE_TRAINING_CONTRACT_20261003.md) 準備自己的同來源 15m CSV。
主來源需要基本 OHLCV 與商品身分欄；成交來源另有 quote_asset_volume、number_of_trades、
taker_buy_base_volume、taker_buy_quote_volume，逐根 OHLCV 必須一致。範例檔不附在 Git。

```powershell
$env:PYTHONPATH='src'
# 不含 --run-research 時只列計畫，輸出目錄必須不存在。
python scripts/run_candidate_training_research.py `
  --source data/raw/btc_15m.csv --flow-source data/raw/btc_15m_flow.csv `
  --feature-study --variants F_existing F_compact_combined --folds 3 `
  --seeds 42 137 2026 --output outputs/candidate_plan_new

# 本機工程測試只跑第一個 seed，每組一輪，不解鎖部署。
python scripts/run_candidate_training_research.py `
  --source data/raw/btc_15m.csv --flow-source data/raw/btc_15m_flow.csv `
  --feature-study --variants F_existing F_compact_combined --folds 3 `
  --seeds 42 137 2026 --smoke --output outputs/candidate_smoke_new

# 只建立私人 Kaggle 白名單封包；這行不提交或開始雲端訓練。
python scripts/build_kaggle_strategy_event.py `
  --source data/raw/btc_15m.csv --flow-source data/raw/btc_15m_flow.csv `
  --candidate-features --config configs/transformer_strategy_event_v2.example.json `
  --username YOUR_KAGGLE_USERNAME --output outputs/kaggle_candidate_new
```

確認自己的資料使用權、私人 Dataset、GPU 配額、實驗配置後才透過 Kaggle CLI 提交。
Notebook 預設 private、GPU、關閉網路；API 憑證由本機認證工具處理，不能放進封包。
雲端完成後下載到新目錄，保留原 staging，再核對：

```powershell
python scripts/audit_candidate_training_research.py `
  outputs/download_new/candidate_feature_results `
  --bundle-root outputs/kaggle_candidate_new/staging
```

稽核要求當前程式與原執行雜湊相符；修改過程式或拿不同版本權重，應明確失敗，不能跳過。
公開程式有去識別調整，私人舊成品需使用原執行快照稽核，不竄改原 manifest 遷就新版。

## 最小下一輪與停止條件

下載後唯讀稽核通過：508 個成品檔案、18 個模型、11,646 列預測、216 個慢速逐筆參考檢查，
並檢查 16,656 列跨政策/成本情境的交易帳務。後者包含重複情境，不是 16,656 筆獨立交易。

先在允許的 Train / Calibration / Selection 區間分開消融 compact、flow、setup，
檢查分數排序、校準與成本後條件期望。若要跨期研究，先固定配置，再評估其餘時間窗；
已用過的歷史一律稱研究測試，新封存需另外前向收集。
沒有跨期穩定改善或壓力成本正期望，就停止增加模型大小、訓練步數或 SAC 自由度。
此輪不改部署模型、不重新訓練 SAC、不升級 Champion、不啟動實盤。
