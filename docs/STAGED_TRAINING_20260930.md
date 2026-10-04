# Transformer 與 SAC 分階段研究

## 目的與狀態

依序固定交易契約、驗證 Transformer、比較同訊號資金配置，最後才考慮額外交易自由度。
這是獨立、不可部署的研究流程；既有 Transformer、SAC、模擬倉、實盤、EXE 都不會被替換。
執行結果另存 `artifacts/reports/20260930/staged_training_run02/result.json` 與同目錄報告。

本批已完成 15 個真實 Transformer 訓練，共耗時約 26.3 分鐘。5 個 seed 均未通過
Transformer 品質閘門，正式 SAC 歷史訓練未啟動，第四階段維持鎖定。
這是研究結果不合格，不是訓練程序未完成；工程測試與獨立稽核通過。
詳細逐筆產物保留於本機，不隨公開版上傳；本頁保留本批成果摘要。

## 一、唯一交易契約

契約名稱：`btc15m_event_entry_sizing_v1`，內容由 `build_contract` 建立並計算 SHA-256。

| 項目 | 本研究固定值 |
|---|---|
| 市場 | Binance USD-M BTCUSDT 永續，15m，UTC |
| 模型 | 既有 Transformer V3，事件研究設定；96 棒序列、64 寬度、2 層 |
| 特徵 | 既有 context_v2 因果特徵，15m／已收盤 1h／4h；不是本次新增九週期模型 |
| 候選規則 | 已收盤 1h 趨勢確認後的 15m 突破，多空對稱 |
| 標籤與出場 | 下一根開盤進場；2 ATR 停損、4 ATR 停利、制度失效退出或最長持有 32 棒後下一開盤退出；同棒停損優先 |
| 模型輸出 | 固定事件淨收益、q10／q50／q90，以及淨收益超過 2 bps 的機率 |
| 接受門檻 | 機率至少 55%，預期淨收益大於 2 bps；不依測試結果調整 |
| 成本 | 每側費用 5 bps、滑價 2 bps、完整價差 1 bp；每八小時 funding 準備金 1 bp |
| SAC 動作 | 0 到 1，映射為 0% 到 20% 入場名目本金比例，無槓桿 |
| 不允許變更 | 方向、成交排程、停損停利、持有時間、槓桿與自由退出 |

選擇既有事件研究模式，是為了真正固定三組的進出場；不能拿一般模型固定 horizon
的報酬預測去宣稱任意出場規則的交易勝率。原始策略仍可能負期望，這是待檢驗假說。
32 棒是最長 8 小時的事件持有窗口，不代表每筆都拿滿 8 小時。
一般 V3 的 5／20／48 棒模型、舊 SAC 的 5 棒模型保留原契約；本研究不更改它們。

## 二、Transformer 訓練與樣本外預測

資料來源為已凍結的五年 BTC 15m CSV；不讀取交易憑證，不上傳到雲端。
使用 seeds 42、137、2026、2027、2028；每個 seed 訓練三個按時間前進的模型，共 15 個。
每個模型最多 20 epochs，早停耐心 5，CPU 2 執行緒。沒有為了拉高分數而更換測試門檻。

- 前 40% 至 90% 的行情分成三個相鄰 OOS 區塊；每個 checkpoint 只預測自己的後方區塊。
- 每次允許的過去區間再切 75% Train、25% Validation；Validation 分校準與選模兩段。
- 沿用訓練器的 label purge 與 32 棒 embargo，避免標籤跨界。
- 每列預測記錄 checkpoint 雜湊、契約雜湊、fit 截止位置與 OOS fold。
- 最後 10% 本研究不訓練、不計績效。但這批歷史曾用於其他研究，不冒充全新的 final holdout。

串接的 OOS 再按 K 棒邊界分為前 60% Train、中 20% Validation、後 20% Test，
而非按模型選中的交易數改切點；去掉退出跨界的交易並保留 embargo。
這裡 OOS Train 是 SAC 可以學習的早期資料，不是把 Transformer 的 in-sample 預測交給 SAC。

### 前置品質閘門

只使用早期 OOS Train 建立常數基準與高分組門檻，在 OOS Validation 檢查：

1. 至少 50 筆通過固定條件、且不重疊的交易。
2. 成本後期望值的區塊 bootstrap 95% 信賴區間下界大於零。
3. 報酬與機率品質都優於較早 OOS Train 的常數基準。
4. 高分候選的實際淨收益高於低分候選，兩組至少各 10 筆；分組門檻只由 OOS Train 決定。
5. 固定交易路徑再加 6 bps 摩擦壓力後，平均淨收益仍為正。
6. 五個預登記 seed 都通過，不把同一行情的五份預測算成五倍樣本。

50 筆是最低研究檢查，不是充分的實盤樣本。信賴區間也依賴區塊抽樣假設。
若失敗，正式 SAC 訓練不啟動，第三步標示 `blocked_by_transformer_gate`，第四步持續鎖定。
這不是程式故障，而是防止在未驗證訊號上繼續花訓練時間。

## 三、同訊號資金配置

若前置閘門通過，只依 OOS Validation 選 Transformer seed，再凍結同一份已接受事件排程。
三組都使用一樣的多空方向、事件時間、成交價格與成本：

| 組別 | 配置方式 |
|---|---|
| Fixed | 每次入場配置帳戶權益 10% |
| Volatility | 10% × 0.3% / 訊號當下 ATR 比例，上限 20% |
| SAC | 在 0% 至 20% 間決定初始配置，可不參與；五個 seeds，各 100,000 步 |

即使 SAC 不參與，仍保留該事件原來占用的時間，不讓它因此挑到基準組看不到的其他交易。
這是刻意受限的比較，不能直接等同真實允許空手後尋找新機會的交易策略。

SAC observation 只有當下 Transformer 預測、因果 ATR 比例、事件方向、已結算權益與回撤。
實際報酬、未來出場時間和成交價格不進入 observation。Scaler 僅 fit OOS Train。
Reward 是成本後結算權益的 log return，不再扣第二次費用，也不獎勵下單次數。
結算回撤達 10% 後持續禁止新風險，直到新 episode；減少資金不會改變事件排程。

### 重要限制

`EventSizingEnv` 是「每次固定事件結算一次」的研究環境，沿用 `strategy_events.py`
的收益與成本，並沒有新增第二套實盤撮合。權益與最大回撤是結算層級，不代表棒內最大回撤。
未模擬資金容量、依下單量改變的滑價、帳戶清算、即時每日停機及稅務。
Funding 是保守準備金，不是歷史實際費率。所有成品 `live_eligible=false`。

## 四、自由度閘門

只有五個 SAC seeds 在 Validation 與保留的 Test 都有正報酬、相對兩個基準的
配對 log return 差異信賴下界為正，且結算回撤不高於兩個基準，才標示
`eligible_for_separate_research_design`。這仍只是可開始另一個研究設計，不會自動開啟加減碼或退出。
增加自由度會改變交易契約，必須另設未看過的評估期間重新驗證，不能修改舊模型 metadata 冒充相容。

## 執行方式

在 `D:\AIQuantTradingSystem`，使用已安裝本專案依賴的 Python：

```powershell
$env:PYTHONPATH = 'src'
python -X utf8 scripts/run_staged_training.py `
  --source outputs/kaggle_strategy_event_20260924/staging/data/btc_15m.csv `
  --config configs/btc15m_staged_research.json `
  --output artifacts/reports/next_staged_training
```

以上是另開新研究的指令，不是查看結果的指令；本批已完成，不必再執行。
輸出目錄必須尚未存在，避免覆寫既有模型。程序會保存 `progress.json`、各模型 checkpoint、
`records.json`、OOS 預測、`transformer_gate.json`、`result.json` 與完整性清單。
目前不支援對此批次原地續跑；中斷的模型仍保留，不會冒充整批完成。

```powershell
python scripts/audit_staged_training.py artifacts/reports/20260930/staged_training_run02
python -m pytest tests/research/test_staged_training.py -q
```

獨立稽核會逐筆核對 OOS 預測與來源模型的預測檔、時間邊界與因果 ATR，並重算品質判斷。
稽核後完整性清單也涵蓋 CSV、XML 與 Markdown。雜湊用於偵測檔案變更，不是數位簽章，
也不能證明行情來源永遠正確。核對完成後不要直接編輯成品；新研究另建輸出目錄。

測試中的 40 步 SAC 使用合成資料，只證明程式可訓練與保存，不能作為歷史獲利證據。

第一次 `staged_training` 執行在保存研究索引時遇到相對與絕對路徑混用，保留失敗進度與
已訓練的第一個模型。修正為入口統一絕對路徑，新增 15 個假模型的完整編排測試後，
改用 `staged_training_run02` 從頭執行。假模型測試只驗證流程，不算入正式 15 個模型。
