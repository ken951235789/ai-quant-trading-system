# 多策略與 SAC 同環境比較

## 基本資訊

- Report ID：`MODEL_RESEARCH-20260930-2AAFD5FF`
- Report Type：`model_research`
- 日期：2026-09-30；時間統一為 UTC。
- Project / Revision：`AIQuantTradingSystem / 76163d3 + working changes`。工作目錄含既有未提交修改，各實驗使用的程式與資料雜湊另存計畫檔。
- Overall Status：`FAIL`（獲利／部署證據不合格，不是回測工作未完成）。工程測試與帳務核對通過；本研究沒有支持實盤或 Champion 升級的獲利證據。
- 範圍：凍結既有模型與特徵，比較不同進出場決策，不訓練、不部署、不下單、不更新 EXE。

## 執行摘要

共完成 **96 組**歷史重播：主實驗 11 種策略／對照 × 2 個區間 × 4 種成本，共 88 組；再補充最終 SAC checkpoint 的 8 組診斷。所有結果都揭露，沒有依測試集挑選新的最佳策略。

重要差別是：整次 SAC 訓練跑了 100,000 步，原本選用的 `best_model` 卻是 **25,000 步 checkpoint**。它在本次驗證與測試區間全部空手。最終 100,000 步模型有交易，但 Test 基本成本後報酬為 **-0.7296%**，不是一樣空手。

固定規則也沒有在本次 Test 基本成本下得到正報酬。這只表示目前這組規則、共同風控及短樣本尚未展現優勢，不能推論所有規則策略或所有強化學習都不可行。

## 模型與樣本

| 項目 | 本次使用內容 |
|---|---|
| 商品 | Binance USD-M BTCUSDT 永續，15m |
| SAC | 2026-09-20 保存的模型；seed 11；112 個特徵、126 維觀察 |
| 神經網路 | `[256, 256, 128]`；learning rate 0.0001；gamma 0.995；batch 512 |
| Transformer | 保存的 V3、未來 5 根毛收益及其他 AI 特徵；不是最新策略事件模型 |
| Transformer 安全起點 | fit／calibration 邊界 2026-07-09 07:15，RL 資料嚴格在此之後 |
| RL Train | 2026-07-09 07:30 至 2026-08-22 02:30，4,205 根 |
| Validation | 2026-08-22 02:45 至 2026-09-02 01:15，1,051 根、1,050 個成交步驟 |
| Test | 2026-09-02 01:30 至 2026-09-13 00:15，1,052 根、1,051 個成交步驟 |
| 初始資金 | 研究假設 1,000 USDT，不對應真實帳戶，也沒有進行匯率換算 |

表中是 K 棒開盤時間；圖表權益標示該棒收盤時間。Test 只有約 11 天，而且是過去已使用過的研究資料，**不是全新、未看過的 final holdout**。

原始目錄記錄於 `artifacts/reports/20260930/strategy_vs_sac/plan.json` 的 `training_dir`，包含模型、三份 split CSV、環境和標準化資訊的 SHA-256。標準化仍使用保存的 train-only 設定，沒有為了提高績效重新 fit。

較早的 500,000 步候選模型壓縮包缺少各 fold 的 `environment.json` 與完整 train／validation／test CSV，無法核對同一輸入和標準化，因此未拿不同期間的舊績效硬比。2026-09-23 的 SAC v34 工作留下缺少 Transformer bundle 的失敗日誌，沒有可供本次評估的新 SAC 權重。

## 固定交易規則

全部在當根收盤決策、下一根開盤由同一 `PortfolioTradingEnv` 成交；不使用未完成高週期 K 棒。同方向已有持倉時，規則對照續抱而非每棒重新滿倉。規則不是獨立為各策略調過最佳參數的正式交易方案。

| 規則 | 進場與退出 |
|---|---|
| Cash | 全程空手 |
| 受風控多單 | 空手建立多單，同向續抱；仍受共同停損、停利與限制，不是純買入持有 |
| EMA 趨勢 | EMA20 > EMA50 且價格 > EMA200 做多；反向做空；不符則平倉 |
| Donchian | 突破前 20 根高／低進場；穿越前 10 根反向通道平倉；通道先 shift 1 |
| RSI | RSI14 < 30 做多、> 70 做空；回到 50 平倉 |
| 布林回歸 | 20 根、母體標準差 2 倍通道外反向進場；回到中線平倉 |
| MACD | 12/26/9 柱體方向與價格相對 EMA200 同向才進場；失效平倉 |
| VWAP 回歸 | ADX14 < 20 且偏離 UTC 當日累積 VWAP 超過 1 ATR 反向進場；回歸 VWAP 或 ADX >= 25 平倉 |
| 1h 趨勢 | 完整收盤的 1h EMA20/50 決定方向 |
| Transformer 毛收益門檻 | 保存的 5 根預期毛收益絕對值 > 0.17% 才順向；否則平倉。固定門檻為往返摩擦 0.15% + 邊際 0.02% |
| SAC | 使用保存的 112 欄標準化特徵及持倉／風控上下文，確定性推論，未修改動作門檻 |

Transformer 門檻對照只用了 `expected_return`，**不等於完整 Transformer 多輸出交易器**，也不使用前一份事件研究的 55% 勝率門檻。指標共用專案 RSI、ATR、ADX 實作，EMA 使用 `adjust=False`，避免不同套件計算定義漂移。

## 共同風控與成本

- 每筆風險上限 0.5%；槓桿設定 2x、最大 3x；保證金比例上限 20%，實際名目曝險仍受部位限制，並非每筆帳戶 2x 滿倉。
- ATR 停損倍數 1.5，距離限制 0.3% 至 0.75%；停利距離 1.5%；單日虧損上限 1.5%；連虧上限 3 筆。
- 最低持倉 4 根、再平衡 deadband 0.02；所有策略走同一組風控。
- 沿用既有完整區間評估模式：原始 10% 硬回撤上限轉為 10% 後近零曝險（乘數 0.0001），評估硬終止設為 100%；不是嚴格在 10% 終止的部署模式。本次 96 組皆未因風控終止區間。
- 期末以市值計價，不強制平倉；尚未平倉部位的未來出場成本尚未扣除。最終 SAC 的 Test 基本成本情境期末仍有少量空單。

| 成本情境 | 單邊手續費 | 單邊滑價 | 完整買賣價差 | Funding |
|---|---:|---:|---:|---|
| Native | 0.04% | 0.02% | 保存設定為 0 | 保存資料 |
| Base | 0.05% | 0.02% | 0.01%，每邊使用半差 | 保存資料 |
| Stress | 0.05% | 0.05% | 0.01%，每邊使用半差 | 保存資料 |
| Zero | 0 | 0 | 0 | 0 |

費率是研究假設，沒有查詢帳戶等級。Funding 沿用現有引擎按持有時間與八小時區間比例攤提，不是逐次真實結算撮合；正數為支付，負數為收取。稅務尚未建模，**以下全部是交易成本後、稅前績效，不能稱為稅後獲利**。

各成本情境會完整重播，費用會改變部位、保護單觸發和風控限制，因此可能出現某些策略 Stress 反而比 Base 虧得少。它不是固定同一串交易後單純相減費用，也不是越貴越好。

## 結果

以下全部是整段**帳戶報酬**，不是每筆交易報酬，也不是槓桿名目本金報酬。已平倉筆數不等於調倉成交次數。

| 策略／模型 | Validation Base | Test Zero | Test Base | Test Stress | Test Base 最大回撤 | 已平倉筆數 |
|---|---:|---:|---:|---:|---:|---:|
| SAC selected 25k | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0 |
| Cash | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0.0000% | 0 |
| SAC final 100k | -3.4645% | +1.1265% | -0.7296% | -2.0843% | 1.6079% | 50 |
| Transformer 毛收益門檻 | -0.8459% | +0.5029% | -0.1073% | -0.2628% | 0.8551% | 11 |
| RSI 回歸 | +0.7191% | +0.0478% | -1.7012% | -1.5766% | 2.3697% | 22 |
| VWAP 回歸 | -2.7646% | +0.2330% | -1.8906% | -2.9036% | 2.1149% | 39 |
| 布林回歸 | -3.0974% | +0.8098% | -2.4626% | -3.3355% | 2.7738% | 53 |
| 1h 趨勢 | -5.4664% | -1.6584% | -3.2089% | -3.9310% | 3.6114% | 28 |
| 受風控多單 | -4.1034% | -0.8555% | -3.2633% | -2.2976% | 4.8026% | 33 |
| MACD 趨勢 | -3.7719% | -1.4733% | -3.5981% | -3.7810% | 3.9890% | 38 |
| EMA 趨勢 | -4.9733% | -2.3668% | -3.6793% | -4.8679% | 3.6846% | 32 |
| Donchian 突破 | -3.1700% | -2.2598% | -3.9625% | -3.8378% | 3.9678% | 32 |

Test Base 的補充分散與交易統計：

| 模型／對照 | 勝率 | Profit Factor | 每筆已平倉平均損益 | 日收益標準差 | 最差單日 |
|---|---:|---:|---:|---:|---:|
| SAC selected | 不可估計 | 不可估計 | 不可估計 | 0% | 0% |
| SAC final | 26.00% | 0.7175 | -0.14085 USDT | 0.3637% | -0.3872% |
| Transformer 毛收益門檻 | 36.36% | 0.8721 | 約 -0.09755 USDT | 0.2763% | -0.7133% |

完整逐日報酬、其他策略統計與逐棒紀錄存於研究目錄。年化 Sharpe、Sortino 雖由原引擎輸出，約 11 天樣本不適合用來推論全年表現。沒有針對大量試驗校正後的顯著獲利結論。

本機研究會生成測試期權益與成本敏感度圖；公開版僅保留上表彙總，不包含私人產物目錄。

## 為什麼 SAC 與固定策略不同

1. **選中的早期模型不是被成本或風控拒單，而是動作本身一直等待。** Test 原始動作介於 -0.02101 至 +0.00840，全落在絕對值 <= 0.03 的 HOLD／WAIT 區間，1,051 步全部 `WAIT_FLAT`，風控原因為空。Validation 同樣如此。零交易沒有可估計的勝率或期望值。
2. **最終模型會交易，但缺少穩健的成本後優勢。** Test 基本成本 50 筆已平倉、勝率 26%、PF 0.7175，另有期末未平倉部位；總成交名目本金約為起始資金的 31.95 倍。手續費 15.9768 USDT、滑價與半價差成本 7.9884 USDT、Funding 0.0826 USDT。零成本重播為 +1.1265%，基本成本變成 -0.7296%，顯示成本敏感性，但不能把兩條不同交易路徑的差額全當成直接費用。Validation 零成本仍為 -0.8064%，所以不是只有交易成本問題，也不能靠降低費用就宣稱解決。
3. **風控與動作需要分開觀察。** 最終模型 Test Base 的方向意圖比例 89.34%，但 83.92% 步驟受到風控調整；777 步記錄連虧限制，另有部位上限、最低持有期等原因，各原因可重疊。多單曝險時間約 19.89%、空單約 1.14%。這不是「SAC 完全不下指令」，也不能因此直接移除風控。
4. **固定規則是受限的決策結構，SAC 還會決定連續目標部位。** 同向規則通常續抱，SAC 可能持續改變目標；所以差異包含調倉頻率與 sizing，而不只有方向準確度。下一輪應固定方向與 sizing 分別做消融，才可辨識原因。
5. **空手比虧損高分，並不矛盾。** 早期空手 checkpoint 優於最終虧損 checkpoint，與驗證選模偏向保守結果一致；但單靠這次比較，不能宣稱已證明 reward 中哪一項導致收斂。正式原因需要 reward／action／cost 的受控消融，不是直接降低部署門檻來強迫交易。

Transformer 對照的 Native Test 為 +0.0166%，改用 Base 便轉負，而且只有 11 筆交易。RSI 則在 Validation 正、Test 負。這兩個例子都不足以升級任何新 Champion，也不支持直接混合策略就會賺錢。

## 修正與驗證

| ID | Severity | Status | 發現、影響與處理 |
|---|---|---|---|
| COST-001 | HIGH | RESOLVED | 永續強制停損／停利退出未套用不利滑價和半價差，會高估績效；已統一退出摩擦並記錄成本，清算分支亦測試 |
| COST-002 | HIGH | RESOLVED | 開倉後才記錄交易起始權益，已平倉損益漏掉開倉費；改為開倉費前權益，平倉損益可對回帳戶權益 |
| MODEL-001 | HIGH | OPEN | selected 25k 全空手，final 100k 成本後負期望；保留研究狀態，不提升部署資格 |
| DATA-001 | MEDIUM | ACCEPTED | 僅單一 seed、約 11 天 Test，且舊 500k 缺完整環境；需較長且可重現的多折、多 seed 檢驗 |

程式證據：`src/ai_quant_trading/reinforcement_learning/environment.py` 的 `_perpetual_rebalance`、`_force_perpetual_exit` 與其呼叫處；回歸測試為 `tests/backtesting/test_strategy_comparison.py`。

| Check | 結果 | 證據 |
|---|---|---|
| 新增比較與成本測試 | PASS | 13 passed；含多空、一般退出、停損、停利、跳空清算、因果性、動作語意 |
| 完整測試 | PASS | `python -m pytest -q --junitxml=artifacts/reports/20260930/strategy_vs_sac/pytest_results.xml`：584 passed、4 subtests passed，138.65 秒 |
| 96 組帳務與資料核對 | PASS | `scripts/audit_strategy_comparison.py`；同時間軸、同風控、來源雜湊、成本加總與平倉損益核對 |
| 靜態檢查 | PASS | Ruff 檢查本次比較、稽核、圖表、補充腳本、環境與測試檔案 |
| 獲利與實盤資格 | FAIL | 沒有通過長期成本後優勢證據；不改品質閘門與 Champion |

Native 也使用修正後的引擎，因此不是原始舊版本績效的逐位元重現。模型權重沒有變更；後續訓練應把這次成本修正視為環境版本變更，重新留下研究紀錄。

## 產物與重現

研究目錄：`artifacts/reports/20260930/strategy_vs_sac/`。

- `plan.json`：主比較在執行前固定的來源、規則、成本與雜湊。
- `comparison.json`、88 份 CSV：主實驗全部結果。
- `final_checkpoint_plan.json`、`final_checkpoint_results.json`、8 份 CSV：檢查到 checkpoint 差異後追加的診斷。不是重新選模或新的 untouched holdout。
- `audit.json`：96 組帳務核對結果。
- `pytest_results.xml`：最後一輪完整測試的 JUnit 紀錄。
- `test_comparison.png`：權益與成本敏感度。
- `report.json`、`report.md`、`artifact_manifest.json`：標準報告與完整性清單。

在專案根目錄執行，使用已安裝本專案依賴的 Python：

```powershell
$env:PYTHONPATH = 'src'
$study = 'artifacts/reports/20260930/strategy_vs_sac'
python scripts/audit_strategy_comparison.py $study
python scripts/plot_strategy_comparison.py $study
```

重新跑回測時必須使用新輸出目錄，不覆寫原實驗：

```powershell
$plan = Get-Content "$study/plan.json" -Raw | ConvertFrom-Json
python scripts/run_strategy_comparison.py --training-dir $plan.training_dir --output artifacts/reports/new_strategy_comparison
python scripts/compare_sac_final_checkpoint.py artifacts/reports/new_strategy_comparison
python scripts/audit_strategy_comparison.py artifacts/reports/new_strategy_comparison
```

圖表或報告更新後完整性清單需透過專案 `build_artifact_manifest` 重新建立；不能忽略來源或模型雜湊不符的錯誤。本次結果不是新的 UI 頁面，亦未複製進已打包的 EXE。

## 下一步與剩餘風險

1. 先建立完整保存環境與 cross-fit Transformer 輸出的較長多區間資料，再跑相同規則與 SAC，分牛、熊、震盪及成本壓力檢驗。不得把這 11 天的勝出者當成未來勝出者。
2. 用 Train／Validation 做固定方向搭配不同 sizing、固定 sizing 搭配不同方向的消融，另檢查再平衡次數與費用。先辨識 SAC 增加的是方向價值、資金配置價值，還是只有成本。
3. 下一次訓練才比較 reward、調倉懲罰與 action 語意；部署端不可偷偷縮小 WAIT 門檻來掩蓋空手，也不以發交易獎金迫使模型下注。
4. 所有規則、門檻與模型選擇在開發區固定後，使用未碰過的未來 holdout；多 seed 與信賴區間不能用重複同一條行情假裝獨立樣本。
5. 仍需更精確的 funding 結算、逐筆或較細粒度滑價、期末清倉成本、稅務與實際帳戶費率驗證。本次不涉及 Testnet、WebSocket、Email、EXE 或正式下單壓力測試，也不宣稱已排除整個專案所有漏洞。
