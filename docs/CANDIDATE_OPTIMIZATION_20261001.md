# 候選進出場與成本後期望值改善研究

## 目的

上次原規則和 Transformer 篩選均未證明正期望。本次先研究規則，不調低 Transformer
接受門檻，不追加 SAC 訓練、不替換機器人模型，也不宣稱模型越大就會獲利。
結果與標準化報告位於 `artifacts/reports/20261001/candidate_optimization/`。

本批已完成：60 組回放、75 個交易 CSV 的完整重播稽核、607 個測試案例及 4 個子測試通過。
五組規則均未通過開發品質閘門，沒有選中策略，也沒有訓練或部署新模型。
完整產物保留於本機，不隨公開版上傳；本頁保留研究方法與彙總結果。

## 固定五組規則

所有策略使用同一份 Binance USD-M BTCUSDT 15m 行情；訊號收盤形成、下一根 15m 開盤成交。
保護單仍逐根 15m 檢查，同棒碰停損停利採停損優先，退出後冷卻 4 根。
趨勢條件沿用已收盤 1h EMA50/EMA200 方向與 ADX >=25，連續兩個小時棒確認。

| 規則 | 進場訊號 | ATR 與最長持有 | 趨勢退出 |
|---|---|---|---|
| C0 原規則 | 順 1h 趨勢，15m 收盤突破前 20 根高低 | 15m ATR；32 根 15m，即 8 小時 | 失去趨勢即退出 |
| C1 小時突破 | 順 1h 趨勢，1h 收盤突破此前 20 根完整小時棒高低 | 1h ATR；64 根 15m，即 16 小時 | 失去趨勢即退出 |
| C2 小時突破 | 同 C1 | 同 C1 | 確認相反趨勢才退出 |
| C3 小時回調 | 順 1h 趨勢，前一小時收盤位於 EMA20 反側，本小時重越且實體同向 | 同 C1 | 失去趨勢即退出 |
| C4 小時回調 | 同 C3 | 同 C1 | 確認相反趨勢才退出 |

五組均為 2 ATR 停損、4 ATR 停利；反向退出仍保留停損、停利與時間上限。
1h 訊號只在整點形成，不將尚未收盤的小時高低價帶入。多空條件對稱，沒有只挑歷史較好的一邊。
新的四組構成「進場 × 趨勢退出」2x2；但相對原規則同時更換訊號頻率、ATR 尺度和持有上限，
不能把全部差異只歸因於其中一項。更長持有不是每筆一定持有 16 小時。

## 成本與量尺

- Zero：全部摩擦為零，用於診斷，不可作為實盤績效。
- Base：每側手續費 5 bps、滑價 2 bps、完整價差 1 bp；每跨 UTC 八小時結算扣 1 bp funding 準備金。
- Stress：每側滑價增為 5 bps，其餘同 Base。
- 每個成本情境重新回放，不只是事後扣固定比例；停損價格與事件排程可能因此改變。
- `net_return` 是每筆名目本金淨收益；`net_r` 是淨收益除以初始計畫停損距離比例。
- `net_r` 不保證真實虧損最多 1R。滑價、跳空及成本可使虧損超過 1R。
- 同時報告每 30 天交易頻率、每筆成本相對停損距離、平均、標準差、最差交易與信賴區間。

不使用「把 taker 改成 maker 費率」假裝省成本，因為尚無限價單成交率及逆向選擇模型。
費率仍是假設，funding 不是實際歷史資料；稅後績效未驗證。

## 驗證順序

1. 執行前在 `plan.json` 固定五組規則、三種成本、程式／資料雜湊及選擇條件。
2. 前 60% 歷史為 Development；只在此處選候選。
3. 候選需基本成本至少 100 筆、每筆淨均值的 95% 區塊 bootstrap 下界 >0、壓力成本均值 >0。
4. Development 再切三段，至少兩段各有 20 筆且均值 >0，防止只由單一行情區間拉高。
5. 後續 60%～70%、70%～80%、80%～90% 為三段固定規則向前回放，起點隔離 64 根，尾端留白 65 根。
6. 預先選中的候選須向前合併至少 100 筆、信賴下界 >0、壓力均值 >0，且每段至少 20 筆與正均值。
7. 即使通過，也只是可建立新標籤契約及模型研究的候選，不會直接升級 Champion。

最後 10% 本次不使用。但整份歷史已多次用於研究，因此本次仍是回溯研究，
不是全新的 final holdout，也未修正所有歷史多重比較。所有規則公開結果，不依 Forward 改選策略。
區塊 bootstrap 使用 2,000 次、固定 seed 20260923，區塊長度為 max(2, floor(sqrt(n)))。
研究不產生完整帳戶收益，故不以每筆收益加總冒充 CAGR、Sharpe 或帳戶回撤。

## 執行方式

在 `D:\AIQuantTradingSystem` 使用已安裝專案依賴的 Python：

```powershell
$env:PYTHONPATH='src'
python -X utf8 scripts/run_candidate_optimization.py `
  --source outputs/kaggle_strategy_event_20260924/staging/data/btc_15m.csv `
  --output artifacts/reports/candidate_optimization_new_run

python -X utf8 scripts/audit_candidate_optimization.py artifacts/reports/candidate_optimization_new_run
python -m pytest tests/backtesting/test_candidate_optimization.py -q
```

輸出目錄必須不存在。程式會保存 `progress.json`、`plan.json`、`selection.json`、
60 個情境交易 CSV、15 個向前合併 CSV、`research.json` 與完整性清單。
稽核完整重播 60 個情境，比對每一筆交易、重新核算費用與 funding，並重現選擇與向前閘門。
執行不需要 API Key、不消耗 Kaggle 額度、不啟動實盤。

## 研究依據與限制

[Liu 與 Tsyvinski 的加密貨幣研究](https://www.nber.org/papers/w24877)觀察到時間序列動能；
[AQR 的時間序列動能研究](https://www.aqr.com/Insights/Research/Journal-Article/Time-Series-Momentum)
研究的是多種期貨與較長期間。這些文獻只能支持測試趨勢類假說，**不能證明這套 BTC 小時規則會獲利**。
本次規則是自行登記的工程研究假說，不是對兩篇論文的策略複製。
