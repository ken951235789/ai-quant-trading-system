# Transformer 與 SAC 精簡特徵升級

## 這次改了什麼

這次只修改程式與做本機功能測試，沒有啟動 Kaggle、正式模型訓練或實盤交易。
既有模型、歷史資料、研究結果與 final holdout 不會被覆寫。

| 項目 | 新版行為 |
|---|---|
| 預期報酬 | 新短線環境使用 `transformer_return_5`；根數、來源、單位與方向保存於模型環境中繼資料 |
| Transformer 挑選器 | 同時輪流分配「時間週期 × 指標類型」，不再讓快速週期先吃掉全部容量 |
| 原始價格 | OHLCV、原始均線、MACD、ATR、VWAP、OBV、布林價格線不進入新版 Transformer 的輸入 |
| SAC 輸入 | 預設預算 112 欄；本機五週期資料實際為 107 欄，取消基礎 15m 與 `u_mtf_15m_*` 的重複表示 |
| 訊號分歧 | 保留正負方向，新增九個市場情境欄位與 Transformer 訊號熵，不把不同訊號硬改成同方向 |

原始價格是從「模型輸入」移除，不是刪掉資料檔。圖表、訓練答案、下一根開盤撮合、
停損停利及交易成本仍需要 OHLCV。新版保留報酬率、價格相對 EMA/VWAP 的距離、
MACD/價格、ATR/價格、K 線實體比例、相對成交量、波動與市場結構等資訊。

## 預期報酬契約

`environment.json` 的 `ai_context.expected_return_contract` 保存：

```json
{
  "schema_version": 1,
  "source_column": "transformer_return_5",
  "horizon_bars": 5,
  "units": "fractional_gross_return",
  "direction": "positive_long_negative_short"
}
```

- `0.01` 是預測毛報酬 1%，不是 1 個百分點單位，也不是扣除成本後的淨報酬。
- 正數偏多、負數偏空；硬風控依下單方向與成本再判斷，不能取絕對值冒充多空都可獲利。
- 15m 決策週期的 5 根代表 75 分鐘窗口；不代表每筆一定持有 75 分鐘或一定到達停利。
- 1、5、20 根角色訊號仍可同時提供給 SAC；只有 `expected_return` 的成本分析來源固定由契約指定。
- 訓練資料若誤填另一個 `expected_return`，資料準備器會依契約重建正確來源。
- 最新指定來源缺失、不是有限數字或無法驗證時，拒絕產生模型交易判斷，不偷偷切換到 20 根。
- 新建且綁定 Transformer 的環境必須有明確契約。舊模型沒有契約時保留舊執行流程的 20 根語意，
  避免改變既有策略；這是相容行為，不代表舊訓練／執行差異已靠修改中繼資料修復。

## 平衡挑選與欄位數

Transformer 的多週期容量依趨勢、動能、波動、成交量、市場結構、因果 SMC、
合約與流動性、市場狀態八種類型輪流分配；保留各週期 `available`、`age_ratio`。
基礎週期與共識情境通常最多占四分之一，其他容量優先留給多週期欄位。
挑選順序固定，不用回測收益或 final holdout 排序，也不因 CSV 欄位順序改變。

本機資料檔：
`data/processed/features_mtf_crypto_binance_futures_BTC-USDT_15m_5m-15m-1h-4h-1d_h1.csv`。
取前 5,000 根檢查後：

| 週期 | Transformer（總預算 256） | SAC |
|---|---:|---:|
| 5m | 38 | 16 |
| 15m | 38 | 16 |
| 1h | 38 | 16 |
| 4h | 37 | 16 |
| 1d | 37 | 16 |
| 基礎／共同情境／AI | 68 | 27 |
| 合計 | 256 | 107 |

SAC 的 107 欄 = 五週期各 14 個技術欄位與 2 個可用性欄位 + 9 個市場情境 + 18 個 Transformer 欄位。
沒有再加入重複的基礎 `u_rsi_14`、`u_return_1`、`u_macd_pct` 或規則持倉捷徑。
Transformer latent 也不再整批塞進精簡版 SAC。
選用 FinBERT 時另外納入四個新聞分數／品質欄位，仍受 112 欄預算控制。

選擇器支援 `1m / 3m / 5m / 15m / 30m / 1h / 4h / 12h / 1d`。
九週期完整 schema 的測試輸入為 112 欄，容量按週期輪流分配；本機現有資料仍是上述五週期，
程式不會捏造不存在的另外四個週期。缺少多週期資料時可退回較小的單週期輸入，
不會用重複欄位硬湊 80 欄。

## 如何保留分歧

| 欄位 | 意義 |
|---|---|
| `market_trend_consensus` | 各有效週期的平均趨勢，-1 偏空、+1 偏多；0 可能是盤整或方向抵銷 |
| `market_fast_trend` | 5m 以下有效週期的平均趨勢 |
| `market_slow_trend` | 4h 以上有效週期的平均趨勢 |
| `market_cross_timeframe_disagreement` | 不同週期趨勢差距，越大代表越分歧 |
| `market_direction_entropy` | 偏多／盤整／偏空票數的標準化熵，越大代表方向分散 |
| `market_reversal_pressure` | RSI 與布林位置形成的反轉壓力摘要，不是已確認反轉訊號 |
| `market_trend_reversal_conflict` | 趨勢方向與反轉壓力是否相反 |
| `market_fast_slow_conflict` | 快、慢週期方向是否相反 |
| `market_timeframe_coverage` | 有多少來源週期有效，避免「沒有資料」被誤認為一致盤整 |

SAC 對應欄位帶 `u_` 前綴。`u_transformer_signal_entropy` 使用 1、5、20 根的
偏多／盤整／偏空機率平均分布計算；它同時反映預測分散與跨窗口分歧，不等於交易勝率。
缺失／過期週期不參與共識投票。市場情境只讀當列已因果對齊的資訊，不讀未來答案，
也不把 Transformer 的預測回灌成它自己的輸入。

## 下次要怎麼使用

1. 使用更新後的專案原始碼。這次沒有重建既有 EXE 或已匯出的可攜 ZIP。
2. 在資料頁建立或更新 BTC 多週期特徵。既有特徵 CSV 也可用，新派生欄位由訓練／推論端建立。
3. 到 Transformer 訓練頁選相同 15m 決策週期的多週期資料，重新訓練 V3；可保持 256 個輸入預算。
4. 使用新 Transformer 產生樣本外推論。不能把它已經看過的訓練／選模期間當成 SAC 的合法樣本外資料。
5. 到強化學習的「環境建構」選新資料與新 Transformer，使用預設精簡特徵，重新建立環境。
6. 到「模型訓練」重新訓練 SAC。舊 332 欄 SAC 不能直接換成新 107 欄，單改 JSON 不會讓權重相容。
7. 用 Walk-forward 與不同 seeds 驗證候選，再依既有一次性 final holdout、Champion 與模擬倉閘門驗收。

可攜訓練來源 `training_portable/workspace.py` 與 Kaggle 聯合／SAC runner 已套用相同契約。
下次上傳前必須重新產生訓練包；已上傳 Kaggle 的舊 Dataset 不會自動更新。
精簡轉換器只建立被選中的 `u_` 欄，降低大型資料集多餘的記憶體占用。

## 已驗證與限制

- 比例化資料在價格等比縮放後保持不變。
- 修改未來列不影響過去市場情境；精簡 CSV 保留完整資料已算出的共識，不以少量欄位重寫。
- 不同 CSV 欄位順序、第一根暖機缺值不改變精簡契約。
- 本機 CPU 測試：Transformer 1 Epoch、SAC 32 steps、模型保存／載入、最新推論與 20 根多空撮合重播。
- 20 根撮合重播使用預先指定的多空目標，另測 SAC 最新 observation；不是自主策略收益實驗。
- 沒有重新做正式收益評估，沒有開啟 final holdout，也沒有證明這些變更能提高勝率或獲利。
- 完整測試與結構化證據見 `artifacts/reports/20260915/compact_ai_inputs.md` 與同名 JSON。
