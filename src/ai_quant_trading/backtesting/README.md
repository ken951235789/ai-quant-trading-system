# AI 歷史回測

此模組只提供 SAC／PPO 與 Transformer 模型回測，不開放傳統規則作為可調交易策略。
固定 EMA 與隨機策略只作同資料基準，用來檢查模型是否真的優於簡單方法及環境是否異常。

## SAC／PPO 回測

- 載入訓練完成的 BTC SAC 或 PPO ZIP。
- 使用建立環境時保存的 train、validation 或 test CSV。
- 沿用模型的特徵順序、標準化資料、持倉限制與風控契約。
- 每根 K 線收盤後決策，下一根開盤成交。
- 計入單邊手續費、滑價與空單持有成本。
- 同時列出模型含成本、同決策無成本、EMA 固定多空、隨機策略及 BTC 買入持有。

測試集才是主要樣本外結果。訓練集只能檢查擬合，驗證集可能已參與模型選擇。

## Transformer 回測

Transformer 本身預測未來報酬、波動率、牛熊機率與不確定性。回測器使用以下條件建立目標部位：

1. 預測報酬超過門檻。
2. 對應方向的牛市或熊市機率超過門檻。
3. 不確定性不超過上限。
4. 部位依預測報酬相對預測波動率縮放。

多空部位會交給與 PPO 相同的 Portfolio Environment 撮合，因此成交時間與成本模型一致。

## 輸出

介面位於 `策略研究 > AI 歷史回測`。每種模型只保存最近一次：

```text
data/processed/model_backtests/
    latest_ppo/
        summary.json
        evaluation.csv
    latest_transformer/
        summary.json
        evaluation.csv
```

再次執行會覆寫同類型結果，避免舊回測持續占用空間。
