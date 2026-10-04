# AI 歷史回測

交易 UI 提供 SAC／PPO 與 Transformer 模型回測，不開放傳統規則作為可調交易策略。
固定 EMA 與隨機策略只作同資料基準，用來檢查模型是否真的優於簡單方法及環境是否異常。

## 固定策略事件回放

`transformer/strategy_events.py` 是獨立選用的研究契約，供 Transformer 事件標籤與過濾效果評估
共用同一份成交規則；不是新的實盤引擎。只接受連續且已收盤的 Binance USD-M BTCUSDT
15m OHLCV，從同源重採樣 1h／4h，不讀取未完成高週期 K 棒。

回放涵蓋下一開盤成交、訊號 ATR 固定保護價格、同棒停損優先、跳空、持有上限、
制度退出、雙邊交易成本、UTC 八小時結算的資金費率準備金。
準備金不是歷史實際 funding；以 15m 推斷成交也不是逐筆撮合。
比較時禁止重疊持倉並保留冷卻時間，輸出每筆名目本金淨報酬而非帳戶權益績效。
操作及限制見 [事件研究說明](../../../docs/transformer_strategy_event_20260924.md)。

### 候選規則與成本診斷

`candidate_research.py` 提供 7 組事前固定的進出場假說，使用零成本、基本成本與壓力
成本完成分段回放；只依開發區間選候選，不根據測試績效調門檻。研究輸出保留逐筆
收益、成本拆解、區間估計、出場原因及凍結模型的拒單診斷，不連接實盤。

使用 `scripts/run_candidate_rule_research.py` 執行、
`scripts/audit_candidate_rule_research.py` 核對。完整設定、結果與限制見
[2026-09-30 候選規則研究](../../../docs/CANDIDATE_RULE_RESEARCH_20260930.md)。

### 低換手候選改善研究

`candidate_optimization.py` 沿用既有 `replay_window` 與 `replay_event`，沒有另建成交引擎。
新增已收盤 1h 突破／回調、1h ATR 保護距離及失效／反向退出的 2x2 固定比較，
並以原策略為對照。提供逐筆成本、淨收益信賴區間、R 倍數、交易頻率及開發／向前閘門。
完整重播稽核另外核對手續費與 funding 準備金；這不是限價撮合或實盤功能。
操作方式見 [低換手研究說明](../../../docs/CANDIDATE_OPTIMIZATION_20261001.md)。

### 大型有限網格研究

`grid_execution.py` 是既有事件成交契約的 NumPy 批次計算版本；以逐筆參考引擎核對，
不取代 Live 或模型回測撮合。`parameter_sweep.py` 窮舉固定的 11,664 組參數，
提供分塊續跑、Search/Validation 選擇隔離、三段 Forward、成本壓力與多重搜尋敏感度診斷。
完整結果含平均、分散、最差交易與簡化結算曲線，不把歷史最高值視為可部署策略。
執行、稽核、繪圖與限制見 [大型參數比較](../../../docs/PARAMETER_SWEEP_20261002.md)。

`complex_strategies.py` 額外提供六個因果複合規則家族，`complex_research.py` 沿用同一
成交與選擇契約，比較 1,152 組新參數及原策略基準。原基準在全部期間/成本必須與前輪一致。
只使用 OHLCV 衍生的 H1/H4 趨勢、波動壓縮、VWAP、成交量及突破回測，
不冒充真實訂單流，也不接入 Live。見 [複合策略定義及操作](../../../docs/COMPLEX_STRATEGIES_20261002.md)。

### 多策略與 SAC 同環境比較

`strategy_comparison.py` 將固定 EMA、Donchian、RSI、布林、MACD、VWAP、1h 趨勢、
Transformer 毛收益門檻與空手／受風控多單對照，接入 SAC 原有撮合環境。
這些是研究基準，不新增 UI 策略選單，也不取代既有部署契約。

使用 `scripts/run_strategy_comparison.py` 執行主比較，
`scripts/compare_sac_final_checkpoint.py` 分開診斷最終 checkpoint，
`scripts/audit_strategy_comparison.py` 核對所有逐棒帳務，
`scripts/plot_strategy_comparison.py` 繪圖。
不能混淆「總訓練步數」與「實際選中的模型步數」，亦不能把空手的零報酬當成合格策略。
完整結果、重現指令與限制見 [SAC 同環境比較報告](../../../docs/STRATEGY_VS_SAC_20260930.md)。

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
