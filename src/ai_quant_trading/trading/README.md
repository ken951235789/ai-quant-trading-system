# 交易決策治理核心

這個模組不是第三個預測模型，而是把每個角色的責任固定下來：

- `contracts.py`：Transformer、SAC、市場狀態與資金管理的資料合約。
- `data_guard.py`：檢查多週期 K 線缺漏、過期、未收盤、異常跳價與 Spread。
- `model_guard.py`：使用訓練期 mean/std 偵測即時模型輸入漂移。
- `regime.py`：把 Transformer 輸出分類成多頭、空頭、震盪或高波動環境。
- `capital.py`：依信心、波動、回撤與市場狀態縮放 SAC 目標部位。
- `orchestrator.py`：依固定順序執行市場狀態分析與資金配置。

固定停損、最大回撤與投資組合限制仍由 `risk/` 負責；實際成交與冪等送單仍由
`paper_trading/`、`live_trading/` 負責，避免在這裡複製第二套交易引擎。
