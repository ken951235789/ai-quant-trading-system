# 交易決策治理核心

這個模組不是第三個預測模型，而是把每個角色的責任固定下來：

- `contracts.py`：Transformer、SAC、市場狀態與資金管理的資料合約。
- `data_guard.py`：檢查多週期 K 線缺漏、過期、未收盤、異常跳價與 Spread。
- `model_guard.py`：使用訓練期 mean/std 偵測即時模型輸入漂移。
- `regime.py`：把 Transformer 輸出分類成多頭、空頭、震盪或高波動環境。
- `capital.py`：依信心、波動、回撤與市場狀態縮放 SAC 目標部位。
- `orchestrator.py`：依固定順序執行市場狀態分析與資金配置。
- `journal.py`：模擬成交的 SQLite 圖文日誌、決策快照與離線研究匯出；不參與下單裁決。

固定停損、最大回撤與投資組合限制仍由 `risk/` 負責；實際成交與冪等送單仍由
`paper_trading/`、`live_trading/` 負責，避免在這裡複製第二套交易引擎。

日誌的適用範圍、檔案位置、成本及 R 的定義、圖片使用限制，見
`docs/JOURNAL_RESEARCH_20261008.md`。
RL 模擬交易透過耐久逐棒提交寫入；實盤另外從已保存的交易所確認成交事件投影，
不把送單紀錄當成交，也不補出未知的完整往返交易 R。

`journal_archive.py` 負責圖文校驗與封存；只在確認 ZIP 與原檔一致後移除衍生散檔，
保留帳務、模型及原始資料，UI 可直接讀取封存中的圖片。
