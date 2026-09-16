# 模擬交易資料

每個子資料夾是一個可持續更新的模擬帳戶，包含：

- `account.json`：現金、持倉、待執行訊號與風控狀態
- `orders.csv`：每筆模擬成交
- `trades.csv`：已完成的來回交易
- `predictions.csv`：每次 AI 上漲機率與訊號
- `performance.csv`：資產、報酬與回撤時間序列
- `positions.csv`：每次執行後的持倉快照

這些檔案是研究資料，不會送出真實委託。
