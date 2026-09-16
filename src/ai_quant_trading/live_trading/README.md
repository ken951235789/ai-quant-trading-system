# Live Trading：Binance USD-M 永續合約

本模組把 PPO／SAC 研究環境的 BTC/USDT USD-M 永續合約目標曝險，接到相同市場的
Binance Futures 下單 API。預設只允許 `BTC/USDT`、Testnet、One-way Mode、Single-Asset
Mode、ISOLATED 逐倉與 2 倍槓桿；程式啟動或載入模組不會自動送單。

## 一致的交易契約

```text
研究資料：Binance USD-M BTCUSDT perpetual
模型輸出：帶方向的 target position fraction（正多、負空）
下單市場：Binance USD-M BTCUSDT perpetual
持倉模式：One-way，positionSide=BOTH
保證金：USDT Single-Asset + ISOLATED
保護單：Mark Price 觸發的 STOP_MARKET / TAKE_PROFIT_MARKET
```

不再把空頭目標當成「賣掉現貨」。同方向目標可增減倉；多空反手必須先以
`reduceOnly=true` 歸零，下一根已收盤 K 線仍維持反向目標才開新倉。

## 安全功能

- `/fapi/v3/account`、`/fapi/v3/positionRisk` 帳戶及實際持倉快照
- `/fapi/v1/order/test` 驗證與 `/fapi/v1/order` 市場單
- HMAC 簽章、伺服器時間校正、唯一 client order id
- 未知成交狀態先查單，禁止直接重送
- One-way、Single-Asset 檢查；逐倉與 1 至 3 倍槓桿硬限制
- 可選的 1x～3x 動態槓桿；未校準 Transformer 預設只能 1x，減倉與平倉不被阻擋
- 增加風險與 reduce-only 減倉分流，禁止單筆直接反手
- `/fapi/v1/algoOrder` 建立多空雙向停損停利
- 交易所持倉、本地受管理持倉與保護單重啟對帳
- Futures 與舊 Spot 持倉命名空間、快照及 Testnet 證據隔離
- 單日虧損、最大回撤、緊急停機、雜湊鏈稽核與每日報告
- CSV 欄位自動遷移、跨程序排他寫入鎖與落盤同步

## 環境

| 模式 | REST base URL | 用途 |
|---|---|---|
| Testnet / Demo | `https://demo-fapi.binance.com` | 無真實資金驗證 |
| Live | `https://fapi.binance.com` | 正式 USD-M 永續 |

`.env` 使用既有環境變數：

```dotenv
BINANCE_TESTNET_API_KEY=
BINANCE_TESTNET_API_SECRET=
BINANCE_DEMO_API_KEY=
BINANCE_DEMO_API_SECRET=
BINANCE_LIVE_API_KEY=
BINANCE_LIVE_API_SECRET=
AI_QUANT_LIVE_TRADING_ENABLED=NO
```

API Key 只開 Futures 交易需要的最小權限，關閉提領並設定 IP 白名單。Live 還需要把
總開關改為 `YES`、輸入 `EXECUTE LIVE`，並通過模型品質與 Testnet 營運證據檢查。

## Email 告警

實盤事件可透過加密 SMTP 寄送 Email。預設只寄 `error` 與 `critical`，包含 Watchdog
異常、事件流失聯、對帳或保護單失敗、緊急停機，以及實盤命令執行錯誤。寄送失敗只會
寫入本地通知紀錄，不會中斷交易主流程。

設定與測試方式見 [`docs/實盤前營運安全與驗收.md`](../../../docs/實盤前營運安全與驗收.md)。
SMTP 密碼只放在 Git 忽略的 `.env`；介面不會顯示帳號、完整收件地址或密碼。

## 無人值守與開機恢復

Testnet 可在介面明確開啟「Windows 登入後自動恢復」。獨立 Supervisor 只接受已保存的
`rl-auto + testnet + EXECUTE TESTNET` 設定；緊急停機、缺少憑證或停用交易所保護單時不會
恢復。Watchdog 與 Supervisor 是不同 Windows 排程，Dashboard 關閉時仍可運作。

桌面版雙擊 `安裝無人值守排程.cmd` 安裝，雙擊 `移除無人值守排程.cmd` 移除。為避免真實
資金在重開機後未經確認自動運作，此功能刻意不支援 Live。

## CLI

查詢 Testnet Futures 帳戶：

```powershell
python scripts\run_live_trading.py status --environment testnet
```

驗證一筆 100 USDT 名目的 BTC 多單，不送入撮合：

```powershell
python scripts\run_live_trading.py order --environment testnet `
  --symbol BTC/USDT --side BUY --quote-amount 100
```

空單使用 `--side SELL`。平多單使用 `SELL --full-exit --reduce-only`；平空單使用
`BUY --full-exit --reduce-only`。實際送到 Testnet 還要加入：

```text
--execute --confirm "EXECUTE TESTNET"
```

使用 SAC／PPO 執行一次相同市場的 AI 輪次：

```powershell
python scripts\run_live_trading.py rl-cycle --environment testnet `
  --training-dir data\processed\rl\environments\你的環境\training\你的成品 `
  --download-latest --leverage 2
```

模型的 `environment.json` 必須是 `exchange=binance_futures`、`symbol=BTC/USDT`；即使只做
Testnet 或 `/order/test` 驗證，Spot 模型也不能送到 Futures Gateway。

## 本地紀錄

```text
data/live_trading/{environment}/
  account_snapshots.csv
  rl_context.csv
  cycles.csv
  orders.csv
  positions.json
  protections.csv
  emergency_halt.json
  audit.jsonl
  reports/
  automation/
```

`positions.json` 的 Futures 鍵使用 `usd_m_futures:BTC/USDT`，避免與歷史 Spot 部位衝突。
CSV 與 JSON 適合目前單機 Testnet；公司級部署應把訂單、成交、持倉、決策與風控狀態
搬到 PostgreSQL，研究用 K 線與特徵改為 Parquet + DuckDB。

`emergency_halt.json` 是持久化安全鎖。保護單恢復、持倉歸零、對帳完成或程序重啟都不會
自動清除；必須由操作介面明確解除。模型品質、資料品質或保護單異常會阻止新增風險，
但不會阻止經帳戶與持倉驗證的 `reduce-only` 減倉或平倉。

私有 WebSocket 的「連線健康」與「事件投影已同步」分開記錄。只有未完成事件重播與 REST
對帳都成功後才會顯示 connected；事件已落資料庫但投影失敗時，重送相同事件仍會再次執行
冪等投影。

## 尚未宣稱完成的部分

- 已有常駐 Futures User Data Stream、事件落 PostgreSQL、斷線重連與週期性 REST 對帳；
  尚未用真實 API Key 做長時間斷網、API 限流與交易所維護演練。
- 尚未用真實 API Key 做 Testnet 整合測試，也沒有足夠 Testnet 天數與輪次，因此 Live
  資金仍應保持鎖定。
- Gateway 一致不代表模型有獲利能力；仍需 walk-forward、成本壓力測試、模型版本治理與
  可重現的 out-of-sample 證據。

完整資料與產品化說明見 `docs/資料覆蓋儲存架構與產品化指南.md`。

## 離線安全演練

以下命令不會向交易所送單，會沿用正式程式路徑驗證斷線重連、保護單契約與 Kill Switch：

```powershell
python scripts\run_live_safety_drills.py
```

報告會寫入 `data/research/live_safety_drills/<UTC時間>/report.json`。離線通過只證明本機
控制流程可運作，不能取代真實 API Key 的 Testnet 長時間驗收。
