# data_collection

## BTC 五年九週期歷史

執行 `python scripts/download_btc_history.py --start 2021-09-08`，可一次補齊
`1m、3m、5m、15m、30m、1h、4h、12h、1d` 的 BTC/USDT USD-M 永續合約資料。
只下載已收盤 K 線，各週期另保留 250 根指標暖機資料。UTC timestamp 代表開盤時間。

大型缺口使用幣安官方月檔並驗證 SHA-256；最近資料和較小缺口使用 Futures REST API。
下載以 SQLite 分批暫存，不一次載入完整分鐘線；中斷後重跑相同命令可續傳。
完成後刪除本次下載暫存，發布到原有固定 CSV，不建立日期版本副本。
既有 Funding/OI 等欄位保留，但本入口不補造或宣稱提供五年訂單簿、OI、新聞資料。

資料位置：`data/raw/crypto/binance_futures/ohlcv/`。
品質與來源清單：同目錄 `btc_futures_history_manifest.json`。
完整下載後執行 `python -X utf8 scripts/verify_btc_history.py` 可獨立重讀全部 CSV，
檢查跨批次排序、去重、覆蓋與雜湊，並產生標準 Markdown/JSON 報告。
詳細說明見 `docs/BTC九週期歷史資料使用說明.md`。

## 進階資料

- `macro.py`：下載 FRED 宏觀資料，依保守發布延遲建立可用時間。
- `sec_fundamentals.py`：下載 SEC Company Facts，以申報日次日建立基本面快照。
- `binance_futures.py`：Funding、OI、spread、多空比、主動買賣比與基差。
- `binance.py`：除 OHLCV 外，也能建立現貨 Order Book 深度與失衡快照。

進階資料使用固定檔名更新。Order Book 無法由 REST 回補歷史，只會從第一次
啟用後開始累積。

## 加密貨幣衍生品脈絡

市場下載頁可額外勾選 `Funding／OI／Spread`。系統使用 Binance USDⓈ-M 公開只讀
端點，不需要 API key，也不包含下單能力。資料保存在：

```text
data/raw/crypto/binance/derivatives_context/
```

同一標的與週期只保留一份 `_latest.csv`，更新時依 timestamp 合併後覆寫。Funding
可依日期分頁；Open Interest 官方端點只提供有限期間；Spread 是目前最佳買賣價快照。
三者合併 K 線時只使用當時以前已公開的資料，不向過去回填。

負責市場資料收集。

Step 2 第一版先支援 Binance Spot 公開市場資料 API。這些資料不需要 API Key，適合在研究初期建立可重現的資料流程。

現在也加入 Yahoo Finance 美股 OHLCV。這個來源同樣不需要 API Key，適合研究與原型驗證；正式商用或實盤前，應改用具明確授權與 SLA 的資料源。

Dashboard 的標的選單不需要手動輸入或自己準備 CSV：加密貨幣會從 Binance `exchangeInfo` 取得目前可交易的全部 USDT 現貨，美股會從 Nasdaq 官方 Symbol Directory 合併 Nasdaq 與其他美國交易所掛牌證券。清單每六小時更新一次；外部服務暫時失敗時會顯示警告並使用內建常用標的備援。

系統直接呼叫 Yahoo Chart JSON API，不需要 `yfinance`。Yahoo 仍可能暫時限流；若出現 Too Many Requests，通常稍後重試即可。穩定大量下載時應改接 Polygon、Alpaca、Tiingo、Alpha Vantage 或券商資料源。

## 目前會收集什麼

### 1. OHLCV K 線資料

來源：

- Binance Spot REST API：`GET /api/v3/klines`
- Binance Spot WebSocket：九週期 `<symbol>@kline_<interval>` 與即時
  `<symbol>@bookTicker`；只保存事件中的已收盤 K 線。

用途：

- 後續計算技術指標
- 建立特徵工程資料集
- 進行歷史回測

CSV 欄位：

- `timestamp`：K 線開盤時間，UTC ISO 格式
- `symbol`：交易對，例如 `BTC/USDT`
- `exchange`：交易所，目前為 `binance`
- `interval`：K 線週期，例如 `1d`
- `open`：開盤價
- `high`：最高價
- `low`：最低價
- `close`：收盤價
- `volume`：base asset 成交量，例如 BTC 數量
- `quote_asset_volume`：quote asset 成交量，例如 USDT 成交額
- `number_of_trades`：該根 K 線內的成交筆數
- `taker_buy_base_volume`：主動買入的 base asset 數量
- `taker_buy_quote_volume`：主動買入的 quote asset 金額
- `open_time_ms`：Binance 原始開盤毫秒時間戳
- `close_time_ms`：Binance 原始收盤毫秒時間戳
- `collected_at`：本系統收集資料的 UTC 時間

### 2. 24 小時 Market Data 快照

來源：

- Binance Spot REST API：`GET /api/v3/ticker/24hr`

用途：

- 觀察最近 24 小時市場狀態
- 後續可做成市場熱度、成交活躍度、流動性特徵

CSV 欄位：

- `timestamp`：快照統計區間結束時間
- `symbol`：交易對
- `exchange`：交易所
- `price_change`：24 小時價格變化
- `price_change_percent`：24 小時價格變化百分比
- `weighted_avg_price`：加權平均價格
- `prev_close_price`：前一收盤價
- `last_price`：最新成交價
- `last_qty`：最新成交數量
- `bid_price`：最佳買價
- `bid_qty`：最佳買量
- `ask_price`：最佳賣價
- `ask_qty`：最佳賣量
- `open_price`：24 小時區間開盤價
- `high_price`：24 小時區間最高價
- `low_price`：24 小時區間最低價
- `volume`：base asset 成交量
- `quote_volume`：quote asset 成交額
- `open_time_ms`：統計區間開始時間
- `close_time_ms`：統計區間結束時間
- `first_trade_id`：第一筆成交 ID
- `last_trade_id`：最後一筆成交 ID
- `trade_count`：成交筆數
- `collected_at`：本系統收集資料的 UTC 時間

### 3. Order Book

Order Book 會在後續階段加入。預計收集：

- bid
- ask
- spread
- depth

### 4. 美股 OHLCV

來源：

- Yahoo Finance Chart JSON API

用途：

- 建立美股技術指標
- 測試跨市場 Feature Engineering
- 回測股票或 ETF 策略

CSV 欄位：

- `timestamp`：資料時間，UTC ISO 格式
- `symbol`：美股代號，例如 `AAPL`
- `exchange`：資料來源，目前為 `yahoo_finance`
- `interval`：K 線週期，例如 `1d`
- `open`：開盤價
- `high`：最高價
- `low`：最低價
- `close`：收盤價
- `adj_close`：調整後收盤價
- `volume`：成交量
- `dividends`：股利
- `stock_splits`：拆股比例
- `collected_at`：本系統收集資料的 UTC 時間

## 輸出位置

OHLCV：

```text
data/raw/crypto/binance/ohlcv/
```

Market Data：

```text
data/raw/crypto/binance/market_data/
```

美股 OHLCV：

```text
data/raw/us_equity/yahoo_finance/ohlcv/
```

## 名稱規則

- 程式與 CSV 欄位中的虛擬貨幣代碼統一使用 `BTC/USDT` 格式。
- 檔名使用 Windows 友善的 `BTC-USDT`，例如 `ohlcv_crypto_binance_BTC-USDT_1d_latest.csv`。
- 美股代碼統一使用大寫 ticker，例如 `AAPL`，檔名為 `ohlcv_us_equity_yahoo_finance_AAPL_1d_latest.csv`。
- Dashboard 同時顯示代碼、中文名稱與英文名稱；CSV 仍保存穩定代碼，避免翻譯文字影響程式處理。

同一來源、標的與 K 線週期只保留一份 `latest.csv`。再次下載時會先讀取既有歷史，
合併新 K 線、依 timestamp 去重，重複時間以新資料為準，再以原子取代方式覆寫。
寫入成功後才清除同市場的舊日期範圍檔，因此更新失敗不會先破壞舊資料。24 小時
市場快照與同預測期的 Feature Dataset 也使用固定檔名，不會每次更新新增一份。

整理舊版 CSV 時，先執行預覽，再確認搬移：

```powershell
python scripts\organize_market_data.py
python scripts\organize_market_data.py --apply
```

整理工具不會覆蓋既有檔案。若新位置已有同名檔案，會保留兩邊並顯示衝突。

## 使用方式

從專案根目錄執行：

```bash
python scripts/download_market_data.py --exchange binance --symbols BTC/USDT ETH/USDT --interval 1d --start 2024-01-01 --end 2024-02-01
```

下載美股日 K：

```bash
python scripts/download_market_data.py --exchange yahoo --symbols AAPL MSFT SPY --interval 1d --start 2024-01-01 --end 2024-02-01
```

更多美股與 ETF 範例：

```bash
python scripts/download_market_data.py --exchange yahoo --symbols NVDA AMD AVGO TSM QQQ VTI GLD --interval 1d --limit 500
```

Dashboard 另有大型科技、半導體、金融、醫療、消費與常用 ETF 快速清單；文字欄仍可輸入 Yahoo Finance 支援的任意 ticker。

下載美股最近 100 筆日 K：

```bash
python scripts/download_market_data.py --exchange yahoo --symbols AAPL MSFT SPY --interval 1d --limit 100
```

只下載最近一批 K 線：

```bash
python scripts/download_market_data.py --symbols BTC/USDT --interval 1h --limit 200
```

只下載 OHLCV，不下載 24 小時 market data：

```bash
python scripts/download_market_data.py --symbols BTC/USDT --interval 1d --skip-market-data
```

## HTTPS 憑證

程式不會關閉 SSL 驗證，也不會在模組匯入時改寫 Python 的全域 SSL context。Windows 會為市場資料 API 的 Requests Session 建立獨立 `truststore.SSLContext`，使用系統信任庫；其他平台使用 Requests 預設驗證。交易系統連接 API 時應維持 HTTPS 驗證，避免資料被中間人竄改。

## 資料品質檢查

目前會檢查：

- 必要欄位是否存在
- timestamp 是否能解析
- OHLCV 是否由舊到新排序
- timestamp 是否重複
- 價格是否大於 0
- volume 是否不小於 0
- high / low 是否符合基本價格邏輯
- trade_count 是否為非負數

後續會加入更完整的缺值、斷線、缺 K、異常尖峰與資料修補流程。

## BTC 即時行情

`binance_stream.py` 使用 combined stream 同時訂閱 BTC/USDT 的九個 K 線週期與
最佳買賣價。連線失敗會自動退避重連；背景 5m 模擬工作會在每次決策前從本機最後
時間戳呼叫 REST 補漏，所以 WebSocket 短暫斷線不會直接造成模型缺 K 線。

WebSocket 會持續收到尚未完成的 K 線更新，但只有 `closed=True` 的事件能寫入模型
資料。這能避免 5m K 線收盤前高低價與成交量仍變動時產生不可重現訊號。
