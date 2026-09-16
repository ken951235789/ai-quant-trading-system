# data/raw

這裡存放從交易所 API 下載的原始市場資料。

第一階段預計儲存 OHLCV CSV，欄位包含：

- timestamp
- open
- high
- low
- close
- volume

原始資料不應被特徵工程或模型訓練程式直接修改。

## Step 2 輸出結構

```text
data/raw/
  crypto/
    binance/
      ohlcv/          Binance 虛擬貨幣 K 線資料
      market_data/    Binance 24 小時市場統計快照
  us_equity/
    yahoo_finance/
      ohlcv/          Yahoo Finance 美股歷史 K 線資料
```

OHLCV 會包含基本 K 線欄位，以及 quote volume、成交筆數與 taker buy volume。Market Data 會包含 24 小時價格變化、成交量、最新 bid / ask 與 trade count。

美股 OHLCV 會包含 open、high、low、close、adj_close、volume、dividends 與 stock_splits。`adj_close`、股利與拆股欄位會在長期回測時特別重要。

虛擬貨幣在資料欄位中使用 `BTC/USDT`，檔名中使用 `BTC-USDT`；美股使用大寫 ticker，例如 `AAPL`。這樣既方便閱讀，也避免 Windows 將 `/` 當成資料夾分隔符號。
