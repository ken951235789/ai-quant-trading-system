# scripts

## 可攜式訓練中心

```powershell
python scripts/build_training_bundle.py
```

輸出在 `dist/AIQuantTrainingPortable` 與 `dist/AIQuantTrainingPortable.zip`。
帶回交易電腦的 SAC/PPO 模型包可用下列命令匯入：

```powershell
python scripts/import_training_package.py outputs/model_packages/模型包.zip --trust-local-model
```

這個資料夾放可執行任務腳本，例如：

- 下載市場資料
- 產生特徵資料集
- 執行模型訓練
- 啟動模擬交易或實盤安全流程
- 整理舊版市場資料名稱
- 啟動 PPO／Transformer 歷史回測介面

腳本只負責串接流程，核心邏輯應放在 `src/ai_quant_trading/` 內。

### database_admin.py

用途：檢查 PostgreSQL、建立本機開發 schema，或把既有實盤 CSV/JSON 冪等匯入。
正式 schema 升級使用 `python -m alembic upgrade head`。

```powershell
python scripts\database_admin.py health
python scripts\database_admin.py import-live --environment testnet
```

## 目前腳本

### download_market_data.py

用途：

- 從 Binance 下載 OHLCV
- 從 Binance 下載 24 小時 market data 快照
- 從 Yahoo Finance 下載美股 OHLCV
- 驗證資料格式
- 儲存 CSV 到 `data/raw/`

範例：

```bash
python scripts/download_market_data.py --symbols BTC/USDT ETH/USDT --interval 1d --start 2024-01-01 --end 2024-02-01
```

下載美股：

```bash
python scripts/download_market_data.py --exchange yahoo --symbols AAPL MSFT SPY --interval 1d --start 2024-01-01 --end 2024-02-01
```

### build_features.py

用途：

- 讀取 Step 2 產生的 OHLCV CSV
- 驗證必要欄位與時間順序
- 計算趨勢、動能、波動率、成交量與市場結構特徵
- 建立未來 N 期報酬與二元 target
- 儲存到 `data/processed/`

範例：

```powershell
python scripts\build_features.py --input data\raw\crypto\binance\ohlcv\ohlcv_crypto_binance_BTC-USDT_1d_20231014_20260709.csv --target-horizon 1
```

將上漲定義為未來 5 期報酬超過 1%：

```powershell
python scripts\build_features.py --input data\raw\us_equity\yahoo_finance\ohlcv\你的檔名.csv --target-horizon 5 --target-threshold 0.01
```

### run_dashboard.py

用途：啟動交易總覽、資料下載、特徵工程、AI 管線、PPO、Transformer、AI 歷史回測與交易介面。

```powershell
python -m pip install -e ".[dashboard,ai,rl]"
python -m streamlit run scripts\run_dashboard.py
```

### run_paper_trading.py

用途：載入 PPO／SAC 成品，以最新已收盤 K 線推進一次模擬帳戶。預設資料保存在 `data/paper_trading/{account_id}/`。

```powershell
python scripts\run_paper_trading.py rl-run --account rl_btc --model-dir data\processed\rl\environments\你的環境\training\你的訓練 --exchange binance_futures --symbol BTC/USDT --interval 15m --download-latest
python scripts\run_paper_trading.py status --account rl_btc
```

常駐執行與安全停止：

```powershell
python scripts\run_paper_trading.py rl-auto --account rl_btc --model-dir data\processed\rl\environments\你的環境\training\你的訓練 --exchange binance_futures --symbol BTC/USDT --interval 15m --download-latest
python scripts\run_paper_trading.py stop --account rl_btc
```

可以用 Windows 工作排程器依模型週期定時呼叫第一行；系統會忽略已經處理過的 K 線。

### verify_trading_pipeline.py

用途：讀取模型保存的原始特徵來源，保留暖機資料後回放最後 20 根 K 線，檢查模型輸入、
Regime、資金管理、固定風控、模擬成交、稽核欄位與重複執行保護。

```powershell
python scripts\verify_trading_pipeline.py `
  --model-dir data\processed\rl\environments\你的環境\training\你的SAC訓練 `
  --bars 20
```

### run_replay_research.py

用途：把歷史 OHLCV 與即時 WebSocket 統一成 `MarketEvent`，執行重複、斷線續傳、
consumer 崩潰、亂序、缺 K 與損壞資料故障注入，產生可重現研究報告。

```powershell
python scripts\run_replay_research.py `
  --input "data\raw\crypto\binance_futures\ohlcv\ohlcv_crypto_binance_futures_BTC-USDT_15m_latest.csv" `
  --limit 500
```

### run_stress_test.py

用途：不連接交易 API，壓測事件匯流排、並行 CSV、雜湊稽核鏈與 PostgreSQL 健康狀態。

```powershell
python scripts\run_stress_test.py --event-count 20000 --csv-rows 1000 --audit-events 200
```

### run_live_trading.py

用途：查詢 Binance USD-M Futures 私有帳戶、驗證市場單，或使用已訓練模型執行一次交易
輪次。預設 Testnet 且只驗證，不會實際送單。

```powershell
python scripts\run_live_trading.py status --environment testnet
python scripts\run_live_trading.py order --environment testnet --symbol BTC/USDT --side BUY --quote-amount 10
python scripts\run_live_trading.py cycle --environment testnet --model-dir data\processed\models\你的模型資料夾 --download-latest
python scripts\run_live_trading.py rl-cycle --environment testnet --training-dir data\processed\rl\environments\環境\training\訓練成品 --download-latest
```

實際送單必須明確加入 `--execute` 和該環境的完整確認文字；Live 另有系統環境變數閘門。密鑰只能放在 `.env` 或系統環境變數，不可寫進命令列、程式碼或 Git。

### run_live_safety_drills.py

用途：完全離線驗證 Futures User Data Stream 斷線重連、交易所保護單參數與 Kill Switch。
不需要 API Key，也不會送出交易所訂單。

```powershell
python scripts\run_live_safety_drills.py
```

### train_rl.py

用途：讀取 Step 9 環境，以 PPO 或 SAC 在 CPU／CUDA 上訓練，定期保存 Checkpoint，並執行驗證與測試評估。全部參數可放在 JSON：

```powershell
python scripts\train_rl.py --environment data\processed\rl\environments\你的環境資料夾 --config configs\rl_training.example.json
```

可用 `--algorithm sac`、`--timesteps 500000`、`--device cuda:0` 覆蓋 JSON，也可用 `--resume 模型.zip` 從既有模型續跑。

### build_universal_rl_environment.py

用途：把至少兩個相同 K 線週期的 Step 3 特徵 CSV 建成通用多市場 RL 環境。
每個市場獨立切分，標準化參數只使用所有市場的訓練段。

```powershell
python scripts\build_universal_rl_environment.py --inputs data\processed\BTC特徵.csv data\processed\ETH特徵.csv data\processed\SPY特徵.csv
```

### organize_market_data.py

用途：將舊版 raw／feature CSV 搬到依市場分類的新路徑並改成可讀檔名。預設只預覽，不會修改檔案。

```powershell
python scripts\organize_market_data.py
python scripts\organize_market_data.py --apply
```

### build_windows_exe.py

用途：使用 PyInstaller 建置可雙擊啟動的 Windows one-folder App。

請使用 Python 3.12。此版本已完成 Windows 桌面版實機驗證；不要使用
Anaconda Python 3.13／OpenSSL 3.6 進行打包，否則內嵌服務可能在啟動後關閉。

```powershell
python -m pip install -e ".[dashboard,ai,rl,package]"
python scripts\build_windows_exe.py
```

輸出位於 `dist/AIQuantTradingSystem/`，使用時必須保留完整資料夾。
