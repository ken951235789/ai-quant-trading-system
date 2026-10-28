# 公開交易資料研究工具

本工具不需要交易所登入、API Key或錢包簽名；下載會呼叫公開唯讀API。來源為 Hyperliquid，不是 Binance。
原始資料只能放在 Git 排除的 `data/research/external_traders/`，報告放在 `artifacts/reports/`；不要強制加入Git。

## 安裝與測試
在儲存庫根目錄執行：
```powershell
python -m pip install -e ".[dev,dashboard]"
$env:PYTHONPATH='src;scripts'
$env:PYTHONIOENCODING='utf-8'
python -m pytest tests/research/test_external_traders.py tests/research/test_external_episodes.py -q
```

Linux/macOS可用 `PYTHONPATH=src:scripts`。離線測試只使用合成資料，不發送請求。

## 下載與配對
```powershell
python scripts/download_external_traders.py --data data/research/external_traders/20261028_pilot_v2 --report artifacts/reports/20261028/external_traders_v2
python scripts/build_external_episodes.py
```

資料夾名稱為協定批次名稱，實際下載的30天區間在首次執行時以 UTC 整點固定於 plan.json。
今天重抓無法復原發布時的全榜樣本或其歷史，不能把新結果當作原研究精確重現。
配對入口預設讀取上述下載目錄。中斷後先確認無同批次程序再續跑；同批次沿用樣本、區間及雜湊快取。
程式或契約變更時使用新目錄，不刪除 plan.json 繞過檢查。

## 資料用途
`training/sequences.npz` 是64棒×16特徵；`samples.csv` 分離標籤、時間、方向及樣本權重。
```python
from ai_quant_trading.research.external_episodes import load_episode_dataset

data = load_episode_dataset(
    "data/research/external_traders/20261028_episodes_v2/training",
    target="return_before_funding",
    allow_missing_funding=True,
)
train = data["metadata"]["split"].eq("train").to_numpy()
x_train, y_train = data["x"][train], data["y"][train]
```

上述是明確允許缺少完整成本的離線研究載入，不是正式淨收益訓練。
僅能用進場前已收盤市場特徵；後續持有時間、出場價、PnL及排行榜績效不能進入輸入。
不把這批資料當 SAC 最佳動作，不自動替換既有 Transformer V3 訓練契約。

## 驗證
```powershell
python scripts/verify_external_traders.py --data data/research/external_traders/20261028_pilot_v2 --report artifacts/reports/20261028/external_traders_v2 --output artifacts/reports/external_download_verification.json
```
若已安裝 PyTorch，可再執行包含批次載入的驗證：
```powershell
python scripts/verify_external_episodes.py --data data/research/external_traders/20261028_episodes_v2 --report artifacts/reports/20261028/external_episodes_v2 --output artifacts/reports/external_episode_verification.json
```
驗證檔名必須尚未存在；PyTorch驗證不訓練模型。完整協定見兩份 `EXTERNAL_*_PROTOCOL_20261028.md`。
