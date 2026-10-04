# 候選特徵精簡與成交資訊研究

## 目的與邊界

驗證不同輸入能否改善固定候選的成本後篩選，不保證營利、不降低55%門檻、不增加槓桿。
沿用候選契約、標籤、V3 主體、校準、三方濾網及成本回放，不另建訓練或撮合引擎。
舊模型、原始資料、研究結果不覆寫。新功能是研究 CLI，未變更 EXE、交易頁或部署模型。

## 五組固定比較

| 版本 | 內容 | 要檢驗的假說 |
|---|---|---|
| F_existing | 既有 context_v2，MSE | 同期重跑基準，不跨資料版本比較 |
| F_compact | 精簡趨勢/波動/位置資訊 | 減少相似表示是否改善泛化 |
| F_compact_flow | 精簡加成交方向/活動 | 交易參與是否增加條件資訊 |
| F_compact_setup | 精簡加已知價位空間/訊號時效 | 避開空間有限或訊號過期是否可學習 |
| F_compact_combined | 同時加入 flow/setup | 兩群是否互補或互相干擾 |

保持 original、trend_pullback、vwap_reversion 三策略各自訓練，不因之前 Forward 排名改挑贏家。
三向前時段乘三策略乘五特徵組乘三 seeds，共135個模型。正式長時間研究要另外確認才執行。
初始特徵集合事先固定；既有常數/相關性修剪與 scaler 只看 Train，不用 Test 挑欄位。
精簡不是宣称被移除的特徵一定無用；完整基準仍存在，可由公平實驗反駁精簡假說。

## 特徵內容

精簡保留 15m/1h/4h 的 return_1、return_4、EMA20/EMA200 距離、EMA50斜率、ATR%、ADX、DI差、RSI、前20棒相對量。
另外保留15m布林位置/寬度、K棒實體/上下影/收盤位置、前日高低距離、當日VWAP距離、時間循環、候選方向及既有成本/ATR。
移除其他同義表示，包括 EMA50 距離、EMA50/200價差、額外波動表示、重複突破距離及趨勢摘要；不刪除原始 OHLCV。

成交群：
- `mtf_{15m,1h,4h}_flow_imbalance`：2乘主動買入基礎量/總基礎量減1。高週期先加總量再計比例，只能在收盤後使用。
- `mtf_15m_flow_trade_activity`：當根成交筆數/前20棒平均，不含當根於分母。
- `mtf_15m_flow_average_trade_relative`：當根報價成交量/筆數，再除以前20棒平均單筆規模。
- `mtf_15m_flow_has_trades`：真實零成交的可觀察旗標，不把缺資料當零成交。
- 明確零量棒 imbalance=0；平均單筆規模不存在時保持缺值遮罩，缺漏原始欄位則整批拒絕。

交易空間群：
- `long_room_atr` / `short_room_atr`：以決策收盤价為參考，到相應方向最近已知價位的距離/ATR。
- 價位集合是前20棒、前96棒高低與前一完整UTC日高低，不使用未來擺動確認或當日最終高低。
- `long_level_known` / `short_level_known`：沒有前方已知價位時距離為NaN且旗標0，不代表空間無限大。
- `pullback_from_high_atr` / `rebound_from_low_atr`：距前20棒高低的有號距離。
- `since_up_break_capped96` / `since_down_break_capped96`：距最近收盤突破的棒數/96，上限1；未觀察到事件同樣為1。
- `regime_age_capped96`：當前已知小時趨勢狀態持續的15m棒數，上限96後除96。
- 欄位前綴均為 `mtf_15m_setup_`。這些是幾何與時間特徵，不是保證獲利空間或真實掛單資訊。

## 資料與相容性

研究主來源：`outputs/kaggle_strategy_event_20260924/staging/data/btc_15m.csv`。
補充來源：`data/raw/crypto/binance_futures/ohlcv/ohlcv_crypto_binance_futures_BTC-USDT_15m_latest.csv`。
補充僅允許 quote_asset_volume、number_of_trades、taker_buy_base_volume、taker_buy_quote_volume 四欄。
逐根核對相同Binance USD-M BTCUSDT、15m時間戳與OHLCV，任何差異停止，不靜默改價或補零。
原始CSV即使含Funding/OI等欄位，也不會自動混入模型；其可得時間與完整性需另案查驗。
plan.json 保存雙來源SHA、程式SHA、特徵版本、切分及配置。frozen_fold CSV保留本次快照，不改原檔。
所有組共用相同候選與標籤；稽核會逐列比對不同特徵組的預測樣本端點及標籤。
舊候選契約省略預設 feature_set 欄位，既有參數雜湊不變；新組明確保存 `candidate_features_v1`。

## 操作

以下在專案根目錄 PowerShell 執行。可將 python 換成自己的 Python 環境完整路徑。
輸出目錄必須不存在；重跑請使用新名稱，禁止覆寫研究。

```powershell
$env:PYTHONPATH = 'src'
python scripts/run_candidate_training_research.py `
  --source outputs/kaggle_strategy_event_20260924/staging/data/btc_15m.csv `
  --flow-source data/raw/crypto/binance_futures/ohlcv/ohlcv_crypto_binance_futures_BTC-USDT_15m_latest.csv `
  --feature-study --seeds 42 137 2026 `
  --output artifacts/reports/20261003/feature_plan_example
```

預設只建計畫。不帶 `--feature-study` 仍走原本特徵/損失四格研究。
加 `--smoke`，並換輸出目錄，執行3策略乘5組共15個一輪模型；只用第一個seed、48,000棒與CPU兩執行緒。
完整正式研究需另獲明確批准，再將 `--smoke` 改為 `--run-research`，使用新目錄；預設每模型最多20輪、早停5輪。
正式訓練前先提出時間與資源方案，不把煙霧秒數等比例推算成正式訓練時間。

```powershell
python scripts/audit_candidate_training_research.py artifacts/reports/20261003/candidate_feature_smoke
python -m pytest -q tests/research/test_candidate_features.py tests/research/test_candidate_contract.py
```

## 怎麼讀結果

`study.json` 的 paired_comparison.feature_effects 記錄精簡、成交、交易空間及交互效果，單位為預測skill差，不是報酬率。
每個 run 的 candidate_evaluation.json 有無AI、簡單統計、Transformer篩選，分零/基本/壓力成本，交易不重疊。
先看高低分組實際收益、校準、分位數覆蓋、成交數、均值/分散/最差交易，再看成本壓力和集中度。
零成交的期望值是null，不是0%獲利；100笔或多seed也不自動等於統計充分。
這些歷史是反覆研究過的研究測試，不能重新命名為全新holdout。

## SAC 評估與停止條件

`study.json.sac_assessment` 只提供是否應繼續的理由，不解除任何既有閘門。
一輪工程煙霧不足以啟動SAC成效訓練。沒有跨期成本後改善、壓力成本證據或合法跨擬合OOS訊號時停止。
即使某組短測獲利，也不能把這組Test當成挑選依據再訓練SAC。
通過後才用同一候選契約與同一OOS事件比較固定部位、波動率部位、SAC入場部位，禁止先加槓桿或自由退出。
沿用 `research/sizing.py` 及 `staged_training.require_sizing_gate`；本次三seed研究不降低既有五seed閘門。
目前候選新契約仍屬隔離研究模型；舊SAC跨擬合入口不可直接拿來混用不同策略標籤。
完整帳戶浮虧、保證金/清算、真實Funding、Maker成交、稅後及實盤資格均不在本次驗證範圍。
