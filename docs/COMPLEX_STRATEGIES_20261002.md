# 複合策略研究：固定條件與使用方式

本輪在第一輪 11,664 組之後新增六個策略家族，不修改第一輪的規則或結果。
不是新增實盤選單，也不訓練 Transformer / SAC；先判斷進出場條件本身的成本後表現。

## 事前固定範圍

六家族 × 兩週期 (15m / 1h) × 兩條件強度 (一般 / 嚴格)
× 停損 (1.5 / 2 ATR) × 目標 (1 / 2 / 3 R) × 持有 (32 / 64 根15m)
× 冷卻 (0 / 4 根15m) × 退出 (強趨勢反向 / 只靠保護價及期限) = **1,152 組**。
另加入原策略基準，合計 **1,153 組、17,295 個期間/成本情境**。

所有高週期由同份 BTCUSD-M 15m 重採樣，排除未完成高週期棒。
EMA 為 `ewm(adjust=False, min_periods=span)`；ATR、ADX、RSI 沿用既有事件引擎公式。
布林通道為 20 棒均值 ± 2 倍母體標準差，非樣本標準差。
RVOL = 當棒成交量 / 前20棒平均量，不含當棒於分母。
暖機 1000 小時，H1 EMA50/200 與 H4 EMA50/200 定義方向。
H4 斜率為 H4 EMA50 相較前4根 H4 的變化率。
VWAP 為開盤日歸屬的 UTC 日內典型價成交量加權累計；不是逐筆真實成交 VWAP。

## 六個家族

| 家族 | 一般條件 | 嚴格條件的額外限制 |
|---|---|---|
| 跨週期趨勢回調 | H1/H4 同向、H1 ADX≥20；碰 EMA20 後同向收回；多 RSI45～70／空30～55；RVOL≥0.8 | ADX≥25、RVOL≥1；多 RSI50～65／空35～50；收盤越過前棒高低且 H4 斜率同向 |
| 波動壓縮突破 | 前6棒曾有 `2σ20 < 1.5 ATR`；H1/H4 同向、ADX≥20；收盤突破前20棒極值；實體同向；RVOL≥1 | 壓縮改為 `2σ20 < 1.2 ATR`、ADX≥25、RVOL≥1.3；突破需超過0.1ATR |
| 區間 VWAP 均值回歸 | H1 ADX<20、H4斜率絕對值<0.3%；前棒在布林帶外，本棒收回帶內；仍離 VWAP 至少0.5ATR；多RSI<45／空>55；RVOL0.5～3；實體同向 | ADX<15、H4斜率<0.2%、RVOL上限2.5；多RSI<40／空>60 |
| 假突破收回 | H1 ADX<25；超過前20棒極值0.05ATR後收回；相應影線≥40%；RVOL≥1；實體同向 | ADX<20、穿透≥0.15ATR、收回≥0.05ATR、影線≥60%、RVOL≥1.3且H4方向不衝突 |
| 突破後回測確認 | H1/H4同向、ADX≥20；RVOL≥1的20棒突破後8棒內回測；容許離價位0.2ATR、失敗界線0.5ATR；RVOL≥0.7；同向收盤且距價位≤1ATR | ADX≥25、突破RVOL≥1.3、4棒內回測；容差0.1ATR、失敗界線0.25ATR、回測RVOL≥0.9 |
| 趨勢／震盪分流 | H1 ADX≥25走上述一般趨勢回調；ADX<18走上述一般均值回歸；中間不交易 | ADX≥30走嚴格趨勢回調；ADX<15走嚴格均值回歸；中間不交易 |

多空條件對稱。假突破收回只描述 OHLCV 價格型態，沒有假稱知道訂單簿、主力或停損位置。
回測突破價只能來自之前已知的突破；穿透失敗界線後永久作廢，等下一個新突破才能重設。
它不使用未來確認的 Swing / Pivot，不會同棒宣告突破又用同棒回測進場。

## 成交與風控邊界

- 訊號收盤後，下一根 15m 開盤成交。
- 使用訊號週期的已收盤 ATR，固定停損及目標，不事後移動保護價格。
- 反向退出只參考已確認兩根小時棒且 ADX≥25 的 H1 EMA50/200 方向。
- 只靠保護價及期限退出時，不因趨勢方向平倉。
- 同棒停利和停損都碰到，取停損；跳空停損取較差價。
- 基本往返約15bp，壓力約21bp，另每跨UTC八小時扣1bp Funding準備金。
- 配置固定10%名目本金、無槓桿，曲線只計平倉結算；不是完整帳戶風控。
- 原策略在五期間、三成本的指標必須與第一輪一致，否則中止研究。

## 選擇與圖表

Search 各家族/週期至少150筆，取壓力月均log收益前五；只在這份名單內做Validation排序。
品質門檻沿用第一輪：成本後正值、月份區塊最大統計量診斷、鄰域正值比例與結算回撤。
每家族只匯出驗證排名最前者供畫圖，避免顯示五條幾乎相同的參數曲線。
沒有滿足最少交易數的家族不降低門檻補選，而是明確標示未有足夠候選。
所有 Forward 數字是回溯研究，不是新的未見封存集；不能事後用 Forward 排名部署。

## 執行

```powershell
Set-Location D:\AIQuantTradingSystem
$env:PYTHONPATH='src'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:OPENBLAS_NUM_THREADS='2'
python -X utf8 scripts/run_complex_strategy_research.py `
  --source outputs/kaggle_strategy_event_20260924/staging/data/btc_15m.csv `
  --previous artifacts/reports/20261002/parameter_sweep `
  --output artifacts/reports/NEW_RUN/complex_strategy_panel
```

輸出必須是新資料夾，避免覆寫先前結果。`progress.json` 記錄階段與進度。
本輪分批保存 CSV，但不支援中斷後合併續跑；中斷請保留原目錄，使用新的研究目錄重跑。
需要 NumPy、pandas，製圖使用 matplotlib，不需 GPU。

## 產生圖表與報告

```powershell
python -X utf8 scripts/audit_complex_strategy_research.py artifacts/reports/NEW_RUN/complex_strategy_panel
python -X utf8 -m pytest -q --junitxml=artifacts/reports/NEW_RUN/complex_strategy_panel/pytest_results.xml
python .agents/skills/ai-quant-engineering/scripts/new_report.py `
  --type model_research --project AIQuantTradingSystem `
  --scope '六家族複合策略研究' --output artifacts/reports/NEW_RUN/complex_strategy_panel/report.json
python -X utf8 scripts/summarize_complex_strategy_research.py artifacts/reports/NEW_RUN/complex_strategy_panel
python .agents/skills/ai-quant-engineering/scripts/validate_report.py artifacts/reports/NEW_RUN/complex_strategy_panel/report.json
```

開啟輸出目錄的 `charts.html` 即可離線查看四張圖表，也可從該頁開啟第一輪的五張圖表。
`report.md` 有完整中文分析；`family_comparison.csv` 與 `round_comparison.csv` 為彙總表，
`pooled_forward.csv` 保留全部參數後續結果。正值、樣本數充足與通過品質閘門是三件不同的事。
