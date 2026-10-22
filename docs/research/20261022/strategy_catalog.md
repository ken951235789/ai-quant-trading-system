> 公開研究快照：文內原始碼行號與歷史文件名稱是本機研究版定位，可能不在此公開版本；不是可直接執行的策略或訓練設定。

# 有限策略研究目錄



156項研究方向／來源候選、20類、27個來源入口。

**不是156套已實作策略，也不是156個獨立證據。所有新增條目均不可直接執行。**

未對照完整歷史不等於從未測過；沒有新增回測、訓練或部署。



## 首批規則定稿

按機制涵蓋、資料可得性與實作可核對性排規則定稿；不是按已看過Test報酬挑選。先語意去重，發現已做過就沿用，不補湊12個新回測。

T07 Ichimoku雲上回復、M03 MACD零軸與已確認擺動、P03 Moving Momentum三步、P06 錨定VWAP回調、P08 縮量回調再放量、R04 線性趨勢殘差回歸、R07 短期報酬反轉、S03 RSI確認背離、S05 Failure Swing反轉、V05 棒級主動量差延續、V06 棒級CVD背離回復、D04 週末區間週一突破

缺完整規則或資料則保持待補；不得自動回測、擴充參數網格、降低成本、增加槓桿或降低既有品質閘門。



## 條目



### 趨勢追蹤



|ID|名稱|進場假說|退出假說|資料|歷史狀態|

|---|---|---|---|---|---|

|T01|雙均線交叉|快均線穿越慢均線後沿交叉方向|反向交叉或事前保護及期限|現有K線欄位可支援設計|已有相關研究，需語意去重|

|T02|H4 EMA20/50交叉|已收盤H4的EMA20/50交叉|沿用八機制研究的固定或移動退出|現有K線欄位可支援設計|既有本地版本已研究|

|T03|價格穿越長均線|收盤由均線一側首次進入另一側|回穿或期限|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|T04|均線帶由分散轉同向|多條均線完成排序且斜率同向後首次確認|排序失效或移動保護|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|T05|Supertrend翻向|ATR趨勢線翻向並通過既有趨勢條件|前棒線追蹤或反向訊號|現有K線欄位可支援設計|既有本地版本已研究|

|T06|Parabolic SAR翻向|SAR方向切換|前棒SAR保護或反向|現有K線欄位可支援設計|既有本地版本已研究|

|T07|Ichimoku雲上回復|價格突破當時可知雲層且轉換線確認|基準線失守或期限|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|T08|Aroon趨勢建立|新高頻率上升且新低頻率下降後首次進場|趨勢條件反轉|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|T09|線性迴歸斜率趨勢|歷史窗斜率經殘差尺度標準化後突破門檻|斜率消退或期限|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|T10|KAMA自適應趨勢|價格穿越因果自適應均線且效率比達標|回穿或保護|現有K線欄位可支援設計|完整歷史尚未逐項比對|



- T01 風險：盤整反覆換手；與H4交叉屬同機制；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：技術指標目錄](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators)；本地研究摘要／改編假說，不宣稱逐字重現來源。

- 歷史定位：`docs/EXTENDED_STRATEGIES_RESULTS_20261014.md`；本地固定版本；funding為準備金。



- T02 風險：高週期訊號不得向前填滿；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：技術指標目錄](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators)；本地研究摘要／改編假說，不宣稱逐字重現來源。

- 歷史定位：`docs/EXTENDED_STRATEGIES_RESULTS_20261014.md`；本地固定版本；funding為準備金。



- T03 風險：與雙均線高度相近；需比較交易指紋；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：技術指標目錄](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- T04 風險：均線數量不等於獨立資訊；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：技術指標目錄](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- T05 風險：ATR初始化與線更新順序影響結果；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：技術指標目錄](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators)；本地研究摘要／改編假說，不宣稱逐字重現來源。

- 歷史定位：`docs/PUBLIC_STRATEGY_REVIEW_20261013.md`；本地改編；舊funding為準備金，不等於歷史實際費。



- T06 風險：盤整鞭打；不以本棒極值更新本棒停損；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：技術指標目錄](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators)；本地研究摘要／改編假說，不宣稱逐字重現來源。

- 歷史定位：`docs/PUBLIC_STRATEGY_REVIEW_20261013.md`；本地改編；舊funding為準備金，不等於歷史實際費。



- T07 風險：雲層前移與遲行線易產生索引洩漏；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：策略與模型目錄](https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- T08 風險：與通道突破重疊，不算獨立勝利；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：技術指標目錄](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- T09 風險：不可用全樣本標準化；門檻未定稿；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：技術指標目錄](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- T10 風險：更多自由參數需計入複雜度；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：技術指標目錄](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators)；本地研究摘要／改編假說，不宣稱逐字重現來源。



### 動能與動能轉折



|ID|名稱|進場假說|退出假說|資料|歷史狀態|

|---|---|---|---|---|---|

|M01|日線7日動能|過去7個完整日報酬正負轉向|固定或移動退出|現有K線欄位可支援設計|既有本地版本已研究|

|M02|多期時間序列動能共識|多個過去報酬窗口同向才持有|共識消失或期限|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|M03|MACD零軸與已確認擺動|MACD過零且突破此前已確認擺動|結構失效或反向|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|M04|RSI區間位移|RSI從弱勢區進入強勢區並持續確認|回到弱勢區|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|M05|ROC加速突破|歷史報酬速度及加速度同步越界|動能衰退或期限|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|M06|TSI訊號交叉|平滑動能穿越訊號線且方向背景一致|反向交叉|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|M07|Fisher轉換反轉|歷史價格位置轉換由極端區回復|中性區或期限|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|M08|Connors RSI複合回歸|價格RSI、連續漲跌與報酬排名共同極端後回復|中性或期限|現有K線欄位可支援設計|完整歷史尚未逐項比對|



- M01 風險：本地日內改編不是學術12月策略；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[AQR：Time Series Momentum](https://www.aqr.com/Insights/Research/Journal-Article/Time-Series-Momentum)；本地研究摘要／改編假說，不宣稱逐字重現來源。

- 歷史定位：`docs/EXTENDED_STRATEGIES_RESULTS_20261014.md`；本地固定版本；funding為準備金。



- M02 風險：各窗口相關，不能當多個獨立票；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[AQR：Time Series Momentum](https://www.aqr.com/Insights/Research/Journal-Article/Time-Series-Momentum)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- M03 風險：pivot須等待右側確認，不得回畫進場；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：策略與模型目錄](https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- M04 風險：與趨勢規則相近；不是超買就反向；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：技術指標目錄](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- M05 風險：差分噪音與高換手；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：技術指標目錄](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- M06 風險：平滑延遲，成本後尚未驗證；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：技術指標目錄](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- M07 風險：只使用歷史窗極值，不能全段縮放；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：技術指標目錄](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- M08 風險：與RSI2同群；排名只能由過去計算；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：技術指標目錄](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators)；本地研究摘要／改編假說，不宣稱逐字重現來源。



### 通道與波動突破



|ID|名稱|進場假說|退出假說|資料|歷史狀態|

|---|---|---|---|---|---|

|B01|Donchian通道突破|收盤首次越過排除當棒的歷史高低|短通道追蹤或共同退出|現有K線欄位可支援設計|既有本地版本已研究|

|B02|H1長通道突破|H4同向時H1首次越過前55棒高低|固定或移動退出|現有K線欄位可支援設計|既有本地版本已研究|

|B03|布林壓縮突破|帶寬收縮後連續收盤突破|移動保護或固定期限|現有K線欄位可支援設計|既有本地版本已研究|

|B04|Keltner通道突破|收盤首次越過EMA加減ATR通道|回到中線或期限|現有K線欄位可支援設計|既有本地版本已研究|

|B05|前日NR7突破|前一完整日為七日最窄日，隔日突破其範圍|固定或移動退出|現有K線欄位可支援設計|既有本地版本已研究|

|B06|Inside Bar突破|已收盤內包棒後突破母棒範圍|母棒另一側或期限|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|B07|ATR價格衝擊延續|實體方向一致且報酬超過事前ATR尺度|衝擊失敗或期限|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|B08|低實現波動分位突破|過去波動分位偏低後突破短區間|波動擴張後追蹤|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|B09|已確認擺動壓縮突破|因果pivot形成收斂結構後越過邊界|收斂結構失效|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|B10|寬幅收盤區間突破|收盤落棒內極端且總範圍超過過去分位|動能失敗或期限|現有K線欄位可支援設計|完整歷史尚未逐項比對|



- B01 風險：通道不能含當棒；與海龜入場相近；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：技術指標目錄](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators)；本地研究摘要／改編假說，不宣稱逐字重現來源。

- 歷史定位：`docs/PUBLIC_STRATEGY_REVIEW_20261013.md`；本地改編；舊funding為準備金，不等於歷史實際費。



- B02 風險：屬Donchian高週期變體，不算新機制；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：技術指標目錄](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators)；本地研究摘要／改編假說，不宣稱逐字重現來源。

- 歷史定位：`docs/EXTENDED_STRATEGIES_RESULTS_20261014.md`；本地固定版本；funding為準備金。



- B03 風險：已研究分支，不因調參再次當新發現；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[John Bollinger：布林帶使用規則](https://www.bollingerbands.com/bollinger-band-rules)；本地研究摘要／改編假說，不宣稱逐字重現來源。

- 歷史定位：`docs/STRATEGY_FACTORY_RESULTS_20261021.md`；45契約0合格；基本測試3個正值，壓力全部負。



- B04 風險：與ATR突破同群；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：技術指標目錄](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators)；本地研究摘要／改編假說，不宣稱逐字重現來源。

- 歷史定位：`docs/PUBLIC_STRATEGY_REVIEW_20261013.md`；本地改編；舊funding為準備金，不等於歷史實際費。



- B05 風險：UTC日界與資料完整性；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：策略與模型目錄](https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models)；本地研究摘要／改編假說，不宣稱逐字重現來源。

- 歷史定位：`docs/EXTENDED_STRATEGIES_RESULTS_20261014.md`；本地固定版本；funding為準備金。



- B06 風險：雙邊同棒觸發未知，需保守處理；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：技術指標目錄](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- B07 風險：衝擊時滑價可能非固定值；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：技術指標目錄](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- B08 風險：與布林壓縮相近，需行為去重；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：技術指標目錄](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- B09 風險：線段選擇主觀，必須固定演算法；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：技術指標目錄](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- B10 風險：尾部行情依賴；需移除最大交易敏感度；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：技術指標目錄](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators)；本地研究摘要／改編假說，不宣稱逐字重現來源。



### 趨勢回調與回測



|ID|名稱|進場假說|退出假說|資料|歷史狀態|

|---|---|---|---|---|---|

|P01|H1/H4趨勢回調|高週期同向，EMA20碰觸後收回|沿用複合契約及獨立退出比較|現有K線欄位可支援設計|既有本地版本已研究|

|P02|H4趨勢H1均線收回|H4定向，H1重新穿回EMA20|八機制既有退出|現有K線欄位可支援設計|既有本地版本已研究|

|P03|Moving Momentum三步|均線定向後Stochastic回調，再由MACD柱確認|需定稿保護、失效及期限|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|P04|布林中軌趨勢回調|先前沿外軌，再回中軌同向收回|中軌失守或期限|現有K線欄位可支援設計|既有本地版本已研究|

|P05|突破後回測確認|先有已知突破，後續限定時間回測價位並收回|價位失效或期限|現有K線欄位可支援設計|既有本地版本已研究|

|P06|錨定VWAP回調|從事前可辨識事件錨定，趨勢回踩VWAP確認|錨定支撐失效或期限|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|P07|趨勢中RSI2回復|長均線定向，短RSI極端後恢復|固定或移動退出|現有K線欄位可支援設計|既有本地版本已研究|

|P08|縮量回調再放量|趨勢內回調量縮，再同向突破前棒且放量|回調低高失效或期限|現有K線欄位可支援設計|完整歷史尚未逐項比對|



- P01 風險：強配對減虧不能直接等於優勢；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：策略與模型目錄](https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models)；本地研究摘要／改編假說，不宣稱逐字重現來源。

- 歷史定位：`artifacts/reports/20261002/complex_strategy_panel/charts.html`；本地複合版本；舊結算權益及funding準備金不可視為完整MTM。



- P02 風險：與P01相關但時間觸發不同；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：策略與模型目錄](https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models)；本地研究摘要／改編假說，不宣稱逐字重現來源。

- 歷史定位：`docs/EXTENDED_STRATEGIES_RESULTS_20261014.md`；本地固定版本；funding為準備金。



- P03 風險：不能把來源入場說明冒充完整BTC策略；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：Moving Momentum](https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models/trading-strategies/moving-momentum)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- P04 風險：本地基本小正值未通過壓力與跨期；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[John Bollinger：布林帶使用規則](https://www.bollingerbands.com/bollinger-band-rules)；本地研究摘要／改編假說，不宣稱逐字重現來源。

- 歷史定位：`docs/STRATEGY_FACTORY_RESULTS_20261021.md`；45契約0合格；基本測試3個正值，壓力全部負。



- P05 風險：不得同棒先突破又假定回測成交；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：策略與模型目錄](https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models)；本地研究摘要／改編假說，不宣稱逐字重現來源。

- 歷史定位：`artifacts/reports/20261002/complex_strategy_panel/charts.html`；本地複合版本；舊結算權益及funding準備金不可視為完整MTM。



- P06 風險：不可事後挑最低點當錨；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：策略與模型目錄](https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- P07 風險：本地停損與來源原版不同；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：策略與模型目錄](https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models)；本地研究摘要／改編假說，不宣稱逐字重現來源。

- 歷史定位：`docs/EXTENDED_STRATEGIES_RESULTS_20261014.md`；本地固定版本；funding為準備金。



- P08 風險：量能分母是否含當棒須明訂；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：策略與模型目錄](https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models)；本地研究摘要／改編假說，不宣稱逐字重現來源。



### 均值回歸



|ID|名稱|進場假說|退出假說|資料|歷史狀態|

|---|---|---|---|---|---|

|R01|布林帶外收回|震盪背景下從帶外收回帶內|中軌或期限|現有K線欄位可支援設計|既有本地版本已研究|

|R02|VWAP偏離回歸|區間背景中偏離日內VWAP後開始收斂|VWAP或保護|現有K線欄位可支援設計|既有本地版本已研究|

|R03|歷史價格Z分數回歸|價格對滾動中心的標準化偏離回復|中性帶或失效|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|R04|線性趨勢殘差回歸|價格偏離僅過去擬合趨勢線後回復|殘差歸零或結構失效|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|R05|CCI極端回復|CCI極端後重返正常區|CCI中性或期限|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|R06|Williams %R區間回復|低趨勢背景下從區間端點回復|區間中點或失效|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|R07|短期報酬反轉|過去短窗極端漲跌後出現反向確認|短期限或波動保護|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|R08|連續收盤超伸反轉|同向收盤連續達門檻後首次反向確認|中心價或期限|現有K線欄位可支援設計|完整歷史尚未逐項比對|



- R01 風險：強趨勢持續走帶會失效；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[John Bollinger：布林帶使用規則](https://www.bollingerbands.com/bollinger-band-rules)；本地研究摘要／改編假說，不宣稱逐字重現來源。

- 歷史定位：`docs/STRATEGY_FACTORY_RESULTS_20261021.md`；45契約0合格；基本測試3個正值，壓力全部負。



- R02 風險：OHLCV近似VWAP非逐筆真實VWAP；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：技術指標目錄](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators)；本地研究摘要／改編假說，不宣稱逐字重現來源。

- 歷史定位：`artifacts/reports/20261002/complex_strategy_panel/charts.html`；本地複合版本；舊結算權益及funding準備金不可視為完整MTM。



- R03 風險：非平穩價格可能無均值回歸；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：技術指標目錄](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- R04 風險：訓練窗與持有期不能事後最佳化；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：技術指標目錄](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- R05 風險：與Z分數回歸屬近似群；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：技術指標目錄](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- R06 風險：與Stochastic同資訊來源；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：技術指標目錄](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- R07 風險：跌深不是低風險，需排除資料錯價；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：技術指標目錄](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- R08 風險：與TD計數相關，不等於原版TD；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：技術指標目錄](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators)；本地研究摘要／改編假說，不宣稱逐字重現來源。



### 價格結構與背離



|ID|名稱|進場假說|退出假說|資料|歷史狀態|

|---|---|---|---|---|---|

|S01|區間假突破收回|刺破既知前高低後收回，影線與方向確認|區間中點或保護|現有K線欄位可支援設計|既有本地版本已研究|

|S02|前日高低假突破|穿越前日高低後收回，當日首次有效訊號|八機制既有退出|現有K線欄位可支援設計|既有本地版本已研究|

|S03|RSI確認背離|兩個已確認價格pivot與RSI方向背離，再突破確認|第二pivot失效或期限|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|S04|MACD確認背離|價格與MACD動能的兩個已確認擺動背離|結構失效或期限|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|S05|Failure Swing反轉|未能延續前高低，反向突破已知中間擺動|失敗端點或期限|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|S06|BOS突破結構|收盤越過已確認結構高低並通過幅度門檻|結構失效|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|S07|CHoCH方向切換|既有結構趨勢首次被反向收盤破壞|新結構失效或期限|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|S08|三棒失衡區回訪|第三棒收盤後確認不重疊區，之後回訪確認|區域失效或期限|現有K線欄位可支援設計|完整歷史尚未逐項比對|



- S01 風險：只描述價格，不推定主力訂單；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：技術指標目錄](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators)；本地研究摘要／改編假說，不宣稱逐字重現來源。

- 歷史定位：`artifacts/reports/20261002/complex_strategy_panel/charts.html`；本地複合版本；舊結算權益及funding準備金不可視為完整MTM。



- S02 風險：前日須已完整收盤；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：技術指標目錄](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators)；本地研究摘要／改編假說，不宣稱逐字重現來源。

- 歷史定位：`docs/EXTENDED_STRATEGIES_RESULTS_20261014.md`；本地固定版本；funding為準備金。



- S03 風險：必須延後至pivot確認，不能回畫交易；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：技術指標目錄](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- S04 風險：與RSI背離同群，不能疊成獨立證據；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：技術指標目錄](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- S05 風險：需明確pivot及確認順序；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：技術指標目錄](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- S06 風險：與Donchian相近；不同名稱不代表不同優勢；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：技術指標目錄](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- S07 風險：趨勢狀態定義不可事後改寫；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：技術指標目錄](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- S08 風險：價格空隙不是可證明的未成交訂單；觸價不等於成交；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：技術指標目錄](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators)；本地研究摘要／改編假說，不宣稱逐字重現來源。



### 成交量與價格分布



|ID|名稱|進場假說|退出假說|資料|歷史狀態|

|---|---|---|---|---|---|

|V01|OBV累積突破|過去量能累積先突破且價格隨後確認|量價條件失效|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|V02|MFI極端回復|量價震盪值由極端恢復|中性或期限|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|V03|CMF方向確認|價格突破伴隨收盤位置加權量能同向|資金流代理反向|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|V04|相對量能突破|價格突破歷史區間且當棒量相對過去均量上升|突破失效或期限|現有K線欄位可支援設計|已有相關研究，需語意去重|

|V05|棒級主動量差延續|taker買賣差累積同向且價格確認突破|量差衰退或期限|部分成交量欄位已有，需核對|完整歷史尚未逐項比對|

|V06|棒級CVD背離回復|累積主動量差與已確認價格擺動背離|價格失效或期限|部分成交量欄位已有，需核對|完整歷史尚未逐項比對|

|V07|成交分布價值區回歸|已完成時段VAH/VAL外回到價值區|POC或區域失效|缺逐筆及深度歷史覆蓋核對|完整歷史尚未逐項比對|

|V08|低成交節點突破接受|突破前一完整分布的低量節點並持續成交確認|重新回到舊價值區|缺逐筆及深度歷史覆蓋核對|完整歷史尚未逐項比對|



- V01 風險：不同交易所成交量不可直接混合；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：技術指標目錄](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- V02 風險：與RSI回歸相關，不是真實資金流；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：技術指標目錄](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- V03 風險：CMF不是主動買賣分類；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：技術指標目錄](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- V04 風險：分母排除當棒；與複合突破有重疊；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：技術指標目錄](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators)；本地研究摘要／改編假說，不宣稱逐字重現來源。

- 歷史定位：`artifacts/reports/20261002/complex_strategy_panel/charts.html`；本地複合版本；舊結算權益及funding準備金不可視為完整MTM。



- V05 風險：買賣量須同單位；不是完整逐價delta；資料缺口：只能推導棒級delta代理，不能冒充逐價Footprint或完整訂單流。

- 來源：[Binance USD-M官方市場資料文件](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- V06 風險：CVD重置點事先固定；不能猜逐筆方向；資料缺口：只能推導棒級delta代理，不能冒充逐價Footprint或完整訂單流。

- 來源：[Binance USD-M官方市場資料文件](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- V07 風險：K線量均分價格只是代理，不可冒充真實Profile；資料缺口：不能用15m高低價猜隊列、撤單、吸收與成交順序。

- 來源：[StockCharts：技術指標目錄](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- V08 風險：需要真實逐價成交分布；資料缺口：不能用15m高低價猜隊列、撤單、吸收與成交順序。

- 來源：[StockCharts：技術指標目錄](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators)；本地研究摘要／改編假說，不宣稱逐字重現來源。



### 時段與事件窗口



|ID|名稱|進場假說|退出假說|資料|歷史狀態|

|---|---|---|---|---|---|

|D01|UTC首小時區間突破|00至01點完整範圍形成後首次收盤突破|固定或移動退出|現有K線欄位可支援設計|既有本地版本已研究|

|D02|時段初始平衡擴張|預先固定市場時段初段區間完成後擴張|回到初始平衡或時段結束|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|D03|時段突破失敗回歸|初始時段突破失敗後回到區間|區間中點或時段結束|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|D04|週末區間週一突破|完整週末高低形成後工作日首次越界|區間失效或固定期限|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|D05|前週高低突破|已完成前週極值被首次收盤突破|前週中點失守或期限|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|D06|資金費結算後反轉|結算後已知價差衝擊開始收斂|短期限或衝擊失效|funding已有；其他衍生資料待補|完整歷史尚未逐項比對|



- D01 風險：BTC無統一開盤競價，UTC劃分是人為假說；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：策略與模型目錄](https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models)；本地研究摘要／改編假說，不宣稱逐字重現來源。

- 歷史定位：`docs/EXTENDED_STRATEGIES_RESULTS_20261014.md`；本地固定版本；funding為準備金。



- D02 風險：與D01同群；夏令時間需交易日曆；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：策略與模型目錄](https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- D03 風險：不是單純反向D02；失敗確認需定稿；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：策略與模型目錄](https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- D04 風險：時區固定；有效樣本稀少；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：策略與模型目錄](https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- D05 風險：與通道突破相關，但事件排程不同；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：策略與模型目錄](https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- D06 風險：結算後費率不能當結算前已知；資料缺口：API有欄位不等於五年全覆蓋；公布前不可使用結算後費率。

- 來源：[Binance USD-M官方市場資料文件](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data)；本地研究摘要／改編假說，不宣稱逐字重現來源。



### 永續合約與槓桿活動



|ID|名稱|進場假說|退出假說|資料|歷史狀態|

|---|---|---|---|---|---|

|F01|已公布資金費極端回歸|已可得費率處歷史極端，價格反向確認|費率恢復或期限|funding已有；其他衍生資料待補|完整歷史尚未逐項比對|

|F02|OI增量價格突破|價格突破且同期已可得OI增加|突破失敗或OI收縮|funding已有；其他衍生資料待補|完整歷史尚未逐項比對|

|F03|OI收縮後價格反轉|槓桿出清代理後價格止跌或止漲確認|再破衝擊端點或期限|funding已有；其他衍生資料待補|完整歷史尚未逐項比對|

|F04|永續Premium均值回歸|溢價指數極端後開始回復|溢價中性或期限|funding已有；其他衍生資料待補|完整歷史尚未逐項比對|

|F05|Mark與成交價偏離回復|已知mark與可成交中價偏離並超過成本|偏離收斂或期限|缺逐筆及深度歷史覆蓋核對|完整歷史尚未逐項比對|

|F06|清算衝擊後反轉|可得清算事件聚集後價格回復|再破衝擊端點|缺逐筆及深度歷史覆蓋核對|完整歷史尚未逐項比對|

|F07|清算衝擊延續|清算聚集同時主動量與價格繼續同向|量價衰竭或期限|缺逐筆及深度歷史覆蓋核對|完整歷史尚未逐項比對|

|F08|帳戶多空擁擠反向確認|已公布多空比例極端且價格失去延續性|比例回復或期限|funding已有；其他衍生資料待補|完整歷史尚未逐項比對|



- F01 風險：極端funding可持續，不能單獨強制反向；資料缺口：API有欄位不等於五年全覆蓋；公布前不可使用結算後費率。

- 來源：[Binance USD-M官方市場資料文件](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- F02 風險：OI增加不能直接識別多空，需價格方向；資料缺口：API有欄位不等於五年全覆蓋；公布前不可使用結算後費率。

- 來源：[Binance USD-M官方市場資料文件](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- F03 風險：OI下降不等於特定方向爆倉；資料缺口：API有欄位不等於五年全覆蓋；公布前不可使用結算後費率。

- 來源：[Binance USD-M官方市場資料文件](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- F04 風險：單腿方向交易不等於套利；資料缺口：API有欄位不等於五年全覆蓋；公布前不可使用結算後費率。

- 來源：[Binance USD-M官方市場資料文件](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- F05 風險：mark不可直接成交，需要bid/ask；資料缺口：不能用15m高低價猜隊列、撤單、吸收與成交順序。

- 來源：[Binance USD-M官方市場資料文件](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- F06 風險：清算流可能抽樣或缺失，不能當完整總量；資料缺口：不能用15m高低價猜隊列、撤單、吸收與成交順序。

- 來源：[Binance USD-M官方市場資料文件](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- F07 風險：與反轉分支分開預登記，禁止事後挑方向；資料缺口：不能用15m高低價猜隊列、撤單、吸收與成交順序。

- 來源：[Binance USD-M官方市場資料文件](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- F08 風險：帳戶比例不是名目持倉比例；資料缺口：API有欄位不等於五年全覆蓋；公布前不可使用結算後費率。

- 來源：[Binance USD-M官方市場資料文件](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data)；本地研究摘要／改編假說，不宣稱逐字重現來源。



### 跨市場與相對價值



|ID|名稱|進場假說|退出假說|資料|歷史狀態|

|---|---|---|---|---|---|

|X01|BTC現貨永續基差收斂|雙腿可成交價差扣全部成本後超門檻|價差收斂或風險期限|需多市場同步資料|完整歷史尚未逐項比對|

|X02|跨交易所永續資金費差|已可得funding差足以支付雙腿交易及持有成本|差額縮小或風險預算用盡|需多市場同步資料|完整歷史尚未逐項比對|

|X03|期貨跨到期月份價差|同資產期限價差偏離事前估計範圍|價差回復或到期前平倉|需多市場同步資料|完整歷史尚未逐項比對|

|X04|BTC與ETH協整價差|僅訓練窗估計穩定關係後，殘差偏離|殘差回歸或關係失效|需多市場同步資料|完整歷史尚未逐項比對|

|X05|加密橫截面動能組合|當時可交易資產排名，買強賣弱|固定再平衡|需多市場同步資料|完整歷史尚未逐項比對|

|X06|現貨領先永續確認|同步時間戳下現貨先移動，永續延後確認|領先差消失或短期限|需多市場同步資料|完整歷史尚未逐項比對|



- X01 風險：funding變動與一腿失敗，不是無風險；資料缺口：不能只用BTC檔案或事後存活名單；時間對齊與可交易性待核對。

- 來源：[Hummingbot：V1策略列表](https://hummingbot.org/strategies/v1-strategies/)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- X02 風險：交易所信用、抵押資金及轉帳限制；資料缺口：不能只用BTC檔案或事後存活名單；時間對齊與可交易性待核對。

- 來源：[Hummingbot：V1策略列表](https://hummingbot.org/strategies/v1-strategies/)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- X03 風險：轉倉、融資與合約規格需建模；資料缺口：不能只用BTC檔案或事後存活名單；時間對齊與可交易性待核對。

- 來源：[CME：Futures Spread Overview](https://www.cmegroup.com/education/courses/understanding-futures-spreads/futures-spread-overview)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- X04 風險：協整檢定也受反覆搜尋偏差；資料缺口：不能只用BTC檔案或事後存活名單；時間對齊與可交易性待核對。

- 來源：[QuantConnect：Strategy Library](https://www.quantconnect.com/docs/v2/writing-algorithms/strategy-library)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- X05 風險：存活偏差與多腿成本，不能只用今日名單；資料缺口：不能只用BTC檔案或事後存活名單；時間對齊與可交易性待核對。

- 來源：[QuantConnect：Strategy Library](https://www.quantconnect.com/docs/v2/writing-algorithms/strategy-library)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- X06 風險：資料延遲可製造假領先，需接收時間；資料缺口：不能只用BTC檔案或事後存活名單；時間對齊與可交易性待核對。

- 來源：[QuantConnect：Strategy Library](https://www.quantconnect.com/docs/v2/writing-algorithms/strategy-library)；本地研究摘要／改編假說，不宣稱逐字重現來源。



### 訂單簿與流動性提供



|ID|名稱|進場假說|退出假說|資料|歷史狀態|

|---|---|---|---|---|---|

|Q01|訂單簿不平衡短期方向|多層買賣深度不平衡且逐筆成交確認|不平衡消失或短期限|缺逐筆及深度歷史覆蓋核對|完整歷史尚未逐項比對|

|Q02|大額主動成交吸收|主動量異常但價格推進有限，反向確認|吸收區失效或短期限|缺逐筆及深度歷史覆蓋核對|完整歷史尚未逐項比對|

|Q03|固定價差雙邊做市|圍繞中價雙邊報價並限制庫存|撤換報價、庫存上限與收盤規則|缺逐筆及深度歷史覆蓋核對|完整歷史尚未逐項比對|

|Q04|Avellaneda-Stoikov庫存報價|依庫存、風險及成交強度調整保留價與價差|動態撤換與庫存處理|缺逐筆及深度歷史覆蓋核對|完整歷史尚未逐項比對|

|Q05|跨交易所做市避險|一側maker報價，成交後另一側taker避險|庫存回復或避險失敗停機|需多市場同步資料|完整歷史尚未逐項比對|

|Q06|有限區間庫存網格|固定區間與有限庫存的分層限價|區間失效強制減倉，禁止無限加碼|缺逐筆及深度歷史覆蓋核對|完整歷史尚未逐項比對|



- Q01 風險：掛單可撤，L2與成交序號須一致；資料缺口：不能用15m高低價猜隊列、撤單、吸收與成交順序。

- 來源：[Binance USD-M官方市場資料文件](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- Q02 風險：不能從15m影線推定真實吸收；資料缺口：不能用15m高低價猜隊列、撤單、吸收與成交順序。

- 來源：[Hummingbot：V1策略列表](https://hummingbot.org/strategies/v1-strategies/)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- Q03 風險：逆向選擇與未成交；不是每次觸價都成交；資料缺口：不能用15m高低價猜隊列、撤單、吸收與成交順序。

- 來源：[Hummingbot：Pure Market Making](https://hummingbot.org/strategies/v1-strategies/pure-market-making/)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- Q04 風險：模型假設非市場真相；參數需只用過去估計；資料缺口：不能用15m高低價猜隊列、撤單、吸收與成交順序。

- 來源：[Avellaneda與Stoikov：High-frequency trading in a limit order book](https://math.nyu.edu/inmemoriam/avellaneda/HighFrequencyTrading.pdf)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- Q05 風險：跨所延遲與單腿風險；資料缺口：不能只用BTC檔案或事後存活名單；時間對齊與可交易性待核對。

- 來源：[Hummingbot：Cross-Exchange Market Making](https://hummingbot.org/strategies/v1-strategies/cross-exchange-market-making/)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- Q06 風險：可隱藏大額浮虧；不得馬丁格爾補倉；資料缺口：不能用15m高低價猜隊列、撤單、吸收與成交順序。

- 來源：[Hummingbot：Pure Market Making](https://hummingbot.org/strategies/v1-strategies/pure-market-making/)；本地研究摘要／改編假說，不宣稱逐字重現來源。



### 期權與波動率結構



|ID|名稱|進場假說|退出假說|資料|歷史狀態|

|---|---|---|---|---|---|

|O01|事件前買入跨式|事前事件下預測波動相對權利金足夠大|事件後或事前損失上限|目前不屬於BTC永續引擎|完整歷史尚未逐項比對|

|O02|Delta避險波動溢價|IV相對僅過去估計的實現波動偏高，有限風險賣方結構|定期避險、波動跳升及風險上限|目前不屬於BTC永續引擎|完整歷史尚未逐項比對|

|O03|期權期限結構價差|近遠月IV相對值偏離，建立事前限定腿數|期限價差回復或到期前|目前不屬於BTC永續引擎|完整歷史尚未逐項比對|

|O04|偏斜回歸價差|相同期限看漲看跌偏斜異常，使用有限風險組合|偏斜回復或期限|目前不屬於BTC永續引擎|完整歷史尚未逐項比對|

|O05|有限風險鐵兀鷹|事前區間與權利金條件符合才建立四腿|組合損失上限、期限或區間破壞|目前不屬於BTC永續引擎|完整歷史尚未逐項比對|

|O06|突破方向垂直價差|已確認方向訊號用限定到期及履約價的價差表達|方向失效或期限|目前不屬於BTC永續引擎|完整歷史尚未逐項比對|



- O01 風險：方向不明不代表買跨式必賺；時間耗損；資料缺口：BTC K線無法重建期權權利金、波動曲面與尾部損失。

- 來源：[CME：Straddles](https://www.cmegroup.com/education/courses/option-strategies/straddles)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- O02 風險：尾部損失、跳躍、gamma成本，不採裸賣無限風險；資料缺口：BTC K線無法重建期權權利金、波動曲面與尾部損失。

- 來源：[CME：25種期權結構](https://www.cmegroup.com/articles/files/2022/25-proven-strategies.pdf)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- O03 風險：不同gamma/vega不等於風險相抵；資料缺口：BTC K線無法重建期權權利金、波動曲面與尾部損失。

- 來源：[CME：Option Calendar Spreads](https://www.cmegroup.com/education/lessons/option-calendar-spreads)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- O04 風險：不同delta與買賣價差需匹配；資料缺口：BTC K線無法重建期權權利金、波動曲面與尾部損失。

- 來源：[CME：25種期權結構](https://www.cmegroup.com/articles/files/2022/25-proven-strategies.pdf)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- O05 風險：低頻大損可吞掉多次小利；不能只看勝率；資料缺口：BTC K線無法重建期權權利金、波動曲面與尾部損失。

- 來源：[CME：25種期權結構](https://www.cmegroup.com/articles/files/2022/25-proven-strategies.pdf)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- O06 風險：是曝險結構而非新的方向優勢；資料缺口：BTC K線無法重建期權權利金、波動曲面與尾部損失。

- 來源：[CME：25種期權結構](https://www.cmegroup.com/articles/files/2022/25-proven-strategies.pdf)；本地研究摘要／改編假說，不宣稱逐字重現來源。



### 外部事件與鏈上背景



|ID|名稱|進場假說|退出假說|資料|歷史狀態|

|---|---|---|---|---|---|

|E01|新聞情緒衝擊延續|首次可得新聞的意外情緒伴隨價格確認|訊息衰退或期限|需有公布時間的外部資料|完整歷史尚未逐項比對|

|E02|搜尋關注度異常|僅可得關注度突增與價格方向配對|關注度衰退或期限|需有公布時間的外部資料|完整歷史尚未逐項比對|

|E03|宏觀公布後衝擊|CPI等公布後實際值相對事前共識及價格確認|短期限或衝擊失效|需有公布時間的外部資料|完整歷史尚未逐項比對|

|E04|ETF已公布資金流|首次可得淨流量異常後下一可交易時點確認|背景反轉或期限|需有公布時間的外部資料|完整歷史尚未逐項比對|

|E05|交易所BTC淨流入背景|已可得鏈上流入異常配合價格失效|流入恢復或期限|需有公布時間的外部資料|完整歷史尚未逐項比對|

|E06|穩定幣供給衝擊背景|已可得供給或流入變化與BTC趨勢確認|背景衰退或期限|需有公布時間的外部資料|完整歷史尚未逐項比對|



- E01 風險：新聞重複、發布延遲、回填與分類錯誤；資料缺口：不可把修訂後宏觀值、日終資金流或回填新聞當事前資訊。

- 來源：[Liu與Tsyvinski：Risks and Returns of Cryptocurrency](https://www.nber.org/papers/w24877)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- E02 風險：趨勢資料常重標準化與修訂；資料缺口：不可把修訂後宏觀值、日終資金流或回填新聞當事前資訊。

- 來源：[Liu與Tsyvinski：Risks and Returns of Cryptocurrency](https://www.nber.org/papers/w24877)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- E03 風險：需即時共識快照，不可用修訂值；資料缺口：不可把修訂後宏觀值、日終資金流或回填新聞當事前資訊。

- 來源：[QuantConnect：Strategy Library](https://www.quantconnect.com/docs/v2/writing-algorithms/strategy-library)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- E04 風險：日終流量不可回填當日開盤；資料缺口：不可把修訂後宏觀值、日終資金流或回填新聞當事前資訊。

- 來源：[QuantConnect：Strategy Library](https://www.quantconnect.com/docs/v2/writing-algorithms/strategy-library)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- E05 風險：地址標籤修訂、內部調撥與不完整覆蓋；資料缺口：不可把修訂後宏觀值、日終資金流或回填新聞當事前資訊。

- 來源：[Liu與Tsyvinski：Risks and Returns of Cryptocurrency](https://www.nber.org/papers/w24877)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- E06 風險：鑄幣不等於即將買入BTC，因果未證實；資料缺口：不可把修訂後宏觀值、日終資金流或回填新聞當事前資訊。

- 來源：[Liu與Tsyvinski：Risks and Returns of Cryptocurrency](https://www.nber.org/papers/w24877)；本地研究摘要／改編假說，不宣稱逐字重現來源。



### 具名公開程式與原版待核對



|ID|名稱|進場假說|退出假說|資料|歷史狀態|

|---|---|---|---|---|---|

|G01|Freqtrade ADXMomentum|ADX、MOM與DI方向組合，沿用既有轉接|共同保護或加作者平多訊號|現有K線欄位可支援設計|既有本地版本已研究|

|G02|Freqtrade AwesomeMacd|AO過零且MACD同向|共同保護或作者平多訊號|現有K線欄位可支援設計|既有本地版本已研究|

|G03|Freqtrade CMCWinner|CCI、MFI與CMO極端組合|共同保護或作者平多訊號|現有K線欄位可支援設計|既有本地版本已研究|

|G04|Freqtrade CofiBitStrategy|低價EMA回調及快Stochastic交叉|共同保護或作者平多訊號|現有K線欄位可支援設計|既有本地版本已研究|

|G05|Freqtrade Simple|MACD、RSI強勢及布林上軌方向|共同保護或作者平多訊號|現有K線欄位可支援設計|既有本地版本已研究|

|G06|Freqtrade EMASkipPump|低於短均線與布林下軌，過量限制|共同保護或作者平多訊號|現有K線欄位可支援設計|既有本地版本已研究|

|G07|Freqtrade CCIStrategy原版|依原始程式待核對|依原始程式待核對|現有K線欄位可支援設計|來源或原版缺件／缺陷|

|G08|Freqtrade TDSequentialStrategy原版|依原始計數與exceed欄位待核對|依原始程式待核對|現有K線欄位可支援設計|來源或原版缺件／缺陷|

|G09|Freqtrade Scalp／SmoothScalp群|低週期均線與震盪指標回調|需核對原生ROI、停損與成交|需核對1m歷史與原規則|已有相關研究，需語意去重|

|G10|作者布林逆勢＋KDJ25|影片與表格尚缺完整進場規則|完整退出、部位與成本缺件|現有K線欄位可支援設計|來源或原版缺件／缺陷|



- G01 風險：只做多，不把平多當開空；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[Freqtrade公開策略庫](https://github.com/freqtrade/freqtrade-strategies)；本地研究摘要／改編假說，不宣稱逐字重現來源。

- 歷史定位：`docs/GITHUB_HISTORY_RESULTS_20261019.md`；六個本地改編已測；非原作者ROI、限價與原生部位完整重現。



- G02 風險：原1h，不能把不同週期績效混在一起；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[Freqtrade公開策略庫](https://github.com/freqtrade/freqtrade-strategies)；本地研究摘要／改編假說，不宣稱逐字重現來源。

- 歷史定位：`docs/GITHUB_HISTORY_RESULTS_20261019.md`；六個本地改編已測；非原作者ROI、限價與原生部位完整重現。



- G03 風險：與均值回歸群相關；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[Freqtrade公開策略庫](https://github.com/freqtrade/freqtrade-strategies)；本地研究摘要／改編假說，不宣稱逐字重現來源。

- 歷史定位：`docs/GITHUB_HISTORY_RESULTS_20261019.md`；六個本地改編已測；非原作者ROI、限價與原生部位完整重現。



- G04 風險：原5m，本地15m改編不等於原版；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[Freqtrade公開策略庫](https://github.com/freqtrade/freqtrade-strategies)；本地研究摘要／改編假說，不宣稱逐字重現來源。

- 歷史定位：`docs/GITHUB_HISTORY_RESULTS_20261019.md`；六個本地改編已測；非原作者ROI、限價與原生部位完整重現。



- G05 風險：原5m，本地15m改編；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[Freqtrade公開策略庫](https://github.com/freqtrade/freqtrade-strategies)；本地研究摘要／改編假說，不宣稱逐字重現來源。

- 歷史定位：`docs/GITHUB_HISTORY_RESULTS_20261019.md`；六個本地改編已測；非原作者ROI、限價與原生部位完整重現。



- G06 風險：ROI、部位與限價未完整重現；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[Freqtrade公開策略庫](https://github.com/freqtrade/freqtrade-strategies)；本地研究摘要／改編假說，不宣稱逐字重現來源。

- 歷史定位：`docs/GITHUB_HISTORY_RESULTS_20261019.md`；六個本地改編已測；非原作者ROI、限價與原生部位完整重現。



- G07 風險：既有審查指出高週期插值洩漏疑慮；不得直接執行；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[Freqtrade公開策略庫](https://github.com/freqtrade/freqtrade-strategies)；本地研究摘要／改編假說，不宣稱逐字重現來源。

- 歷史定位：`docs/GITHUB_HISTORY_RESULTS_20261019.md`；六個本地改編已測；非原作者ROI、限價與原生部位完整重現。



- G08 風險：既有審查指出第9棒更新及讀值不一致；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[Freqtrade公開策略庫](https://github.com/freqtrade/freqtrade-strategies)；本地研究摘要／改編假說，不宣稱逐字重現來源。

- 歷史定位：`docs/GITHUB_HISTORY_RESULTS_20261019.md`；六個本地改編已測；非原作者ROI、限價與原生部位完整重現。



- G09 風險：兩來源先合併同群，不虛增兩個機制；資料缺口：15m不可拆成真實1m；歷史下載完整度未核對。

- 來源：[Freqtrade公開策略庫](https://github.com/freqtrade/freqtrade-strategies)；本地研究摘要／改編假說，不宣稱逐字重現來源。

- 歷史定位：`docs/GITHUB_HISTORY_RESULTS_20261019.md`；六個本地改編已測；非原作者ROI、限價與原生部位完整重現。



- G10 風險：812%不能用本地BTC改編證實或證偽；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[指標與策略：使用者提供影片](https://www.youtube.com/watch?v=ifWUYXnWJtY)；本地研究摘要／改編假說，不宣稱逐字重現來源。

- 歷史定位：`docs/STRATEGY_FACTORY_RESULTS_20261021.md`；原版812%重現BLOCKED，不能用本地BTC研究代替。



### 確認式圖形與K線型態



|ID|名稱|進場假說|退出假說|資料|歷史狀態|

|---|---|---|---|---|---|

|C01|雙頂／雙底頸線確認|兩個已確認端點形成後突破中間頸線|右端點失效或期限|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|C02|頭肩型態頸線確認|三個因果擺動的比例與次序符合後越過頸線|右肩失效或期限|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|C03|三角收斂突破|僅已確認端點構成收斂，首次收盤離開|回到收斂區或期限|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|C04|旗形／旗面延續|先有方向衝擊，再有受限整理，突破整理邊界|旗面失效或期限|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|C05|楔形失效反轉|同向收斂端點形成後反向突破|楔形極端或期限|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|C06|支撐壓力吞沒確認|先有已知價位，再有實體吞沒及下一步確認|型態端點失效或期限|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|C07|長影線拒絕確認|在事前價位形成比例受限長影線，後續確認|影線端點或期限|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|C08|三棒轉折確認|預定三棒實體與位置順序完成後才觸發|型態失效或期限|現有K線欄位可支援設計|完整歷史尚未逐項比對|



- C01 風險：多空合併同項，不能事後挑相似高低；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts圖形型態](https://chartschool.stockcharts.com/table-of-contents/chart-analysis/chart-patterns)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- C02 風險：肩頭容差與pivot延遲需預定，不依圖挑單；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts圖形型態](https://chartschool.stockcharts.com/table-of-contents/chart-analysis/chart-patterns)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- C03 風險：與既有壓縮結構相關，不算獨立證據；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts圖形型態](https://chartschool.stockcharts.com/table-of-contents/chart-analysis/chart-patterns)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- C04 風險：不是看到後續大漲才定義旗桿；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts圖形型態](https://chartschool.stockcharts.com/table-of-contents/chart-analysis/chart-patterns)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- C05 風險：與三角分類可能重疊，需互斥辨識；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts圖形型態](https://chartschool.stockcharts.com/table-of-contents/chart-analysis/chart-patterns)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- C06 風險：只記錄完整收盤K線，不能宣稱訂單資訊；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts K線型態](https://chartschool.stockcharts.com/table-of-contents/chart-analysis/candlestick-charts)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- C07 風險：與假突破收回可能同交易，需去重；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts K線型態](https://chartschool.stockcharts.com/table-of-contents/chart-analysis/candlestick-charts)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- C08 風險：星線跳空定義不能硬套24小時BTC；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts K線型態](https://chartschool.stockcharts.com/table-of-contents/chart-analysis/candlestick-charts)；本地研究摘要／改編假說，不宣稱逐字重現來源。



### 統計估計與因果狀態



|ID|名稱|進場假說|退出假說|資料|歷史狀態|

|---|---|---|---|---|---|

|H01|OU殘差回歸|訓練窗確認均值回歸殘差，再越過成本調整界線|殘差回歸或模型失效|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|H02|Kalman局部趨勢偏離|僅向前濾波估計趨勢，偏離後回復|狀態失效或期限|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|H03|因果變點後趨勢|線上偵測分布改變後，方向訊號確認|後續變點或期限|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|H04|HMM趨勢／震盪分流|只用當時filtered狀態決定固定規則分支|沿分支原契約，不事後改動|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|H05|AR報酬殘差反轉|過去估計一步報酬殘差異常後回復|短期限或誤差持續失效|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|H06|歷史時段標準化衝擊|報酬相對過去同時段分布異常後確認|回到正常帶或期限|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|H07|波動跳躍後恢復|以過去波動為基準識別衝擊，再等待回復|再次擴張或期限|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|H08|Hurst狀態配對規則|滾動估計持續性後選固定趨勢或回歸分支|沿各分支契約|現有K線欄位可支援設計|完整歷史尚未逐項比對|



- H01 風險：不能直接假定BTC價格平穩；參數與界線需隔離；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[QuantConnect Optimal Pairs Trading](https://www.quantconnect.com/research/15294/optimal-pairs-trading/)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- H02 風險：禁止使用看完未來的smoother；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[QuantConnect Kalman Pairs Trading](https://www.quantconnect.com/forum/discussion/6826/from-research-to-production-kalman-filters-and-pairs-trading/p1)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- H03 風險：離線分段會偷看未來，偵測有延遲；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[QuantConnect：Strategy Library](https://www.quantconnect.com/docs/v2/writing-algorithms/strategy-library)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- H04 風險：只做分類不保證報酬；與既有分流比較；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[QuantConnect Hidden Markov Models](https://www.quantconnect.com/docs/v2/research-environment/applying-research/hidden-markov-models)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- H05 風險：估計窗與模型階數必須先凍結；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[QuantConnect：Strategy Library](https://www.quantconnect.com/docs/v2/writing-algorithms/strategy-library)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- H06 風險：不能用整段歷史估算時段平均；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[QuantConnect：Strategy Library](https://www.quantconnect.com/docs/v2/writing-algorithms/strategy-library)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- H07 風險：低頻K線只是跳躍代理，需明列估計誤差；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[QuantConnect：Strategy Library](https://www.quantconnect.com/docs/v2/writing-algorithms/strategy-library)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- H08 風險：短樣本Hurst不穩，與HMM為不同估計器而非獨立優勢；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[QuantConnect：Strategy Library](https://www.quantconnect.com/docs/v2/writing-algorithms/strategy-library)；本地研究摘要／改編假說，不宣稱逐字重現來源。



### 具名系統與圖表表示



|ID|名稱|進場假說|退出假說|資料|歷史狀態|

|---|---|---|---|---|---|

|I01|Connors RSI2來源重現|先取得來源完整趨勢與極端RSI條件|來源離場與風險設定另核對|現有K線欄位可支援設計|已有相關研究，需語意去重|

|I02|Last Stochastic來源候選|依來源固定最後一次有效震盪確認|完整離場規則待定|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|I03|Elder Impulse確認|EMA方向與MACD柱變化一致，配合已知高週期|方向不一致或保護期限|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|I04|Ichimoku雲層多條件版|雲層、基準與轉換線以當時可得時間同步確認|基準或雲層失效|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|I05|箱體建立後突破|箱頂底先因果確認，再突破固定箱界|返回箱體或期限|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|I06|點數圖突破|依預定格距及反轉格數形成突破|反轉列或真實價格保護|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|I07|Renko方向切換確認|事先固定磚距，已確認磚變向後真實行情確認|真實價保護或反向|需核對1m歷史與原規則|完整歷史尚未逐項比對|

|I08|Heikin-Ashi趨勢確認|合成K線只作已收盤訊號，真實價下一步成交|合成方向消失或真實保護|現有K線欄位可支援設計|完整歷史尚未逐項比對|



- I01 風險：既有P07是改編，不能重命名當原版完成；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：策略與模型目錄](https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models)；本地研究摘要／改編假說，不宣稱逐字重現來源。

- 歷史定位：`docs/EXTENDED_STRATEGIES_RESULTS_20261014.md`；本地固定版本；funding為準備金。



- I02 風險：來源名稱不是可執行規則，需逐條核對；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：策略與模型目錄](https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- I03 風險：原系統含濾網用途，完整交易契約需另外定稿；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts Elder Impulse](https://chartschool.stockcharts.com/table-of-contents/chart-analysis/chart-types/elder-impulse-system)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- I04 風險：與T07同群；只有新增完整規則才是新版本；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：策略與模型目錄](https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- I05 風險：不可每根事後改箱；與Donchian相關；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：策略與模型目錄](https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- I06 風險：合成價格不可直接撮合，棒內順序未知；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts Point and Figure Patterns](https://chartschool.stockcharts.com/table-of-contents/chart-analysis/point-and-figure-charts/classic-patterns)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- I07 風險：不可用合成磚收盤成交；需更細歷史；資料缺口：15m不可拆成真實1m；歷史下載完整度未核對。

- 來源：[StockCharts：策略與模型目錄](https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- I08 風險：合成OHLC不能當成交價，與普通趨勢需公平比較；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[StockCharts：策略與模型目錄](https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models)；本地研究摘要／改編假說，不宣稱逐字重現來源。



### 多腿套利與投組研究



|ID|名稱|進場假說|退出假說|資料|歷史狀態|

|---|---|---|---|---|---|

|J01|現貨與定期期貨正向套利|可成交到期基差扣融資與雙腿成本後達門檻|到期收斂或風險上限|需多市場同步資料|完整歷史尚未逐項比對|

|J02|反向到期基差套利|負基差扣借幣及雙腿成本後才考慮反向組合|收斂或借貸條件失效|需多市場同步資料|完整歷史尚未逐項比對|

|J03|跨交易所現貨價差|兩邊可成交報價扣庫存與成本後有差額|同步雙腿或預定未對沖風險控制|需多市場同步資料|完整歷史尚未逐項比對|

|J04|三角匯率循環|三腿實際bid/ask與深度組成可行循環|完成三腿或失敗時減風險|需多市場同步資料|完整歷史尚未逐項比對|

|J05|AMM與CEX價差|池報價對交易所報價扣gas與衝擊後有差額|雙腿完成或失敗處理|需多市場同步資料|完整歷史尚未逐項比對|

|J06|橫截面Carry組合|歷史可得carry排名形成受限雙腿組合|預定再平衡|需多市場同步資料|完整歷史尚未逐項比對|

|J07|PCA因子殘差組合|只用訓練窗估計共同因子，交易殘差偏離|殘差收斂或結構失效|需多市場同步資料|完整歷史尚未逐項比對|

|J08|Kalman動態配對避險|向前估計避險比率後交易價差偏離|價差回復或關係失效|需多市場同步資料|完整歷史尚未逐項比對|



- J01 風險：永續不是到期合約；資本占用需計入；資料缺口：不能只用BTC檔案或事後存活名單；時間對齊與可交易性待核對。

- 來源：[CME：Futures Spread Overview](https://www.cmegroup.com/education/courses/understanding-futures-spreads/futures-spread-overview)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- J02 風險：借幣不可得就不能假設成交；資料缺口：不能只用BTC檔案或事後存活名單；時間對齊與可交易性待核對。

- 來源：[CME：Futures Spread Overview](https://www.cmegroup.com/education/courses/understanding-futures-spreads/futures-spread-overview)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- J03 風險：預置資產、提款與交易所信用風險；資料缺口：不能只用BTC檔案或事後存活名單；時間對齊與可交易性待核對。

- 來源：[Hummingbot：V1策略列表](https://hummingbot.org/strategies/v1-strategies/)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- J04 風險：不能以三個中價相乘當獲利；最小單位與延遲；資料缺口：不能只用BTC檔案或事後存活名單；時間對齊與可交易性待核對。

- 來源：[Hummingbot：V1策略列表](https://hummingbot.org/strategies/v1-strategies/)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- J05 風險：區塊排序、MEV與交易失敗成本；資料缺口：不能只用BTC檔案或事後存活名單；時間對齊與可交易性待核對。

- 來源：[Hummingbot AMM Arbitrage](https://hummingbot.org/strategies/v1-strategies/amm-arbitrage/)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- J06 風險：carry高可能代表風險高；存活偏差；資料缺口：不能只用BTC檔案或事後存活名單；時間對齊與可交易性待核對。

- 來源：[QuantConnect：Strategy Library](https://www.quantconnect.com/docs/v2/writing-algorithms/strategy-library)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- J07 風險：成分選擇、因子數與成本都需凍結；資料缺口：不能只用BTC檔案或事後存活名單；時間對齊與可交易性待核對。

- 來源：[QuantConnect PCA and Pairs Trading](https://www.quantconnect.com/docs/v2/research-environment/applying-research/pca-and-pairs-trading)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- J08 風險：動態比率增加換手，需與固定配對比較；資料缺口：不能只用BTC檔案或事後存活名單；時間對齊與可交易性待核對。

- 來源：[QuantConnect Kalman Pairs Trading](https://www.quantconnect.com/forum/discussion/6826/from-research-to-production-kalman-filters-and-pairs-trading/p1)；本地研究摘要／改編假說，不宣稱逐字重現來源。



### 跨市場事件與市場中斷



|ID|名稱|進場假說|退出假說|資料|歷史狀態|

|---|---|---|---|---|---|

|K01|CME休市缺口回復|市場重開後可得缺口開始收斂才觸發|缺口回補或期限|需有公布時間的外部資料|完整歷史尚未逐項比對|

|K02|ETF與期貨基差偏離|ETF可成交價格與期貨避險價差異常|差額回復或時段結束|需多市場同步資料|完整歷史尚未逐項比對|

|K03|ETF溢折價事件|當時可得NAV代理與可成交價格偏離|價差回復或期限|需有公布時間的外部資料|完整歷史尚未逐項比對|

|K04|期權到期Gamma背景|已可得未平倉分布形成事前背景，價格確認|到期或背景失效|目前不屬於BTC永續引擎|完整歷史尚未逐項比對|

|K05|ETF收盤窗口衝擊|交易日曆固定窗口內量價偏離後確認|窗口結束或保護|需有公布時間的外部資料|完整歷史尚未逐項比對|

|K06|幣種上架公告事件|首次可得公告後依固定等待期及流動性確認|短期限或失效|需多市場同步資料|完整歷史尚未逐項比對|

|K07|鏈上升級事件風險溢價|事先公告的事件窗口與價格條件配對|事件後或風險上限|需有公布時間的外部資料|完整歷史尚未逐項比對|

|K08|交易所中斷後價差恢復|可驗證恢復時間後跨所報價重新對齊|價差回復或流動性失效|需多市場同步資料|完整歷史尚未逐項比對|



- K01 風險：BTC連續行情與CME交易日曆不可混淆；資料缺口：不可把修訂後宏觀值、日終資金流或回填新聞當事前資訊。

- 來源：[QuantConnect：Strategy Library](https://www.quantconnect.com/docs/v2/writing-algorithms/strategy-library)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- K02 風險：ETF交易時間、匯率、追蹤誤差及借券；資料缺口：不能只用BTC檔案或事後存活名單；時間對齊與可交易性待核對。

- 來源：[QuantConnect：Strategy Library](https://www.quantconnect.com/docs/v2/writing-algorithms/strategy-library)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- K03 風險：公布NAV不可回填到較早時間；資料缺口：不可把修訂後宏觀值、日終資金流或回填新聞當事前資訊。

- 來源：[QuantConnect：Strategy Library](https://www.quantconnect.com/docs/v2/writing-algorithms/strategy-library)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- K04 風險：OI不揭露交易商方向，Gamma僅情境估計；資料缺口：BTC K線無法重建期權權利金、波動曲面與尾部損失。

- 來源：[QuantConnect：Strategy Library](https://www.quantconnect.com/docs/v2/writing-algorithms/strategy-library)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- K05 風險：不能用當日完整流量提前交易；資料缺口：不可把修訂後宏觀值、日終資金流或回填新聞當事前資訊。

- 來源：[QuantConnect：Strategy Library](https://www.quantconnect.com/docs/v2/writing-algorithms/strategy-library)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- K06 風險：不是BTC單標的；須包含失敗、下架與幸存偏差；資料缺口：不能只用BTC檔案或事後存活名單；時間對齊與可交易性待核對。

- 來源：[QuantConnect：Strategy Library](https://www.quantconnect.com/docs/v2/writing-algorithms/strategy-library)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- K07 風險：事件數少、版本修訂及事後分類；資料缺口：不可把修訂後宏觀值、日終資金流或回填新聞當事前資訊。

- 來源：[QuantConnect：Strategy Library](https://www.quantconnect.com/docs/v2/writing-algorithms/strategy-library)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- K08 風險：中斷時不一定可成交，需真實狀態與延遲；資料缺口：不能只用BTC檔案或事後存活名單；時間對齊與可交易性待核對。

- 來源：[QuantConnect：Strategy Library](https://www.quantconnect.com/docs/v2/writing-algorithms/strategy-library)；本地研究摘要／改編假說，不宣稱逐字重現來源。



### 成交時鐘與拍賣結構



|ID|名稱|進場假說|退出假說|資料|歷史狀態|

|---|---|---|---|---|---|

|L01|日內VWAP重新站回|可得日內VWAP被重新收回並確認方向|再次失守或期限|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|L02|VWAP偏離帶延續|事前估計偏離帶外持續成交且方向確認|回到帶內或期限|現有K線欄位可支援設計|完整歷史尚未逐項比對|

|L03|價值區遷移延續|完整過去時段POC與價值區同向移動|遷移失敗或期限|缺逐筆及深度歷史覆蓋核對|完整歷史尚未逐項比對|

|L04|隔夜庫存代理回歸|固定時段成交方向偏置後下一時段回復|偏置消退或期限|缺逐筆及深度歷史覆蓋核對|完整歷史尚未逐項比對|

|L05|連續價位主動量不平衡|逐價Footprint多層不平衡後價格確認|不平衡區失效或期限|缺逐筆及深度歷史覆蓋核對|完整歷史尚未逐項比對|

|L06|掛單補充吸收代理|可觀測同價位反覆補單與主動成交後反向確認|價位失效或期限|缺逐筆及深度歷史覆蓋核對|完整歷史尚未逐項比對|

|L07|成交金額棒動能|因果累積到固定金額才封棒，再產生動能訊號|預定事件棒或真實時間期限|缺逐筆及深度歷史覆蓋核對|完整歷史尚未逐項比對|

|L08|報價更新強度突破|深度與報價更新強度異常，伴隨成交突破|強度衰退或短期限|缺逐筆及深度歷史覆蓋核對|完整歷史尚未逐項比對|



- L01 風險：與VWAP回調同群，不能靠改名增加證據；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[Binance USD-M官方市場資料文件](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- L02 風險：與均值回歸方向相反，事前指定市場背景；資料缺口：仍須逐契約驗證區間、缺口、費用與時間戳。

- 來源：[Binance USD-M官方市場資料文件](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- L03 風險：需逐價成交，不能把K線總量均分當真值；資料缺口：不能用15m高低價猜隊列、撤單、吸收與成交順序。

- 來源：[Binance USD-M官方市場資料文件](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- L04 風險：只是交易流代理，不知道參與者真實庫存；資料缺口：不能用15m高低價猜隊列、撤單、吸收與成交順序。

- 來源：[Binance USD-M官方市場資料文件](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- L05 風險：方向分類與逐筆完整性不足就停止；資料缺口：不能用15m高低價猜隊列、撤單、吸收與成交順序。

- 來源：[Binance USD-M官方市場資料文件](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- L06 風險：L2不能辨識隱藏委託身分，不稱已偵測冰山；資料缺口：不能用15m高低價猜隊列、撤單、吸收與成交順序。

- 來源：[Binance USD-M官方市場資料文件](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- L07 風險：歷史成交粒度、封棒時間與15m比較需公平；資料缺口：不能用15m高低價猜隊列、撤單、吸收與成交順序。

- 來源：[Binance USD-M官方市場資料文件](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data)；本地研究摘要／改編假說，不宣稱逐字重現來源。



- L08 風險：不能以K線交易筆數代替撤掛單事件數；資料缺口：不能用15m高低價猜隊列、撤單、吸收與成交順序。

- 來源：[Binance USD-M官方市場資料文件](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data)；本地研究摘要／改編假說，不宣稱逐字重現來源。



## 來源與授權



- SC [StockCharts：策略與模型目錄](https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models)：策略分類與機制參考，不提供本地BTC版本的盈利證據。只保存連結及本地摘要；未複製程式。

- SCI [StockCharts：技術指標目錄](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators)：僅指標背景，本地進出場假說另訂。只保存連結及本地摘要。

- MM [StockCharts：Moving Momentum](https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models/trading-strategies/moving-momentum)：均線定向、Stochastic回調、MACD確認的順序；本地版本未重現。未複製全文或程式。

- BB [John Bollinger：布林帶使用規則](https://www.bollingerbands.com/bollinger-band-rules)：布林帶背景，觸軌不自動等於買賣訊號。未複製全文或程式。

- TSM [AQR：Time Series Momentum](https://www.aqr.com/Insights/Research/Journal-Article/Time-Series-Momentum)：多資產長期動能研究；不能移植其績效到BTC15m。僅書目與本地研究假說。

- CRYPTO [Liu與Tsyvinski：Risks and Returns of Cryptocurrency](https://www.nber.org/papers/w24877)：加密動能與關注度研究，不是本地日內策略規格。僅書目與本地研究假說。

- QC [QuantConnect：Strategy Library](https://www.quantconnect.com/docs/v2/writing-algorithms/strategy-library)：配對及跨資產研究入口；單篇規則與授權尚待逐項核對。只登錄，未匯入程式。

- FT [Freqtrade公開策略庫](https://github.com/freqtrade/freqtrade-strategies)：來源固定commit f3340ce11f5bdf62f598522e64d1f5638eaa13f5；本地改編與原版分開。GPL-3.0；既有轉接維持隔離，新增項目不複製或執行外部程式。

- HB [Hummingbot：V1策略列表](https://hummingbot.org/strategies/v1-strategies/)：做市、現貨永續與跨市場機制；文件註明維護狀態需逐策略核對。只保存連結；程式版本及授權另查。

- PMM [Hummingbot：Pure Market Making](https://hummingbot.org/strategies/v1-strategies/pure-market-making/)：掛撤單與庫存調整，OHLCV不足以證明限價成交。只保存連結及摘要。

- XEMM [Hummingbot：Cross-Exchange Market Making](https://hummingbot.org/strategies/v1-strategies/cross-exchange-market-making/)：一地掛單、一地避險；不是無風險收益。只保存連結及摘要。

- AS [Avellaneda與Stoikov：High-frequency trading in a limit order book](https://math.nyu.edu/inmemoriam/avellaneda/HighFrequencyTrading.pdf)：庫存風險報價模型；成交強度及市場假設需重新估計。僅書目與機制摘要。

- BIN [Binance USD-M官方市場資料文件](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data)：只證明欄位/API存在，不保證五年歷史可取得、不支持策略有效性。遵守資料來源條款；本輪不下載行情。

- CMEF [CME：Futures Spread Overview](https://www.cmegroup.com/education/courses/understanding-futures-spreads/futures-spread-overview)：跨到期月份價差背景，BTC版本及融資成本待定。只保存連結及摘要。

- CMEO [CME：Straddles](https://www.cmegroup.com/education/courses/option-strategies/straddles)：跨式組合及損益風險背景；不是BTC波動預測模型。只保存連結及摘要。

- CMEC [CME：Option Calendar Spreads](https://www.cmegroup.com/education/lessons/option-calendar-spreads)：期權跨期限結構；入場條件為本地待驗假說。只保存連結及摘要。

- CME25 [CME：25種期權結構](https://www.cmegroup.com/articles/files/2022/25-proven-strategies.pdf)：僅組合損益結構；標題不等於策略有經驗證的超額收益。不複製圖片或原文。

- AUTHOR [指標與策略：使用者提供影片](https://www.youtube.com/watch?v=ifWUYXnWJtY)：只核對指標及部分參數；完整交易規則、商品、部位與成本缺件。未下載影片或繞過存取限制。

- PATTERN [StockCharts圖形型態](https://chartschool.stockcharts.com/table-of-contents/chart-analysis/chart-patterns)：型態背景；本地因果辨識、參數及退出仍待定稿。只存摘要與連結。

- CANDLE [StockCharts K線型態](https://chartschool.stockcharts.com/table-of-contents/chart-analysis/candlestick-charts)：K線型態定義，不支持單靠型態有BTC交易優勢。只存摘要與連結。

- ELDER [StockCharts Elder Impulse](https://chartschool.stockcharts.com/table-of-contents/chart-analysis/chart-types/elder-impulse-system)：趨勢與動能濾網，不是完整成交契約。不複製原程式。

- OU [QuantConnect Optimal Pairs Trading](https://www.quantconnect.com/research/15294/optimal-pairs-trading/)：OU配對模型背景，股票結果不直接移植BTC。僅書目，未匯入外部程式。

- KALMAN [QuantConnect Kalman Pairs Trading](https://www.quantconnect.com/forum/discussion/6826/from-research-to-production-kalman-filters-and-pairs-trading/p1)：動態避險比例的研究入口，非本地已驗證策略。僅書目；程式及引用來源授權另核對。

- PCA [QuantConnect PCA and Pairs Trading](https://www.quantconnect.com/docs/v2/research-environment/applying-research/pca-and-pairs-trading)：因子殘差背景；需歷史資產名單、訓練切分與多腿成本。僅書目與假說。

- HMM [QuantConnect Hidden Markov Models](https://www.quantconnect.com/docs/v2/research-environment/applying-research/hidden-markov-models)：市場狀態研究方法；不得使用看完未來的平滑狀態。僅書目與假說。

- AMM [Hummingbot AMM Arbitrage](https://hummingbot.org/strategies/v1-strategies/amm-arbitrage/)：跨AMM與交易所價差及gas背景，沒有執行或部署。只存連結與摘要。

- PF [StockCharts Point and Figure Patterns](https://chartschool.stockcharts.com/table-of-contents/chart-analysis/point-and-figure-charts/classic-patterns)：點數圖背景；不能把合成格價格當真實成交。只存連結與摘要。



## 別名與相關性



- Golden Cross／Death Cross → T01：方向是同一均線交叉機制，不另加兩套。

- Turtle通道入場 → B01：只對應通道進場概念；完整海龜加碼與風控尚未重現。

- BB Squeeze／布林收口 → B03：同家族，參數及退出不同需契約版本。

- SMC流動性掃單價格代理 → S01：目前僅價格假突破，不能宣稱知道止損訂單。

- FVG／Fair Value Gap → S08：同三棒區域假說，不額外計數。

- Order Flow Delta代理 → V05：棒級taker量差與真實Footprint分開。

- BTC/ETH配對 → X04：同協整群；換回看期不是新策略。

- AS做市 → Q04：同庫存報價模型。

- 不同均線長度／不同K線週期 → T01：代表同機制參數，不自動計為獨立策略。

- W底／M頂 → C01：方向為同一確認式雙端點機制。

- 牛旗／熊旗 → C04：方向不另計數。

- 冰山偵測 → L06：僅列可觀測補單代理，不能宣稱識別隱藏訂單。

- 動態Beta配對 → J08：同動態避險比率假說。



## 共同缺件



- 除明列既有實作外，尚需定稿指標公式、參數、決策時間、訊號衝突與退出契約

- 新條目必須先核對歷史語意重複，再凍結資料、規則、成本與嘗試預算

- 所有已研究歷史只能稱研究測試，不得冒充全新final holdout

- 來源僅支持機制或指標定義，不代表本目錄的本地改編有正期望

- 同家族、同進出場不同參數、長短方向及不同週期不可算成獨立策略證據
