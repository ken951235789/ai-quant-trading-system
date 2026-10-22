> 公開研究快照：文內原始碼行號與歷史文件名稱是本機研究版定位，可能不在此公開版本；不是可直接執行的策略或訓練設定。

# 訓練特徵細分類



41細群、210個欄位或候選名稱；本次實際啟用0。

字串存在只證明原始碼引用，不證明已計算、已選入模型、資料可得或沒有洩漏；未找到也不證明動態欄位不存在。



Transformer/SAC允許表示角色條件下可研究，不代表已被目前模型選用。

MTF名稱先解析週期再分類；不是把每群複製九次。



## 報酬與變化速度

角色：因果市場輸入候選。決策時可得、特徵契約相容、資料覆蓋通過；不是自動啟用

可用時間：當根收盤後。尺度：報酬比例或對數報酬；重複表達需消融。

風險：長短窗高度相關，不全部塞入。



|欄位|原始碼對照|Transformer候選|SAC候選|

|---|---|---|---|

|return_1|src/ai_quant_trading/features/indicators.py:30、src/ai_quant_trading/features/indicators.py:109|True|True|

|log_return_1|src/ai_quant_trading/features/intraday.py:10、src/ai_quant_trading/features/intraday.py:187|True|True|

|return_3|src/ai_quant_trading/features/intraday.py:11|True|True|

|return_5|src/ai_quant_trading/features/intraday.py:12、src/ai_quant_trading/features/multitimeframe.py:61|True|True|

|return_20|src/ai_quant_trading/features/intraday.py:13|True|True|

|roc_12_pct|src/ai_quant_trading/features/indicators.py:43、src/ai_quant_trading/features/indicators.py:141|True|True|



## K線實體與範圍

角色：因果市場輸入候選。決策時可得、特徵契約相容、資料覆蓋通過；不是自動啟用

可用時間：收盤後，不可提前知道high/low。尺度：價格比率；同義欄位擇一。

風險：同義欄位會重複加權。



|欄位|原始碼對照|Transformer候選|SAC候選|

|---|---|---|---|

|candle_body_pct|src/ai_quant_trading/features/intraday.py:14、src/ai_quant_trading/features/intraday.py:193|True|True|

|range_pct|src/ai_quant_trading/features/intraday.py:15、src/ai_quant_trading/features/intraday.py:194|True|True|

|price_body_ratio|src/ai_quant_trading/features/market_context.py:29、src/ai_quant_trading/features/market_context.py:98|True|True|

|price_range_ratio|src/ai_quant_trading/features/market_context.py:30、src/ai_quant_trading/features/market_context.py:99|True|True|



## 影線與收盤位置

角色：因果市場輸入候選。決策時可得、特徵契約相容、資料覆蓋通過；不是自動啟用

可用時間：完整收盤。尺度：以全棒範圍或價格縮放。

風險：零範圍需遮罩；不等於真實吸收。



|欄位|原始碼對照|Transformer候選|SAC候選|

|---|---|---|---|

|upper_wick_pct|src/ai_quant_trading/features/intraday.py:16、src/ai_quant_trading/features/intraday.py:195|True|True|

|lower_wick_pct|src/ai_quant_trading/features/intraday.py:17、src/ai_quant_trading/features/intraday.py:196|True|True|

|close_location_value|src/ai_quant_trading/features/intraday.py:18、src/ai_quant_trading/features/intraday.py:197|True|True|

|price_close_location|src/ai_quant_trading/features/market_context.py:31、src/ai_quant_trading/features/market_context.py:100|True|True|

|wick_imbalance|src/ai_quant_trading/features/multitimeframe.py:101|True|True|



## 均線距離

角色：因果市場輸入候選。決策時可得、特徵契約相容、資料覆蓋通過；不是自動啟用

可用時間：對應週期已收盤且暖機完整。尺度：比例或ATR倍數，擇一版本。

風險：不能與原始均線價格重複混入。



|欄位|原始碼對照|Transformer候選|SAC候選|

|---|---|---|---|

|ema_20_gap|src/ai_quant_trading/features/intraday.py:19、src/ai_quant_trading/features/intraday.py:202|True|True|

|ema_50_gap|src/ai_quant_trading/features/intraday.py:20、src/ai_quant_trading/features/intraday.py:203|True|True|

|ema_200_gap|src/ai_quant_trading/features/intraday.py:21、src/ai_quant_trading/features/intraday.py:204|True|True|

|ema_20_atr|src/ai_quant_trading/features/multitimeframe.py:40|True|True|

|ema_50_atr|src/ai_quant_trading/features/multitimeframe.py:41|True|True|

|ema_200_atr|src/ai_quant_trading/features/multitimeframe.py:42|True|True|

|short_ema_5_gap|src/ai_quant_trading/features/indicators.py:22、src/ai_quant_trading/features/indicators.py:251|True|True|

|short_ema_200_gap|src/ai_quant_trading/features/indicators.py:23、src/ai_quant_trading/features/indicators.py:252|True|True|



## 趨勢斜率與排列

角色：因果市場輸入候選。決策時可得、特徵契約相容、資料覆蓋通過；不是自動啟用

可用時間：只用過去窗口。尺度：比例變化或ATR尺度。

風險：斜率定義不同需獨立版本。



|欄位|原始碼對照|Transformer候選|SAC候選|

|---|---|---|---|

|ema_20_slope_5|src/ai_quant_trading/features/intraday.py:24、src/ai_quant_trading/features/intraday.py:207|True|True|

|linear_regression_slope_20|src/ai_quant_trading/features/intraday.py:33、src/ai_quant_trading/features/intraday.py:224|True|True|

|ema_20_50_atr|src/ai_quant_trading/features/intraday.py:22、src/ai_quant_trading/features/intraday.py:205|True|True|

|ema_50_200_atr|src/ai_quant_trading/features/intraday.py:23、src/ai_quant_trading/features/intraday.py:206|True|True|



## 方向強度與效率

角色：因果市場輸入候選。決策時可得、特徵契約相容、資料覆蓋通過；不是自動啟用

可用時間：收盤後；初始化暖機。尺度：核對0至1或0至100原契約。

風險：ADX不代表方向，不能自行改尺度。



|欄位|原始碼對照|Transformer候選|SAC候選|

|---|---|---|---|

|adx_14|src/ai_quant_trading/features/intraday.py:25、src/ai_quant_trading/features/intraday.py:208|True|True|

|plus_di_14|src/ai_quant_trading/features/intraday.py:26、src/ai_quant_trading/features/intraday.py:209|True|True|

|minus_di_14|src/ai_quant_trading/features/intraday.py:27、src/ai_quant_trading/features/intraday.py:210|True|True|

|choppiness_14|src/ai_quant_trading/features/intraday.py:41、src/ai_quant_trading/features/intraday.py:277|True|True|

|efficiency_ratio_candidate|未找到同名字串，候選或動態／內部狀態待核對|True|True|



## 趨勢線距離

角色：因果市場輸入候選。決策時可得、特徵契約相容、資料覆蓋通過；不是自動啟用

可用時間：前棒可得保護線。尺度：方向與ATR距離。

風險：欄名不保證實際ATR窗口，需核對實作。



|欄位|原始碼對照|Transformer候選|SAC候選|

|---|---|---|---|

|supertrend_direction_10_3|src/ai_quant_trading/features/intraday.py:28、src/ai_quant_trading/features/intraday.py:213|True|True|

|supertrend_distance_atr|src/ai_quant_trading/features/intraday.py:29、src/ai_quant_trading/features/intraday.py:214|True|True|

|psar_distance_atr_candidate|未找到同名字串，候選或動態／內部狀態待核對|True|True|



## 震盪位置

角色：因果市場輸入候選。決策時可得、特徵契約相容、資料覆蓋通過；不是自動啟用

可用時間：收盤後。尺度：固定有界尺度；只在訓練窗擬合其他轉換。

風險：超買不是必跌；相似指標逐組比較。



|欄位|原始碼對照|Transformer候選|SAC候選|

|---|---|---|---|

|rsi_14|src/ai_quant_trading/features/indicators.py:39、src/ai_quant_trading/features/indicators.py:126|True|True|

|short_rsi_2|src/ai_quant_trading/features/indicators.py:21、src/ai_quant_trading/features/indicators.py:250|True|True|

|stoch_rsi_k_14|src/ai_quant_trading/features/intraday.py:34、src/ai_quant_trading/features/intraday.py:239|True|True|

|stoch_rsi_d_3|src/ai_quant_trading/features/intraday.py:35、src/ai_quant_trading/features/intraday.py:240|True|True|

|cci_normalized_candidate|未找到同名字串，候選或動態／內部狀態待核對|True|True|

|williams_r_candidate|未找到同名字串，候選或動態／內部狀態待核對|True|True|



## 平滑動能與柱變化

角色：因果市場輸入候選。決策時可得、特徵契約相容、資料覆蓋通過；不是自動啟用

可用時間：對應週期收盤。尺度：除價格或ATR，禁止用絕對價格捷徑。

風險：MACD派生欄相關度高。



|欄位|原始碼對照|Transformer候選|SAC候選|

|---|---|---|---|

|macd_pct|src/ai_quant_trading/features/market_context.py:25、src/ai_quant_trading/features/market_context.py:86|True|True|

|macd_signal_pct|src/ai_quant_trading/features/market_context.py:26、src/ai_quant_trading/features/market_context.py:87|True|True|

|macd_histogram_pct|src/ai_quant_trading/features/market_context.py:27、src/ai_quant_trading/features/market_context.py:88|True|True|

|macd_histogram_atr|src/ai_quant_trading/features/multitimeframe.py:64|True|True|

|tsi_candidate|未找到同名字串，候選或動態／內部狀態待核對|True|True|



## 確認式背離與失敗擺動

角色：因果市場輸入候選。決策時可得、特徵契約相容、資料覆蓋通過；不是自動啟用

可用時間：右側pivot確認後，不回填pivot當時。尺度：方向、強度及確認後經過棒數。

風險：本群多為待實作，不能用事後ZigZag。



|欄位|原始碼對照|Transformer候選|SAC候選|

|---|---|---|---|

|rsi_price_divergence_confirmed_candidate|未找到同名字串，候選或動態／內部狀態待核對|True|True|

|macd_price_divergence_confirmed_candidate|未找到同名字串，候選或動態／內部狀態待核對|True|True|

|failure_swing_confirmed_candidate|未找到同名字串，候選或動態／內部狀態待核對|True|True|

|pivot_age_bars_candidate|未找到同名字串，候選或動態／內部狀態待核對|True|True|



## 波動幅度

角色：因果市場輸入候選。決策時可得、特徵契約相容、資料覆蓋通過；不是自動啟用

可用時間：只用截至當時報酬。尺度：百分比與年化慣例明確分開。

風險：不是未來波動；OHLC估計各有假設。



|欄位|原始碼對照|Transformer候選|SAC候選|

|---|---|---|---|

|atr_pct|src/ai_quant_trading/features/market_context.py:28、src/ai_quant_trading/features/market_context.py:89|True|True|

|historical_volatility_20|src/ai_quant_trading/features/indicators.py:49、src/ai_quant_trading/features/indicators.py:175|True|True|

|realized_volatility_20|src/ai_quant_trading/features/intraday.py:37、src/ai_quant_trading/features/intraday.py:257|True|True|

|parkinson_volatility_20|src/ai_quant_trading/features/intraday.py:38、src/ai_quant_trading/features/intraday.py:261|True|True|



## 壓縮與波動狀態

角色：因果市場輸入候選。決策時可得、特徵契約相容、資料覆蓋通過；不是自動啟用

可用時間：分位數不得用全樣本。尺度：歷史排名或訓練窗門檻。

風險：低波動不代表必定突破。



|欄位|原始碼對照|Transformer候選|SAC候選|

|---|---|---|---|

|bb_width_20|src/ai_quant_trading/features/indicators.py:48、src/ai_quant_trading/features/indicators.py:170|True|True|

|squeeze_on|src/ai_quant_trading/features/intraday.py:40、src/ai_quant_trading/features/intraday.py:266|True|True|

|high_volatility_regime|src/ai_quant_trading/features/multitimeframe.py:124|True|True|

|low_volatility_regime|src/ai_quant_trading/features/multitimeframe.py:125|True|True|

|volatility_percentile_candidate|未找到同名字串，候選或動態／內部狀態待核對|True|True|



## 通道位置與偏離

角色：因果市場輸入候選。決策時可得、特徵契約相容、資料覆蓋通過；不是自動啟用

可用時間：通道邊界是否排除當棒須固定。尺度：Z分數、通道相對位置。

風險：相關欄位重複，不各算獨立優勢。



|欄位|原始碼對照|Transformer候選|SAC候選|

|---|---|---|---|

|bb_zscore_20|src/ai_quant_trading/features/market_context.py:135、src/ai_quant_trading/features/multitimeframe.py:72|True|True|

|short_bb_zscore_20|src/ai_quant_trading/features/indicators.py:24、src/ai_quant_trading/features/indicators.py:253|True|True|

|keltner_position_20|src/ai_quant_trading/features/intraday.py:39、src/ai_quant_trading/features/intraday.py:264|True|True|

|donchian_position_20|src/ai_quant_trading/features/intraday.py:30、src/ai_quant_trading/features/intraday.py:218|True|True|



## 已知價位與區間位置

角色：因果市場輸入候選。決策時可得、特徵契約相容、資料覆蓋通過；不是自動啟用

可用時間：前日／前週必須完整結束。尺度：ATR距離或區間比率。

風險：當日最終高低不可提前使用。



|欄位|原始碼對照|Transformer候選|SAC候選|

|---|---|---|---|

|previous_high_20_atr|src/ai_quant_trading/features/multitimeframe.py:105|True|True|

|previous_low_20_atr|src/ai_quant_trading/features/multitimeframe.py:106|True|True|

|range_position_20|src/ai_quant_trading/features/multitimeframe.py:107|True|True|

|previous_day_high_distance_candidate|未找到同名字串，候選或動態／內部狀態待核對|True|True|

|previous_day_low_distance_candidate|未找到同名字串，候選或動態／內部狀態待核對|True|True|



## 確認擺動與結構方向

角色：因果市場輸入候選。決策時可得、特徵契約相容、資料覆蓋通過；不是自動啟用

可用時間：原SMC右側確認後再依原契約可用。尺度：距離ATR、方向離散值。

風險：相同名稱可能有不同確認延遲。



|欄位|原始碼對照|Transformer候選|SAC候選|

|---|---|---|---|

|smc_swing_high_distance_atr|src/ai_quant_trading/features/smc.py:12、src/ai_quant_trading/features/smc.py:107|True|True|

|smc_swing_low_distance_atr|src/ai_quant_trading/features/smc.py:13、src/ai_quant_trading/features/smc.py:108|True|True|

|smc_structure_direction|src/ai_quant_trading/features/smc.py:14、src/ai_quant_trading/features/smc.py:109|True|True|

|higher_high|src/ai_quant_trading/features/indicators.py:53、src/ai_quant_trading/features/indicators.py:200|True|True|

|higher_low|src/ai_quant_trading/features/indicators.py:54、src/ai_quant_trading/features/indicators.py:201|True|True|



## 突破與結構改變事件

角色：因果市場輸入候選。決策時可得、特徵契約相容、資料覆蓋通過；不是自動啟用

可用時間：收盤確認；不得向前複製高週期訊號。尺度：當棒事件與持續狀態分開。

風險：多條事件可能同筆交易。



|欄位|原始碼對照|Transformer候選|SAC候選|

|---|---|---|---|

|donchian_breakout_up_20|src/ai_quant_trading/features/intraday.py:31、src/ai_quant_trading/features/intraday.py:219|True|True|

|donchian_breakout_down_20|src/ai_quant_trading/features/intraday.py:32、src/ai_quant_trading/features/intraday.py:220|True|True|

|breakout_20|src/ai_quant_trading/features/indicators.py:55、src/ai_quant_trading/features/indicators.py:205|True|True|

|smc_bos_up|src/ai_quant_trading/features/smc.py:15、src/ai_quant_trading/features/smc.py:110|True|True|

|smc_bos_down|src/ai_quant_trading/features/smc.py:16、src/ai_quant_trading/features/smc.py:111|True|True|

|smc_choch_up|src/ai_quant_trading/features/smc.py:17、src/ai_quant_trading/features/smc.py:112|True|True|

|smc_choch_down|src/ai_quant_trading/features/smc.py:18、src/ai_quant_trading/features/smc.py:113|True|True|



## 價格掃過與失衡區代理

角色：因果市場輸入候選。決策時可得、特徵契約相容、資料覆蓋通過；不是自動啟用

可用時間：完整形態形成後。尺度：方向與ATR。

風險：僅價格型態，不宣稱知道主力或未成交單。



|欄位|原始碼對照|Transformer候選|SAC候選|

|---|---|---|---|

|smc_liquidity_sweep_high|src/ai_quant_trading/features/smc.py:19、src/ai_quant_trading/features/smc.py:114|True|True|

|smc_liquidity_sweep_low|src/ai_quant_trading/features/smc.py:20、src/ai_quant_trading/features/smc.py:117|True|True|

|smc_range_position_20|src/ai_quant_trading/features/smc.py:21、src/ai_quant_trading/features/smc.py:120|True|True|

|smc_bullish_fvg_atr|src/ai_quant_trading/features/smc.py:22、src/ai_quant_trading/features/smc.py:123|True|True|

|smc_bearish_fvg_atr|src/ai_quant_trading/features/smc.py:23、src/ai_quant_trading/features/smc.py:124|True|True|



## 相對成交量與異常

角色：因果市場輸入候選。決策時可得、特徵契約相容、資料覆蓋通過；不是自動啟用

可用時間：當棒總量收盤後；分母定義另核對。尺度：相對歷史均量或Z分數。

風險：不同市場成交量不可混用。



|欄位|原始碼對照|Transformer候選|SAC候選|

|---|---|---|---|

|volume_change|src/ai_quant_trading/features/indicators.py:50、src/ai_quant_trading/features/indicators.py:184|True|True|

|relative_volume_20|src/ai_quant_trading/features/intraday.py:42、src/ai_quant_trading/features/intraday.py:284|True|True|

|volume_zscore_20|src/ai_quant_trading/features/intraday.py:43、src/ai_quant_trading/features/intraday.py:285|True|True|

|volume_oscillator_5_20|src/ai_quant_trading/features/intraday.py:46、src/ai_quant_trading/features/intraday.py:295|True|True|



## 量價累積代理

角色：因果市場輸入候選。決策時可得、特徵契約相容、資料覆蓋通過；不是自動啟用

可用時間：收盤後。尺度：有界量價比率／斜率。

風險：不是帳戶資金流或主動買賣。



|欄位|原始碼對照|Transformer候選|SAC候選|

|---|---|---|---|

|mfi_14|src/ai_quant_trading/features/intraday.py:36、src/ai_quant_trading/features/intraday.py:251|True|True|

|cmf_20|src/ai_quant_trading/features/intraday.py:44、src/ai_quant_trading/features/intraday.py:288|True|True|

|ad_line_slope_20|src/ai_quant_trading/features/intraday.py:45、src/ai_quant_trading/features/intraday.py:292|True|True|

|obv_pressure_5|src/ai_quant_trading/features/multitimeframe.py:88|True|True|



## VWAP與錨定位置

角色：因果市場輸入候選。決策時可得、特徵契約相容、資料覆蓋通過；不是自動啟用

可用時間：只累積到決策時，錨點須事前已知。尺度：ATR或價格比率。

風險：K線近似VWAP與逐筆VWAP分開。



|欄位|原始碼對照|Transformer候選|SAC候選|

|---|---|---|---|

|rolling_vwap_gap_atr|src/ai_quant_trading/features/intraday.py:47、src/ai_quant_trading/features/intraday.py:299|True|True|

|vwap_gap_20|src/ai_quant_trading/features/multitimeframe.py:87|True|True|

|anchored_vwap_gap_candidate|未找到同名字串，候選或動態／內部狀態待核對|True|True|

|session_vwap_band_position_candidate|未找到同名字串，候選或動態／內部狀態待核對|True|True|



## 主動量差與CVD

角色：因果市場輸入候選。決策時可得、特徵契約相容、資料覆蓋通過；不是自動啟用

可用時間：棒級資料收盤可得；缺值遮罩。尺度：同單位買賣量比例。

風險：棒級代理不是逐價Footprint。



|欄位|原始碼對照|Transformer候選|SAC候選|

|---|---|---|---|

|taker_buy_ratio|src/ai_quant_trading/features/intraday.py:48、src/ai_quant_trading/features/intraday.py:307|True|True|

|volume_delta_ratio|src/ai_quant_trading/features/intraday.py:49、src/ai_quant_trading/features/intraday.py:308|True|True|

|cvd_pressure_20|src/ai_quant_trading/features/intraday.py:50、src/ai_quant_trading/features/intraday.py:309|True|True|



## 成交速度與大小

角色：因果市場輸入候選。決策時可得、特徵契約相容、資料覆蓋通過；不是自動啟用

可用時間：逐筆或完整棒成交統計。尺度：相對過去強度。

風險：交易筆數不是委託撤單數。



|欄位|原始碼對照|Transformer候選|SAC候選|

|---|---|---|---|

|trade_intensity_20|src/ai_quant_trading/features/intraday.py:51、src/ai_quant_trading/features/intraday.py:311|True|True|

|average_trade_size_ratio_20|src/ai_quant_trading/features/intraday.py:52、src/ai_quant_trading/features/intraday.py:313|True|True|

|large_trade_fraction_candidate|未找到同名字串，候選或動態／內部狀態待核對|True|True|



## 流動性與成本尺度

角色：因果市場輸入候選。決策時可得、特徵契約相容、資料覆蓋通過；不是自動啟用

可用時間：決策前有效bid/ask與深度。尺度：價格比例、ATR成本比。

風險：收盤代理不能替代真實可成交深度。



|欄位|原始碼對照|Transformer候選|SAC候選|

|---|---|---|---|

|amihud_20|src/ai_quant_trading/features/intraday.py:53、src/ai_quant_trading/features/intraday.py:317|True|True|

|spread_fraction|src/ai_quant_trading/features/intraday.py:59、src/ai_quant_trading/features/intraday.py:337|True|True|

|spread_to_atr_candidate|未找到同名字串，候選或動態／內部狀態待核對|True|True|

|depth_imbalance_candidate|未找到同名字串，候選或動態／內部狀態待核對|True|True|



## 訂單簿補單與撤單

角色：因果市場輸入候選。決策時可得、特徵契約相容、資料覆蓋通過；不是自動啟用

可用時間：完整序號的L2增量與接收時間。尺度：深度或時間尺度。

風險：本群待補事件資料，不能用K線推定。



|欄位|原始碼對照|Transformer候選|SAC候選|

|---|---|---|---|

|book_replenishment_candidate|未找到同名字串，候選或動態／內部狀態待核對|True|True|

|cancel_intensity_candidate|未找到同名字串，候選或動態／內部狀態待核對|True|True|

|microprice_gap_candidate|未找到同名字串，候選或動態／內部狀態待核對|True|True|

|order_flow_imbalance_candidate|未找到同名字串，候選或動態／內部狀態待核對|True|True|



## 資金費與相對極端

角色：因果市場輸入候選。決策時可得、特徵契約相容、資料覆蓋通過；不是自動啟用

可用時間：區分已公布預估與結算後實際費率。尺度：有符號費率、歷史排名、剩餘時間。

風險：歷史實際費可作帳務，不一定可提前作預測。



|欄位|原始碼對照|Transformer候選|SAC候選|

|---|---|---|---|

|funding_rate|src/ai_quant_trading/features/intraday.py:322、src/ai_quant_trading/features/multitimeframe.py:141|True|True|

|funding_percentile_200|src/ai_quant_trading/features/intraday.py:55、src/ai_quant_trading/features/intraday.py:325|True|True|

|funding_zscore_200|src/ai_quant_trading/features/intraday.py:56、src/ai_quant_trading/features/intraday.py:326|True|True|

|minutes_to_funding_ratio|src/ai_quant_trading/features/intraday.py:69、src/ai_quant_trading/features/intraday.py:355|True|True|



## 持倉量與槓桿活動代理

角色：因果市場輸入候選。決策時可得、特徵契約相容、資料覆蓋通過；不是自動啟用

可用時間：來源公布後backward as-of。尺度：相對變化，不用絕對市值。

風險：OI不識別多空方向，歷史覆蓋未證明。



|欄位|原始碼對照|Transformer候選|SAC候選|

|---|---|---|---|

|open_interest_change_1|src/ai_quant_trading/features/intraday.py:57、src/ai_quant_trading/features/intraday.py:334|True|True|

|open_interest_change|src/ai_quant_trading/features/intraday.py:331、src/ai_quant_trading/features/intraday.py:339|True|True|

|open_interest_zscore_50|src/ai_quant_trading/features/intraday.py:58、src/ai_quant_trading/features/intraday.py:335|True|True|

|oi_price_joint_state_candidate|未找到同名字串，候選或動態／內部狀態待核對|True|True|



## 基差與溢價

角色：因果市場輸入候選。決策時可得、特徵契約相容、資料覆蓋通過；不是自動啟用

可用時間：各來源時間與可交易性一致。尺度：同時間同商品比例。

風險：mark不可成交；跨所接收延遲可能是假訊號。



|欄位|原始碼對照|Transformer候選|SAC候選|

|---|---|---|---|

|basis_rate|src/ai_quant_trading/features/advanced.py:307|True|True|

|premium_index_candidate|未找到同名字串，候選或動態／內部狀態待核對|True|True|

|mark_index_gap_candidate|未找到同名字串，候選或動態／內部狀態待核對|True|True|

|term_slope_candidate|未找到同名字串，候選或動態／內部狀態待核對|True|True|



## 清算與擁擠代理

角色：因果市場輸入候選。決策時可得、特徵契約相容、資料覆蓋通過；不是自動啟用

可用時間：只使用當時收到的事件。尺度：按成交量或OI尺度。

風險：清算資料可能抽樣；帳戶比率不是持倉比例。



|欄位|原始碼對照|Transformer候選|SAC候選|

|---|---|---|---|

|liquidation_long_volume_candidate|未找到同名字串，候選或動態／內部狀態待核對|True|True|

|liquidation_short_volume_candidate|未找到同名字串，候選或動態／內部狀態待核對|True|True|

|long_short_ratio_candidate|未找到同名字串，候選或動態／內部狀態待核對|True|True|



## 時鐘與交易時段

角色：因果市場輸入候選。決策時可得、特徵契約相容、資料覆蓋通過；不是自動啟用

可用時間：事前已知日曆，UTC與DST明確。尺度：週期編碼與二元狀態。

風險：未經資料證明不能把時段當必然優勢。



|欄位|原始碼對照|Transformer候選|SAC候選|

|---|---|---|---|

|hour_sin|src/ai_quant_trading/features/intraday.py:61、src/ai_quant_trading/features/intraday.py:346|True|True|

|hour_cos|src/ai_quant_trading/features/intraday.py:62、src/ai_quant_trading/features/intraday.py:347|True|True|

|weekday_sin|src/ai_quant_trading/features/intraday.py:63、src/ai_quant_trading/features/intraday.py:348|True|True|

|weekday_cos|src/ai_quant_trading/features/intraday.py:64、src/ai_quant_trading/features/intraday.py:349|True|True|

|is_weekend|src/ai_quant_trading/features/intraday.py:65、src/ai_quant_trading/features/intraday.py:350|True|True|

|asia_session|src/ai_quant_trading/features/intraday.py:66、src/ai_quant_trading/features/intraday.py:351|True|True|

|europe_session|src/ai_quant_trading/features/intraday.py:67、src/ai_quant_trading/features/intraday.py:352|True|True|

|us_session|src/ai_quant_trading/features/intraday.py:68、src/ai_quant_trading/features/intraday.py:353|True|True|



## 跨週期共識與衝突

角色：因果市場輸入候選。決策時可得、特徵契約相容、資料覆蓋通過；不是自動啟用

可用時間：每個來源週期都已收盤。尺度：方向分布、分歧及熵。

風險：分歧是資訊，不應強迫所有週期同向。



|欄位|原始碼對照|Transformer候選|SAC候選|

|---|---|---|---|

|market_trend_consensus|src/ai_quant_trading/features/market_context.py:14、src/ai_quant_trading/features/market_context.py:171|True|True|

|market_fast_trend|src/ai_quant_trading/features/market_context.py:15、src/ai_quant_trading/features/market_context.py:172|True|True|

|market_slow_trend|src/ai_quant_trading/features/market_context.py:16、src/ai_quant_trading/features/market_context.py:173|True|True|

|market_cross_timeframe_disagreement|src/ai_quant_trading/features/market_context.py:17、src/ai_quant_trading/features/market_context.py:174|True|True|

|market_direction_entropy|src/ai_quant_trading/features/market_context.py:18、src/ai_quant_trading/features/market_context.py:178|True|True|

|market_reversal_pressure|src/ai_quant_trading/features/market_context.py:19、src/ai_quant_trading/features/market_context.py:179|True|True|

|market_trend_reversal_conflict|src/ai_quant_trading/features/market_context.py:20、src/ai_quant_trading/features/market_context.py:180|True|True|

|market_fast_slow_conflict|src/ai_quant_trading/features/market_context.py:21、src/ai_quant_trading/features/market_context.py:181|True|True|



## 統計狀態與估計信心

角色：因果市場輸入候選。決策時可得、特徵契約相容、資料覆蓋通過；不是自動啟用

可用時間：只能filtered，不能smoothed或事後分段。尺度：只用過去估計的尺度。

風險：新增模型也需樣本外估計，不能用全資料擬合。



|欄位|原始碼對照|Transformer候選|SAC候選|

|---|---|---|---|

|kalman_innovation_candidate|未找到同名字串，候選或動態／內部狀態待核對|True|True|

|ou_half_life_candidate|未找到同名字串，候選或動態／內部狀態待核對|True|True|

|filtered_regime_probability_candidate|未找到同名字串，候選或動態／內部狀態待核對|True|True|

|change_point_probability_candidate|未找到同名字串，候選或動態／內部狀態待核對|True|True|



## 跨資產價差與領先

角色：因果市場輸入候選。決策時可得、特徵契約相容、資料覆蓋通過；不是自動啟用

可用時間：同步且歷史可交易名單。尺度：訓練窗比率與因子殘差。

風險：接收延遲與存活偏差。



|欄位|原始碼對照|Transformer候選|SAC候選|

|---|---|---|---|

|pair_spread_zscore_candidate|未找到同名字串，候選或動態／內部狀態待核對|True|True|

|hedge_ratio_candidate|未找到同名字串，候選或動態／內部狀態待核對|True|True|

|cross_asset_lead_lag_candidate|未找到同名字串，候選或動態／內部狀態待核對|True|True|

|breadth_available|src/ai_quant_trading/features/advanced.py:104|True|True|



## 新聞／宏觀／鏈上發布

角色：因果市場輸入候選。決策時可得、特徵契約相容、資料覆蓋通過；不是自動啟用

可用時間：首次可得時間，不用修訂版本回填。尺度：只用訓練窗尺度及發布前共識。

風險：此處列欄位候選，不代表資料或FinBERT已接通。



|欄位|原始碼對照|Transformer候選|SAC候選|

|---|---|---|---|

|sentiment_score|未找到同名字串，候選或動態／內部狀態待核對|True|True|

|macro_surprise_candidate|未找到同名字串，候選或動態／內部狀態待核對|True|True|

|etf_flow_surprise_candidate|未找到同名字串，候選或動態／內部狀態待核對|True|True|

|exchange_netflow_candidate|未找到同名字串，候選或動態／內部狀態待核對|True|True|



## 可用性與資料新鮮度

角色：資料品質與缺值遮罩。缺值不等於零或中性，遮罩須和原始欄位一起驗證

可用時間：決策時可測得。尺度：遮罩與非負資料年齡。

風險：缺失填零必須保留遮罩，不得當安全市場。



|欄位|原始碼對照|Transformer候選|SAC候選|

|---|---|---|---|

|microstructure_available|src/ai_quant_trading/features/intraday.py:54、src/ai_quant_trading/features/intraday.py:320|True|True|

|derivatives_available|src/ai_quant_trading/features/intraday.py:60、src/ai_quant_trading/features/intraday.py:338|True|True|

|funding_available|src/ai_quant_trading/features/multitimeframe.py:117|True|True|

|open_interest_available|src/ai_quant_trading/features/multitimeframe.py:118|True|True|

|spread_available|src/ai_quant_trading/features/multitimeframe.py:119|True|True|

|market_timeframe_coverage|src/ai_quant_trading/features/market_context.py:22、src/ai_quant_trading/features/market_context.py:182|True|True|

|orderbook_available|src/ai_quant_trading/features/advanced.py:342|True|True|

|orderbook_age_days|src/ai_quant_trading/features/advanced.py:339|True|True|

|derivatives_age_days|src/ai_quant_trading/features/advanced.py:318|True|True|



## 候選身分與風險條件

角色：候選交易條件。事前固定條件，不得以事後最佳方向或停利替代

可用時間：進場前固定。尺度：版本化類別與比例。

風險：計畫盈虧比不是事後實現盈虧比。



|欄位|原始碼對照|Transformer候選|SAC候選|

|---|---|---|---|

|candidate_strategy_id_candidate|未找到同名字串，候選或動態／內部狀態待核對|True|True|

|candidate_side_candidate|未找到同名字串，候選或動態／內部狀態待核對|True|True|

|planned_stop_atr_candidate|未找到同名字串，候選或動態／內部狀態待核對|True|True|

|planned_reward_risk_candidate|未找到同名字串，候選或動態／內部狀態待核對|True|True|

|planned_holding_bars_candidate|未找到同名字串，候選或動態／內部狀態待核對|True|True|



## Transformer樣本外分析輸出

角色：合法樣本外模型輸出。必須附checkpoint、horizon、OOS時間及校準證據；禁止Transformer自吃輸出

可用時間：合法樣本外預測且早於SAC決策。尺度：依保存horizon、校準與單位契約。

風險：禁止Transformer自身輸入循環或訓練內預測給SAC。



|欄位|原始碼對照|Transformer候選|SAC候選|

|---|---|---|---|

|transformer_return_1|src/ai_quant_trading/transformer/features.py:9|False|True|

|transformer_return_5|src/ai_quant_trading/transformer/features.py:10|False|True|

|transformer_return_20|src/ai_quant_trading/transformer/features.py:11|False|True|

|transformer_volatility|src/ai_quant_trading/transformer/features.py:12|False|True|

|transformer_uncertainty|src/ai_quant_trading/transformer/features.py:15|False|True|

|transformer_bull_probability|src/ai_quant_trading/transformer/features.py:13|False|True|

|transformer_bear_probability|src/ai_quant_trading/transformer/features.py:14|False|True|

|transformer_available|src/ai_quant_trading/transformer/features.py:16、src/ai_quant_trading/transformer/features.py:58|False|True|

|expected_return|未找到同名字串，候選或動態／內部狀態待核對|False|True|



## 部位與持有狀態

角色：決策前帳戶狀態。本研究分工為SAC狀態；需與真實帳戶/模擬事件時間一致

可用時間：決策前帳戶快照。尺度：權益比例與事前風險尺度。

風險：既有環境有部分觀測但不是同名CSV欄；不可假稱已對接。



|欄位|原始碼對照|Transformer候選|SAC候選|

|---|---|---|---|

|position_fraction_candidate|未找到同名字串，候選或動態／內部狀態待核對|False|True|

|unrealized_return_candidate|未找到同名字串，候選或動態／內部狀態待核對|False|True|

|holding_bars_ratio_candidate|未找到同名字串，候選或動態／內部狀態待核對|False|True|

|distance_to_stop_candidate|未找到同名字串，候選或動態／內部狀態待核對|False|True|



## 風險預算與曝險

角色：決策前帳戶狀態。本研究分工為SAC狀態；需與真實帳戶/模擬事件時間一致

可用時間：決策前，獨立風控仍保留否決權。尺度：當前權益或核准限額。

風險：模型不能自行提高硬性風險上限。



|欄位|原始碼對照|Transformer候選|SAC候選|

|---|---|---|---|

|drawdown_fraction_candidate|未找到同名字串，候選或動態／內部狀態待核對|False|True|

|daily_loss_remaining_candidate|未找到同名字串，候選或動態／內部狀態待核對|False|True|

|margin_buffer_candidate|未找到同名字串，候選或動態／內部狀態待核對|False|True|

|net_exposure_candidate|未找到同名字串，候選或動態／內部狀態待核對|False|True|

|gross_exposure_candidate|未找到同名字串，候選或動態／內部狀態待核對|False|True|



## 成交與營運品質

角色：執行及營運診斷。本版不作模型輸入；事後成交品質不得回流當筆決策

可用時間：下單或成交後。尺度：只供診斷；未來若用延遲版本必須新契約。

風險：當筆成交後資訊不能預測當筆進場。



|欄位|原始碼對照|Transformer候選|SAC候選|

|---|---|---|---|

|realized_slippage_bps_candidate|未找到同名字串，候選或動態／內部狀態待核對|False|False|

|fill_ratio_candidate|未找到同名字串，候選或動態／內部狀態待核對|False|False|

|order_latency_ms_candidate|未找到同名字串，候選或動態／內部狀態待核對|False|False|

|reconnect_count_candidate|未找到同名字串，候選或動態／內部狀態待核對|False|False|



## 未來答案與交易後路徑

角色：未來標籤／事後診斷。可當答案或績效，不得作決策特徵

可用時間：持倉結束及觀察窗結束之後。尺度：依交易契約與標籤結束時間。

風險：只能當標籤或診斷，不可輸入Transformer或SAC。



|欄位|原始碼對照|Transformer候選|SAC候選|

|---|---|---|---|

|future_return|src/ai_quant_trading/features/indicators.py:296、src/ai_quant_trading/features/indicators.py:326|False|False|

|target|src/ai_quant_trading/features/indicators.py:301、src/ai_quant_trading/features/indicators.py:326|False|False|

|target_timestamp|src/ai_quant_trading/features/indicators.py:294|False|False|

|max_favorable_excursion|未找到同名字串，候選或動態／內部狀態待核對|False|False|

|max_adverse_excursion|未找到同名字串，候選或動態／內部狀態待核對|False|False|

|take_profit_hit|未找到同名字串，候選或動態／內部狀態待核對|False|False|

|stop_loss_hit|未找到同名字串，候選或動態／內部狀態待核對|False|False|



## 原始價量與絕對指標

角色：原始尺度／撮合資料。新版不直接當模型輸入，保留撮合與比例化計算；不刪原始資料

可用時間：行情收盤或報價收到後。尺度：先轉成比例或距離，新模型不直接使用。

風險：保留撮合資料不等於允許模型靠價格記憶時間。



|欄位|原始碼對照|Transformer候選|SAC候選|

|---|---|---|---|

|open|src/ai_quant_trading/features/indicators.py:94、src/ai_quant_trading/features/intraday.py:171|False|False|

|high|src/ai_quant_trading/features/indicators.py:94、src/ai_quant_trading/features/indicators.py:189|False|False|

|low|src/ai_quant_trading/features/indicators.py:94、src/ai_quant_trading/features/indicators.py:189|False|False|

|close|src/ai_quant_trading/features/indicators.py:94、src/ai_quant_trading/features/indicators.py:108|False|False|

|volume|src/ai_quant_trading/features/indicators.py:94、src/ai_quant_trading/features/indicators.py:184|False|False|

|sma_20|src/ai_quant_trading/features/indicators.py:32|False|False|

|ema_20|src/ai_quant_trading/features/indicators.py:36、src/ai_quant_trading/features/multitimeframe.py:410|False|False|

|atr_14|src/ai_quant_trading/features/indicators.py:44、src/ai_quant_trading/features/indicators.py:163|False|False|

|macd|src/ai_quant_trading/features/indicators.py:40、src/ai_quant_trading/features/indicators.py:130|False|False|

|obv|src/ai_quant_trading/features/indicators.py:51、src/ai_quant_trading/features/indicators.py:187|False|False|

|vwap|src/ai_quant_trading/features/indicators.py:52、src/ai_quant_trading/features/indicators.py:191|False|False|

|mark_price|src/ai_quant_trading/features/market_context.py:46、src/ai_quant_trading/features/advanced.py:296|False|False|

|index_price|src/ai_quant_trading/features/market_context.py:47、src/ai_quant_trading/features/advanced.py:309|False|False|



## 消融邊界

- max_groups_per_round：4

- baseline：沿用凍結的既有特徵與模型契約，不在目錄工具改動

- method：一次加入或移除一個細群；相同候選、成本、切分、seed與預算；看成本後收益和不確定性

- timeframe_rule：以mtf_<週期>_<特徵>對照；高週期收盤後才可用。不是所有群都複製九次；外部發布資料另用as-of

- stop：資料不足、未來洩漏、行為重複或跨期成本後無改善就停止；不因目錄欄位多就增加模型維度
