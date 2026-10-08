# 研究與事件重播

既有 `replay.py`、`faults.py`、`reporting.py` 負責歷史／即時共用重播、故障注入與報告。
它們的既有行為沒有因下列模型研究而修改。

## 分階段模型研究

`staged_training.py` 固定 Transformer 與資金配置的交易定義，先跑多種子、跨時間
樣本外預測，再依較早 OOS Train／Validation 決定是否允許訓練 SAC。
`sizing.py` 只研究固定候選事件的入場資金比例，沿用既有事件回放產生的方向、
成交價格、退出及成本。它不是新的實盤撮合器，也沒有自由出場、槓桿或棒內清算能力。

本研究的 horizon 是既有事件模型最長 32 棒，不是舊 SAC 的 5 棒，也不是一般 V3 的
20 棒主要目標。欄位表示名目本金的交易淨報酬，不能塞到舊模型預期毛收益欄位。
完整契約、模型雜湊、資料邊界與原始設定會寫入 `plan.json` 與 OOS CSV。

主要指令：`scripts/run_staged_training.py`；事後核對：`scripts/audit_staged_training.py`。
詳細設計與操作見 [分階段模型研究](../../../docs/STAGED_TRAINING_20260930.md)。

## 候選契約與成本歸因

`candidate_contract.py` 將原突破、趨勢回調與 VWAP 均值回歸接入既有 Transformer 事件訓練。
`event_attribution.py` 提供固定路徑成本拆解與 MFE/MAE 保守界線；不可作模型輸入。
`candidate_evaluation.py` 比較無 AI、Train-only 統計濾網與 Transformer，成本情境各自重播。
`scripts/run_candidate_training_research.py` 預設僅產生計畫，`--smoke` 才執行本機一輪測試。
每個模型固定一種策略；研究模型不能直接用於 SAC／Paper／Live。
操作與限制見 [候選契約研究](../../../docs/CANDIDATE_TRAINING_CONTRACT_20261003.md)。

## 候選特徵研究新增入口

`candidate_features.py` 提供固定精簡、成交方向/活動、交易空間/訊號時效特徵。
沿用 `scripts/run_candidate_training_research.py --feature-study`，沒有第二套訓練引擎。
中文操作、五組比較與 SAC 前置條件見 [特徵研究](../../../docs/CANDIDATE_FEATURE_STUDY_20261003.md)。

## 機率門檻敏感度

`probability_sweep.py` 使用已保存模型預測，重用候選撮合與不重疊排程，研究 0% 至 100% 門檻。
`scripts/run_probability_sweep.py` 先稽核輸入，再輸出逐模型、成本與區間比較，不變更正式設定。
操作、成本假設及結算權益限制見 [門檻研究](../../../docs/PROBABILITY_THRESHOLD_SWEEP.md)。

## 逐筆進出場圖

`trade_review.py` 與 `trade_review.html` 將既有排程轉為離線 K 線檢視器，不重新撮合。
`scripts/render_trade_review.py` 先核對資料 / 契約雜湊、帳務、時間戳與已保存統計，
再標出進出場、固定停損停利、預定盈虧比及本筆扣成本後的 R。
操作與可重現命令見 [逐筆圖表](../../../docs/TRADE_ENTRY_EXIT_REVIEW.md)。

## 訓練交接與退出研究

- `candidate_handoff.py`：從既有 Dataset 保存全候選、序列索引與分離標籤，核對來源、scaler 及 OOS 時間。
- `exit_selection.py`：相同候選下選擇不同 R 契約的模型預測，保持 55% 與 2bps 門檻；只輸出研究提案，不送單。
- `account_replay.py`：固定成交路徑逐棒權益、持倉浮虧、成本一致性；不聲稱已驗證標記價格清算或真實 funding。

操作指令與限制見 [交接說明](../../../docs/JOURNAL_RESEARCH_20261008.md)。
