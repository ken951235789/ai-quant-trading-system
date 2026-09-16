# 可攜式訓練模組

BTC 環境建構預設使用 5 根預期報酬契約與最多 112 個精簡特徵，與主程式共用挑選器。
`BTCEnvironmentSettings.expected_return_horizon` 與 `feature_budget` 會於建立時驗證。
新特徵需重訓 Transformer 與 SAC，並重新產生可攜包；舊 ZIP 不會自動更新。

此模組只負責 BTC 多週期資料、Transformer、SAC/PPO、樣本外回測與模型匯出。
它不讀取交易所 API Key，也不包含下單、模擬倉或實盤服務。

請使用 `scripts/build_training_bundle.py` 產生可複製到其他 Windows 電腦的完整訓練包。
