# 依賴鎖定

`windows-py313-cpu.txt` 是目前 Windows、Python 3.13、CPU 環境已通過測試的完整依賴閉包。
它是 constraints lock，不會取代 `pyproject.toml` 的功能分組。

安裝時使用：

```powershell
python -m pip install -e ".[dev,security,dashboard,ai,rl,database,package]" `
  -c requirements\locks\windows-py313-cpu.txt
```

重新產生目前環境的 lock：

```powershell
python scripts\export_dependency_lock.py `
  --output requirements\locks\windows-py313-cpu.txt
```

驗證目前環境是否仍與 lock 完全一致：

```powershell
python scripts\export_dependency_lock.py `
  --output requirements\locks\windows-py313-cpu.txt --check
```

CUDA 版必須在目標 GPU、CUDA 與 Python 版本都確定後，在乾淨虛擬環境內重新產生獨立 lock，
不要直接把 CPU 的 `torch` wheel 當成 CUDA 訓練環境。更新 lock 後必須重新執行完整測試與訓練煙霧測試。
