"""建立可複製到其他 Windows 電腦的獨立模型訓練包。"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "dist" / "AIQuantTrainingPortable"
BUNDLE_MARKER = ".ai_quant_training_bundle"
SOURCE_PACKAGES = (
    "ai_pipeline",
    "backtesting",
    "data_collection",
    "features",
    "reinforcement_learning",
    "risk",
    "sentiment",
    "trading",
    "training_portable",
    "transformer",
)
DASHBOARD_FILES = (
    "__init__.py",
    "model_activity.py",
    "model_backtest_page.py",
    "model_job_lock.py",
    "rl_training_jobs.py",
    "rl_training_page.py",
    "services.py",
    "transformer_training_jobs.py",
    "transformer_training_page.py",
    "ui.py",
)
PAPER_VALIDATION_FILES = (
    "config.py",
    "state.py",
    "storage.py",
)


TRAINING_APP = '''"""可攜式訓練中心的 Streamlit 入口。"""

from ai_quant_trading.training_portable.app import main

main()
'''


TRAINING_LAUNCHER = '''"""可攜式訓練中心的桌面視窗入口。"""

from ai_quant_trading.training_portable.launcher import main

raise SystemExit(main())
'''


REQUIREMENTS = """# PyTorch 由 setup_training_env.ps1 依顯卡另外安裝。
numpy>=1.26
pandas>=2.2
requests>=2.33.0
urllib3>=2.7.0,<3.0
idna>=3.15,<4.0
defusedxml>=0.7.1,<1.0
python-dotenv>=1.2
websocket-client>=1.8,<2.0
truststore>=0.10.4; sys_platform == 'win32'
safetensors>=0.4,<1.0
filelock>=3.20.3
transformers>=5.0,<6.0
gymnasium>=1.1,<2.0
stable-baselines3>=2.7,<3.0
streamlit>=1.55,<2.0
plotly>=5.24,<7.0
pyarrow>=23,<25
pillow>=12
pywebview>=6.2,<7.0
"""


SETUP_POWERSHELL = r'''param(
    [ValidateSet("auto", "rtx50", "rtx30_40", "cpu")]
    [string]$Profile = "auto"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $Root

function Find-Python312 {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.12 -c "import sys; print(sys.executable)" *> $null
        if ($LASTEXITCODE -eq 0) { return @("py", "-3.12") }
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        & python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)"
        if ($LASTEXITCODE -eq 0) { return @("python") }
    }
    throw "找不到 Python 3.12。請先從 python.org 安裝 64 位元 Python 3.12。"
}

$PythonCommand = Find-Python312
if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
    Write-Host "建立 Python 3.12 虛擬環境..."
    if ($PythonCommand.Count -eq 2) {
        & $PythonCommand[0] $PythonCommand[1] -m venv .venv
    } else {
        & $PythonCommand[0] -m venv .venv
    }
    if ($LASTEXITCODE -ne 0) { throw "建立虛擬環境失敗" }
}

$Python = Join-Path $Root ".venv\Scripts\python.exe"
& $Python -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "更新 pip 失敗" }

if ($Profile -eq "auto") {
    $GpuName = ""
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $GpuName = (& nvidia-smi --query-gpu=name --format=csv,noheader | Select-Object -First 1)
    }
    if ($GpuName -match "RTX 50") {
        $Profile = "rtx50"
    } elseif ($GpuName -match "RTX (30|40)") {
        $Profile = "rtx30_40"
    } elseif ($GpuName) {
        $Profile = "rtx30_40"
    } else {
        $Profile = "cpu"
    }
    Write-Host "偵測到：$GpuName；安裝方案：$Profile"
}

switch ($Profile) {
    "rtx50" {
        & $Python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
    }
    "rtx30_40" {
        & $Python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu126
    }
    "cpu" {
        & $Python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu
    }
}
if ($LASTEXITCODE -ne 0) { throw "PyTorch 安裝失敗" }

& $Python -m pip install -r requirements-training.txt
if ($LASTEXITCODE -ne 0) { throw "訓練套件安裝失敗" }

$env:PYTHONPATH = Join-Path $Root "src"
& $Python -c "import torch, streamlit, stable_baselines3, transformers; print('PyTorch', torch.__version__); print('CUDA', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
if ($LASTEXITCODE -ne 0) { throw "訓練環境驗證失敗" }

Write-Host ""
Write-Host "安裝完成。現在可以雙擊 啟動訓練中心.cmd。" -ForegroundColor Green
Read-Host "按 Enter 關閉"
'''


INSTALL_CMD = r'''@echo off
chcp 65001 >nul
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_training_env.ps1"
if errorlevel 1 pause
'''


START_CMD = r'''@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo 尚未安裝訓練環境，請先雙擊「安裝訓練環境.cmd」。
  pause
  exit /b 1
)
set "AI_QUANT_TRAINING_ROOT=%CD%"
set "PYTHONPATH=%CD%\src"
".venv\Scripts\python.exe" training_launcher.py
if errorlevel 1 (
  echo 啟動失敗，請查看 logs\training-launcher.log 與 logs\training-server.log。
  pause
)
'''


README = """# AI Quant 獨立可攜訓練中心

這個資料夾可以複製到另一台 Windows 電腦，專門訓練 BTC Transformer、SAC 與 PPO。
它不包含交易所 API Key、下單、模擬倉、WebSocket 機器人或實盤服務。

## 第一次使用

1. 安裝 64 位元 Python 3.12，安裝時勾選 Add Python to PATH。
2. 更新 NVIDIA 驅動程式。
3. 雙擊 `安裝訓練環境.cmd`。
4. 安裝完成後雙擊 `啟動訓練中心.cmd`。

安裝腳本會自動判斷顯卡：

- RTX 50 系列使用 CUDA 12.8 PyTorch。
- RTX 30／40 系列使用 CUDA 12.6 PyTorch。
- 沒有 NVIDIA 顯卡時使用 CPU 版 PyTorch。

## 操作順序

1. `BTC 訓練資料`：下載 5m、15m、1h、4h、1d 並建立多週期特徵。
2. `Transformer`：設定架構與 epochs，完成時間序列模型。
3. `建立 RL 環境`：把 Transformer 歷史預測加入 SAC/PPO 狀態。
4. `SAC/PPO`：先 smoke test，再執行多 Seed 與 Walk-forward。
5. `模型回測`：只看 validation/test 樣本外績效。
6. `模型匯出`：建立含特徵、標準化、風控與 SHA-256 的 ZIP。

## 資料夾

- `data/raw`：Binance Futures 原始 K 線。
- `data/processed`：技術特徵、多週期資料、Transformer 與 RL 環境。
- `outputs/model_packages`：帶回交易電腦的模型 ZIP。
- `logs`：桌面視窗與 Streamlit 服務日誌。
- `src`：訓練核心原始碼，註解採中文。

## 注意

- 訓練期間必須保持訓練中心開啟，關閉視窗會停止背景程序。
- 使用 `--include-data` 建立的可攜包會附帶目前已準備好的 BTC Futures 資料與特徵，可直接在新電腦開始訓練。
- 只有需要補齊搬移後的新行情時，才需要在 `BTC 訓練資料` 頁面執行增量更新；不必重新下載全部歷史。
- 模型 ZIP 含 PyTorch／Stable-Baselines3 成品，只能匯入你自己信任的模型包。
- 模型訓練完成不代表可以實盤，仍需樣本外回測與至少 4 至 8 週模擬驗證。
"""


STREAMLIT_CONFIG = """[server]
headless = true
fileWatcherType = "none"

[browser]
gatherUsageStats = false

[theme]
base = "light"
primaryColor = "#2f6fad"
backgroundColor = "#f5f7f9"
secondaryBackgroundColor = "#ffffff"
textColor = "#263442"
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="建立獨立可攜式 AI Quant 訓練包")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--include-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否附帶目前 BTC raw 與 feature CSV",
    )
    parser.add_argument("--no-zip", action="store_true", help="不要建立 ZIP")
    return parser


def _copy_source(destination: Path) -> None:
    source_root = PROJECT_ROOT / "src" / "ai_quant_trading"
    package_root = destination / "src" / "ai_quant_trading"
    package_root.mkdir(parents=True)
    for filename in [
        "__init__.py",
        "market_clock.py",
        "performance.py",
        "persistence.py",
    ]:
        shutil.copy2(source_root / filename, package_root / filename)
    for directory in SOURCE_PACKAGES:
        shutil.copytree(
            source_root / directory,
            package_root / directory,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
    dashboard = package_root / "dashboard"
    dashboard.mkdir()
    for filename in DASHBOARD_FILES:
        shutil.copy2(source_root / "dashboard" / filename, dashboard / filename)
    operations = package_root / "operations"
    operations.mkdir()
    for filename in ["__init__.py", "integrity.py"]:
        shutil.copy2(source_root / "operations" / filename, operations / filename)
    # RL 品質登錄器只需要模擬帳戶的資料契約，不包含撮合、機器人或下單程式。
    paper_support = package_root / "paper_trading"
    paper_support.mkdir()
    (paper_support / "__init__.py").write_text(
        '"""只供訓練品質驗證使用的模擬帳戶資料契約。"""\n',
        encoding="utf-8",
    )
    for filename in PAPER_VALIDATION_FILES:
        shutil.copy2(source_root / "paper_trading" / filename, paper_support / filename)


def _copy_training_data(destination: Path) -> None:
    raw_target = destination / "data" / "raw"
    processed_target = destination / "data" / "processed"
    raw_target.mkdir(parents=True, exist_ok=True)
    processed_target.mkdir(parents=True, exist_ok=True)
    raw_source = PROJECT_ROOT / "data" / "raw"
    if raw_source.exists():
        for path in raw_source.rglob("*"):
            if path.is_file() and "BTC" in path.name.upper():
                relative = path.relative_to(raw_source)
                target = raw_target / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    processed_source = PROJECT_ROOT / "data" / "processed"
    if processed_source.exists():
        for pattern in ["features_*BTC*.csv", "features_*BTC*.json"]:
            for path in processed_source.glob(pattern):
                shutil.copy2(path, processed_target / path.name)
        sentiment = processed_source / "sentiment"
        if sentiment.is_dir():
            shutil.copytree(sentiment, processed_target / "sentiment")
    config = PROJECT_ROOT / "data" / "config" / "ai_pipeline.json"
    if config.is_file():
        target = destination / "data" / "config" / config.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config, target)


def _write_runtime_files(destination: Path) -> None:
    files = {
        "training_app.py": TRAINING_APP,
        "training_launcher.py": TRAINING_LAUNCHER,
        "requirements-training.txt": REQUIREMENTS,
        "setup_training_env.ps1": SETUP_POWERSHELL,
        "安裝訓練環境.cmd": INSTALL_CMD,
        "啟動訓練中心.cmd": START_CMD,
        "README.md": README,
        ".streamlit/config.toml": STREAMLIT_CONFIG,
    }
    for relative, content in files.items():
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        # Windows PowerShell 5.1 需要 BOM 才能穩定辨識含中文的 UTF-8 腳本。
        encoding = "utf-8-sig" if path.suffix.lower() in {".ps1", ".cmd"} else "utf-8"
        path.write_text(content, encoding=encoding)
    for relative in [
        "data/raw",
        "data/processed/transformer/models",
        "data/processed/rl/environments",
        "data/processed/model_backtests",
        "outputs/model_packages",
        "logs",
    ]:
        (destination / relative).mkdir(parents=True, exist_ok=True)


def _copy_reference_files(destination: Path) -> None:
    for directory in ["configs", "docs"]:
        source = PROJECT_ROOT / directory
        if source.is_dir():
            shutil.copytree(source, destination / directory)


def main(arguments: list[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    destination = args.output.resolve()
    if destination == PROJECT_ROOT or PROJECT_ROOT.is_relative_to(destination):
        raise ValueError("輸出資料夾不可包含專案根目錄")
    if destination.exists() and (
        destination != DEFAULT_OUTPUT.resolve()
        and not (destination / BUNDLE_MARKER).is_file()
    ):
        raise ValueError("拒絕覆蓋不是由本工具建立的輸出資料夾")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    (destination / BUNDLE_MARKER).write_text("AI Quant training bundle\n", encoding="ascii")
    _copy_source(destination)
    _copy_reference_files(destination)
    if args.include_data:
        _copy_training_data(destination)
    _write_runtime_files(destination)

    archive = None
    if not args.no_zip:
        archive_path = destination.with_suffix(".zip")
        archive_path.unlink(missing_ok=True)
        archive = Path(shutil.make_archive(str(destination), "zip", destination.parent, destination.name))
    size = sum(path.stat().st_size for path in destination.rglob("*") if path.is_file())
    print(f"訓練包：{destination}")
    print(f"檔案大小：{size / 1024 / 1024:.1f} MB")
    if archive is not None:
        print(f"ZIP：{archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
