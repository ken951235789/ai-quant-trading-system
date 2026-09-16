# -*- mode: python ; coding: utf-8 -*-
"""AI Quant Trading System 的 PyInstaller one-folder 建置規格。"""

from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)


PROJECT_ROOT = Path(SPECPATH).resolve().parent
datas = [
    (str(PROJECT_ROOT / "packaging" / "streamlit_entry.py"), "app"),
]
binaries = []
hiddenimports = collect_submodules("ai_quant_trading")
hiddenimports += collect_submodules("streamlit.runtime")
hiddenimports += collect_submodules("streamlit.web")
hiddenimports += collect_submodules("streamlit.components.v1")
hiddenimports += collect_submodules("gymnasium")
hiddenimports += collect_submodules("stable_baselines3")
hiddenimports += collect_submodules("tokenizers")
hiddenimports += collect_submodules("huggingface_hub")
hiddenimports += collect_submodules("safetensors")
hiddenimports += collect_submodules("websocket")
hiddenimports += collect_submodules("sqlalchemy")
hiddenimports += collect_submodules("psycopg")
hiddenimports += collect_submodules("alembic")
hiddenimports += [
    "transformers.models.auto.configuration_auto",
    "transformers.models.auto.modeling_auto",
    "transformers.models.auto.tokenization_auto",
    "transformers.models.bert.configuration_bert",
    "transformers.models.bert.modeling_bert",
    "transformers.models.bert.tokenization_bert",
    "webview.platforms.winforms",
    "webview.platforms.edgechromium",
]

for package_name in ["scipy", "tokenizers", "safetensors", "psycopg", "duckdb"]:
    binaries += collect_dynamic_libs(package_name)

# 只收集實際介面需要的靜態資源，避免把套件測試與大型 AI 框架一併打包。
for package_name in [
    "streamlit",
    "plotly",
    "altair",
    "stable_baselines3",
    "transformers",
]:
    datas += collect_data_files(package_name, include_py_files=False)

for distribution_name in [
    "streamlit",
    "plotly",
    "altair",
    "pyarrow",
    "pywebview",
    "pythonnet",
    "gymnasium",
    "stable-baselines3",
    "torch",
    "transformers",
    "tokenizers",
    "huggingface-hub",
    "safetensors",
    "websocket-client",
    "SQLAlchemy",
    "psycopg",
    "psycopg-binary",
    "alembic",
    "duckdb",
]:
    datas += copy_metadata(distribution_name)

a = Analysis(
    [str(PROJECT_ROOT / "src" / "ai_quant_trading" / "dashboard" / "launcher.py")],
    pathex=[str(PROJECT_ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "cv2",
        "cryptography",
        "curl_cffi",
        "dask",
        "hypothesis",
        "IPython",
        "jupyter",
        "kivy",
        "notebook",
        "pygame",
        "polars",
        "pytest",
        "_pytest",
        "PyQt5",
        "PyQt6",
        "PySide2",
        "PySide6",
        "tensorflow",
        "torchvision",
        "sklearn",
        "webview.platforms.android",
        "webview.platforms.cef",
        "webview.platforms.cocoa",
        "webview.platforms.gtk",
        "webview.platforms.qt",
        "yfinance",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AIQuantTradingSystem",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="AIQuantTradingSystem",
)
