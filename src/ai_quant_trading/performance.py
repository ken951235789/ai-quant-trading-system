"""互動介面、模型訓練與背景交易共用的效能限制。"""

from __future__ import annotations

from collections.abc import MutableMapping
import os


THREAD_ENVIRONMENT_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def available_cpu_threads() -> int:
    """回傳目前作業系統可用的邏輯處理器數量。"""
    return max(int(os.cpu_count() or 1), 1)


def clamp_cpu_threads(requested: int) -> int:
    """將使用者設定限制在本機實際可用的 CPU 範圍內。"""
    if requested <= 0:
        raise ValueError("cpu_threads 必須大於 0")
    return min(int(requested), available_cpu_threads())


def apply_thread_environment(
    environment: MutableMapping[str, str],
    cpu_threads: int,
    *,
    overwrite: bool = False,
) -> MutableMapping[str, str]:
    """在子程序載入 NumPy／PyTorch 前限制底層數學函式庫執行緒。"""
    value = str(clamp_cpu_threads(cpu_threads))
    for key in THREAD_ENVIRONMENT_KEYS:
        if overwrite or key not in environment:
            environment[key] = value
    return environment


def configure_torch_cpu_threads(cpu_threads: int) -> int:
    """限制 PyTorch CPU 執行緒，保留核心給 Dashboard 與行情程序。"""
    resolved = clamp_cpu_threads(cpu_threads)
    import torch

    torch.set_num_threads(resolved)
    try:
        # inter-op 只能在第一次平行運算前設定；已初始化時保留既有值即可。
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return resolved
