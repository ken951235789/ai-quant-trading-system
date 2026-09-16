"""偵測 Stable-Baselines3、PyTorch 與可用 CUDA 裝置。"""

from __future__ import annotations

from dataclasses import dataclass
import warnings


@dataclass(frozen=True, slots=True)
class TrainingBackendStatus:
    """訓練後端與 GPU 裝置狀態。"""

    available: bool
    torch_version: str | None
    stable_baselines_version: str | None
    cuda_available: bool
    cuda_devices: tuple[str, ...]
    cuda_device_indices: tuple[int, ...] = ()
    cuda_errors: tuple[str, ...] = ()
    error: str | None = None

    @property
    def device_options(self) -> tuple[str, ...]:
        options = ["auto", "cpu"]
        indices = self.cuda_device_indices or tuple(range(len(self.cuda_devices)))
        options.extend(f"cuda:{index}" for index in indices)
        return tuple(options)


def inspect_training_backend() -> TrainingBackendStatus:
    """延遲載入大型套件，回傳 CPU／CUDA 可用狀態。"""
    try:
        import stable_baselines3
        import torch
    except ImportError as exc:
        return TrainingBackendStatus(False, None, None, False, (), str(exc))

    devices: list[str] = []
    device_indices: list[int] = []
    cuda_errors: list[str] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            name = f"未知 GPU {index}"
            try:
                # is_available 只驗證驅動；實際張量探針才能發現 wheel 未包含舊 GPU kernel。
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    name = torch.cuda.get_device_name(index)
                    probe = torch.ones(1, device=f"cuda:{index}")
                    _ = float(probe.square().item())
                    torch.cuda.synchronize(index)
            except Exception as exc:
                first_line = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
                cuda_errors.append(f"CUDA:{index} · {name}：{first_line}")
            else:
                devices.append(name)
                device_indices.append(index)
    return TrainingBackendStatus(
        True,
        str(torch.__version__),
        str(stable_baselines3.__version__),
        bool(devices),
        tuple(devices),
        tuple(device_indices),
        tuple(cuda_errors),
    )


def resolve_training_device(
    requested: str,
    status: TrainingBackendStatus | None = None,
) -> str:
    """解析 auto／CPU／CUDA，禁止 CUDA 不可用時靜默退回 CPU。"""
    backend = status or inspect_training_backend()
    if not backend.available:
        raise ImportError(f"強化學習訓練後端尚未安裝：{backend.error or '未知錯誤'}")
    normalized = requested.lower()
    if normalized == "auto":
        if not backend.cuda_available:
            return "cpu"
        available_indices = backend.cuda_device_indices or tuple(range(len(backend.cuda_devices)))
        return f"cuda:{available_indices[0]}"
    if normalized == "cpu":
        return "cpu"
    if normalized == "cuda":
        normalized = "cuda:0"
    if normalized.startswith("cuda:"):
        try:
            index = int(normalized.split(":", maxsplit=1)[1])
        except ValueError as exc:
            raise ValueError("CUDA 裝置格式必須是 cuda:N") from exc
        if not backend.cuda_available:
            raise RuntimeError("目前 PyTorch 或硬體未提供 CUDA，不能使用 GPU 訓練")
        available_indices = backend.cuda_device_indices or tuple(range(len(backend.cuda_devices)))
        if index not in available_indices:
            raise RuntimeError(f"CUDA:{index} 不可用，可用裝置為 {list(available_indices)}")
        return normalized
    raise ValueError("device 必須是 auto、cpu、cuda 或 cuda:N")
