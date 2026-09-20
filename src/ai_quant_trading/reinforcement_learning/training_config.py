"""PPO／SAC 訓練、評估與模型保存參數。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Literal


AlgorithmName = Literal["ppo", "sac"]
ActivationName = Literal["tanh", "relu", "elu", "leaky_relu"]
ActionNoiseName = Literal["none", "normal", "ornstein_uhlenbeck"]


@dataclass(frozen=True, slots=True)
class RLTrainingConfig:
    """涵蓋 Stable-Baselines3 PPO 與 SAC 的可調純量參數。"""

    algorithm: AlgorithmName = "ppo"
    total_timesteps: int = 100_000
    device: str = "auto"
    seed: int = 42
    n_envs: int = 1
    cpu_threads: int = 4
    learning_rate: float = 0.0003
    gamma: float = 0.99
    batch_size: int = 256
    net_arch: tuple[int, ...] = (128, 128)
    activation_fn: ActivationName = "relu"
    stats_window_size: int = 100
    checkpoint_freq: int = 25_000
    evaluation_freq: int = 10_000
    n_eval_episodes: int = 1
    deterministic_eval: bool = True
    save_replay_buffer: bool = True

    # PPO 參數
    ppo_n_steps: int = 1024
    ppo_n_epochs: int = 10
    ppo_gae_lambda: float = 0.95
    ppo_clip_range: float = 0.2
    ppo_clip_range_vf: float | None = None
    ppo_normalize_advantage: bool = True
    ppo_ent_coef: float = 0.005
    ppo_vf_coef: float = 0.5
    ppo_max_grad_norm: float = 0.5
    ppo_use_sde: bool = False
    ppo_sde_sample_freq: int = -1
    ppo_target_kl: float | None = None

    # SAC 參數
    sac_buffer_size: int = 400_000
    sac_learning_starts: int = 20_000
    sac_tau: float = 0.005
    sac_train_freq: int = 1
    sac_train_freq_unit: Literal["step", "episode"] = "step"
    sac_gradient_steps: int = 1
    sac_optimize_memory_usage: bool = False
    sac_n_steps: int = 1
    sac_ent_coef: str | float = "auto_0.01"
    sac_target_update_interval: int = 1
    sac_target_entropy: str | float = "auto"
    sac_use_sde: bool = False
    sac_sde_sample_freq: int = -1
    sac_use_sde_at_warmup: bool = False
    sac_action_noise: ActionNoiseName = "normal"
    sac_action_noise_sigma: float = 0.05

    def __post_init__(self) -> None:
        if self.algorithm not in {"ppo", "sac"}:
            raise ValueError("algorithm 只支援 ppo 或 sac")
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps 必須大於 0")
        if not re.fullmatch(r"auto|cpu|cuda(?::\d+)?", self.device.lower()):
            raise ValueError("device 必須是 auto、cpu、cuda 或 cuda:N")
        if self.n_envs <= 0:
            raise ValueError("n_envs 必須大於 0")
        if self.cpu_threads <= 0:
            raise ValueError("cpu_threads 必須大於 0")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate 必須大於 0")
        if not 0 < self.gamma <= 1:
            raise ValueError("gamma 必須大於 0 且不超過 1")
        if self.batch_size <= 1:
            raise ValueError("batch_size 必須大於 1")
        if not self.net_arch or any(width <= 0 for width in self.net_arch):
            raise ValueError("net_arch 至少需要一層正整數寬度")
        if self.activation_fn not in {"tanh", "relu", "elu", "leaky_relu"}:
            raise ValueError("不支援的 activation_fn")
        if self.stats_window_size <= 0:
            raise ValueError("stats_window_size 必須大於 0")
        if self.checkpoint_freq < 0 or self.evaluation_freq < 0:
            raise ValueError("checkpoint_freq 與 evaluation_freq 不可小於 0")
        if self.n_eval_episodes <= 0:
            raise ValueError("n_eval_episodes 必須大於 0")
        if self.algorithm == "ppo":
            self._validate_ppo()
        else:
            self._validate_sac()

    def _validate_ppo(self) -> None:
        rollout_size = self.ppo_n_steps * self.n_envs
        if self.ppo_n_steps <= 1 or self.ppo_n_epochs <= 0:
            raise ValueError("PPO n_steps 必須大於 1，n_epochs 必須大於 0")
        if rollout_size % self.batch_size != 0:
            raise ValueError("PPO 的 n_steps × n_envs 必須可被 batch_size 整除")
        if not 0 < self.ppo_gae_lambda <= 1:
            raise ValueError("PPO gae_lambda 必須大於 0 且不超過 1")
        if self.ppo_clip_range <= 0:
            raise ValueError("PPO clip_range 必須大於 0")
        if self.ppo_clip_range_vf is not None and self.ppo_clip_range_vf <= 0:
            raise ValueError("PPO clip_range_vf 必須大於 0 或設為 None")
        if self.ppo_ent_coef < 0 or self.ppo_vf_coef < 0 or self.ppo_max_grad_norm <= 0:
            raise ValueError("PPO loss 係數不可為負，max_grad_norm 必須大於 0")
        if self.ppo_sde_sample_freq < -1:
            raise ValueError("PPO sde_sample_freq 不可小於 -1")
        if self.ppo_target_kl is not None and self.ppo_target_kl <= 0:
            raise ValueError("PPO target_kl 必須大於 0 或設為 None")

    def _validate_sac(self) -> None:
        if self.sac_buffer_size <= 0 or self.sac_learning_starts < 0:
            raise ValueError("SAC buffer_size 必須大於 0，learning_starts 不可小於 0")
        if self.sac_learning_starts >= self.sac_buffer_size:
            raise ValueError("SAC learning_starts 必須小於 buffer_size")
        if not 0 < self.sac_tau <= 1:
            raise ValueError("SAC tau 必須大於 0 且不超過 1")
        if self.sac_train_freq <= 0 or self.sac_gradient_steps < -1:
            raise ValueError("SAC train_freq 必須大於 0，gradient_steps 不可小於 -1")
        if self.sac_train_freq_unit not in {"step", "episode"}:
            raise ValueError("SAC train_freq_unit 只支援 step 或 episode")
        if self.sac_train_freq_unit == "episode" and self.n_envs != 1:
            raise ValueError("SAC 依 episode 更新時只支援單一環境")
        if self.sac_n_steps <= 0 or self.sac_target_update_interval <= 0:
            raise ValueError("SAC n_steps 與 target_update_interval 必須大於 0")
        if self.sac_action_noise not in {"none", "normal", "ornstein_uhlenbeck"}:
            raise ValueError("不支援的 SAC action noise")
        if self.sac_action_noise_sigma < 0:
            raise ValueError("SAC action_noise_sigma 不可小於 0")
        self._validate_auto_or_number(self.sac_ent_coef, "SAC ent_coef")
        self._validate_auto_or_number(
            self.sac_target_entropy,
            "SAC target_entropy",
            allow_negative=True,
            allow_auto_initial=False,
        )

    @staticmethod
    def _validate_auto_or_number(
        value: str | float,
        name: str,
        *,
        allow_negative: bool = False,
        allow_auto_initial: bool = True,
    ) -> None:
        if isinstance(value, str):
            if value == "auto" or (allow_auto_initial and value.startswith("auto_")):
                return
            try:
                numeric = float(value)
            except ValueError as exc:
                allowed = "auto、auto_N 或數值" if allow_auto_initial else "auto 或數值"
                raise ValueError(f"{name} 必須是 {allowed}") from exc
        else:
            numeric = float(value)
        if not allow_negative and numeric < 0:
            raise ValueError(f"{name} 不可小於 0")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
