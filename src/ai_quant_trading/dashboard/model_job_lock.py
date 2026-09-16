"""協調 PPO 與 Transformer 背景工作，避免同時搶用運算資源。"""

from threading import RLock


MODEL_JOB_LOCK = RLock()
