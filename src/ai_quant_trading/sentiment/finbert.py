"""FinBERT 延遲載入與新聞文字推論。"""

from __future__ import annotations

import os
import ssl
import sys
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ai_quant_trading.sentiment.config import FinBERTConfig


def _configure_huggingface_truststore() -> None:
    """讓 Hugging Face 在 Windows 使用系統憑證庫，且維持 TLS 驗證。"""
    if sys.platform != "win32":
        return
    try:
        import httpx
        import truststore
        from huggingface_hub import set_client_factory
    except ImportError:
        return

    def create_client() -> httpx.Client:
        context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        return httpx.Client(
            verify=context,
            follow_redirects=True,
            timeout=httpx.Timeout(60.0, connect=15.0),
        )

    set_client_factory(create_client)


@dataclass(frozen=True, slots=True)
class FinBERTBackendStatus:
    """目前電腦是否具備 FinBERT 推論環境。"""

    transformers_installed: bool
    torch_installed: bool
    cuda_available: bool
    ready: bool
    message: str


def finbert_backend_status() -> FinBERTBackendStatus:
    """檢查套件與 GPU 狀態，不下載任何模型。"""
    transformers_installed = find_spec("transformers") is not None
    torch_installed = find_spec("torch") is not None
    cuda_available = False
    if torch_installed:
        import torch

        cuda_available = bool(torch.cuda.is_available())
    ready = transformers_installed and torch_installed
    if ready:
        message = "FinBERT 推論環境已就緒"
    else:
        missing = [
            name
            for name, installed in [
                ("transformers", transformers_installed),
                ("torch", torch_installed),
            ]
            if not installed
        ]
        message = f"缺少套件：{', '.join(missing)}"
    return FinBERTBackendStatus(
        transformers_installed,
        torch_installed,
        cuda_available,
        ready,
        message,
    )


class FinBERTAnalyzer:
    """按下推論按鈕後才載入模型，避免啟動 App 時占用記憶體。"""

    def __init__(self, config: FinBERTConfig | None = None) -> None:
        self.config = config or FinBERTConfig()
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._device = ""
        self._label_indices: dict[str, int] = {}

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def resolved_device(self) -> str:
        return self._device or "尚未載入"

    def _resolve_device(self, torch_module: object) -> str:
        requested = self.config.device.lower()
        if requested == "auto":
            return "cuda" if torch_module.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch_module.cuda.is_available():
            raise RuntimeError("目前 PyTorch 無法使用 CUDA，請改用 auto 或 cpu")
        return requested

    @staticmethod
    def _resolve_label_indices(id2label: dict[object, object]) -> dict[str, int]:
        normalized = {
            str(label).lower(): int(index)
            for index, label in id2label.items()
        }
        indices: dict[str, int] = {}
        for target in ["positive", "negative", "neutral"]:
            match = next(
                (index for label, index in normalized.items() if target in label),
                None,
            )
            if match is None:
                raise ValueError(f"FinBERT 模型缺少 {target} 標籤")
            indices[target] = match
        return indices

    def load(self) -> None:
        """從 Hugging Face 快取或網路載入 tokenizer 與模型。"""
        if self.loaded:
            return
        status = finbert_backend_status()
        if not status.ready:
            raise RuntimeError(status.message)
        if sys.platform == "win32":
            # 普通 Windows 帳號無法建立 Hugging Face 快取的符號連結。
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        _configure_huggingface_truststore()
        cache_dir = Path(self.config.cache_dir).resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            cache_dir=cache_dir,
            revision=self.config.revision,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            cache_dir=cache_dir,
            revision=self.config.revision,
            # ProsusAI/finbert 目前只有 .bin；新版 Transformers 會以
            # weights_only=True 安全載入，並用固定 revision 防止版本漂移。
            weights_only=True,
        )
        device = self._resolve_device(torch)
        model.to(device)
        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch
        self._device = device
        self._label_indices = self._resolve_label_indices(model.config.id2label)

    def analyze(self, texts: Sequence[str]) -> pd.DataFrame:
        """輸出正面、負面、中性機率及 -1 到 1 的連續情緒分數。"""
        normalized = [str(text).strip() for text in texts]
        if not normalized or any(not text for text in normalized):
            raise ValueError("texts 必須包含至少一筆非空文字")
        self.load()
        rows: list[dict[str, float | str]] = []
        if self._torch is None or self._tokenizer is None or self._model is None:
            raise RuntimeError("FinBERT 載入完成後仍缺少推論元件")
        for start in range(0, len(normalized), self.config.batch_size):
            batch = normalized[start : start + self.config.batch_size]
            tokens = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            tokens = {key: value.to(self._device) for key, value in tokens.items()}
            with self._torch.inference_mode():
                logits = self._model(**tokens).logits
                probabilities = self._torch.softmax(logits, dim=-1).cpu().numpy()
            for text, probability in zip(batch, probabilities, strict=True):
                positive = float(probability[self._label_indices["positive"]])
                negative = float(probability[self._label_indices["negative"]])
                neutral = float(probability[self._label_indices["neutral"]])
                confidence = float(np.max(probability))
                rows.append(
                    {
                        "text": text,
                        "finbert_positive": positive,
                        "finbert_negative": negative,
                        "finbert_neutral": neutral,
                        "finbert_sentiment": positive - negative,
                        "finbert_confidence": confidence,
                        "finbert_confident": float(
                            confidence >= self.config.confidence_threshold
                        ),
                    }
                )
        return pd.DataFrame(rows)

    def score_news(
        self,
        news: pd.DataFrame,
        *,
        title_column: str = "title",
        text_column: str = "text",
    ) -> pd.DataFrame:
        """保留新聞時間與來源欄，並附加 FinBERT 分數。"""
        if title_column not in news and text_column not in news:
            raise ValueError("新聞資料至少需要 title 或 text 欄位")
        title = (
            news[title_column].fillna("").astype(str)
            if title_column in news
            else pd.Series("", index=news.index)
        )
        body = (
            news[text_column].fillna("").astype(str)
            if text_column in news
            else pd.Series("", index=news.index)
        )
        combined = (title.str.strip() + ". " + body.str.strip()).str.strip(". ")
        if (combined == "").any():
            raise ValueError("新聞標題與內容不可同時為空")
        scored = self.analyze(combined.tolist()).drop(columns=["text"])
        result = news.reset_index(drop=True).copy()
        for column in scored:
            result[column] = scored[column]
        return result
