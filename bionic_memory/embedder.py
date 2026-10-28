"""向量化封装：默认使用 fastembed 在本地运行 bge-small-zh。

无需 GPU，首次运行会自动下载模型（约 100MB），之后可离线使用。
"""
from __future__ import annotations

import threading


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5",
                 device: str = "cpu", cache_dir: str | None = None):
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            from fastembed import TextEmbedding

            kwargs = {"model_name": self.model_name}
            if self.cache_dir:
                kwargs["cache_dir"] = self.cache_dir
            if self.device and self.device.lower() != "cpu":
                kwargs["device"] = self.device
            try:
                self._model = TextEmbedding(**kwargs)
            except TypeError:
                # 旧版 fastembed 不支持 device / cache_dir 参数
                kwargs.pop("device", None)
                kwargs.pop("cache_dir", None)
                self._model = TextEmbedding(**kwargs)
            return self._model

    def embed(self, text: str) -> list[float]:
        model = self._load()
        vector = next(model.embed([text]))
        return [float(x) for x in vector]
