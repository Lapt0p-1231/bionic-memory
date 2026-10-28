"""向量化封装：默认使用 fastembed 在本地运行 bge-small-zh。

无需 GPU，首次运行会自动下载模型（约 100MB），之后可离线使用。
带 LRU 缓存：重复文本（尤其是高频查询）不会重复推理。
"""
from __future__ import annotations

import threading
from collections import OrderedDict


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5",
                 device: str = "cpu", cache_dir: str | None = None,
                 cache_size: int = 1024):
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.cache_size = max(0, int(cache_size))
        self._model = None
        self._load_lock = threading.Lock()
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._cache_lock = threading.Lock()

    def _load(self):
        """懒加载模型，避免 import 时就占用内存/触发下载。"""
        if self._model is not None:
            return self._model
        with self._load_lock:
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

    def _cache_get(self, text: str) -> list[float] | None:
        if self.cache_size <= 0:
            return None
        with self._cache_lock:
            vector = self._cache.get(text)
            if vector is not None:
                self._cache.move_to_end(text)
                return list(vector)
        return None

    def _cache_put(self, text: str, vector: list[float]) -> None:
        if self.cache_size <= 0:
            return
        with self._cache_lock:
            self._cache[text] = list(vector)
            self._cache.move_to_end(text)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

    def embed(self, text: str) -> list[float]:
        """把文本编码为向量。结果会按文本做 LRU 缓存。"""
        cached = self._cache_get(text)
        if cached is not None:
            return cached
        model = self._load()
        vector = [float(x) for x in next(model.embed([text]))]
        self._cache_put(text, vector)
        return list(vector)

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache.clear()
