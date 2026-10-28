"""Pydantic 数据模型。"""
from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field


class Emotion(BaseModel):
    """一条记忆附带的三维情绪标签。"""

    valence: float = Field(default=0.0, description="情绪效价，-1 到 1")
    arousal: float = Field(default=0.0, description="情绪唤醒度，0 到 1")
    importance: float = Field(default=0.5, description="重要度，0 到 1，越低越容易被遗忘")

    def clamped(self) -> "Emotion":
        return Emotion(
            valence=max(-1.0, min(1.0, float(self.valence))),
            arousal=max(0.0, min(1.0, float(self.arousal))),
            importance=max(0.0, min(1.0, float(self.importance))),
        )


class Memory(BaseModel):
    """一条长期记忆（存储在向量库中）。"""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    user_id: str
    content: str
    type: Literal["dialogue", "musing", "summary"] = "dialogue"
    emotion: Emotion = Field(default_factory=Emotion)
    source_id: str | None = Field(default=None, description="派生自哪条记忆（走神用）")
    confidence: float = Field(default=1.0, description="置信度，走神产物固定偏低")
    created_at: float = Field(default_factory=time.time)


class RecalledMemory(BaseModel):
    """召回结果：记忆本体 + 相似度分数。"""

    memory: Memory
    score: float
