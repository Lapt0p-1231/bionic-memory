"""BionicMemory 核心类：仿生冷热记忆。

- 热记忆（SQLite）：最近 N 轮对话，始终完整参与上下文
- 冷记忆（ChromaDB）：每轮对话切片 + 三维情绪打分，按语义召回
- 走神（musing）：定时回忆高重要度记忆，生成内心独白再存回冷记忆

本库不绑定任何 LLM：所有需要模型的地方都通过 `llm_callback` 回调。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Awaitable, Callable

from .database import Storage
from .embedder import Embedder
from .models import Emotion, Memory
from .prompts import EMOTION_PROMPT, MUSING_PROMPT

logger = logging.getLogger("bionic_memory")

LLMCallback = Callable[[str], Awaitable[str]]

DEFAULT_CONFIG: dict = {
    "embedding": {
        "model": "BAAI/bge-small-zh-v1.5",
        "device": "cpu",
        "cache_dir": None,
    },
    "memory": {
        "short_term_limit": 30,
        "recall_top_k": 3,
        "recall_threshold": 0.55,
    },
    "musing": {
        "enabled": True,
        "interval_minutes": 120,
        "min_importance": 0.6,
        "max_per_round": 3,
    },
    "storage": {
        "data_dir": "./bionic_memory_data",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config(config_path: str | None, config: dict | None) -> dict:
    loaded: dict = {}
    if config_path:
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
    elif config:
        loaded = config
    return _deep_merge(DEFAULT_CONFIG, loaded)


def _strip_json(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.removeprefix("```json").removeprefix("```")
        raw = raw.removesuffix("```").strip()
    return raw


def _parse_emotion(raw: str) -> Emotion:
    data = json.loads(_strip_json(raw))
    if not isinstance(data, dict):
        raise ValueError("情绪分析结果不是 JSON 对象")
    return Emotion(
        valence=float(data.get("valence", 0.0)),
        arousal=float(data.get("arousal", 0.0)),
        importance=float(data.get("importance", 0.5)),
    ).clamped()


class BionicMemory:
    def __init__(self, user_id: str, config_path: str | None = None,
                 config: dict | None = None, data_dir: str | None = None):
        self.user_id = str(user_id)
        self.config = _load_config(config_path, config)

        storage_cfg = self.config["storage"]
        self.data_dir = os.path.abspath(data_dir or storage_cfg["data_dir"])

        embed_cfg = self.config["embedding"]
        self.embedder = Embedder(
            model_name=embed_cfg["model"],
            device=str(embed_cfg.get("device", "cpu")),
            cache_dir=embed_cfg.get("cache_dir"),
        )
        self.db = Storage(self.data_dir)
        self._musing_task: asyncio.Task | None = None
        self._closed = False

    # ---------- 对话前 ----------
    async def build_context(self, user_message: str) -> dict:
        """返回组装好的上下文。

        {
            "short_term": [{"role": "user", "content": "..."}, ...],
            "recalled_memories": [{"content": "...", "type": "...",
                                   "score": 0.72, "importance": 0.8}, ...],
            "emotion_state": {"valence": 0.2, "arousal": 0.4, "importance": 0.6}
        }
        """
        mem_cfg = self.config["memory"]
        short_term = self.db.get_short_term(
            self.user_id, keep_turns=mem_cfg["short_term_limit"]
        )

        recalled: list[dict] = []
        if user_message and user_message.strip():
            embedding = await asyncio.to_thread(self.embedder.embed, user_message)
            hits = self.db.recall(
                self.user_id,
                embedding,
                top_k=mem_cfg["recall_top_k"],
                threshold=mem_cfg["recall_threshold"],
            )
            recalled = [
                {
                    "content": memory.content,
                    "type": memory.type,
                    "score": round(score, 4),
                    "importance": memory.emotion.importance,
                }
                for memory, score in hits
            ]

        return {
            "short_term": short_term,
            "recalled_memories": recalled,
            "emotion_state": self.db.get_profile(self.user_id),
        }

    # ---------- 对话后 ----------
    async def save_interaction(self, user_message: str, ai_reply: str,
                               llm_callback: LLMCallback | None = None) -> Memory:
        """写入热记忆并固化到冷记忆。

        llm_callback 可选：传入时用于分析情绪与重要度；
        不传时使用中性默认值（仍会入库，可被召回）。
        """
        keep = self.config["memory"]["short_term_limit"]
        self.db.append_turn(self.user_id, "user", user_message, keep)
        self.db.append_turn(self.user_id, "assistant", ai_reply, keep)

        exchange = f"用户：{user_message}\n助手：{ai_reply}"
        emotion = Emotion()
        if llm_callback is not None:
            try:
                raw = await llm_callback(f"{EMOTION_PROMPT}\n\n对话：\n{exchange}")
                emotion = _parse_emotion(raw)
            except Exception as exc:
                logger.warning("情绪分析失败，使用默认值：%s", exc)

        embedding = await asyncio.to_thread(self.embedder.embed, exchange)
        memory = self.db.add_memory(self.user_id, exchange, embedding,
                                    memory_type="dialogue", emotion=emotion)
        self.db.set_profile(self.user_id, {
            "valence": emotion.valence,
            "arousal": emotion.arousal,
            "importance": emotion.importance,
        })
        return memory

    # ---------- 走神 ----------
    async def trigger_musing(self, llm_callback: LLMCallback,
                             min_importance: float | None = None,
                             max_per_round: int | None = None) -> list[Memory]:
        """手动触发一次走神：回忆高重要度记忆，生成内心独白并入库。"""
        musing_cfg = self.config["musing"]
        min_importance = (
            musing_cfg["min_importance"] if min_importance is None else min_importance
        )
        max_per_round = (
            musing_cfg["max_per_round"] if max_per_round is None else max_per_round
        )
        memories = self.db.get_important(self.user_id, limit=max_per_round,
                                         min_importance=min_importance)
        created: list[Memory] = []
        for memory in memories:
            try:
                raw = await llm_callback(f"{MUSING_PROMPT}\n\n往事：{memory.content}")
                text = raw.strip()
                if not text:
                    continue
                embedding = await asyncio.to_thread(self.embedder.embed, text)
                created.append(self.db.add_memory(
                    self.user_id, text, embedding, memory_type="musing",
                    emotion=Emotion(importance=0.6),
                ))
            except Exception as exc:
                logger.warning("走神生成失败：%s", exc)
        return created

    def start_musing_loop(self, llm_callback: LLMCallback) -> None:
        """启动后台走神定时器（配置 musing.enabled 为 true 时才有意义）。"""
        if not self.config["musing"]["enabled"] or self._musing_task is not None:
            return
        interval = max(1, int(self.config["musing"]["interval_minutes"])) * 60

        async def loop():
            while True:
                await asyncio.sleep(interval)
                try:
                    created = await self.trigger_musing(llm_callback)
                    if created:
                        logger.info("走神生成了 %d 条独白", len(created))
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning("走神任务异常：%s", exc)

        self._musing_task = asyncio.create_task(loop())

    def stop_musing_loop(self) -> None:
        if self._musing_task is not None:
            self._musing_task.cancel()
            self._musing_task = None

    # ---------- 其他 ----------
    def clear(self) -> None:
        """清空该用户的全部记忆（热 + 冷 + 情绪状态）。"""
        self.db.clear_user(self.user_id)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.stop_musing_loop()
        self.db.close()
