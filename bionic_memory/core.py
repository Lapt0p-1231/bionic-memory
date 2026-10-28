"""BionicMemory 核心类：仿生冷热记忆。

- 热记忆（SQLite）：最近 N 轮对话，始终完整参与上下文
- 冷记忆（ChromaDB）：每轮对话切片 + 三维情绪打分，按语义召回
- 走神（musing）：定时回忆高重要度记忆，生成内心独白再存回冷记忆

本库不绑定任何 LLM：所有需要模型的地方都通过 `llm_callback` 回调。
所有阻塞操作（SQLite/Chroma/embedding）都在线程池里执行，不阻塞事件循环。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Awaitable, Callable

from .database import Storage
from .embedder import Embedder
from .models import Emotion, Memory
from .prompts import EMOTION_PROMPT, MUSING_PROMPT

logger = logging.getLogger("bionic_memory")

LLMCallback = Callable[[str], Awaitable[str]]

DREAM_PREFIX = "（好像做过一个类似的梦，画面很模糊）"

DEFAULT_CONFIG: dict = {
    "embedding": {
        "model": "BAAI/bge-small-zh-v1.5",
        "device": "cpu",
        "cache_dir": None,
        "cache_size": 1024,
    },
    "memory": {
        "short_term_limit": 30,
        "recall_top_k": 3,
        "recall_threshold": 0.55,
        "max_musing_recall": 1,
    },
    "musing": {
        "enabled": True,
        "interval_minutes": 120,
        "min_importance": 0.6,
        "max_per_round": 3,
        "timeout_seconds": 60,
    },
    "storage": {
        "data_dir": "./bionic_memory_data",
        "hnsw": {
            "space": "cosine",
            "M": 32,
            "construction_ef": 200,
            "search_ef": 100,
        },
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


def _to_float(value, default: float) -> float:
    """把可能是字符串/布尔/None 的字段安全转成 float。"""
    if isinstance(value, bool) or value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_emotion(raw: str) -> Emotion:
    """解析情绪 JSON，脏输出自动降级。

    兼容：```json 包裹、前后夹带解释文字、数字是字符串、缺字段。
    完全解析失败时返回中性默认值，不抛异常。
    """
    text = _strip_json(raw)
    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                data = None
    if not isinstance(data, dict):
        logger.warning("情绪分析输出无法解析，使用默认值：%r", raw[:120])
        return Emotion()
    emotion = Emotion(
        valence=_to_float(data.get("valence"), 0.0),
        arousal=_to_float(data.get("arousal"), 0.0),
        importance=_to_float(data.get("importance"), 0.5),
    ).clamped()
    return emotion


def format_recall(items: list[dict], dream_prefix: str = DREAM_PREFIX) -> str:
    """把召回结果格式化成提示词片段，走神记忆自动加"梦"的批注。

    >>> format_recall([{"content": "一起去过海边", "type": "dialogue"},
    ...                {"content": "好像有人说过要离开", "type": "musing"}])
    '- 一起去过海边\\n- （好像做过一个类似的梦，画面很模糊）好像有人说过要离开'
    """
    lines = []
    for item in items:
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        if item.get("type") == "musing":
            lines.append(f"- {dream_prefix}{content}")
        else:
            lines.append(f"- {content}")
    return "\n".join(lines)


class BionicMemory:
    """一个用户/角色的仿生记忆实例。

    参数：
        user_id: 记忆归属（QQ 号、群号或任意字符串）
        config_path: YAML 配置路径；与 config 二选一
        config: 直接传 dict 配置（会与默认配置深合并）
        data_dir: 覆盖配置里的存储目录

    用法：
        memory = BionicMemory("123", config_path="config.yaml")
        ctx = await memory.build_context("今天天气真差")
        await memory.save_interaction("今天天气真差", reply, llm_callback=my_llm)
        await memory.close()
    """

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
            cache_size=int(embed_cfg.get("cache_size", 1024)),
        )
        self.db = Storage(self.data_dir, hnsw=storage_cfg.get("hnsw"))
        self._musing_task: asyncio.Task | None = None
        self._closed = False

    # ---------- 对话前 ----------
    async def build_context(self, user_message: str) -> dict:
        """组装对话前的上下文（不写入任何数据）。

        返回：
            {
                "short_term": [{"role": "user", "content": "..."}, ...],
                "recalled_memories": [
                    {"content": "...", "type": "dialogue", "score": 0.72,
                     "importance": 0.8, "confidence": 1.0}, ...
                ],
                "emotion_state": {"valence": 0.2, "arousal": 0.4, ...}
            }

        提示：把 recalled_memories 交给 format_recall() 生成带"梦"批注的片段。
        """
        mem_cfg = self.config["memory"]
        short_term = await asyncio.to_thread(
            self.db.get_short_term, self.user_id,
            keep_turns=mem_cfg["short_term_limit"],
        )

        recalled: list[dict] = []
        if user_message and user_message.strip():
            embedding = await asyncio.to_thread(self.embedder.embed, user_message)
            hits = await asyncio.to_thread(
                self.db.recall,
                self.user_id,
                embedding,
                top_k=mem_cfg["recall_top_k"],
                threshold=mem_cfg["recall_threshold"],
                max_musing=mem_cfg.get("max_musing_recall", 1),
            )
            recalled = [
                {
                    "content": memory.content,
                    "type": memory.type,
                    "score": round(score, 4),
                    "importance": memory.emotion.importance,
                    "confidence": memory.confidence,
                }
                for memory, score in hits
            ]

        emotion_state = await asyncio.to_thread(self.db.get_profile, self.user_id)
        return {
            "short_term": short_term,
            "recalled_memories": recalled,
            "emotion_state": emotion_state,
        }

    # ---------- 对话后 ----------
    async def save_interaction(self, user_message: str, ai_reply: str,
                               llm_callback: LLMCallback | None = None) -> Memory:
        """写入一轮对话：热记忆 + 冷记忆（带情绪/重要度）。

        llm_callback 可选：传入时用于分析情绪与重要度；
        不传时使用中性默认值（仍会入库，可被召回）。
        回调失败时自动降级为默认值，不影响记忆写入。
        """
        keep = self.config["memory"]["short_term_limit"]
        await asyncio.to_thread(
            self.db.append_turn, self.user_id, "user", user_message, keep
        )
        await asyncio.to_thread(
            self.db.append_turn, self.user_id, "assistant", ai_reply, keep
        )

        exchange = f"用户：{user_message}\n助手：{ai_reply}"
        emotion = Emotion()
        if llm_callback is not None:
            timeout = float(self.config["musing"].get("timeout_seconds", 60))
            try:
                raw = await asyncio.wait_for(
                    llm_callback(f"{EMOTION_PROMPT}\n\n对话：\n{exchange}"),
                    timeout=timeout,
                )
                emotion = _parse_emotion(raw)
            except asyncio.TimeoutError:
                logger.warning("情绪分析超时（%ss），使用默认值", timeout)
            except Exception as exc:
                logger.warning("情绪分析失败，使用默认值：%s", exc)

        embedding = await asyncio.to_thread(self.embedder.embed, exchange)
        memory = await asyncio.to_thread(
            self.db.add_memory, self.user_id, exchange, embedding,
            "dialogue", emotion,
        )
        await asyncio.to_thread(self.db.set_profile, self.user_id, {
            "valence": emotion.valence,
            "arousal": emotion.arousal,
            "importance": emotion.importance,
        })
        return memory

    # ---------- 走神 ----------
    async def trigger_musing(self, llm_callback: LLMCallback,
                             min_importance: float | None = None,
                             max_per_round: int | None = None) -> list[Memory]:
        """手动触发一次走神：回忆高重要度记忆，生成内心独白并入库。

        单条记忆失败/超时只跳过该条，不影响其他记忆。
        """
        musing_cfg = self.config["musing"]
        min_importance = (
            musing_cfg["min_importance"] if min_importance is None else min_importance
        )
        max_per_round = (
            musing_cfg["max_per_round"] if max_per_round is None else max_per_round
        )
        timeout = float(musing_cfg.get("timeout_seconds", 60))
        memories = await asyncio.to_thread(
            self.db.get_important, self.user_id,
            limit=max_per_round, min_importance=min_importance,
        )
        created: list[Memory] = []
        for memory in memories:
            try:
                raw = await asyncio.wait_for(
                    llm_callback(f"{MUSING_PROMPT}\n\n往事：{memory.content}"),
                    timeout=timeout,
                )
                text = (raw or "").strip()
                if not text:
                    continue
                embedding = await asyncio.to_thread(self.embedder.embed, text)
                created.append(await asyncio.to_thread(
                    self.db.add_memory,
                    self.user_id, text, embedding,
                    "musing", Emotion(importance=0.6),
                    memory.id, 0.4,
                ))
            except asyncio.TimeoutError:
                logger.warning("走神生成超时（%ss），跳过该条", timeout)
            except Exception as exc:
                logger.warning("走神生成失败：%s", exc)
        if created:
            logger.info("走神生成了 %d 条独白", len(created))
        return created

    def start_musing_loop(self, llm_callback: LLMCallback) -> None:
        """启动后台走神定时器。

        配置 musing.enabled 为 false 或 interval_minutes <= 0 时不启动；
        重复调用无副作用。
        异常会按指数退避重试（60s → 120s → ... 上限 1 小时），不会杀死任务。
        """
        if not self.config["musing"]["enabled"] or self._musing_task is not None:
            return
        interval_minutes = float(self.config["musing"]["interval_minutes"])
        if interval_minutes <= 0:
            return
        interval = interval_minutes * 60

        async def loop():
            backoff = 60
            while True:
                await asyncio.sleep(interval if backoff == 60 else backoff)
                try:
                    await self.trigger_musing(llm_callback)
                    backoff = 60
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning("走神任务异常，%ss 后重试：%s", backoff, exc)
                    backoff = min(backoff * 2, 3600)

        self._musing_task = asyncio.create_task(loop(), name="bionic-memory-musing")

    def stop_musing_loop(self) -> None:
        """停止后台走神定时器。"""
        if self._musing_task is not None:
            self._musing_task.cancel()
            self._musing_task = None

    # ---------- 其他 ----------
    def clear(self) -> None:
        """清空该用户的全部记忆（热 + 冷 + 情绪状态）。"""
        self.db.clear_user(self.user_id)

    async def close(self) -> None:
        """停止后台任务并关闭数据库，释放资源。"""
        if self._closed:
            return
        self._closed = True
        self.stop_musing_loop()
        await asyncio.to_thread(self.db.close)
