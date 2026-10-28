"""存储层：SQLite 负责热记忆与人设状态，ChromaDB 负责冷记忆向量检索。"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time

from .models import Emotion, Memory


class Storage:
    """SQLite + ChromaDB 的持久化封装。

    热记忆（short_term / profile）在 SQLite，冷记忆（memories）在 ChromaDB。
    本类所有方法都是同步的，异步调用方请用 asyncio.to_thread 包装。
    """

    def __init__(self, data_dir: str, hnsw: dict | None = None):
        os.makedirs(data_dir, exist_ok=True)
        self.data_dir = data_dir
        self._lock = threading.Lock()

        self._conn = sqlite3.connect(
            os.path.join(data_dir, "memory.db"), check_same_thread=False
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_tables()

        import chromadb

        self._client = chromadb.PersistentClient(path=os.path.join(data_dir, "chroma"))
        # HNSW 参数显式配置：写入性能与召回率平衡
        hnsw_cfg = {
            "space": "cosine",
            "M": 32,
            "construction_ef": 200,
            "search_ef": 100,
        }
        hnsw_cfg.update(hnsw or {})
        metadata = {
            "hnsw:space": str(hnsw_cfg["space"]),
            "hnsw:M": int(hnsw_cfg["M"]),
            "hnsw:construction_ef": int(hnsw_cfg["construction_ef"]),
            "hnsw:search_ef": int(hnsw_cfg["search_ef"]),
        }
        self._collection = self._client.get_or_create_collection(
            "memories", metadata=metadata
        )

    def _init_tables(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    type TEXT NOT NULL,
                    valence REAL DEFAULT 0,
                    arousal REAL DEFAULT 0,
                    importance REAL DEFAULT 0.5,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memories_user
                    ON memories (user_id, importance DESC);
                CREATE TABLE IF NOT EXISTS short_term (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_short_user
                    ON short_term (user_id, id);
                CREATE TABLE IF NOT EXISTS profile (
                    user_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL
                );
                """
            )

    # ---------- 热记忆 ----------
    def append_turn(self, user_id: str, role: str, content: str,
                    keep_turns: int = 30) -> None:
        """追加一条短期记忆，并把超出 keep_turns 轮的旧记录裁掉。"""
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO short_term (user_id, role, content, created_at) VALUES (?,?,?,?)",
                (user_id, role, content, time.time()),
            )
            self._conn.execute(
                """
                DELETE FROM short_term
                WHERE user_id = ? AND id NOT IN (
                    SELECT id FROM short_term WHERE user_id = ?
                    ORDER BY id DESC LIMIT ?
                )
                """,
                (user_id, user_id, keep_turns * 2),
            )

    def get_short_term(self, user_id: str, keep_turns: int = 30) -> list[dict]:
        """按时间正序返回最近 keep_turns 轮对话。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content FROM short_term WHERE user_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (user_id, keep_turns * 2),
            ).fetchall()
        return [{"role": r, "content": c} for r, c in reversed(rows)]

    # ---------- 冷记忆 ----------
    def add_memory(self, user_id: str, content: str, embedding: list[float],
                   memory_type: str = "dialogue",
                   emotion: Emotion | None = None,
                   source_id: str | None = None,
                   confidence: float | None = None) -> Memory:
        """写入一条冷记忆（SQLite 元数据 + ChromaDB 向量）。

        走神产物（type="musing"）默认 confidence=0.4，其余默认 1.0。
        """
        emotion = (emotion or Emotion()).clamped()
        if confidence is None:
            # 走神产物是"梦"，默认低置信度
            confidence = 0.4 if memory_type == "musing" else 1.0
        confidence = max(0.0, min(1.0, float(confidence)))
        memory = Memory(user_id=user_id, content=content, type=memory_type,
                        emotion=emotion, source_id=source_id,
                        confidence=confidence)
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO memories (id, user_id, content, type, valence, arousal,"
                " importance, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (memory.id, user_id, content, memory_type, emotion.valence,
                 emotion.arousal, emotion.importance, memory.created_at),
            )
        self._collection.add(
            ids=[memory.id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[{
                "user_id": user_id,
                "type": memory_type,
                "valence": emotion.valence,
                "arousal": emotion.arousal,
                "importance": emotion.importance,
                "source_id": str(source_id or ""),
                "confidence": confidence,
                "created_at": memory.created_at,
            }],
        )
        return memory

    def recall(self, user_id: str, embedding: list[float], top_k: int = 3,
               threshold: float = 0.55,
               max_musing: int = 1) -> list[tuple[Memory, float]]:
        """语义召回。

        真实对话优先；走神产物（梦）最多返回 max_musing 条，
        避免虚构内容挤掉真实记忆。
        """
        total = self._collection.count()
        if total == 0:
            return []
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=max(top_k * 2, top_k),
            where={"user_id": user_id},
            include=["documents", "metadatas", "distances"],
        )
        ids = result.get("ids") or [[]]
        docs = result.get("documents") or [[]]
        metas = result.get("metadatas") or [[]]
        dists = result.get("distances") or [[]]

        dialogues: list[tuple[Memory, float]] = []
        musings: list[tuple[Memory, float]] = []
        for mid, doc, meta, dist in zip(ids[0], docs[0], metas[0], dists[0]):
            score = 1.0 - float(dist)
            if score < threshold:
                continue
            meta = meta or {}
            memory = Memory(
                id=mid,
                user_id=user_id,
                content=doc,
                type=meta.get("type", "dialogue"),
                emotion=Emotion(
                    valence=float(meta.get("valence", 0)),
                    arousal=float(meta.get("arousal", 0)),
                    importance=float(meta.get("importance", 0.5)),
                ),
                source_id=meta.get("source_id") or None,
                confidence=float(meta.get("confidence", 1.0)),
                created_at=float(meta.get("created_at", 0)),
            )
            # 真实对话优先：走神产物最多召回 max_musing 条，梦境不能挤掉事实
            if memory.type == "musing":
                musings.append((memory, score))
            else:
                dialogues.append((memory, score))

        picks = (dialogues + musings[:max(0, max_musing)])[:top_k]
        return picks

    def get_important(self, user_id: str, limit: int = 3,
                      min_importance: float = 0.6) -> list[Memory]:
        """取高重要度记忆作为走神素材（排除走神产物，避免梦生梦）。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, content, type, valence, arousal, importance, created_at "
                "FROM memories WHERE user_id = ? AND importance >= ? "
                "AND type != 'musing' "
                "ORDER BY importance DESC, created_at DESC LIMIT ?",
                (user_id, min_importance, limit),
            ).fetchall()
        return [
            Memory(id=r[0], user_id=user_id, content=r[1], type=r[2],
                   emotion=Emotion(valence=r[3], arousal=r[4], importance=r[5]),
                   created_at=r[6])
            for r in rows
        ]

    # ---------- 人设状态 ----------
    def set_profile(self, user_id: str, state: dict) -> None:
        """覆盖写入用户情绪状态。"""
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO profile (user_id, state) VALUES (?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET state = excluded.state",
                (user_id, json.dumps(state, ensure_ascii=False)),
            )

    def get_profile(self, user_id: str) -> dict:
        """读取用户情绪状态，没有则返回空 dict。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT state FROM profile WHERE user_id = ?", (user_id,)
            ).fetchone()
        if not row:
            return {}
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return {}

    # ---------- 清理 ----------
    def clear_user(self, user_id: str) -> None:
        """删除该用户的全部热记忆、冷记忆与情绪状态。"""
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM short_term WHERE user_id = ?", (user_id,))
            self._conn.execute("DELETE FROM profile WHERE user_id = ?", (user_id,))
            rows = self._conn.execute(
                "SELECT id FROM memories WHERE user_id = ?", (user_id,)
            ).fetchall()
            self._conn.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
        ids = [r[0] for r in rows]
        if ids:
            self._collection.delete(ids=ids)

    def close(self) -> None:
        """关闭 SQLite 连接。"""
        with self._lock:
            self._conn.close()
