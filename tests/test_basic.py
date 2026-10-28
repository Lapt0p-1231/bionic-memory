"""单元测试：不加载真实嵌入模型，用确定性假向量跑完整逻辑。

运行：pip install pytest && pytest -q
"""
import asyncio
import zlib

import pytest

from bionic_memory import BionicMemory, Emotion, format_recall
from bionic_memory.core import _parse_emotion


class FakeEmbedder:
    """字符袋假向量：文本越相似，余弦越高。不加载真实模型。"""

    def __init__(self):
        self.calls = 0

    def embed(self, text):
        self.calls += 1
        vec = [0.0] * 64
        for ch in text:
            vec[zlib.crc32(ch.encode("utf-8")) % 64] += 1.0
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]


def make_memory(tmp_path, **overrides):
    cfg = {
        "memory": {"recall_top_k": 3, "recall_threshold": 0.5,
                   "max_musing_recall": 1, "short_term_limit": 2},
        "musing": {"min_importance": 0.9, "max_per_round": 1,
                   "timeout_seconds": 5},
        "storage": {"data_dir": str(tmp_path)},
    }
    for section, values in overrides.items():
        cfg.setdefault(section, {}).update(values)
    memory = BionicMemory("u1", config=cfg)
    memory.embedder = FakeEmbedder()
    return memory


def fake_llm(prompt):
    async def _call(_):
        if "情绪分析器" in prompt:
            return '```json\n{"valence": "0.7", "arousal": 0.5, "importance": "0.95"}\n```'
        if "独自发呆" in prompt:
            return "那天的站台很冷，但我好像没那么怕了。"
        return "嗯"
    return _call(prompt)


def test_save_build_and_persistence(tmp_path):
    async def run():
        memory = make_memory(tmp_path)
        await memory.save_interaction("我拿到offer了", "恭喜你！", llm_callback=fake_llm)

        ctx = await memory.build_context("我拿到offer了")
        assert len(ctx["short_term"]) == 2
        assert ctx["emotion_state"]["valence"] == pytest.approx(0.7)
        assert ctx["recalled_memories"]
        assert ctx["recalled_memories"][0]["confidence"] == 1.0
        await memory.close()

        reopened = make_memory(tmp_path)
        ctx2 = await reopened.build_context("我拿到offer了")
        assert ctx2["recalled_memories"]
        await reopened.close()

    asyncio.run(run())


def test_short_term_limit(tmp_path):
    async def run():
        memory = make_memory(tmp_path)
        for i in range(3):
            await memory.save_interaction(f"消息{i}", f"回复{i}")
        ctx = await memory.build_context("消息2")
        assert len(ctx["short_term"]) == 4  # 2 轮 = 4 条
        await memory.close()

    asyncio.run(run())


def test_user_isolation(tmp_path):
    async def run():
        memory = make_memory(tmp_path)
        await memory.save_interaction("只属于u1", "嗯")
        other = BionicMemory("u2", config={
            "storage": {"data_dir": str(tmp_path)},
        })
        other.embedder = FakeEmbedder()
        assert not (await other.build_context("只属于u1"))["recalled_memories"]
        await memory.close()
        await other.close()

    asyncio.run(run())


def test_musing_source_confidence_and_no_snowball(tmp_path):
    async def run():
        memory = make_memory(tmp_path)
        await memory.save_interaction("我下周一去北京出差", "注意安全", llm_callback=fake_llm)

        created = await memory.trigger_musing(fake_llm)
        assert len(created) == 1
        assert created[0].type == "musing"
        assert created[0].source_id
        assert created[0].confidence == 0.4

        important = memory.db.get_important("u1", limit=10, min_importance=0.5)
        assert all(m.type != "musing" for m in important)

        await memory.close()

    asyncio.run(run())


def test_recall_caps_musing(tmp_path):
    async def run():
        memory = make_memory(tmp_path, musing={"min_importance": 0.5})
        await memory.save_interaction("站台很冷", "多穿点", llm_callback=fake_llm)
        for _ in range(3):
            await memory.trigger_musing(fake_llm)

        vec = memory.embedder.embed("那天的站台很冷，但我好像没那么怕了。")
        hits = memory.db.recall("u1", vec, top_k=3, threshold=0.2, max_musing=1)
        types = [m.type for m, _ in hits]
        assert types.count("musing") == 1, types
        assert "dialogue" in types
        await memory.close()

    asyncio.run(run())


def test_clear(tmp_path):
    async def run():
        memory = make_memory(tmp_path)
        await memory.save_interaction("会被清掉", "嗯")
        memory.clear()
        ctx = await memory.build_context("会被清掉")
        assert not ctx["recalled_memories"] and not ctx["short_term"]
        await memory.close()

    asyncio.run(run())


def test_hnsw_metadata(tmp_path):
    memory = make_memory(tmp_path)
    meta = memory.db._collection.metadata
    assert meta["hnsw:M"] == 32
    assert meta["hnsw:construction_ef"] == 200
    assert meta["hnsw:search_ef"] == 100
    asyncio.run(memory.close())


def test_musing_loop_interval_zero(tmp_path):
    async def run():
        memory = make_memory(tmp_path, musing={"interval_minutes": 0})
        memory.start_musing_loop(fake_llm)
        assert memory._musing_task is None, "间隔为 0 时不应启动走神任务"
        await memory.close()

    asyncio.run(run())


def test_embedder_cache():
    from bionic_memory.embedder import Embedder

    embedder = Embedder(cache_size=2)
    embedder._model = type("M", (), {
        "embed": lambda self, texts: iter([[1.0, 2.0]])
    })()
    assert embedder.embed("hello") == [1.0, 2.0]
    assert embedder.embed("hello") == [1.0, 2.0]
    embedder._cache.pop("hello")  # 模拟缓存失效后重新推理
    assert embedder.embed("hello") == [1.0, 2.0]
    embedder.clear_cache()
    assert embedder.embed("hello") == [1.0, 2.0]


@pytest.mark.parametrize("raw,expected", [
    ('{"valence": 0.5, "arousal": 0.5, "importance": 0.8}', (0.5, 0.5, 0.8)),
    ('```json\n{"valence": -1.5, "arousal": 2, "importance": 999}\n```', (-1.0, 1.0, 1.0)),
    ('结果如下 {"valence": "0.2", "arousal": "0.3", "importance": "0.4"} 完毕', (0.2, 0.3, 0.4)),
    ('{"valence": true, "arousal": null}', (0.0, 0.0, 0.5)),
    ('完全不是JSON', (0.0, 0.0, 0.5)),
])
def test_parse_emotion_dirty(raw, expected):
    emotion = _parse_emotion(raw)
    assert isinstance(emotion, Emotion)
    assert (emotion.valence, emotion.arousal, emotion.importance) == expected


def test_format_recall():
    text = format_recall([
        {"content": "一起去过海边", "type": "dialogue"},
        {"content": "好像有人说过要离开", "type": "musing"},
    ])
    assert text.splitlines()[0] == "- 一起去过海边"
    assert "梦" in text.splitlines()[1]
    assert "好像有人说过要离开" in text.splitlines()[1]
