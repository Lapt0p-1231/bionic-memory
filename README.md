# bionic-memory

背景：受够了老一套“短期记忆+长期记忆”打法，而且即使找一个副模型做总结也是无脑总结，上下文一长这人机味就出来了。所以东拼西凑地做了一套本地化方案，用于接入聊天软件机器人
优势：纯本地、理论上无需 GPU、不需要大模型 API。这本质上只是个本地embedding方案。

- 热记忆：最近 N 轮对话留在 SQLite 里始终完整参与上下文
- 冷记忆：每轮对话切一片存进 ChromaDB，带着“情绪效价” / “唤醒度” / “重要度”三个标签
- 语义召回：聊天时用最新一句话做向量检索，只把"此刻想起来的事"注入提示词
- 走神机制：空闲时翻出高重要度的旧记忆，生成一小段内心独白再存回去——增加记忆混乱度让它更像人一点

没有 GPU 也能跑（bge-small-zh，CPU 约 100MB），ChromaDB 零配置开箱即用。

接下来是让AI帮我总结的，如果出错了不要怪我QAQ

## 特性

- 纯本地：SQLite + ChromaDB，数据不出机器
- 零 LLM 绑定：只负责记忆管理，模型调用以回调形式传入
- 异步接口：`async` / `await`，可直接塞进任何 asyncio 应用
- 可配置：召回条数、相似度阈值、短期记忆长度、走神间隔
- 中文友好：默认嵌入模型为 `BAAI/bge-small-zh-v1.5`
- 解耦：不依赖 LangChain / LlamaIndex，依赖只有 4 个

## 安装

```bash
pip install -r requirements.txt
```

首次运行会自动下载嵌入模型（约 100MB），之后可完全离线。

## 快速开始

```python
import asyncio
from bionic_memory import BionicMemory

async def my_llm(prompt: str) -> str:
    # 你的模型调用，任意厂商，只要返回字符串
    ...

async def main():
    memory = BionicMemory(user_id="123456", config_path="./config.yaml")

    # 1. 对话前：拿组装好的上下文
    context = await memory.build_context(user_message="今天天气真差")
    # context == {
    #   "short_term": [{"role": "user", "content": "..."}, ...],
    #   "recalled_memories": [{"content": "...", "type": "dialogue",
    #                          "score": 0.72, "importance": 0.8}],
    #   "emotion_state": {"valence": -0.3, "arousal": 0.2, "importance": 0.6}
    # }

    # 2. 你自己调用模型生成回复
    reply = await my_llm("今天天气真差")

    # 3. 对话后：存记忆（同时让 LLM 分析情绪和重要度）
    await memory.save_interaction("今天天气真差", reply, llm_callback=my_llm)

    await memory.close()

asyncio.run(main())
```

## 配置

复制 `config.example.yaml` 为 `config.yaml`：

```yaml
embedding:
  model: "BAAI/bge-small-zh-v1.5"
  device: "cpu"          # 有 GPU 可改 "cuda"
  cache_dir: null

memory:
  short_term_limit: 30   # 短期记忆保留轮数
  recall_top_k: 3        # RAG 召回条数
  recall_threshold: 0.55 # 相似度阈值（低于此值不召回）

musing:
  enabled: true
  interval_minutes: 120  # 走神间隔
  min_importance: 0.6
  max_per_round: 3

storage:
  data_dir: "./bionic_memory_data"
```

也可以完全不写 YAML，直接传 dict：

```python
memory = BionicMemory(user_id="123", config={
    "memory": {"recall_top_k": 5},
    "storage": {"data_dir": "./my_data"},
})
```

## 与 LLM 解耦

本库**从不直接调用模型**。所有需要模型的地方都通过 `llm_callback` 传入：

```python
async def my_llm(prompt: str) -> str: ...
```

约定非常简单：**收到一个完整提示词字符串，返回字符串**。库内部已经拼好了提示词（见 `bionic_memory/prompts.py`），你只需要把请求发出去：

- `save_interaction(..., llm_callback=my_llm)`：分析情绪与重要度
- `trigger_musing(my_llm)`：生成走神独白
- `start_musing_loop(my_llm)`：启动后台定时走神

完整可运行示例：

- [`examples/basic_usage.py`](examples/basic_usage.py)（假 LLM，无 API Key 可直接跑）
- [`examples/openai_usage.py`](examples/openai_usage.py)
- [`examples/deepseek_usage.py`](examples/deepseek_usage.py)

## API

| 方法 | 说明 |
|---|---|
| `BionicMemory(user_id, config_path=None, config=None, data_dir=None)` | 初始化 |
| `await build_context(user_message)` | 返回 `short_term` / `recalled_memories` / `emotion_state` |
| `await save_interaction(user_message, ai_reply, llm_callback=None)` | 写热记忆 + 打分入库冷记忆 |
| `await trigger_musing(llm_callback)` | 手动触发一次走神 |
| `start_musing_loop(llm_callback)` | 启动后台走神定时器 |
| `stop_musing_loop()` | 停止定时器 |
| `clear()` | 清空该用户全部记忆 |
| `await close()` | 关闭数据库与后台任务 |

## 工作原理

```
                 ┌──────────────────────────────┐
  新消息 ───────▶│  embed(bge-small-zh, 本地CPU) │
                 └──────────────┬───────────────┘
                                ▼
                   ChromaDB 语义检索（Top-K, 阈值过滤）
                                │
   最近30轮(SQLite) ────────────┼───────▶ build_context()
                                │
                  情绪状态(SQLite profile) ─┘
                                ▼
                        你的 LLM 生成回复
                                ▼
                 ┌──────────────────────────────┐
                 │ 情绪/重要度打分(llm_callback) │
                 │ 对话切片 → ChromaDB          │
                 │ 对话轮次 → SQLite 热层       │
                 └──────────────────────────────┘
                                │
                     定时器（默认2小时）触发走神
                                ▼
             高重要度记忆 → 内心独白 → 存回 ChromaDB
```

为什么不用"总结"？摘要有损压缩，而且压完就只剩"性格描述"，具体事件全没了。切片 + 向量召回相当于**保留原件，只按需取出相关的那几页**，重要度低的自然沉底，不会污染上下文。

## 目录结构

```
bionic-memory/
├── README.md
├── requirements.txt
├── config.example.yaml
├── bionic_memory/
│   ├── __init__.py
│   ├── core.py          # 核心类 BionicMemory
│   ├── embedder.py      # 封装 bge-small-zh 加载与推理
│   ├── database.py      # SQLite + ChromaDB 的 CRUD 封装
│   ├── models.py        # Pydantic 数据模型 (Memory, Emotion)
│   └── prompts.py       # 内置的情绪提取与走神提示词
└── examples/
    ├── basic_usage.py
    ├── openai_usage.py
    └── deepseek_usage.py
```

## 已知限制

- 嵌入模型默认只针对中文优化；英文场景请把 `embedding.model` 换成 `BAAI/bge-small-en-v1.5`
- 相似度阈值与 `top_k` 需要根据你的语料微调，默认值偏保守（宁可不召回，不乱召回）
- 冷记忆目前只增不删，长期运行请关注存储体积（可按 `importance` 定期清理）
- 多进程同时写同一份数据不支持，请按进程隔离 `data_dir`

## 致谢 / 依赖

- [ChromaDB](https://github.com/chroma-core/chroma) — 向量存储
- [FastEmbed](https://github.com/qdrant/fastembed) — 本地嵌入推理
- [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5) — 中文嵌入模型
