"""最小可运行示例：用一个假 LLM 演示完整流程。

运行：
    pip install -r requirements.txt
    python examples/basic_usage.py
"""
import asyncio
import json

from bionic_memory import BionicMemory


async def fake_llm(prompt: str) -> str:
    """假装是 LLM。真实场景请换成你的模型调用。

    本库约定：回调接收一个完整提示词字符串，返回字符串。
    """
    if "情绪分析器" in prompt:
        return json.dumps({"valence": -0.3, "arousal": 0.2, "importance": 0.6})
    if "独自发呆" in prompt:
        return "那天的雨好像也没那么讨厌，至少伞下是暖的。"
    return "嗯，我在听。"


async def main() -> None:
    memory = BionicMemory(
        user_id="demo_user",
        config_path="./config.example.yaml",
    )

    # 1. 对话前：拿上下文（此时还没存任何东西，召回为空）
    context = await memory.build_context("今天天气真差")
    print("首次上下文：", context)

    # 2. 对话后：存入记忆，同时让 LLM 分析情绪
    await memory.save_interaction(
        user_message="今天天气真差",
        ai_reply="（叹气）是啊，出门记得带伞哦。",
        llm_callback=fake_llm,
    )

    # 3. 再聊一轮，之后用相关的话去召回
    await memory.save_interaction(
        user_message="我最讨厌下雨了，鞋都湿了",
        ai_reply="下次记得穿防水的鞋呀。",
        llm_callback=fake_llm,
    )
    context = await memory.build_context("下雨天真烦")
    print("\n召回结果：")
    for item in context["recalled_memories"]:
        print(f"  [{item['type']}] ({item['score']:.3f}) {item['content']}")
    print("情绪状态：", context["emotion_state"])

    # 4. 手动触发一次走神
    musings = await memory.trigger_musing(fake_llm, min_importance=0.5)
    print("\n走神独白：")
    for m in musings:
        print("  ", m.content)

    await memory.close()


if __name__ == "__main__":
    asyncio.run(main())
