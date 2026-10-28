"""接入 DeepSeek（OpenAI 兼容接口）的完整示例。

运行：
    pip install -r requirements.txt openai
    set DEEPSEEK_API_KEY=sk-xxx
    python examples/deepseek_usage.py
"""
import asyncio
import os

from openai import AsyncOpenAI

from bionic_memory import BionicMemory

client = AsyncOpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)


async def deepseek_llm(prompt: str) -> str:
    response = await client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
    )
    return response.choices[0].message.content or ""


async def main() -> None:
    memory = BionicMemory(user_id="deepseek_user", config_path="./config.yaml")

    user_message = "今天工作好累，被老板骂了"
    context = await memory.build_context(user_message)

    # 用召回的记忆拼一个最小的系统提示（你可以换成任何自己的逻辑）
    system_parts = ["你是一个温柔的朋友。"]
    if context["recalled_memories"]:
        system_parts.append("你突然想起这些往事：")
        for m in context["recalled_memories"]:
            # 走神产物是梦，措辞上带点不确定，别当成事实
            if m["type"] == "musing":
                system_parts.append(f"- （好像做过一个类似的梦，画面很模糊）{m['content']}")
            else:
                system_parts.append(f"- {m['content']}")
    messages = [{"role": "system", "content": "\n".join(system_parts)}]
    messages += context["short_term"]
    messages.append({"role": "user", "content": user_message})

    response = await client.chat.completions.create(
        model="deepseek-chat", messages=messages, temperature=0.8
    )
    reply = response.choices[0].message.content or ""

    await memory.save_interaction(user_message, reply, llm_callback=deepseek_llm)
    print("回复：", reply)

    # 把走神定时器挂上，之后每 interval_minutes 自动胡思乱想
    memory.start_musing_loop(deepseek_llm)
    await asyncio.sleep(5)

    await memory.close()


if __name__ == "__main__":
    asyncio.run(main())
