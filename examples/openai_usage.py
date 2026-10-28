"""接入 OpenAI 的完整示例。

运行：
    pip install -r requirements.txt openai
    set OPENAI_API_KEY=sk-xxx
    python examples/openai_usage.py
"""
import asyncio
import os

from openai import AsyncOpenAI

from bionic_memory import BionicMemory

client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])


async def openai_llm(prompt: str) -> str:
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
    )
    return response.choices[0].message.content or ""


async def main() -> None:
    memory = BionicMemory(user_id="openai_user", config_path="./config.yaml")

    user_message = "你还记得我上次说想去哪里旅行吗？"
    context = await memory.build_context(user_message)

    system_parts = ["你是一个记性很好的伙伴。"]
    if context["recalled_memories"]:
        system_parts.append("你想起了这些：")
        system_parts += [f"- {m['content']}" for m in context["recalled_memories"]]
    messages = [{"role": "system", "content": "\n".join(system_parts)}]
    messages += context["short_term"]
    messages.append({"role": "user", "content": user_message})

    response = await client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, temperature=0.8
    )
    reply = response.choices[0].message.content or ""

    await memory.save_interaction(user_message, reply, llm_callback=openai_llm)
    print("回复：", reply)

    await memory.close()


if __name__ == "__main__":
    asyncio.run(main())
