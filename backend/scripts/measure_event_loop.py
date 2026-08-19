"""
事件循环阻塞测量脚本

原理：把 qa_query 替换为 sleep 2 秒的假函数（模拟 LLM + 检索的耗时），
向 /api/chat 发一个请求，并在它处理期间并发轮询 /health：

- 若 /api/chat 的同步调用直接跑在事件循环上：/health 要等 chat 结束才响应，延迟 ≈ 2s
- 若同步调用被 asyncio.to_thread 扔进线程池：/health 照常毫秒级响应

用法（在 backend/ 目录下）：
    py -3.12 scripts/measure_event_loop.py
"""
import asyncio
import sys
import time
from pathlib import Path

# 保证能 import app 包（脚本位于 backend/scripts/ 下）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from app.main import app
from app.api import routes
from app.core.config import settings
from app.services.token_tracker import TokenCost

FAKE_DELAY = 2.0  # 假 QA 的耗时，模拟 LLM 推理


def fake_qa_query(**kwargs):
    """模拟一次耗时 2s 的同步 QA 调用"""
    time.sleep(FAKE_DELAY)
    return {
        "answer": "（假答案）",
        "sources": [],
        "token_cost": {},
        "reasoning_steps": [],
        "queries": [kwargs.get("question", "")],
    }


def fake_generate_session_title(question: str):
    """标题生成也替换成假函数，避免真的去调 LLM"""
    return ("测试会话", TokenCost())


async def main():
    settings.ensure_dirs()  # ASGITransport 不触发 lifespan，目录在这里确保

    # 替换 routes 模块里引用的函数名（monkeypatch）
    routes.qa_query = fake_qa_query
    routes.generate_session_title = fake_generate_session_title

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        chat_task = asyncio.create_task(
            client.post("/api/chat", json={"question": "测试并发"}, timeout=30)
        )
        await asyncio.sleep(0.2)  # 确保 chat 先进入处理

        latencies = []
        for _ in range(5):
            start = time.perf_counter()
            resp = await client.get("/api/health", timeout=30)
            assert resp.status_code == 200
            latencies.append(time.perf_counter() - start)
            await asyncio.sleep(0.2)

        chat_resp = await chat_task
        assert chat_resp.status_code == 200, chat_resp.text

    worst = max(latencies)
    print(f"chat 假处理耗时: {FAKE_DELAY:.1f}s")
    print("chat 处理期间 /health 的 5 次延迟: "
          + ", ".join(f"{x * 1000:.0f}ms" for x in latencies))
    if worst > FAKE_DELAY / 2:
        print("结论: /health 被拖住 —— 事件循环被阻塞 [BLOCKED]")
    else:
        print("结论: /health 毫秒级响应 —— 事件循环畅通 [OK]")


if __name__ == "__main__":
    asyncio.run(main())
