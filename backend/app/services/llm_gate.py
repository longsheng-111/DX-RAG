"""
LLM 并发闸（阶段 4）

worker 与 API 进程都会调 LLM，加全局信号量防止并发打爆上游限额，
429 时指数退避重试（1s / 2s / 4s，最多 3 次重试）。

实现说明：方案文档写的是 asyncio.Semaphore，但 LLM 调用走同步 OpenAI SDK
（API 侧经 asyncio.to_thread 执行，worker 侧同样在事件循环里同步阻塞），
跨线程/跨事件循环都可用的正确等价物是 threading.Semaphore。
"""
import logging
import threading
import time
from typing import Callable, TypeVar

from openai import RateLimitError

from app.core.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

# 模块级单例信号量：进程内全部 LLM 调用共用同一额度
_semaphore = threading.Semaphore(settings.llm_max_concurrent)

# 429 退避序列：第 1/2/3 次重试前分别等 1s / 2s / 4s
_BACKOFF_SECONDS = (1, 2, 4)


def call_with_gate(fn: Callable[[], T]) -> T:
    """在全局信号量额度内执行 LLM 调用，429 指数退避重试后仍失败则抛出"""
    with _semaphore:
        for attempt, delay in enumerate(_BACKOFF_SECONDS):
            try:
                return fn()
            except RateLimitError:
                logger.warning(
                    "LLM 触发 429 限流，%ds 后第 %d 次重试", delay, attempt + 1
                )
                time.sleep(delay)
        # 最后一次（或重试额度耗尽后的兜底一次）不再捕获，异常原样抛出
        return fn()
