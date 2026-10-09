"""
摄入任务队列入口（阶段 4：ARQ + Redis）

API 进程经本模块把摄入任务投递到 Redis 队列，由独立 worker 进程
（arq app.worker.WorkerSettings）消费。连接池惰性创建：
- 不上传就不碰 Redis，本地开发不起 Redis 时其余接口不受影响
- Redis 不可达时上传接口返回 503（路由层捕获），不会拖垮整个应用
"""
import logging
from typing import Optional

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings

from app.core.config import settings

logger = logging.getLogger(__name__)

_pool: Optional[ArqRedis] = None


async def get_pool() -> ArqRedis:
    """惰性创建全局 ARQ 连接池（Redis 不可达时抛异常，由调用方兜底）"""
    global _pool
    if _pool is None:
        _pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    return _pool


async def enqueue_ingest(job_id: str) -> None:
    """投递摄入任务；Redis 不可达抛异常（路由层转为 503）"""
    pool = await get_pool()
    await pool.enqueue_job("ingest_file", job_id)


async def close_pool() -> None:
    """关闭连接池（应用关停钩子用）"""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
