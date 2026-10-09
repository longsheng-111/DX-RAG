"""
ARQ worker（阶段 4：摄入队列化）

独立进程运行：`arq app.worker.WorkerSettings`（docker-compose 的 worker 服务，
或本地开发另开窗口）。与 backend 共用镜像和代码，仅 CMD 不同。

消费上传接口投递的 ingest_file 任务：
  queued → running（prepare_ingest 清旧数据）→ process_file（阶段边界刷进度/心跳）
  → done / failed（失败由 ARQ 重试 1 次，仍失败保持 failed）/ cancelled（同名重传撤单）
"""
import asyncio
import logging
from datetime import datetime

from arq.connections import RedisSettings

from app.core.config import settings
from app.core.storage import job_store
from app.services.ingest import process_file

logger = logging.getLogger(__name__)

# 重试上限：max_tries=2 → 首次执行 + 失败重试 1 次
MAX_TRIES = 2


def _now() -> str:
    return datetime.now().isoformat()


class IngestCancelled(Exception):
    """任务在执行期间被撤单（同名重传/删除文件），正常中止不计失败"""


async def ingest_file(ctx: dict, job_id: str) -> None:
    job = job_store.get_job(job_id)
    if job is None:
        logger.warning("摄入任务不存在，跳过: %s", job_id)
        return
    if job["status"] != "queued":
        # cancelled（撤单）或重复投递：直接跳过
        logger.info("摄入任务 %s 状态为 %s，跳过执行", job_id, job["status"])
        return

    job_store.update_job(job_id, status="running", progress=0.05)

    def progress_cb(p: float) -> None:
        # 每个阶段边界检查撤单并刷心跳（updated_at 由 update_job 自动刷新）
        current = job_store.get_job(job_id)
        if current and current["status"] == "cancelled":
            raise IngestCancelled(job_id)
        job_store.update_job(job_id, progress=p)

    try:
        # 同名重传：清掉旧向量/旧 chunks（排队期间旧内容仍在服务检索）
        job_store.prepare_ingest(job_id)
        result = await asyncio.to_thread(
            process_file, job["file_path"], job["collection_name"], progress_cb
        )
        job_store.update_job(
            job_id,
            status="done",
            progress=1.0,
            chunk_count=result["chunks"],
            error=None,
            finished_at=_now(),
        )
        logger.info(
            "摄入完成: %s -> %s（%d 个切片）",
            job["file_name"], job["collection_name"], result["chunks"],
        )
    except IngestCancelled:
        logger.info("摄入任务已撤单，中止: %s", job_id)
    except Exception as e:
        job_try = ctx.get("job_try", 1)
        if job_try >= MAX_TRIES:
            # 重试额度耗尽：落 failed + error，不再重试
            job_store.update_job(
                job_id, status="failed", error=str(e), finished_at=_now()
            )
            logger.exception("摄入任务最终失败: %s", job_id)
        else:
            # 回滚为 queued 等待 ARQ 重试（retry 语义见 WorkerSettings.max_tries）
            job_store.update_job(job_id, status="queued", error=str(e))
            logger.warning(
                "摄入任务第 %d 次执行失败，等待重试: %s (%s)", job_try, job_id, e
            )
        raise


class WorkerSettings:
    """ARQ worker 配置：max_jobs=2（embedding 是 CPU/内存密集，多了反而拖垮机器）"""

    functions = [ingest_file]
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    max_jobs = 2
    max_tries = MAX_TRIES
    job_timeout = 3600  # 单任务最长 1 小时（大 PDF OCR 场景）
