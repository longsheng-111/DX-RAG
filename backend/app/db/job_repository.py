"""
摄入任务存储（postgres 后端，阶段 4：摄入队列化）

基于 ingest_jobs 表（阶段 1 已建）+ files 表实现，接口与
app/core/job_store.py 的 JSON 文件版逐字兼容，由 core/storage.py 分派。

与文件版的关键差异（由 files 表 UNIQUE(collection_id, file_name) 约束推导）：
- create_job 复用同知识库同名文件的既有 File 行（同名重传场景），不新建第二行；
  旧 chunks 的清理推迟到 worker 启动时（prepare_ingest），排队期间旧内容仍可检索
- prepare_ingest 只清 chunks 不删 File 行（job 经 file_id 外键挂在 File 上，
  删行会级联删掉任务本身），File.status 转为 processing
"""
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from app.db.base import new_session
from app.db.models import Chunk, Collection, File, IngestJob
from app.db.vector_repository import _get_or_create_collection, _get_collection

# 任务状态机：queued → running → done / failed / cancelled
JOB_STATUS_ACTIVE = ("queued", "running")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt) -> Optional[str]:
    return dt.isoformat() if dt else None


def _to_dict(session, job: IngestJob) -> dict:
    file_row = session.get(File, job.file_id)
    col = session.get(Collection, file_row.collection_id) if file_row else None
    chunk_count = None
    if job.status == "done" and file_row is not None:
        chunk_count = (
            session.query(Chunk).filter(Chunk.file_id == file_row.id).count()
        )
    return {
        "job_id": str(job.id),
        "file_name": file_row.file_name if file_row else "",
        "collection_name": col.name if col else "",
        "file_path": (file_row.storage_path or "") if file_row else "",
        "status": job.status,
        "progress": job.progress,
        "error": job.error,
        "chunk_count": chunk_count,
        "created_at": _iso(job.created_at),
        "updated_at": _iso(job.updated_at),
        "finished_at": _iso(job.finished_at),
    }


class JobStore:
    """摄入任务存储（DB 版）"""

    def create_job(self, file_name: str, collection_name: str, file_path: str) -> dict:
        """创建 queued 状态任务；同知识库同名文件复用既有 File 行（见模块 docstring）"""
        with new_session() as session:
            col = _get_or_create_collection(session, collection_name)
            file_row = (
                session.query(File)
                .filter(File.collection_id == col.id, File.file_name == file_name)
                .first()
            )
            if file_row is None:
                size = 0
                try:
                    size = os.path.getsize(file_path)
                except OSError:
                    pass
                file_row = File(
                    collection_id=col.id,
                    file_name=file_name,
                    storage_path=file_path,
                    size_bytes=size,
                    status="pending",
                )
                session.add(file_row)
                session.flush()
            else:
                # 同名重传：更新落盘路径指向新文件；status 留给 worker 流转，
                # 旧 chunks 在 prepare_ingest 前仍在服务检索
                file_row.storage_path = file_path

            job = IngestJob(
                file_id=file_row.id,
                status="queued",
                progress=0.0,
                updated_at=_now(),
            )
            session.add(job)
            session.commit()
            session.refresh(job)
            return _to_dict(session, job)

    def get_job(self, job_id: str) -> Optional[dict]:
        try:
            job_uuid = uuid.UUID(job_id)
        except (ValueError, AttributeError):
            return None
        with new_session() as session:
            job = session.get(IngestJob, job_uuid)
            if job is None:
                return None
            return _to_dict(session, job)

    def update_job(self, job_id: str, **fields) -> Optional[dict]:
        """更新任务字段（自动刷新 updated_at 心跳），任务不存在返回 None"""
        try:
            job_uuid = uuid.UUID(job_id)
        except (ValueError, AttributeError):
            return None
        # dict 接口字段 → ORM 列映射；非列字段（file_name 等）静默忽略
        allowed = {"status", "progress", "error", "finished_at"}
        with new_session() as session:
            job = session.get(IngestJob, job_uuid)
            if job is None:
                return None
            for key, value in fields.items():
                if key not in allowed:
                    continue
                if key == "finished_at" and isinstance(value, str):
                    value = datetime.fromisoformat(value)
                setattr(job, key, value)
            job.updated_at = _now()
            session.commit()
            session.refresh(job)
            return _to_dict(session, job)

    def find_active_job(self, file_name: str, collection_name: str) -> Optional[dict]:
        """查找同文件同知识库的活跃（queued/running）任务，用于同名重传撤单"""
        with new_session() as session:
            col = _get_collection(session, collection_name)
            if col is None:
                return None
            job = (
                session.query(IngestJob)
                .join(File, IngestJob.file_id == File.id)
                .filter(
                    File.collection_id == col.id,
                    File.file_name == file_name,
                    IngestJob.status.in_(JOB_STATUS_ACTIVE),
                )
                .order_by(IngestJob.created_at.desc())
                .first()
            )
            if job is None:
                return None
            return _to_dict(session, job)

    def prepare_ingest(self, job_id: str) -> None:
        """worker 启动时调用：清掉旧 chunks（保留 File 行与任务外键），File 转 processing"""
        try:
            job_uuid = uuid.UUID(job_id)
        except (ValueError, AttributeError):
            return
        with new_session() as session:
            job = session.get(IngestJob, job_uuid)
            if job is None:
                return
            session.query(Chunk).filter(Chunk.file_id == job.file_id).delete(
                synchronize_session=False
            )
            file_row = session.get(File, job.file_id)
            if file_row is not None:
                file_row.status = "processing"
                file_row.error = None
            session.commit()


# 全局单例（无状态，方法各自开 DB Session）
job_store = JobStore()
