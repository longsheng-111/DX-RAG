"""
摄入任务存储（file 后端，阶段 4：摄入队列化）

每个任务保存为一个 JSON 文件，存储在 backend/data/ingest_jobs/ 目录下，
与 session.py 的文件持久化模式一致（持锁 + 原子写入）。
postgres 后端的对应实现是 app/db/job_repository.py（ingest_jobs 表），
两者接口逐字兼容，由 core/storage.py 按 STORAGE_BACKEND 分派。
"""
import json
import logging
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

# 任务状态机：queued → running → done / failed / cancelled
JOB_STATUS_ACTIVE = ("queued", "running")


def _now() -> str:
    return datetime.now().isoformat()


class JobStore:
    """摄入任务存储（JSON 文件版）"""

    def __init__(self):
        self.job_dir = Path(settings.job_dir)
        self.job_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _job_path(self, job_id: str) -> Path:
        # 防路径穿越：job_id 只允许本模块生成的格式
        if not job_id.startswith("job_") or "/" in job_id or "\\" in job_id or ".." in job_id:
            raise ValueError(f"Invalid job ID format: {job_id}")
        return self.job_dir / f"{job_id}.json"

    def create_job(self, file_name: str, collection_name: str, file_path: str) -> dict:
        """创建 queued 状态的摄入任务并落盘"""
        job = {
            "job_id": f"job_{uuid.uuid4().hex[:16]}",
            "file_name": file_name,
            "collection_name": collection_name,
            "file_path": file_path,
            "status": "queued",
            "progress": 0.0,
            "error": None,
            "chunk_count": None,
            "created_at": _now(),
            # updated_at 兼作 worker 心跳：worker 每次推进进度都会刷新它
            "updated_at": _now(),
            "finished_at": None,
        }
        self._save(job)
        return job

    def get_job(self, job_id: str) -> Optional[dict]:
        path = self._job_path(job_id)
        if not path.exists():
            return None
        with self._lock:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                logger.warning("读取摄入任务失败: %s", job_id, exc_info=True)
                return None

    def update_job(self, job_id: str, **fields) -> Optional[dict]:
        """更新任务字段（自动刷新 updated_at 心跳），任务不存在返回 None"""
        with self._lock:
            path = self._job_path(job_id)
            if not path.exists():
                return None
            try:
                with open(path, "r", encoding="utf-8") as f:
                    job = json.load(f)
            except Exception:
                logger.warning("读取摄入任务失败: %s", job_id, exc_info=True)
                return None
            job.update(fields)
            job["updated_at"] = _now()
            self._save_locked(job)
            return job

    def find_active_job(self, file_name: str, collection_name: str) -> Optional[dict]:
        """查找同文件同知识库的活跃（queued/running）任务，用于同名重传撤单"""
        for path in self.job_dir.glob("job_*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    job = json.load(f)
            except Exception:
                continue
            if (
                job.get("file_name") == file_name
                and job.get("collection_name") == collection_name
                and job.get("status") in JOB_STATUS_ACTIVE
            ):
                return job
        return None

    def prepare_ingest(self, job_id: str) -> None:
        """worker 启动时调用：删除同知识库同名文件的旧向量（同名重传场景）。

        与 DB 版语义差异：file 后端没有 files 表行与 job 外键的耦合，
        直接整体删除旧文件数据即可；删除推迟到 worker 启动（而非上传时），
        排队期间旧内容仍可检索。
        """
        job = self.get_job(job_id)
        if not job:
            return
        # 延迟 import：本模块被 storage.py 分派引用，避免 import 期环依赖
        from app.core.vector_store import delete_file, get_files

        existing = {f["file_name"] for f in get_files(job["collection_name"])}
        if job["file_name"] in existing:
            delete_file(job["file_name"], job["collection_name"])

    def _save(self, job: dict) -> None:
        with self._lock:
            self._save_locked(job)

    def _save_locked(self, job: dict) -> None:
        path = self._job_path(job["job_id"])
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(job, f, ensure_ascii=False, indent=2)
        tmp.replace(path)


# 全局单例
job_store = JobStore()
