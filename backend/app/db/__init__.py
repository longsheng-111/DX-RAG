"""
数据库层（阶段 1：数据库化）——对外暴露 Base / Engine 入口 / 全部模型

用法：
    from app.db import get_db            # FastAPI Depends
    from app.db import new_session       # 脚本/服务内手动获取 Session
    from app.db.models import Chunk      # 具体模型
"""
from app.db.base import Base, SessionLocal, dispose_engine, get_db, get_engine, new_session
from app.db.models import Chunk, Collection, Feedback, File, IngestJob, Session

__all__ = [
    "Base",
    "SessionLocal",
    "get_engine",
    "new_session",
    "get_db",
    "dispose_engine",
    "Collection",
    "File",
    "Chunk",
    "Session",
    "Feedback",
    "IngestJob",
]
