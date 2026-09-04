"""
会话 DB Repository（阶段 1：数据库化，波次 2）

替代 app.core.session.SessionManager 的本地 JSON 文件存储，
公开接口（方法名 / 参数 / 返回类型）与 SessionManager 逐字兼容：
- 返回值仍是 app.core.session.Session 领域对象，不泄露 ORM 对象
- session_id 格式校验 sess_ + 16 位 hex，非法 id 抛 ValueError（与文件版一致）
- created_at / updated_at 由本类显式赋 datetime.now()（naive 本地时间），
  前端按本地时区解析 isoformat 字符串，不使用 timezone-aware 或 DB server_default
- 每会话一把 threading.Lock（沿用文件版并发语义），写操作走 DB 事务
"""
import logging
import re
import threading
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import select

from app.core.config import settings
from app.core.session import Session
from app.db.base import new_session
from app.db.models import Session as DbSession

logger = logging.getLogger(__name__)

# 会话 ID 格式: sess_ + 16 位 hex（与 app.core.session 保持一致）
_SESSION_ID_RE = re.compile(r"^sess_[a-f0-9]{16,}$")


def _naive_iso(dt: datetime) -> str:
    """DB 时间 → naive 本地时间 isoformat（PostgreSQL timestamptz 读回可能带时区，统一剥掉）"""
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    return dt.isoformat()


def _to_domain(row: DbSession) -> Session:
    """ORM 行 → 领域对象（复用 from_dict 的字段过滤，与文件版读取路径一致）"""
    return Session.from_dict({
        "session_id": row.id,
        "title": row.title,
        "persona": row.persona,
        "kb_id": row.kb_id,
        "mode": row.mode,
        "messages": row.messages or [],
        "compressed_history": row.compressed_history or [],
        "exam_state": row.exam_state or {},
        "total_tokens": row.total_tokens or 0,
        "created_at": _naive_iso(row.created_at),
        "updated_at": _naive_iso(row.updated_at),
    })


def _fill_row(row: DbSession, session: Session) -> DbSession:
    """领域对象 → ORM 行；时间显式解析回 naive datetime 落库，不依赖 server_default"""
    row.title = session.title
    row.persona = session.persona
    row.kb_id = session.kb_id
    row.mode = session.mode
    row.messages = session.messages
    row.compressed_history = session.compressed_history
    row.exam_state = session.exam_state
    row.total_tokens = session.total_tokens
    row.created_at = datetime.fromisoformat(session.created_at)
    row.updated_at = datetime.fromisoformat(session.updated_at)
    return row


class SessionRepository:
    """会话 DB 存储，公开接口与 app.core.session.SessionManager 完全一致"""

    _instance: Optional["SessionRepository"] = None

    def __new__(cls) -> "SessionRepository":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        # 每个会话一把锁，防止并发读写互相覆盖丢消息（沿用文件版并发语义）
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _get_lock(self, session_id: str) -> threading.Lock:
        """获取指定会话的锁，不存在则创建"""
        with self._locks_guard:
            lock = self._locks.get(session_id)
            if lock is None:
                lock = threading.Lock()
                self._locks[session_id] = lock
            return lock

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        """校验会话 ID 格式（与文件版 _session_path 的校验行为一致）"""
        if not _SESSION_ID_RE.match(session_id):
            raise ValueError(f"Invalid session ID format: {session_id}")

    def create_session(
        self,
        persona: str = "default",
        kb_id: Optional[str] = None,
        title: str = "新会话",
        mode: str = "qa",
        exam_state: Optional[dict] = None,
    ) -> Session:
        """创建新会话，支持直接指定考官模式配置"""
        session_id = f"sess_{uuid.uuid4().hex[:16]}"
        session = Session(
            session_id=session_id,
            title=title,
            persona=persona,
            kb_id=kb_id or settings.vector_collection,
            mode=mode,
            exam_state=exam_state or {},
        )
        self.save_session(session)
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """获取会话；不存在时返回 None"""
        self._validate_session_id(session_id)
        with self._get_lock(session_id):
            with new_session() as db:
                row = db.get(DbSession, session_id)
                return _to_domain(row) if row is not None else None

    def save_session(self, session: Session) -> None:
        """保存会话（持锁 + DB 事务 upsert，防并发写丢消息）"""
        self._validate_session_id(session.session_id)
        session.updated_at = datetime.now().isoformat()
        with self._get_lock(session.session_id):
            with new_session() as db:
                row = db.get(DbSession, session.session_id)
                if row is None:
                    row = DbSession(id=session.session_id)
                    db.add(row)
                _fill_row(row, session)
                db.commit()

    def list_sessions(self) -> list[Session]:
        """列出所有会话，按更新时间倒序"""
        with new_session() as db:
            rows = db.scalars(select(DbSession)).all()
        sessions = [_to_domain(row) for row in rows]
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions

    def delete_session(self, session_id: str) -> bool:
        """删除会话（反馈经 ORM 关系 cascade + FK ondelete CASCADE 级联删除）"""
        self._validate_session_id(session_id)
        with self._get_lock(session_id):
            with new_session() as db:
                row = db.get(DbSession, session_id)
                if row is None:
                    return False
                db.delete(row)
                db.commit()
                return True

    def switch_persona(
        self,
        session_id: str,
        new_persona: str,
        clear_history: bool = False,
    ) -> Optional[Session]:
        """切换会话角色"""
        session = self.get_session(session_id)
        if not session:
            return None
        session.persona = new_persona
        if clear_history:
            session.messages = []
            session.compressed_history = []
        self.save_session(session)
        return session

    def update_title(self, session_id: str, title: str) -> Optional[Session]:
        """更新会话标题"""
        session = self.get_session(session_id)
        if not session:
            return None
        session.title = title
        self.save_session(session)
        return session


# 全局单例（与文件版 session_manager 用法一致，供波次 3 接线）
session_repository = SessionRepository()
