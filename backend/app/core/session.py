"""
会话管理模块 - 本地 JSON 文件持久化

每个会话保存为一个 JSON 文件，存储在 backend/data/sessions/ 目录下。
适合当前阶段的轻量级持久化需求，无需引入数据库。
"""
import json
import logging
import os
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

# 会话 ID 格式: sess_ + 16 位 hex
_SESSION_ID_RE = re.compile(r'^sess_[a-f0-9]{16,}$')


class Session:
    """会话数据模型"""

    def __init__(
        self,
        session_id: str,
        title: str = "新会话",
        persona: str = "default",
        kb_id: str = "",
        mode: str = "qa",
        messages: Optional[list[dict]] = None,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        total_tokens: int = 0,
        compressed_history: Optional[list[dict]] = None,
        exam_state: Optional[dict] = None,
    ):
        self.session_id = session_id
        self.title = title
        self.persona = persona
        self.kb_id = kb_id or settings.vector_collection
        self.mode = mode
        self.messages = messages or []
        self.created_at = created_at or datetime.now().isoformat()
        self.updated_at = updated_at or datetime.now().isoformat()
        self.total_tokens = total_tokens
        self.compressed_history = compressed_history or []
        self.exam_state = exam_state or {}

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "title": self.title,
            "persona": self.persona,
            "kb_id": self.kb_id,
            "mode": self.mode,
            "messages": self.messages,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "total_tokens": self.total_tokens,
            "compressed_history": self.compressed_history,
            "exam_state": self.exam_state,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        """从字典构造会话，忽略未知键以兼容旧版本数据"""
        known_keys = {
            "session_id", "title", "persona", "kb_id", "mode", "messages",
            "created_at", "updated_at", "total_tokens", "compressed_history",
            "exam_state",
        }
        return cls(**{k: v for k, v in data.items() if k in known_keys})

    def add_message(
        self,
        role: str,
        content: str,
        token_cost: Optional[dict] = None,
        sources: Optional[list[dict]] = None,
        reasoning_steps: Optional[list[dict]] = None,
    ) -> None:
        """添加一条消息并更新时间戳，可选保存来源和推理路径"""
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        if token_cost:
            msg["token_cost"] = token_cost
        if sources:
            msg["sources"] = sources
        if reasoning_steps:
            msg["reasoning_steps"] = reasoning_steps
        self.messages.append(msg)
        self.updated_at = datetime.now().isoformat()

        if token_cost:
            self.total_tokens += token_cost.get("total_tokens", 0)

    def update_total_tokens(self, delta: int) -> None:
        """累加 Token 消耗"""
        self.total_tokens += delta
        self.updated_at = datetime.now().isoformat()


class SessionManager:
    """会话管理器单例"""

    _instance: Optional["SessionManager"] = None

    def __new__(cls) -> "SessionManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        self.session_dir = Path(settings.session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        # 每个会话一把锁，防止并发读写互相覆盖丢消息
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

    def _session_path(self, session_id: str) -> Path:
        # 校验格式 + 防路径穿越
        if not _SESSION_ID_RE.match(session_id):
            raise ValueError(f"Invalid session ID format: {session_id}")
        resolved = (self.session_dir / f"{session_id}.json").resolve()
        if not str(resolved).startswith(str(self.session_dir.resolve())):
            raise ValueError("Path traversal detected")
        return resolved

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
        """获取会话；不存在或读取失败时返回 None"""
        path = self._session_path(session_id)
        if not path.exists():
            return None
        with self._get_lock(session_id):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return Session.from_dict(data)
            except json.JSONDecodeError:
                # 文件损坏：备份为 .corrupt 后按不存在处理，避免反复读到坏文件
                corrupt = path.with_name(path.name + ".corrupt")
                os.replace(path, corrupt)
                logger.warning("会话文件损坏，已备份为 %s: %s", corrupt, session_id)
                return None
            except Exception:
                logger.warning("读取会话失败: %s", session_id, exc_info=True)
                return None

    def save_session(self, session: Session) -> None:
        """保存会话到 JSON 文件（持锁 + 原子写入，防并发写丢消息）"""
        path = self._session_path(session.session_id)
        session.updated_at = datetime.now().isoformat()
        with self._get_lock(session.session_id):
            tmp = path.with_name(path.name + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)

    def list_sessions(self) -> list[Session]:
        """列出所有会话，按更新时间倒序"""
        sessions = []
        for path in self.session_dir.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                sessions.append(Session.from_dict(data))
            except Exception:
                continue
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions

    def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        path = self._session_path(session_id)
        if path.exists():
            path.unlink()
            return True
        return False

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


# 全局单例
session_manager = SessionManager()
