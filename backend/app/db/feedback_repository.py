"""
反馈 DB Repository（阶段 1：数据库化，波次 2）

替代 app.services.feedback.FeedbackStore 的本地 JSON 文件存储，
公开接口（add_feedback / get_session_feedback / format_feedback_for_prompt）
与 FeedbackStore 逐字兼容，返回字典的字段集与字段顺序保持一致。

DB 化后天然修复审计项 #13：反馈按行独立存储，单条记录写入失败/损坏
不再像单文件 JSON 那样清空全部会话的反馈。
"""
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import select

from app.db.base import new_session
from app.db.models import Feedback as DbFeedback


def _naive_iso(dt: datetime) -> str:
    """DB 时间 → naive 本地时间 isoformat（PostgreSQL timestamptz 读回可能带时区，统一剥掉）"""
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    return dt.isoformat()


def _to_dict(row: DbFeedback) -> dict:
    """ORM 行 → 反馈字典（字段集与字段顺序与文件版 FeedbackStore 一致）"""
    return {
        "id": row.id,
        "session_id": row.session_id,
        "original_chunk_id": row.original_chunk_id,
        "target_chunk_id": row.target_chunk_id,
        "type": row.type,
        "original_content": row.original_content,
        "target_content": row.target_content,
        "note": row.note,
        "created_at": _naive_iso(row.created_at),
    }


class FeedbackRepository:
    """反馈 DB 存储，公开接口与 app.services.feedback.FeedbackStore 完全一致"""

    _instance: Optional["FeedbackRepository"] = None

    def __new__(cls) -> "FeedbackRepository":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        # 无本地状态；Engine 由 app.db.base 惰性创建
        pass

    def add_feedback(
        self,
        session_id: str,
        original_chunk_id: str,
        target_chunk_id: str,
        feedback_type: str,
        original_content: str = "",
        target_content: str = "",
        note: str = "",
    ) -> dict:
        """追加一条反馈记录"""
        row = DbFeedback(
            id=f"fb_{uuid.uuid4().hex[:12]}",
            session_id=session_id,
            original_chunk_id=original_chunk_id,
            target_chunk_id=target_chunk_id,
            type=feedback_type,
            original_content=original_content,
            target_content=target_content,
            note=note,
            # 显式赋 naive 本地时间，与文件版 datetime.now().isoformat() 语义一致
            created_at=datetime.now(),
        )
        with new_session() as db:
            db.add(row)
            db.commit()
        return _to_dict(row)

    def get_session_feedback(self, session_id: str) -> list[dict]:
        """获取指定会话的全部历史反馈（按写入时间正序，与文件版追加顺序一致）"""
        with new_session() as db:
            rows = db.scalars(
                select(DbFeedback)
                .where(DbFeedback.session_id == session_id)
                .order_by(DbFeedback.created_at.asc())
            ).all()
        return [_to_dict(row) for row in rows]

    def format_feedback_for_prompt(self, session_id: str) -> str:
        """把历史反馈格式化为 System Prompt 中的事实修正约束"""
        feedbacks = self.get_session_feedback(session_id)
        if not feedbacks:
            return ""

        lines = ["【历史反馈 / 事实修正约束】"]
        for i, fb in enumerate(feedbacks, 1):
            lines.append(f"{i}. 反馈类型：{fb['type']}")
            if fb.get("original_content"):
                lines.append(f"   原引用片段：{fb['original_content'][:200]}")
            if fb.get("target_content"):
                lines.append(f"   应优先采用的片段：{fb['target_content'][:200]}")
            if fb.get("note"):
                lines.append(f"   用户备注：{fb['note']}")
        lines.append("以上约束优先级高于检索结果，后续回答请优先遵守。")
        return "\n".join(lines)


# 全局单例（与文件版 feedback_store 用法一致，供波次 3 接线）
feedback_repository = FeedbackRepository()
