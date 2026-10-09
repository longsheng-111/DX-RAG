"""
数据库 ORM 模型（阶段 1：数据库化）

对应 docs/LangGraph重构设计.md 3.1 的六张业务表。与设计文档的偏差（以现有真实代码/数据为准）：
- chunks.embedding 维度为 512（bge-small-zh-v1.5 实际维度，文档中 1024 为草图）
- sessions.kb_id 沿用现有 Session 模型的知识库名称字符串，非 collection UUID 外键
  （阶段 1 铁律：对外接口与数据语义不变）
- sessions 用 JSONB 列整体存 messages / exam_state / compressed_history
  （LangGraph checkpoint 拆表属阶段 2/3）
- feedbacks 字段以 app/services/feedback.py 的真实数据结构为准
  （type=replace|inaccurate、original_chunk_id、target_chunk_id 等），非文档草图的 rating/comment
"""
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON, TypeDecorator

from app.db.base import Base

# bge-small-zh-v1.5 的向量维度（设计文档中 1024 为草图，以代码实际模型为准）
EMBEDDING_DIM = 512


class VectorVariant(TypeDecorator):
    """
    跨方言向量列：
    - postgresql 方言下映射 pgvector.sqlalchemy.Vector(512)
    - 其他方言（如本机测试用 SQLite）退化为 JSON 存 list[float]
    """

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            # 延迟导入，SQLite 测试环境也装有 pgvector，此处仅为方言分发
            from pgvector.sqlalchemy import Vector as PgVector

            return dialect.type_descriptor(PgVector(EMBEDDING_DIM))
        return dialect.type_descriptor(JSON())


# 会话 JSON 列：PostgreSQL 用 JSONB，其余方言（SQLite）退化为普通 JSON
JsonbVariant = JSON().with_variant(JSONB(), "postgresql")


class Collection(Base):
    """知识库"""

    __tablename__ = "collections"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    # 现有向量存储以 name 作为业务标识（_collections.json 的键），保留唯一约束
    name: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    # 现有数据只有 name 没有展示名，允许为空，迁移时可回填为 name
    display_name: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    files: Mapped[list["File"]] = relationship(
        back_populates="collection", cascade="all, delete-orphan"
    )
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="collection", cascade="all, delete-orphan"
    )


class File(Base):
    """上传文件（原始文件仍落盘，DB 记元数据）"""

    __tablename__ = "files"
    __table_args__ = (
        UniqueConstraint("collection_id", "file_name", name="uq_files_collection_file_name"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    collection_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("collections.id", ondelete="CASCADE"), nullable=False
    )
    file_name: Mapped[str] = mapped_column(Text, nullable=False)
    # 原始文件落盘路径，历史数据可能为空字符串
    storage_path: Mapped[str] = mapped_column(Text, nullable=False, default="")
    # sha256，同名重传判重；历史文件数据无此值，允许为空
    file_hash: Mapped[Optional[str]] = mapped_column(Text)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="pending")  # pending/processing/done/failed
    error: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    collection: Mapped[Collection] = relationship(back_populates="files")
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="file", cascade="all, delete-orphan"
    )
    ingest_jobs: Mapped[list["IngestJob"]] = relationship(
        back_populates="file", cascade="all, delete-orphan"
    )


class Chunk(Base):
    """切片 + 向量"""

    __tablename__ = "chunks"
    __table_args__ = (
        Index("idx_chunks_collection", "collection_id"),
        # HNSW 向量索引仅对 postgresql 方言生效，其余方言（SQLite）自动忽略
        Index(
            "idx_chunks_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    file_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("files.id", ondelete="CASCADE"), nullable=False
    )
    collection_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("collections.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # 允许为空：embedding 故障时切片仍可落库，后续重建向量
    embedding: Mapped[Optional[list]] = mapped_column(VectorVariant)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    file: Mapped[File] = relationship(back_populates="chunks")
    collection: Mapped[Collection] = relationship(back_populates="chunks")


class Session(Base):
    """会话（阶段 1 整体存 JSONB 列；LangGraph checkpoint 拆表属阶段 2/3）"""

    __tablename__ = "sessions"

    # 现有格式 sess_ + 16 位 hex，保持兼容
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    title: Mapped[str] = mapped_column(Text, nullable=False, default="新会话")
    persona: Mapped[str] = mapped_column(Text, nullable=False, default="default")
    # 现有 Session 以知识库名称字符串引用（见 core/session.py），非 UUID 外键
    kb_id: Mapped[str] = mapped_column(Text, nullable=False, default="")
    mode: Mapped[str] = mapped_column(Text, nullable=False, default="qa")  # qa / exam
    messages: Mapped[list] = mapped_column(JsonbVariant, nullable=False, default=list)
    compressed_history: Mapped[list] = mapped_column(JsonbVariant, nullable=False, default=list)
    exam_state: Mapped[dict] = mapped_column(JsonbVariant, nullable=False, default=dict)
    total_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    feedbacks: Mapped[list["Feedback"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class Feedback(Base):
    """反馈（替代 feedback.json，字段以 app/services/feedback.py 真实数据为准）"""

    __tablename__ = "feedbacks"

    # 现有格式 fb_ + 12 位 hex
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    session_id: Mapped[str] = mapped_column(
        Text, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    type: Mapped[str] = mapped_column(Text, nullable=False)  # replace / inaccurate
    original_chunk_id: Mapped[str] = mapped_column(Text, nullable=False, default="")
    target_chunk_id: Mapped[str] = mapped_column(Text, nullable=False, default="")
    original_content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    target_content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    note: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    session: Mapped[Session] = relationship(back_populates="feedbacks")


class IngestJob(Base):
    """摄入任务（上传接口只入库 job，实际摄入由阶段 4 的 ARQ worker 消费）"""

    __tablename__ = "ingest_jobs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    file_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("files.id", ondelete="CASCADE"), nullable=False
    )
    # queued/running/done/failed/cancelled
    status: Mapped[str] = mapped_column(Text, nullable=False, default="queued")
    progress: Mapped[float] = mapped_column(default=0.0, nullable=False)
    error: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    # 兼作 worker 心跳：worker 每次推进进度都刷新它，超时不变视为 worker 已死
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    file: Mapped[File] = relationship(back_populates="ingest_jobs")


__all__ = [
    "EMBEDDING_DIM",
    "VectorVariant",
    "JsonbVariant",
    "Collection",
    "File",
    "Chunk",
    "Session",
    "Feedback",
    "IngestJob",
]
