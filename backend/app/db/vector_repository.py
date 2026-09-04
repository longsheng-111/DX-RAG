"""
向量库的数据库实现（阶段 1：数据库化 波次 2）

对外公开接口与 app/core/vector_store.py（FAISS 文件版）逐字兼容：
routes.py / services/retrieval.py / services/ingest.py 只需改 import 来源即可切换
（接线由波次 3 统一执行，本模块不改动任何消费方）。

与文件版的行为差异（均为审计修复项 / 已定案裁决的有意为之）：
- 审计 #10：rename_collection 在 DB 下是一次 UPDATE，不再触碰文件系统目录，
  Windows 下重命名 FileExistsError 一类问题随之消除
- 审计 #11：不存在"索引文件损坏封死"问题（无独立索引文件，向量就在 chunks 表内）
- 审计 #12：delete_file 是单事务 DELETE，无"索引已改、元数据未写"的中间态
- 审计 P2：add_texts 增加 texts / metadatas / ids 平行数组等长校验（不等长 ValueError）
- TESTING.md 裁决①：delete_collection 对不存在的知识库抛 ValueError（404 语义，
  与 rename_collection 的语义对齐；由 routes.py 在波次 3 映射为 404）
- UNIQUE(collection_id, file_name)：同名文件重复 add_texts 抛 ValueError。
  routes.py 现行流程本就是"先 delete_file 再 add_texts"，API 层行为不变

检索按方言分派（同一方法内分派，对外无感）：
- postgresql：pgvector 余弦算子 Chunk.embedding.cosine_distance(...) ORDER BY + LIMIT，
  距离 = 1 - 余弦相似度，返回前转换回相似度
- 其他方言（本机测试用 SQLite）：Python 内 numpy 计算余弦相似度
"""
import os
import re
import uuid
from typing import Optional

# 默认使用 HuggingFace 镜像，避免国内网络下载 BGE 模型超时
# 用户仍可通过环境变量 HF_ENDPOINT 覆盖此设置（与 core/vector_store.py 一致）
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
from sqlalchemy import Float, func

from app.core.config import settings
from app.db import get_engine, new_session
from app.db.models import Chunk, Collection, File


# ============================================================
#  嵌入模型单例（与 core/vector_store.py 同一套加载逻辑）
# ============================================================

_embed_model = None  # SentenceTransformer，延迟导入以便测试环境无需安装该依赖


def get_embed_model():
    """获取嵌入模型（懒加载，优先使用本地 ModelScope 缓存）"""
    global _embed_model
    if _embed_model is None:
        # 延迟导入：无 sentence-transformers 的环境（如 SQLite 测试）可正常 import 本模块，
        # 模型缺失时的异常在首次编码时才抛出，与文件版行为一致
        from sentence_transformers import SentenceTransformer

        model_name = settings.embed_model
        # 优先尝试本地路径（ModelScope 缓存）
        local_path = os.path.join(
            os.path.expanduser("~"), ".cache", "modelscope", "BAAI", "bge-small-zh-v1.5"
        )
        if os.path.isdir(local_path):
            model_name = local_path
            print(f"[Embed] 使用本地模型: {model_name}")
        else:
            print(f"[Embed] 加载模型: {model_name}")
        _embed_model = SentenceTransformer(model_name)
        print(f"[Embed] 模型加载完成，维度: {_embed_model.get_sentence_embedding_dimension()}")
    return _embed_model


def encode_texts(texts: list[str]) -> np.ndarray:
    """将文本列表编码为归一化向量矩阵 (N, dim)"""
    model = get_embed_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.astype(np.float32)


def encode_query(query: str) -> np.ndarray:
    """将查询文本编码为归一化向量 (dim,)"""
    return encode_texts([query])[0]


# ============================================================
#  名称校验与内部查询辅助
# ============================================================

# 知识库名称白名单：中英文字符、数字、下划线、连字符（与 core/vector_store.py 一致）
_COLLECTION_NAME_RE = re.compile(r'^[\w一-鿿-]+$')


def _validate_collection_name(collection_name: str) -> None:
    """知识库名称白名单校验，非法名抛 ValueError（对齐文件版 _collection_dir 的行为）"""
    if not collection_name or not _COLLECTION_NAME_RE.match(collection_name):
        raise ValueError(f"非法知识库名称: {collection_name!r}")


def _get_collection(session, name: str) -> Optional[Collection]:
    """按名称查知识库，不存在返回 None"""
    return session.query(Collection).filter(Collection.name == name).first()


def _get_or_create_collection(session, name: str) -> Collection:
    """按名称查知识库，不存在则创建（对齐文件版 add_texts 自动注册知识库的行为）"""
    col = _get_collection(session, name)
    if col is None:
        col = Collection(name=name)
        session.add(col)
        session.flush()  # 立刻拿到 col.id 供外键使用
    return col


def _parse_chunk_id(raw) -> uuid.UUID:
    """调用方传入的 ids 解析为 UUID；非 UUID 字符串退化为随机生成"""
    try:
        return uuid.UUID(str(raw))
    except (ValueError, AttributeError, TypeError):
        return uuid.uuid4()


# ============================================================
#  知识库管理
# ============================================================

def list_collections() -> list[dict]:
    """列出所有知识库（结构与文件版一致：{name, chunk_count}）"""
    with new_session() as session:
        rows = (
            session.query(Collection.name, func.count(Chunk.id))
            .outerjoin(Chunk, Chunk.collection_id == Collection.id)
            .group_by(Collection.id, Collection.name)
            .order_by(Collection.created_at)
            .all()
        )
        return [{"name": name, "chunk_count": count} for name, count in rows]


def create_collection(name: str) -> dict:
    """创建新知识库"""
    _validate_collection_name(name)
    with new_session() as session:
        if _get_collection(session, name) is not None:
            raise ValueError(f"知识库 '{name}' 已存在")
        session.add(Collection(name=name))
        session.commit()
        return {"name": name, "status": "created"}


def delete_collection(name: str) -> dict:
    """
    删除知识库及其所有数据。
    单事务删 chunks + files + collection（无中间态）；
    知识库不存在抛 ValueError（TESTING.md 裁决①的 404 语义，与 rename 对齐）。
    """
    _validate_collection_name(name)
    with new_session() as session:
        col = _get_collection(session, name)
        if col is None:
            raise ValueError(f"知识库 '{name}' 不存在")
        session.query(Chunk).filter(Chunk.collection_id == col.id).delete(
            synchronize_session=False
        )
        session.query(File).filter(File.collection_id == col.id).delete(
            synchronize_session=False
        )
        session.delete(col)
        session.commit()
        return {"name": name, "status": "deleted"}


def rename_collection(old_name: str, new_name: str) -> dict:
    """重命名知识库（DB 改名 = 一次 UPDATE，审计 #10 随之消除）"""
    _validate_collection_name(old_name)
    _validate_collection_name(new_name)
    with new_session() as session:
        col = _get_collection(session, old_name)
        if col is None:
            raise ValueError(f"知识库 '{old_name}' 不存在")
        if _get_collection(session, new_name) is not None:
            raise ValueError(f"知识库 '{new_name}' 已存在")
        col.name = new_name
        session.commit()
        return {"old_name": old_name, "new_name": new_name, "status": "renamed"}


# ============================================================
#  文本向量操作
# ============================================================

def add_texts(
    texts: list[str],
    metadatas: list[dict],
    collection_name: str,
    ids: Optional[list[str]] = None,
) -> int:
    """
    批量添加文本向量到知识库。
    - texts / metadatas / ids 平行数组必须等长，不等长抛 ValueError（审计 P2 修复）
    - 知识库不存在时自动创建（与文件版一致）
    - 同一知识库内 file_name 已存在时抛 ValueError（UNIQUE 约束语义；
      需要重入库请先 delete_file，与 routes.py 现行流程一致）
    """
    if not texts:
        return 0

    _validate_collection_name(collection_name)
    if len(metadatas) != len(texts):
        raise ValueError(
            f"texts 与 metadatas 长度不一致: {len(texts)} != {len(metadatas)}"
        )
    if ids is not None and len(ids) != len(texts):
        raise ValueError(f"texts 与 ids 长度不一致: {len(texts)} != {len(ids)}")

    embeddings = encode_texts(texts)

    with new_session() as session:
        col = _get_or_create_collection(session, collection_name)
        file_rows: dict[str, File] = {}  # 本批次内按 file_name 复用 File 行
        chunks: list[Chunk] = []

        for i, (text, md) in enumerate(zip(texts, metadatas)):
            md = md or {}
            file_name = md.get("file_name", "unknown")
            file_row = file_rows.get(file_name)
            if file_row is None:
                existing = (
                    session.query(File)
                    .filter(File.collection_id == col.id, File.file_name == file_name)
                    .first()
                )
                if existing is not None:
                    raise ValueError(
                        f"文件 '{file_name}' 已存在于知识库 '{collection_name}'，"
                        "请先 delete_file 再重新入库"
                    )
                file_row = File(
                    collection_id=col.id,
                    file_name=file_name,
                    storage_path=md.get("source", "") or "",
                    status="done",
                )
                session.add(file_row)
                session.flush()  # 立刻拿到 file_row.id 供 chunks 外键使用
                file_rows[file_name] = file_row

            chunks.append(
                Chunk(
                    id=_parse_chunk_id(ids[i]) if ids is not None else uuid.uuid4(),
                    file_id=file_row.id,
                    collection_id=col.id,
                    chunk_index=md.get("chunk_index", i),
                    content=text,
                    embedding=np.asarray(embeddings[i], dtype=np.float32).tolist(),
                )
            )

        session.add_all(chunks)
        session.commit()

    return len(texts)


# ============================================================
#  检索（按方言分派，对外无感）
# ============================================================

def vector_search(
    query: str,
    collection_name: str,
    top_k: int = 10,
) -> list[tuple[str, str, float]]:
    """
    向量相似度检索，返回 [(file_name, content, similarity), ...] 按相似度降序。
    知识库不存在或无数据时返回 []（与文件版一致）。
    """
    _validate_collection_name(collection_name)
    query_vec = encode_query(query)

    if get_engine().dialect.name == "postgresql":
        # pgvector 余弦距离算子 <=>（距离 = 1 - 余弦相似度），ORDER BY + LIMIT 下推 DB。
        # 注意：Chunk.embedding 是 VectorVariant（TypeDecorator impl=JSON），ORM 层
        # 比较器不带 pgvector 的 cosine_distance 方法，因此直接显式拼 <=> 运算符，
        # 语义与 cosine_distance 完全等价（后者编译结果即 embedding <=> :param）
        distance = Chunk.embedding.op("<=>", return_type=Float)(
            np.asarray(query_vec, dtype=np.float32).tolist()
        ).label("distance")
        with new_session() as session:
            col = _get_collection(session, collection_name)
            if col is None:
                return []
            rows = (
                session.query(File.file_name, Chunk.content, distance)
                .join(File, Chunk.file_id == File.id)
                .filter(Chunk.collection_id == col.id, Chunk.embedding.isnot(None))
                .order_by(distance)
                .limit(top_k)
                .all()
            )
        return [
            (file_name, content, 1.0 - float(dist))
            for file_name, content, dist in rows
        ]

    # SQLite 等方言：取出全部向量，Python 内 numpy 算余弦（仅用于本机测试）
    with new_session() as session:
        col = _get_collection(session, collection_name)
        if col is None:
            return []
        rows = (
            session.query(File.file_name, Chunk.content, Chunk.embedding)
            .join(File, Chunk.file_id == File.id)
            .filter(Chunk.collection_id == col.id, Chunk.embedding.isnot(None))
            .all()
        )
    if not rows:
        return []

    q = np.asarray(query_vec, dtype=np.float32)
    mat = np.asarray(
        [np.asarray(emb, dtype=np.float32) for _, _, emb in rows], dtype=np.float32
    )
    # 防御性归一化：即使个别向量未归一，结果仍是真正的余弦相似度（与 PG 路径一致）
    q = q / (np.linalg.norm(q) + 1e-12)
    mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    sims = mat @ q

    order = np.argsort(sims)[::-1][:top_k]
    return [
        (rows[i][0], rows[i][1], float(sims[i]))
        for i in order
    ]


def get_store_version(collection_name: str) -> tuple[int, float]:
    """
    检索/BM25 缓存的版本令牌：(chunk 总数, 最新 chunk 创建时间戳)。
    增删数据即变化，用于替代文件版的 metadata.json mtime
    （retrieval.py 的 _build_index 在波次 3 接线时改用它做缓存 key）。
    """
    _validate_collection_name(collection_name)
    with new_session() as session:
        col = _get_collection(session, collection_name)
        if col is None:
            return (0, 0.0)
        count, max_created = (
            session.query(func.count(Chunk.id), func.max(Chunk.created_at))
            .filter(Chunk.collection_id == col.id)
            .one()
        )
        return (count, max_created.timestamp() if max_created else 0.0)


# ============================================================
#  文件管理
# ============================================================

def get_files(collection_name: str) -> list[dict]:
    """获取知识库中的所有文件列表（结构：{file_name, chunk_count, collection_name}）"""
    _validate_collection_name(collection_name)
    with new_session() as session:
        col = _get_collection(session, collection_name)
        if col is None:
            return []
        rows = (
            session.query(File.file_name, func.count(Chunk.id))
            .outerjoin(Chunk, Chunk.file_id == File.id)
            .filter(File.collection_id == col.id)
            .group_by(File.id, File.file_name)
            .order_by(File.created_at)
            .all()
        )
        return [
            {
                "file_name": file_name,
                "chunk_count": count,
                "collection_name": collection_name,
            }
            for file_name, count in rows
        ]


def delete_file(file_name: str, collection_name: str) -> dict:
    """
    从知识库中删除指定文件的所有向量。
    单事务删 chunks + files 行（一个 commit，审计 #12 的中间态随之消除）；
    文件不存在返回 deleted_count=0（与文件版一致）。
    """
    _validate_collection_name(collection_name)
    with new_session() as session:
        col = _get_collection(session, collection_name)
        if col is None:
            return {"file_name": file_name, "deleted_count": 0}
        file_row = (
            session.query(File)
            .filter(File.collection_id == col.id, File.file_name == file_name)
            .first()
        )
        if file_row is None:
            return {"file_name": file_name, "deleted_count": 0}
        deleted_count = (
            session.query(Chunk)
            .filter(Chunk.file_id == file_row.id)
            .delete(synchronize_session=False)
        )
        session.delete(file_row)
        session.commit()
        return {"file_name": file_name, "deleted_count": deleted_count}


def get_vector_store(collection_name: str) -> dict:
    """
    兼容旧接口 - 返回 collection 的元数据摘要 {documents, metadatas, ids}，
    供 AdvancedRetriever 构建 BM25 倒排索引（与文件版逐字段对齐）。
    """
    _validate_collection_name(collection_name)
    with new_session() as session:
        col = _get_collection(session, collection_name)
        if col is None:
            return {"documents": [], "metadatas": [], "ids": []}
        rows = (
            session.query(Chunk, File)
            .join(File, Chunk.file_id == File.id)
            .filter(Chunk.collection_id == col.id)
            .order_by(File.created_at, Chunk.chunk_index)
            .all()
        )
        documents: list[str] = []
        metadatas: list[dict] = []
        ids: list[str] = []
        for chunk, file_row in rows:
            documents.append(chunk.content)
            metadatas.append(
                {
                    "file_name": file_row.file_name,
                    "source": file_row.storage_path,
                    "chunk_index": chunk.chunk_index,
                    "collection_name": collection_name,
                }
            )
            ids.append(str(chunk.id))
        return {"documents": documents, "metadatas": metadatas, "ids": ids}
