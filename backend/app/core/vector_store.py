"""
向量存储模块 - FAISS + JSON 元数据存储
项目使用 FAISS 作为向量检索引擎，元数据以 JSON 文件形式存储。
"""
import os
import json
import re
import uuid
import shutil
from pathlib import Path
from typing import Optional

# 默认使用 HuggingFace 镜像，避免国内网络下载 BGE 模型超时
# 用户仍可通过环境变量 HF_ENDPOINT 覆盖此设置
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from app.core.config import settings


# ============================================================
#  嵌入模型单例
# ============================================================

_embed_model: Optional[SentenceTransformer] = None


def get_embed_model() -> SentenceTransformer:
    """获取嵌入模型（懒加载，优先使用本地 ModelScope 缓存）"""
    global _embed_model
    if _embed_model is None:
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
#  FAISS 向量存储
# ============================================================

def _get_store_dir() -> Path:
    """向量存储根目录"""
    d = Path(settings.vector_persist_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _collections_index_path() -> Path:
    """知识库索引文件路径"""
    return _get_store_dir() / "_collections.json"


def _atomic_write_json(path: Path, data: dict) -> None:
    """原子写入 JSON：先写临时文件再 os.replace，避免进程中断留下半个文件"""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _read_json_tolerant(path: Path, default: dict) -> dict:
    """读取 JSON 文件；内容损坏时备份为 .corrupt 并按空数据处理，不再裸抛"""
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        corrupt = path.with_name(path.name + ".corrupt")
        os.replace(path, corrupt)
        print(f"[VectorStore] JSON 文件损坏，已备份为 {corrupt}，按空数据重建: {path}")
        return default


def _load_collections_index() -> dict:
    """加载知识库索引"""
    return _read_json_tolerant(_collections_index_path(), {})


def _save_collections_index(data: dict) -> None:
    """保存知识库索引"""
    _atomic_write_json(_collections_index_path(), data)


# 知识库名称白名单：中英文字符、数字、下划线、连字符（与 routes.py 一致）
_COLLECTION_NAME_RE = re.compile(r'^[\w一-鿿-]+$')


def _collection_dir(collection_name: str) -> Path:
    """某个知识库的数据目录（名称白名单 + 路径前缀校验，防路径穿越）"""
    if not collection_name or not _COLLECTION_NAME_RE.match(collection_name):
        raise ValueError(f"非法知识库名称: {collection_name!r}")
    base = _get_store_dir().resolve()
    d = (base / collection_name).resolve()
    if not str(d).startswith(str(base)):
        raise ValueError(f"知识库路径越界: {collection_name!r}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _faiss_path(collection_name: str) -> Path:
    """FAISS 索引文件路径"""
    return _collection_dir(collection_name) / "index.faiss"


def _meta_path(collection_name: str) -> Path:
    """元数据 JSON 文件路径"""
    return _collection_dir(collection_name) / "metadata.json"


def _read_faiss_index(collection_name: str) -> Optional[faiss.Index]:
    """直接读取 FAISS 索引文件，不存在则返回 None（不做一致性校验）"""
    path = _faiss_path(collection_name)
    if path.exists():
        return faiss.read_index(str(path))
    return None


def _save_faiss_index(collection_name: str, index: faiss.Index) -> None:
    """保存 FAISS 索引"""
    faiss.write_index(index, str(_faiss_path(collection_name)))


def _load_metadata(collection_name: str) -> dict:
    """加载元数据（documents、file 信息等）"""
    return _read_json_tolerant(
        _meta_path(collection_name),
        {"documents": [], "metadatas": [], "ids": []},
    )


def _save_metadata(collection_name: str, data: dict) -> None:
    """保存元数据"""
    _atomic_write_json(_meta_path(collection_name), data)


def _rebuild_index_from_metadata(collection_name: str, meta: dict) -> faiss.Index:
    """从元数据全量重建 FAISS 索引（需重新编码，仅在索引与元数据不一致时使用）"""
    dim = get_embed_model().get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(dim)
    if meta["documents"]:
        index.add(encode_texts(meta["documents"]))
    _save_faiss_index(collection_name, index)
    return index


def _load_faiss_index(collection_name: str) -> Optional[faiss.Index]:
    """加载 FAISS 索引，不存在则返回 None；索引与元数据数量不一致时自动重建"""
    index = _read_faiss_index(collection_name)
    if index is None:
        return None
    meta = _load_metadata(collection_name)
    if index.ntotal != len(meta["ids"]):
        print(
            f"[VectorStore] 索引与元数据不一致 (ntotal={index.ntotal}, "
            f"ids={len(meta['ids'])})，已从元数据重建: {collection_name}"
        )
        index = _rebuild_index_from_metadata(collection_name, meta)
    return index


def _get_or_create_index(collection_name: str) -> faiss.Index:
    """获取或创建 FAISS 索引"""
    index = _load_faiss_index(collection_name)
    if index is None:
        dim = get_embed_model().get_sentence_embedding_dimension()
        index = faiss.IndexFlatIP(dim)  # Inner Product = Cosine (归一化后)
    return index


# ============================================================
#  知识库管理
# ============================================================

def list_collections() -> list[dict]:
    """列出所有知识库"""
    idx = _load_collections_index()
    result = []
    for name, info in idx.items():
        meta = _load_metadata(name)
        result.append({
            "name": name,
            "chunk_count": len(meta.get("ids", [])),
        })
    return result


def create_collection(name: str) -> dict:
    """创建新知识库"""
    idx = _load_collections_index()
    if name in idx:
        raise ValueError(f"知识库 '{name}' 已存在")

    idx[name] = {"created": True}
    _save_collections_index(idx)

    # 初始化空索引和元数据
    dim = get_embed_model().get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(dim)
    _save_faiss_index(name, index)
    _save_metadata(name, {"documents": [], "metadatas": [], "ids": []})

    return {"name": name, "status": "created"}


def delete_collection(name: str) -> dict:
    """删除知识库及其所有数据"""
    idx = _load_collections_index()
    if name in idx:
        del idx[name]
        _save_collections_index(idx)

    col_dir = _collection_dir(name)
    if col_dir.exists():
        shutil.rmtree(col_dir)

    _invalidate_search_cache(name)

    return {"name": name, "status": "deleted"}


def rename_collection(old_name: str, new_name: str) -> dict:
    """重命名知识库"""
    idx = _load_collections_index()
    if old_name not in idx:
        raise ValueError(f"知识库 '{old_name}' 不存在")
    if new_name in idx:
        raise ValueError(f"知识库 '{new_name}' 已存在")

    # 重命名目录
    old_dir = _collection_dir(old_name)
    new_dir = _collection_dir(new_name)
    if old_dir.exists():
        old_dir.rename(new_dir)

    # 更新索引
    idx[new_name] = idx.pop(old_name)
    _save_collections_index(idx)

    _invalidate_search_cache(old_name)
    _invalidate_search_cache(new_name)

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
    """批量添加文本向量到知识库"""
    if not texts:
        return 0

    if ids is None:
        ids = [str(uuid.uuid4()) for _ in texts]

    # 编码
    embeddings = encode_texts(texts)

    # 更新 FAISS 索引
    index = _get_or_create_index(collection_name)
    index.add(embeddings)
    _save_faiss_index(collection_name, index)

    # 更新元数据
    meta = _load_metadata(collection_name)
    meta["documents"].extend(texts)
    meta["metadatas"].extend(metadatas)
    meta["ids"].extend(ids)
    _save_metadata(collection_name, meta)

    # 索引与元数据均已落盘，使检索缓存失效
    _invalidate_search_cache(collection_name)

    # 确保知识库已注册
    idx = _load_collections_index()
    if collection_name not in idx:
        idx[collection_name] = {"created": True}
        _save_collections_index(idx)

    return len(texts)


# ============================================================
#  检索缓存
# ============================================================

# collection_name -> (index, metadata, index.faiss 的 mtime_ns)
# 仅当索引文件的 mtime 变化时才重读磁盘，避免每次检索都全量加载
_search_cache: dict[str, tuple[faiss.Index, dict, int]] = {}


def _invalidate_search_cache(collection_name: str) -> None:
    """使指定知识库的检索缓存失效"""
    _search_cache.pop(collection_name, None)


def _get_cached_search_data(collection_name: str) -> tuple[Optional[faiss.Index], dict]:
    """获取检索用的索引和元数据，优先命中 mtime 缓存"""
    path = _faiss_path(collection_name)
    if not path.exists():
        _invalidate_search_cache(collection_name)
        return None, {"documents": [], "metadatas": [], "ids": []}

    cached = _search_cache.get(collection_name)
    if cached is not None and cached[2] == path.stat().st_mtime_ns:
        return cached[0], cached[1]

    index = _load_faiss_index(collection_name)  # 含一致性校验，必要时自动重建
    meta = _load_metadata(collection_name)
    if index is not None:
        # 一致性重建可能改写了索引文件，重新取 mtime 再缓存
        _search_cache[collection_name] = (index, meta, path.stat().st_mtime_ns)
    return index, meta


def vector_search(
    query: str,
    collection_name: str,
    top_k: int = 10,
) -> list[tuple[str, str, float]]:
    """向量相似度检索（索引与元数据走 mtime 缓存，避免每次全量读盘）"""
    index, meta = _get_cached_search_data(collection_name)
    if index is None or index.ntotal == 0:
        return []

    if not meta["ids"]:
        return []

    # 编码查询
    query_vec = encode_query(query).reshape(1, -1)

    # 检索
    k = min(top_k, index.ntotal)
    distances, indices = index.search(query_vec, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(meta["documents"]):
            # 正常不会越界：索引加载时已校验 ntotal 与元数据数量一致，
            # 此处仅为防御性兜底
            continue
        file_name = meta["metadatas"][idx].get("file_name", "")
        content = meta["documents"][idx]
        similarity = float(dist)  # IP → [-1, 1], 归一化后等价余弦相似度
        results.append((file_name, content, similarity))

    return results


# ============================================================
#  文件管理
# ============================================================

def get_files(collection_name: str) -> list[dict]:
    """获取知识库中的所有文件列表"""
    meta = _load_metadata(collection_name)
    if not meta["metadatas"]:
        return []

    file_map: dict[str, dict] = {}
    for m in meta["metadatas"]:
        fname = m.get("file_name", "unknown")
        if fname not in file_map:
            file_map[fname] = {
                "file_name": fname,
                "chunk_count": 0,
                "collection_name": collection_name,
            }
        file_map[fname]["chunk_count"] += 1

    return list(file_map.values())


def delete_file(file_name: str, collection_name: str) -> dict:
    """从知识库中删除指定文件的所有向量"""
    meta = _load_metadata(collection_name)

    # 找出要保留的索引
    keep_indices = [
        i for i, m in enumerate(meta["metadatas"])
        if m.get("file_name") != file_name
    ]
    deleted_count = len(meta["ids"]) - len(keep_indices)

    if deleted_count == 0:
        return {"file_name": file_name, "deleted_count": 0}

    # 重建 FAISS 索引（FAISS 不支持直接删除）
    dim = get_embed_model().get_sentence_embedding_dimension()
    new_index = faiss.IndexFlatIP(dim)

    if keep_indices:
        # 优先从旧索引 reconstruct 保留项的向量，省去重新跑 embedding；
        # 旧索引缺失、与元数据不一致或不支持 reconstruct 时回退到重新编码
        old_index = _read_faiss_index(collection_name)
        rebuilt = False
        if old_index is not None and old_index.ntotal == len(meta["ids"]):
            try:
                vectors = np.vstack([old_index.reconstruct(i) for i in keep_indices])
                new_index.add(vectors)
                rebuilt = True
            except (NotImplementedError, RuntimeError):
                rebuilt = False
        if not rebuilt:
            keep_texts = [meta["documents"][i] for i in keep_indices]
            new_index.add(encode_texts(keep_texts))

    _save_faiss_index(collection_name, new_index)
    _invalidate_search_cache(collection_name)

    # 更新元数据
    new_meta = {
        "documents": [meta["documents"][i] for i in keep_indices],
        "metadatas": [meta["metadatas"][i] for i in keep_indices],
        "ids": [meta["ids"][i] for i in keep_indices],
    }
    _save_metadata(collection_name, new_meta)

    return {"file_name": file_name, "deleted_count": deleted_count}


def get_vector_store(collection_name: str) -> dict:
    """
    兼容旧接口 - 返回 collection 的元数据摘要。
    用于 HybridRetriever 构建倒排索引。
    """
    meta = _load_metadata(collection_name)
    return {
        "documents": meta["documents"],
        "metadatas": meta["metadatas"],
        "ids": meta["ids"],
    }
