"""
PgVectorStore（app/db/vector_repository.py）测试

- SQLite 方言（本机无 PostgreSQL），向量列走 JSON 退化路径
- 无 sentence-transformers / torch：monkeypatch 打桩假嵌入函数（维度 512，
  确定性生成，已知向量用于断言最近邻排序）
- 必须在 import 任何 app 模块之前设置 DATABASE_URL（settings 是 import 期单例）
"""
import hashlib
import os
import sys
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest

# ---- 测试环境准备：必须在 import app 之前完成 ----
_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

_TMP_DIR = tempfile.mkdtemp(prefix="vector_repo_test_")
os.environ["DATABASE_URL"] = f"sqlite:///{Path(_TMP_DIR, 'test.db').as_posix()}"

from app.db import Base, get_engine, new_session  # noqa: E402
from app.db.models import Chunk, Collection, File  # noqa: E402
from app.db import vector_repository as vr  # noqa: E402

Base.metadata.create_all(get_engine())

EMB_DIM = 512


# ============================================================
#  假嵌入：确定性向量，已知方向用于断言最近邻
# ============================================================

def _unit(i: int) -> np.ndarray:
    v = np.zeros(EMB_DIM, dtype=np.float32)
    v[i] = 1.0
    return v


_FAKE_VECS = {
    "苹果": _unit(0),
    # 与"苹果"高度相似但不完全相同
    "苹果派": (0.98 * _unit(0) + 0.02 * _unit(1)).astype(np.float32),
    "香蕉": _unit(1),
    "苹果，香蕉": (0.7 * _unit(0) + 0.7 * _unit(1)).astype(np.float32),
}


def _fake_vec(text: str) -> np.ndarray:
    """已知文本返回固定向量；未知文本用 sha256 做确定性随机向量（归一化）"""
    if text in _FAKE_VECS:
        return _FAKE_VECS[text]
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "little")
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMB_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _fake_encode_texts(texts: list[str]) -> np.ndarray:
    return np.stack([_fake_vec(t) for t in texts]).astype(np.float32)


def _fake_encode_query(query: str) -> np.ndarray:
    return _fake_vec(query)


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch):
    """打桩嵌入编码，避免加载真实 BGE 模型（无 torch/sentence-transformers）"""
    monkeypatch.setattr(vr, "encode_texts", _fake_encode_texts)
    monkeypatch.setattr(vr, "encode_query", _fake_encode_query)


@pytest.fixture(autouse=True)
def clean_db():
    """每个用例前清空业务表，保证隔离"""
    with new_session() as s:
        for model in (Chunk, File, Collection):
            s.query(model).delete(synchronize_session=False)
        s.commit()
    yield


def _add_two_files(collection: str = "kb1") -> None:
    """预置两个文件共 4 个 chunk"""
    vr.add_texts(
        ["苹果", "苹果派"],
        [
            {"file_name": "a.txt", "source": "/up/a.txt", "chunk_index": 0},
            {"file_name": "a.txt", "source": "/up/a.txt", "chunk_index": 1},
        ],
        collection,
    )
    vr.add_texts(
        ["香蕉", "苹果，香蕉"],
        [
            {"file_name": "b.txt", "source": "/up/b.txt", "chunk_index": 0},
            {"file_name": "b.txt", "source": "/up/b.txt", "chunk_index": 1},
        ],
        collection,
    )


# ============================================================
#  知识库管理
# ============================================================

def test_create_and_list_collections():
    assert vr.create_collection("kb1") == {"name": "kb1", "status": "created"}
    assert vr.list_collections() == [{"name": "kb1", "chunk_count": 0}]

    # 重名创建 → ValueError（与文件版一致）
    with pytest.raises(ValueError, match="已存在"):
        vr.create_collection("kb1")


def test_list_collections_chunk_count():
    _add_two_files()
    assert vr.list_collections() == [{"name": "kb1", "chunk_count": 4}]


def test_invalid_collection_name():
    # 名称白名单：空串、路径穿越、空格、分号等非法字符一律 ValueError（与文件版一致）
    for bad in ("", "a/b", "..", "a b", "drop;table"):
        with pytest.raises(ValueError, match="非法知识库名称"):
            vr.create_collection(bad)
        with pytest.raises(ValueError, match="非法知识库名称"):
            vr.delete_collection(bad)
        with pytest.raises(ValueError, match="非法知识库名称"):
            vr.vector_search("苹果", bad)
        with pytest.raises(ValueError, match="非法知识库名称"):
            vr.add_texts(["x"], [{"file_name": "f"}], bad)


def test_rename_collection():
    vr.create_collection("kb1")
    assert vr.rename_collection("kb1", "kb2") == {
        "old_name": "kb1",
        "new_name": "kb2",
        "status": "renamed",
    }
    assert [c["name"] for c in vr.list_collections()] == ["kb2"]

    # 审计 #10：DB 改名 = UPDATE，不触碰文件系统，
    # 反复来回重命名不再有 Windows 目录重命名的 FileExistsError
    vr.rename_collection("kb2", "kb3")
    vr.rename_collection("kb3", "kb2")
    vr.rename_collection("kb2", "kb1")
    assert [c["name"] for c in vr.list_collections()] == ["kb1"]


def test_rename_collection_data_preserved():
    _add_two_files("kb1")
    vr.rename_collection("kb1", "kb2")
    # 重命名后数据原样可查
    assert vr.list_collections() == [{"name": "kb2", "chunk_count": 4}]
    assert {f["file_name"] for f in vr.get_files("kb2")} == {"a.txt", "b.txt"}


def test_rename_collection_errors():
    vr.create_collection("kb1")
    vr.create_collection("kb2")
    # 重命名不存在的库 → ValueError（与文件版一致）
    with pytest.raises(ValueError, match="不存在"):
        vr.rename_collection("nope", "kb3")
    # 目标名已存在 → ValueError（与文件版一致）
    with pytest.raises(ValueError, match="已存在"):
        vr.rename_collection("kb1", "kb2")


def test_delete_collection():
    _add_two_files("kb1")
    assert vr.delete_collection("kb1") == {"name": "kb1", "status": "deleted"}
    assert vr.list_collections() == []

    # 审计 #12 同族：chunks / files / collection 一个事务删净，无残留
    with new_session() as s:
        assert s.query(Chunk).count() == 0
        assert s.query(File).count() == 0
        assert s.query(Collection).count() == 0


def test_delete_nonexistent_collection():
    # TESTING.md 裁决①：删除不存在的库 → ValueError（404 语义，与 rename 对齐）
    with pytest.raises(ValueError, match="不存在"):
        vr.delete_collection("nope")


# ============================================================
#  add_texts / vector_search
# ============================================================

def test_add_texts_auto_creates_collection():
    # 知识库不存在时自动注册（与文件版一致）
    n = vr.add_texts(["苹果"], [{"file_name": "a.txt", "chunk_index": 0}], "kb1")
    assert n == 1
    assert vr.list_collections() == [{"name": "kb1", "chunk_count": 1}]


def test_add_texts_empty():
    assert vr.add_texts([], [], "kb1") == 0
    assert vr.list_collections() == []  # 空批次不建库


def test_add_texts_length_mismatch():
    # 审计 P2 修复：平行数组不等长 → ValueError
    with pytest.raises(ValueError, match="长度不一致"):
        vr.add_texts(["a", "b"], [{"file_name": "f"}], "kb1")
    with pytest.raises(ValueError, match="长度不一致"):
        vr.add_texts(
            ["a"],
            [{"file_name": "f"}],
            "kb1",
            ids=[str(uuid.uuid4()), str(uuid.uuid4())],
        )
    # 校验失败不落任何数据
    assert vr.list_collections() == []


def test_add_texts_duplicate_file_conflict():
    # UNIQUE(collection_id, file_name)：同名文件重复入库 → ValueError，原数据不受影响
    _add_two_files("kb1")
    with pytest.raises(ValueError, match="已存在"):
        vr.add_texts(["新内容"], [{"file_name": "a.txt", "chunk_index": 0}], "kb1")
    assert vr.list_collections() == [{"name": "kb1", "chunk_count": 4}]
    # 先删后加是合法路径（与 routes.py 现行流程一致）
    vr.delete_file("a.txt", "kb1")
    assert vr.add_texts(["新内容"], [{"file_name": "a.txt", "chunk_index": 0}], "kb1") == 1


def test_vector_search_structure_and_ranking():
    _add_two_files("kb1")
    results = vr.vector_search("苹果", "kb1", top_k=3)

    # 返回结构：[(file_name, content, similarity)]，三元组逐字段对齐文件版
    assert len(results) == 3
    for file_name, content, score in results:
        assert isinstance(file_name, str)
        assert isinstance(content, str)
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0

    # 排序正确性：完全同向的"苹果"第一（相似度≈1），"苹果派"第二，"香蕉"最后
    assert results[0][1] == "苹果"
    assert results[0][0] == "a.txt"
    assert results[0][2] == pytest.approx(1.0, abs=1e-5)
    assert results[1][1] == "苹果派"
    assert results[2][1] == "苹果，香蕉"
    assert results[0][2] > results[1][2] > results[2][2]

    # top_k 截断
    assert len(vr.vector_search("苹果", "kb1", top_k=1)) == 1
    # top_k 超过总数时返回全部
    assert len(vr.vector_search("苹果", "kb1", top_k=100)) == 4


def test_vector_search_empty_and_missing_collection():
    assert vr.vector_search("苹果", "nope") == []
    vr.create_collection("kb1")
    assert vr.vector_search("苹果", "kb1") == []


# ============================================================
#  文件管理
# ============================================================

def test_get_files():
    _add_two_files("kb1")
    files = {f["file_name"]: f for f in vr.get_files("kb1")}
    assert set(files) == {"a.txt", "b.txt"}
    assert files["a.txt"]["chunk_count"] == 2
    assert files["a.txt"]["collection_name"] == "kb1"
    # 不存在的知识库 → 空列表（与文件版一致）
    assert vr.get_files("nope") == []


def test_delete_file():
    _add_two_files("kb1")
    assert vr.delete_file("a.txt", "kb1") == {"file_name": "a.txt", "deleted_count": 2}

    # 级联删 chunks：库里只剩 b.txt 的 2 个 chunk
    assert vr.list_collections() == [{"name": "kb1", "chunk_count": 2}]
    assert [f["file_name"] for f in vr.get_files("kb1")] == ["b.txt"]
    with new_session() as s:
        assert s.query(Chunk).count() == 2
        assert s.query(File).count() == 1

    # 删除后检索只命中剩余文件
    results = vr.vector_search("苹果", "kb1")
    assert {r[0] for r in results} == {"b.txt"}

    # 文件不存在 → deleted_count=0（与文件版一致）
    assert vr.delete_file("a.txt", "kb1") == {"file_name": "a.txt", "deleted_count": 0}
    assert vr.delete_file("x.txt", "nope") == {"file_name": "x.txt", "deleted_count": 0}


def test_get_vector_store():
    _add_two_files("kb1")
    data = vr.get_vector_store("kb1")
    assert set(data.keys()) == {"documents", "metadatas", "ids"}
    assert len(data["documents"]) == len(data["metadatas"]) == len(data["ids"]) == 4
    assert "苹果" in data["documents"]
    for md in data["metadatas"]:
        assert md["file_name"] in {"a.txt", "b.txt"}
        assert md["collection_name"] == "kb1"
        assert isinstance(md["chunk_index"], int)
    # ids 为 chunk 主键字符串
    assert all(uuid.UUID(i) for i in data["ids"])
    # 不存在的知识库 → 空结构（与文件版一致）
    assert vr.get_vector_store("nope") == {"documents": [], "metadatas": [], "ids": []}


def test_pg_branch_sql_compiles():
    """
    PG 方言路径的 SQL 编译检查（本机无 PostgreSQL，不执行，仅验证语句可生成）：
    显式拼的 <=> 即 pgvector 余弦距离算子，与 cosine_distance 编译结果一致
    """
    from sqlalchemy import Float
    from sqlalchemy.dialects import postgresql

    distance = Chunk.embedding.op("<=>", return_type=Float)([0.1] * EMB_DIM).label("distance")
    with new_session() as s:
        q = (
            s.query(File.file_name, Chunk.content, distance)
            .join(File, Chunk.file_id == File.id)
            .filter(Chunk.embedding.isnot(None))
            .order_by(distance)
            .limit(5)
        )
        sql = str(q.statement.compile(dialect=postgresql.dialect()))
    assert "<=>" in sql
    assert "ORDER BY" in sql
    assert "LIMIT" in sql


def test_get_store_version():
    # 缓存版本令牌：增删数据即变化（替代文件版 metadata.json mtime）
    v0 = vr.get_store_version("kb1")
    assert v0 == (0, 0.0)

    _add_two_files("kb1")
    v1 = vr.get_store_version("kb1")
    assert v1[0] == 4 and v1 != v0

    vr.delete_file("a.txt", "kb1")
    v2 = vr.get_store_version("kb1")
    assert v2[0] == 2 and v2 != v1
