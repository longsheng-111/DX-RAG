"""
PostgreSQL 真实集成测试（阶段 1 收口验证）

与既有 SQLite 测试（test_session_feedback_repo.py / test_vector_repo.py）的关系：
- 既有测试在 SQLite 上验证行为契约与 SQL 编译，本文件把同一批关键路径放到
  真实 PostgreSQL + pgvector 上端到端执行（真实 INSERT vector 列、真实 <=> 算子、
  真实 FK CASCADE）。
- 门控：仅在环境变量 DATABASE_URL 显式指向 postgresql:// / postgresql+psycopg2://
  时运行；CI / 无库环境下整模块 skip，不影响既有 SQLite 回归套件。
- settings 是 import 期单例：运行本文件必须在进程启动前设置 DATABASE_URL，例如
    cd backend && DATABASE_URL=postgresql+psycopg2://postgres@127.0.0.1:5433/dxrag_test \
        venv_pg/Scripts/python.exe -m pytest tests/test_pg_integration.py -q
- 嵌入用 monkeypatch 的确定性假 512 维向量（同 test_vector_repo 的思路），
  不加载真实 BGE 模型；向量检索断言排序正确 + 实际发出的 SQL 含 pgvector <=> 算子。

实测定案（本机 PG 16.15 + pgvector 0.8.0 + psycopg2 + pgvector-python 0.4.2）：
  pgvector.sqlalchemy.Vector 自带 bind_processor/result_processor，SQLAlchemy 层
  已把 list[float] 序列化为 '[...]' 字符串、把返回文本解析为 numpy.ndarray，
  psycopg2 只见字符串参数，因此无需 register_vector(conn) 注册即可正确读写。
"""
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# ---- 门控：必须早于一切 app import ----
_PG_URL = os.environ.get("DATABASE_URL", "")
if not _PG_URL.startswith("postgresql"):
    pytest.skip(
        "PG 集成测试：需 DATABASE_URL 显式指向 PostgreSQL（默认 skip，CI/无库环境安全）",
        allow_module_level=True,
    )

pytest.importorskip("psycopg2", reason="PG 集成测试需要 psycopg2 驱动")
pytest.importorskip("pgvector", reason="PG 集成测试需要 pgvector-python")

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from sqlalchemy import event, func, select  # noqa: E402

from app.core.session import Session  # noqa: E402
from app.db import Base, get_engine, new_session  # noqa: E402
from app.db import vector_repository as vr  # noqa: E402
from app.db.feedback_repository import FeedbackRepository  # noqa: E402
from app.db.models import Chunk, Collection, File  # noqa: E402
from app.db.models import Feedback as DbFeedback  # noqa: E402
from app.db.models import Session as DbSession  # noqa: E402
from app.db.session_repository import SessionRepository  # noqa: E402

EMB_DIM = 512


@pytest.fixture(scope="session", autouse=True)
def _create_tables():
    """建表（等价 alembic upgrade head 的 schema；已有表时幂等）并确认引擎指向 PG"""
    engine = get_engine()
    assert engine.dialect.name == "postgresql", f"引擎方言异常: {engine.url}"
    Base.metadata.create_all(engine)


@pytest.fixture(autouse=True)
def _clean_tables():
    """每个用例前清空业务表（先子后父，bulk delete 不走 ORM 级联）"""
    with new_session() as db:
        for model in (DbFeedback, DbSession, Chunk, File, Collection):
            db.query(model).delete(synchronize_session=False)
        db.commit()
    yield


# ============================================================
#  假嵌入：确定性 512 维向量，已知方向用于断言最近邻排序
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
    if text in _FAKE_VECS:
        return _FAKE_VECS[text]
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "little")
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMB_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch):
    """打桩嵌入编码，避免加载真实 BGE 模型"""
    monkeypatch.setattr(
        vr, "encode_texts",
        lambda texts: np.stack([_fake_vec(t) for t in texts]).astype(np.float32),
    )
    monkeypatch.setattr(vr, "encode_query", lambda q: _fake_vec(q))


@pytest.fixture()
def repo() -> SessionRepository:
    return SessionRepository()


@pytest.fixture()
def fb_repo() -> FeedbackRepository:
    return FeedbackRepository()


def _add_two_files(collection: str = "kb_pg") -> None:
    """预置两个文件共 4 个 chunk（假嵌入）"""
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
#  ① SessionRepository：PG 上 CRUD 往返（JSONB 列真实读写）
# ============================================================

def test_session_crud_roundtrip(repo):
    exam_state = {"status": "running", "scores": [8, 9], "weak_points": ["事务"]}
    s = repo.create_session(persona="backend", kb_id="kb_pg", title="PG 面试",
                            mode="examiner", exam_state=exam_state)
    loaded = repo.get_session(s.session_id)
    assert loaded is not None
    assert loaded.title == "PG 面试"
    assert loaded.mode == "examiner"
    assert loaded.exam_state == exam_state
    assert loaded.messages == []

    # 追加消息 + token 累加后 upsert
    s.add_message("user", "什么是索引", token_cost={"total_tokens": 10})
    s.add_message("assistant", "索引是……",
                  token_cost={"total_tokens": 20},
                  sources=[{"chunk_id": "c1", "score": 0.9}],
                  reasoning_steps=[{"step": "retrieve"}])
    repo.save_session(s)
    loaded = repo.get_session(s.session_id)
    assert [m["role"] for m in loaded.messages] == ["user", "assistant"]
    assert loaded.messages[1]["sources"] == [{"chunk_id": "c1", "score": 0.9}]
    assert loaded.total_tokens == 30
    assert loaded.updated_at > s.created_at  # save 刷新 updated_at

    # list / 删除
    assert [x.session_id for x in repo.list_sessions()] == [s.session_id]
    assert repo.delete_session(s.session_id) is True
    assert repo.get_session(s.session_id) is None
    assert repo.delete_session(s.session_id) is False


# ============================================================
#  ② FeedbackRepository：增查 + 删会话级联（PG 真实 FK CASCADE）
# ============================================================

def test_feedback_add_query_and_cascade(repo, fb_repo):
    s1 = repo.create_session()
    s2 = repo.create_session()
    fb_repo.add_feedback(s1.session_id, "c1", "c2", "replace", note="n1")
    fb_repo.add_feedback(s1.session_id, "c3", "c4", "inaccurate", note="n2")
    fb_repo.add_feedback(s2.session_id, "c9", "c9", "replace", note="other")

    fbs = fb_repo.get_session_feedback(s1.session_id)
    assert [f["note"] for f in fbs] == ["n1", "n2"]  # 按写入时间正序
    assert {f["session_id"] for f in fbs} == {s1.session_id}

    # prompt 格式化（PG JSONB 读回路径）
    text = fb_repo.format_feedback_for_prompt(s1.session_id)
    assert "【历史反馈 / 事实修正约束】" in text
    assert "反馈类型：replace" in text

    # 删除 s1 → 反馈物理级联删除（直连 DB 验证，s2 的反馈不受影响）
    assert repo.delete_session(s1.session_id) is True
    assert fb_repo.get_session_feedback(s1.session_id) == []
    assert len(fb_repo.get_session_feedback(s2.session_id)) == 1
    with new_session() as db:
        assert db.scalar(select(func.count()).select_from(DbFeedback)) == 1
        assert db.scalar(select(func.count()).select_from(DbSession)) == 1


# ============================================================
#  ③ vector_repository：add_texts 真实写 vector 列 + 读回维度
# ============================================================

def test_add_texts_writes_real_vector_column():
    vr.add_texts(["苹果"], [{"file_name": "a.txt", "chunk_index": 0}], "kb_pg")

    with new_session() as db:
        chunk = db.scalar(select(Chunk))
        assert chunk is not None
        # pgvector.sqlalchemy 的 result_processor 把 vector 文本解析为 numpy.ndarray
        assert chunk.embedding is not None
        emb = np.asarray(chunk.embedding, dtype=np.float32)
        assert emb.shape == (EMB_DIM,)
        assert np.allclose(emb, _FAKE_VECS["苹果"], atol=1e-6)

        # 直连 PG 确认列类型确实是 vector(512)，而非 JSON 退化
        from sqlalchemy import text as sa_text
        typmod = db.execute(sa_text(
            "SELECT atttypmod FROM pg_attribute "
            "WHERE attrelid='chunks'::regclass AND attname='embedding'"
        )).scalar()
        assert typmod == EMB_DIM


# ============================================================
#  ④ vector_search 的 PG 方言分支：真实 <=> 算子 + 排序正确
# ============================================================

def test_vector_search_pg_dialect_uses_pgvector_operator():
    _add_two_files("kb_pg")

    # 事件监听捕获实际发往 PG 的 SQL，断言检索真实走 pgvector <=> 余弦距离算子
    engine = get_engine()
    assert engine.dialect.name == "postgresql"
    captured: list[str] = []

    @event.listens_for(engine, "before_cursor_execute")
    def _capture(conn, cursor, statement, parameters, context, executemany):
        captured.append(statement)

    try:
        results = vr.vector_search("苹果", "kb_pg", top_k=3)
    finally:
        event.remove(engine, "before_cursor_execute", _capture)

    # ① 真实下推 pgvector 算子（而非取回全表在 Python 里算）
    assert any("<=>" in stmt for stmt in captured), \
        "未捕获到含 <=> 的 SQL，PG 方言分支疑似未生效"

    # ② 排序正确：同向"苹果"第一（相似度≈1），"苹果派"第二，混合向量最后
    assert [r[1] for r in results] == ["苹果", "苹果派", "苹果，香蕉"]
    assert results[0][0] == "a.txt"
    assert results[0][2] == pytest.approx(1.0, abs=1e-5)
    assert results[0][2] > results[1][2] > results[2][2]

    # ③ top_k 截断 / 超量返回全部 / 空库与不存在库
    assert len(vr.vector_search("苹果", "kb_pg", top_k=1)) == 1
    assert len(vr.vector_search("苹果", "kb_pg", top_k=100)) == 4
    vr.create_collection("kb_empty")
    assert vr.vector_search("苹果", "kb_empty") == []
    assert vr.vector_search("苹果", "nope") == []


# ============================================================
#  ⑤ get_store_version：缓存版本令牌随增删变化
# ============================================================

def test_get_store_version():
    assert vr.get_store_version("kb_pg") == (0, 0.0)
    _add_two_files("kb_pg")
    v1 = vr.get_store_version("kb_pg")
    assert v1[0] == 4 and v1[1] > 0.0
    vr.delete_file("a.txt", "kb_pg")
    v2 = vr.get_store_version("kb_pg")
    assert v2[0] == 2 and v2 != v1


# ============================================================
#  ⑥ rename / delete_file / delete_collection 级联（PG 真实事务）
# ============================================================

def test_rename_and_delete_cascades():
    _add_two_files("kb_pg")

    # rename：一次 UPDATE，数据原样可查
    assert vr.rename_collection("kb_pg", "kb_pg2")["status"] == "renamed"
    assert vr.list_collections() == [{"name": "kb_pg2", "chunk_count": 4}]
    assert {f["file_name"] for f in vr.get_files("kb_pg2")} == {"a.txt", "b.txt"}

    # delete_file：单事务删 chunks + files 行
    assert vr.delete_file("a.txt", "kb_pg2") == {"file_name": "a.txt", "deleted_count": 2}
    with new_session() as db:
        assert db.scalar(select(func.count()).select_from(Chunk)) == 2
        assert db.scalar(select(func.count()).select_from(File)) == 1
    # 删除后检索只命中剩余文件（PG 分支实时反映删除）
    assert {r[0] for r in vr.vector_search("苹果", "kb_pg2")} == {"b.txt"}
    # 文件不存在 → deleted_count=0
    assert vr.delete_file("a.txt", "kb_pg2") == {"file_name": "a.txt", "deleted_count": 0}

    # delete_collection：整库删净；不存在的库抛 ValueError（404 语义）
    assert vr.delete_collection("kb_pg2") == {"name": "kb_pg2", "status": "deleted"}
    with new_session() as db:
        assert db.scalar(select(func.count()).select_from(Chunk)) == 0
        assert db.scalar(select(func.count()).select_from(File)) == 0
        assert db.scalar(select(func.count()).select_from(Collection)) == 0
    with pytest.raises(ValueError, match="不存在"):
        vr.delete_collection("kb_pg2")
