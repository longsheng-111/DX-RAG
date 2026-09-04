"""
SessionRepository / FeedbackRepository 行为契约测试（阶段 1：数据库化，波次 2）

在 SQLite 临时库上验证 DB 版与文件版（SessionManager / FeedbackStore）的可观测行为一致。

注意：app.core.config.settings 是 import 期单例，DATABASE_URL 必须在 import app
任何模块之前设置；因此这里无法用 tmp_path fixture（fixture 运行时 import 已完成），
改用 import 期的 tempfile.mkdtemp 临时目录，效果等同。
"""
import os
import re
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import pytest

# ---- 必须早于一切 app import：指向临时 SQLite 库（绝对路径）----
_TEST_DIR = tempfile.mkdtemp(prefix="dxrag_repo_test_")
os.environ["DATABASE_URL"] = f"sqlite:///{(Path(_TEST_DIR) / 'test.db').as_posix()}"

from sqlalchemy import func, select  # noqa: E402

from app.core.session import Session  # noqa: E402
from app.db import Base, get_engine, new_session  # noqa: E402
from app.db.feedback_repository import FeedbackRepository  # noqa: E402
from app.db.models import Feedback as DbFeedback  # noqa: E402
from app.db.models import Session as DbSession  # noqa: E402
from app.db.session_repository import SessionRepository  # noqa: E402

_SESSION_ID_RE = re.compile(r"^sess_[a-f0-9]{16,}$")


def _session_to_dict(session) -> dict:
    """与 app.api.routes._session_to_dict 字段集逐字一致（验证 repo 返回值可直接复用）"""
    return {
        "session_id": session.session_id,
        "title": session.title,
        "persona": session.persona,
        "kb_id": session.kb_id,
        "mode": session.mode,
        "exam_state": session.exam_state,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "total_tokens": session.total_tokens,
        "message_count": len(session.messages),
    }


def _tick() -> None:
    """Windows 上 datetime.now() 粒度较粗（约 15ms），排序类断言前让时间往前走一点"""
    time.sleep(0.05)


@pytest.fixture(scope="session", autouse=True)
def _create_tables():
    Base.metadata.create_all(get_engine())


@pytest.fixture(autouse=True)
def _clean_tables():
    # 先删子表再删父表（bulk delete 不走 ORM 级联，需手动保证顺序）
    with new_session() as db:
        db.query(DbFeedback).delete()
        db.query(DbSession).delete()
        db.commit()
    yield


@pytest.fixture()
def repo() -> SessionRepository:
    return SessionRepository()


@pytest.fixture()
def fb_repo() -> FeedbackRepository:
    return FeedbackRepository()


# ============================================================
#  会话 CRUD
# ============================================================

def test_create_session_defaults(repo):
    s = repo.create_session()
    assert _SESSION_ID_RE.match(s.session_id)
    assert s.title == "新会话"
    assert s.persona == "default"
    assert s.kb_id == "knowledge_chunks"  # settings.vector_collection 默认值
    assert s.mode == "qa"
    assert s.messages == []
    assert s.compressed_history == []
    assert s.exam_state == {}
    assert s.total_tokens == 0
    # naive 本地时间 isoformat（前端按本地时区解析）
    assert datetime.fromisoformat(s.created_at).tzinfo is None
    assert datetime.fromisoformat(s.updated_at).tzinfo is None


def test_create_session_with_exam_state_roundtrip(repo):
    exam_state = {"status": "running", "scores": [1, 2], "max_follow_up": 2}
    s = repo.create_session(persona="interviewer", kb_id="kb_x", title="面试",
                            mode="examiner", exam_state=exam_state)
    loaded = repo.get_session(s.session_id)
    assert loaded is not None
    assert loaded.session_id == s.session_id
    assert loaded.title == "面试"
    assert loaded.persona == "interviewer"
    assert loaded.kb_id == "kb_x"
    assert loaded.mode == "examiner"
    assert loaded.exam_state == exam_state
    assert loaded.created_at == s.created_at
    assert loaded.updated_at == s.updated_at


def test_get_session_not_found_returns_none(repo):
    assert repo.get_session("sess_" + "f" * 16) is None


def test_save_session_upsert_and_refresh_updated_at(repo):
    s = repo.create_session()
    old_updated_at = s.updated_at
    _tick()
    s.title = "改标题"
    s.add_message("user", "你好")
    repo.save_session(s)
    loaded = repo.get_session(s.session_id)
    assert loaded.title == "改标题"
    assert [m["content"] for m in loaded.messages] == ["你好"]
    assert loaded.updated_at > old_updated_at  # save 会刷新 updated_at


def test_save_session_creates_when_missing(repo):
    # 与文件版一致：save 对不存在的会话执行创建（upsert）
    s = Session(session_id="sess_" + "a" * 16, title="手工构造")
    repo.save_session(s)
    loaded = repo.get_session(s.session_id)
    assert loaded is not None
    assert loaded.title == "手工构造"


def test_message_with_sources_roundtrip(repo):
    s = repo.create_session()
    s.add_message(
        "assistant", "回答",
        token_cost={"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
        sources=[{"chunk_id": "c1", "score": 0.9}],
        reasoning_steps=[{"step": "retrieve"}],
    )
    repo.save_session(s)
    loaded = repo.get_session(s.session_id)
    msg = loaded.messages[0]
    assert msg["role"] == "assistant"
    assert msg["token_cost"]["total_tokens"] == 12
    assert msg["sources"] == [{"chunk_id": "c1", "score": 0.9}]
    assert msg["reasoning_steps"] == [{"step": "retrieve"}]
    assert "timestamp" in msg


# ============================================================
#  非法 / 不存在 id
# ============================================================

@pytest.mark.parametrize("bad_id", [
    "not-a-session",
    "sess_short",
    "sess_" + "a" * 15,          # 位数不足
    "sess_" + "A" * 16,          # 大写非法
    "sess_" + "g" * 16,          # 非 hex
    "../sess_" + "a" * 16,       # 路径穿越
])
def test_invalid_session_id_raises_value_error(repo, bad_id):
    with pytest.raises(ValueError, match="Invalid session ID format"):
        repo.get_session(bad_id)
    with pytest.raises(ValueError, match="Invalid session ID format"):
        repo.delete_session(bad_id)
    with pytest.raises(ValueError, match="Invalid session ID format"):
        repo.save_session(Session(session_id=bad_id))


def test_delete_session_missing_returns_false(repo):
    assert repo.delete_session("sess_" + "b" * 16) is False


# ============================================================
#  token 累加 / 大整数
# ============================================================

def test_add_message_accumulates_total_tokens(repo):
    s = repo.create_session()
    s.add_message("user", "问题1", token_cost={"total_tokens": 10})
    s.add_message("assistant", "回答1", token_cost={"total_tokens": 15})
    s.add_message("user", "问题2")  # 无 token_cost 不累加
    assert s.total_tokens == 25
    repo.save_session(s)
    assert repo.get_session(s.session_id).total_tokens == 25


def test_update_total_tokens(repo):
    s = repo.create_session()
    s.update_total_tokens(7)
    s.update_total_tokens(3)
    repo.save_session(s)
    assert repo.get_session(s.session_id).total_tokens == 10


def test_total_tokens_bigint(repo):
    s = repo.create_session()
    s.total_tokens = 2**40 + 12345  # 超过 32 位，验证 BIGINT 映射
    repo.save_session(s)
    assert repo.get_session(s.session_id).total_tokens == 2**40 + 12345


# ============================================================
#  list / 排序 / 返回结构
# ============================================================

def test_list_sessions_sorted_by_updated_at_desc(repo):
    s1 = repo.create_session(title="第一批")
    _tick()
    s2 = repo.create_session(title="第二批")
    _tick()
    s1.title = "第一批-改"
    repo.save_session(s1)  # 再保存 s1，使其成为最新
    ids = [s.session_id for s in repo.list_sessions()]
    assert ids == [s1.session_id, s2.session_id]


def test_list_sessions_field_set_compatible_with_routes(repo):
    repo.create_session(title="字段校验")
    listed = repo.list_sessions()
    assert len(listed) == 1
    d = _session_to_dict(listed[0])
    assert set(d.keys()) == {
        "session_id", "title", "persona", "kb_id", "mode", "exam_state",
        "created_at", "updated_at", "total_tokens", "message_count",
    }
    assert d["message_count"] == 0
    assert datetime.fromisoformat(d["updated_at"]).tzinfo is None


def test_list_sessions_empty(repo):
    assert repo.list_sessions() == []


# ============================================================
#  persona / 标题 / 删除
# ============================================================

def test_switch_persona(repo):
    s = repo.create_session(persona="default")
    s.add_message("user", "hi")
    repo.save_session(s)

    updated = repo.switch_persona(s.session_id, "teacher")
    assert updated.persona == "teacher"
    assert len(updated.messages) == 1  # 默认不清历史

    cleared = repo.switch_persona(s.session_id, "default", clear_history=True)
    assert cleared.messages == []
    assert cleared.compressed_history == []
    assert repo.get_session(s.session_id).persona == "default"

    assert repo.switch_persona("sess_" + "c" * 16, "teacher") is None


def test_update_title(repo):
    s = repo.create_session()
    updated = repo.update_title(s.session_id, "新标题")
    assert updated.title == "新标题"
    assert repo.get_session(s.session_id).title == "新标题"
    assert repo.update_title("sess_" + "d" * 16, "x") is None


def test_delete_session(repo):
    s = repo.create_session()
    assert repo.delete_session(s.session_id) is True
    assert repo.get_session(s.session_id) is None
    assert repo.delete_session(s.session_id) is False


# ============================================================
#  并发保存（两线程交替 add_message + save）
# ============================================================

def test_concurrent_add_message_save(repo):
    s = repo.create_session()
    errors = []
    rounds = 30

    def worker(tag: str) -> None:
        try:
            local = repo.get_session(s.session_id)
            for i in range(rounds):
                local.add_message("user", f"{tag}-{i}",
                                  token_cost={"total_tokens": 3})
                repo.save_session(local)
        except Exception as e:  # pragma: no cover - 断言在下方统一做
            errors.append(e)

    t1 = threading.Thread(target=worker, args=("A",))
    t2 = threading.Thread(target=worker, args=("B",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert errors == []
    final = repo.get_session(s.session_id)
    # 整行最后一次写入获胜（与文件版原子替换语义一致）：
    # 最终状态必为某一线程的完整快照，不允许出现撕裂/混搭损坏
    assert len(final.messages) >= rounds
    assert all(re.fullmatch(r"[AB]-\d+", m["content"]) for m in final.messages)
    assert final.total_tokens == 3 * len(final.messages)


# ============================================================
#  反馈：增 / 查 / prompt 格式化
# ============================================================

def test_add_feedback_return_structure(repo, fb_repo):
    s = repo.create_session()
    fb = fb_repo.add_feedback(
        session_id=s.session_id,
        original_chunk_id="chunk-1",
        target_chunk_id="chunk-2",
        feedback_type="replace",
        original_content="原片段",
        target_content="应采用片段",
        note="用户备注",
    )
    assert list(fb.keys()) == [
        "id", "session_id", "original_chunk_id", "target_chunk_id",
        "type", "original_content", "target_content", "note", "created_at",
    ]
    assert fb["id"].startswith("fb_") and len(fb["id"]) == len("fb_") + 12
    assert fb["session_id"] == s.session_id
    assert fb["type"] == "replace"
    assert datetime.fromisoformat(fb["created_at"]).tzinfo is None


def test_get_session_feedback_aggregation(repo, fb_repo):
    s1 = repo.create_session()
    s2 = repo.create_session()
    fb_repo.add_feedback(s1.session_id, "c1", "c2", "replace", note="n1")
    _tick()
    fb_repo.add_feedback(s1.session_id, "c3", "c4", "inaccurate", note="n2")
    _tick()
    fb_repo.add_feedback(s2.session_id, "c9", "c9", "replace", note="other")

    fbs = fb_repo.get_session_feedback(s1.session_id)
    assert [f["note"] for f in fbs] == ["n1", "n2"]  # 按写入时间正序
    assert {f["session_id"] for f in fbs} == {s1.session_id}
    assert fbs[0]["type"] == "replace"
    assert fbs[1]["type"] == "inaccurate"
    # 无反馈的会话返回空列表
    assert fb_repo.get_session_feedback("sess_" + "e" * 16) == []


def test_format_feedback_for_prompt(repo, fb_repo):
    s = repo.create_session()
    assert fb_repo.format_feedback_for_prompt(s.session_id) == ""
    fb_repo.add_feedback(
        s.session_id, "c1", "c2", "replace",
        original_content="错误片段", target_content="正确片段", note="注意口径",
    )
    text = fb_repo.format_feedback_for_prompt(s.session_id)
    assert "【历史反馈 / 事实修正约束】" in text
    assert "反馈类型：replace" in text
    assert "原引用片段：错误片段" in text
    assert "应优先采用的片段：正确片段" in text
    assert "用户备注：注意口径" in text
    assert text.endswith("以上约束优先级高于检索结果，后续回答请优先遵守。")


# ============================================================
#  审计修复的 DB 等价物
# ============================================================

def test_feedback_writes_are_row_isolated(repo, fb_repo):
    """#13 的 DB 语义：反馈按行存储，会话间的写入互不影响、不会整体丢失"""
    s1 = repo.create_session()
    s2 = repo.create_session()
    first = fb_repo.add_feedback(s1.session_id, "c1", "c2", "replace", note="s1-首条")
    for i in range(5):
        fb_repo.add_feedback(s2.session_id, "x", "y", "inaccurate", note=f"s2-{i}")
    fb_repo.add_feedback(s1.session_id, "c3", "c4", "replace", note="s1-次条")
    s1_fbs = fb_repo.get_session_feedback(s1.session_id)
    assert [f["note"] for f in s1_fbs] == ["s1-首条", "s1-次条"]
    assert s1_fbs[0] == first  # 首条反馈内容未被后续任何写入影响
    assert len(fb_repo.get_session_feedback(s2.session_id)) == 5


def test_delete_session_cascades_feedback(repo, fb_repo):
    """删除会话级联删除反馈（FK ondelete CASCADE / ORM cascade 生效）"""
    s = repo.create_session()
    fb_repo.add_feedback(s.session_id, "c1", "c2", "replace", note="n1")
    fb_repo.add_feedback(s.session_id, "c3", "c4", "inaccurate", note="n2")

    assert repo.delete_session(s.session_id) is True
    assert fb_repo.get_session_feedback(s.session_id) == []
    # 直连 DB 验证反馈行确实被物理删除，而非仅查询过滤
    with new_session() as db:
        remaining = db.scalar(select(func.count()).select_from(DbFeedback))
    assert remaining == 0


def test_clear_session_keeps_feedback(repo, fb_repo):
    """保持现状：清空会话消息不触碰反馈（该语义的变更属另一审计项，不在本波次）"""
    s = repo.create_session()
    fb_repo.add_feedback(s.session_id, "c1", "c2", "replace", note="保留我")
    s.messages = []
    s.compressed_history = []
    repo.save_session(s)  # 等价于 routes 的 /sessions/{id}/clear
    assert len(fb_repo.get_session_feedback(s.session_id)) == 1
