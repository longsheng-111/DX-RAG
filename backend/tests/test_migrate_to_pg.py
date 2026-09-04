"""
migrate_to_pg.py（文件存储 → DB 一次性迁移脚本）测试

- 合成夹具：FAISS IndexFlatIP(512) 写确定性向量 + metadata.json + _collections.json
  + 两个 sess_*.json + feedbacks.json（含一条孤儿反馈），全部放临时目录
- 必须在 import 任何 app 模块之前设置 DATABASE_URL / VECTOR_PERSIST_DIR /
  SESSION_DIR / UPLOAD_DIR（settings 是 import 期单例）；脚本经 importlib 按路径加载
- SQLite 方言：向量列走 JSON 退化路径，直接比对 list[float]
- 用例按文件顺序共享同一库：首轮迁移 → 幂等重跑 → 退出码 → 丢失检测（放最后，
  因为它会故意删一行制造丢失）
"""
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path

import numpy as np
import faiss
import pytest

# ---- 测试环境准备：必须在 import app / 加载脚本之前完成 ----
_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

_TMP = Path(tempfile.mkdtemp(prefix="migrate_to_pg_test_"))
_VECTOR_DIR = _TMP / "vector_db"
_SESSION_DIR = _TMP / "sessions"
_FEEDBACK_DIR = _TMP / "feedback"   # 与 session 目录同级，沿用 FeedbackStore 的约定
_UPLOAD_DIR = _TMP / "uploads"

os.environ["DATABASE_URL"] = f"sqlite:///{(_TMP / 'test.db').as_posix()}"
os.environ["VECTOR_PERSIST_DIR"] = str(_VECTOR_DIR)
os.environ["SESSION_DIR"] = str(_SESSION_DIR)
os.environ["UPLOAD_DIR"] = str(_UPLOAD_DIR)

EMB_DIM = 512


def _unit(i: int) -> np.ndarray:
    """第 i 个基向量（确定性，便于 reconstruct 后逐位比对）"""
    v = np.zeros(EMB_DIM, dtype=np.float32)
    v[i] = 1.0
    return v


# ============================================================
#  合成夹具
# ============================================================

def _write_kb(name: str, docs, metas, ids, vecs) -> None:
    """写一个知识库：index.faiss（IndexFlatIP）+ metadata.json"""
    col_dir = _VECTOR_DIR / name
    col_dir.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(EMB_DIM)
    index.add(np.stack(vecs).astype(np.float32))
    faiss.write_index(index, str(col_dir / "index.faiss"))
    (col_dir / "metadata.json").write_text(
        json.dumps({"documents": docs, "metadatas": metas, "ids": ids},
                   ensure_ascii=False),
        encoding="utf-8",
    )


def _write_session(sid: str, data: dict) -> None:
    _SESSION_DIR.mkdir(parents=True, exist_ok=True)
    (_SESSION_DIR / f"{sid}.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )


# 知识库 kb_alpha：a.txt（2 切片）+ b.txt（2 切片，原始文件缺失）；kb_beta：c.txt（1 切片）
_ALPHA_IDS = [str(uuid.uuid4()) for _ in range(4)]
_BETA_IDS = [str(uuid.uuid4())]
_VECS = [_unit(i) for i in range(4)] + [_unit(10)]

_A_TXT = _UPLOAD_DIR / "a.txt"
_C_TXT = _UPLOAD_DIR / "c.txt"

_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
_A_TXT.write_text("苹果文件原始内容", encoding="utf-8")
_C_TXT.write_text("橙子文件原始内容", encoding="utf-8")

_VECTOR_DIR.mkdir(parents=True, exist_ok=True)
(_VECTOR_DIR / "_collections.json").write_text(
    json.dumps({"kb_alpha": {"created": True}, "kb_beta": {"created": True}},
               ensure_ascii=False),
    encoding="utf-8",
)
_write_kb(
    "kb_alpha",
    ["苹果 第一段", "苹果 第二段", "香蕉 第一段", "香蕉 第二段"],
    [
        {"file_name": "a.txt", "source": str(_A_TXT), "chunk_index": 0,
         "collection_name": "kb_alpha"},
        {"file_name": "a.txt", "source": str(_A_TXT), "chunk_index": 1,
         "collection_name": "kb_alpha"},
        # b.txt 的 source 路径不存在，且 uploads/ 下也没有 → file_hash 应置空
        {"file_name": "b.txt", "source": str(_UPLOAD_DIR / "不存在_b.txt"),
         "chunk_index": 0, "collection_name": "kb_alpha"},
        {"file_name": "b.txt", "source": str(_UPLOAD_DIR / "不存在_b.txt"),
         "chunk_index": 1, "collection_name": "kb_alpha"},
    ],
    _ALPHA_IDS,
    _VECS[:4],
)
_write_kb(
    "kb_beta",
    ["橙子 第一段"],
    [{"file_name": "c.txt", "source": str(_C_TXT), "chunk_index": 0,
      "collection_name": "kb_beta"}],
    _BETA_IDS,
    [_VECS[4]],
)

# 会话：一个考官模式（含 exam_state / 带 token_cost 的消息），一个普通 QA
SID1 = "sess_" + uuid.uuid4().hex[:16]
SID2 = "sess_" + uuid.uuid4().hex[:16]
SID_ORPHAN = "sess_" + uuid.uuid4().hex[:16]  # 只在 feedbacks.json 里出现

_S1_MESSAGES = [
    {"role": "user", "content": "开始面试", "timestamp": "2026-08-01T10:00:00.5"},
    {"role": "assistant", "content": "第一题：什么是索引？",
     "timestamp": "2026-08-01T10:00:01.5",
     "token_cost": {"total_tokens": 1234},
     "sources": [{"file_name": "a.txt", "content": "苹果 第一段"}],
     "reasoning_steps": [{"step": "检索", "detail": "命中 a.txt"}]},
]
_S1_EXAM_STATE = {"q_index": 2, "scores": [8, 9], "weak_points": ["事务"]}
_write_session(SID1, {
    "session_id": SID1,
    "title": "后端面试",
    "persona": "backend",
    "kb_id": "kb_alpha",
    "mode": "examiner",
    "messages": _S1_MESSAGES,
    "created_at": "2026-08-01T10:00:00.123456",
    "updated_at": "2026-08-02T11:22:33.654321",
    "total_tokens": 1234,
    "compressed_history": [{"summary": "已压缩一轮"}],
    "exam_state": _S1_EXAM_STATE,
})
_write_session(SID2, {
    "session_id": SID2,
    "title": "随便问问",
    "persona": "default",
    "kb_id": "kb_beta",
    "mode": "qa",
    "messages": [{"role": "user", "content": "你好", "timestamp": "2026-08-03T09:00:00"}],
    "created_at": "2026-08-03T09:00:00",
    "updated_at": "2026-08-03T09:00:00",
    "total_tokens": 0,
    "compressed_history": [],
    "exam_state": {},
})

# 反馈：一条属于 SID1，一条孤儿（session 不存在）
FB_OK = "fb_" + uuid.uuid4().hex[:12]
FB_ORPHAN = "fb_" + uuid.uuid4().hex[:12]
_FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
(_FEEDBACK_DIR / "feedbacks.json").write_text(
    json.dumps({
        SID1: [{
            "id": FB_OK,
            "session_id": SID1,
            "original_chunk_id": _ALPHA_IDS[0],
            "target_chunk_id": _ALPHA_IDS[2],
            "type": "replace",
            "original_content": "苹果 第一段",
            "target_content": "香蕉 第一段",
            "note": "引用错了",
            "created_at": "2026-08-04T12:00:00",
        }],
        SID_ORPHAN: [{
            "id": FB_ORPHAN,
            "session_id": SID_ORPHAN,
            "original_chunk_id": "",
            "target_chunk_id": "",
            "type": "inaccurate",
            "original_content": "",
            "target_content": "",
            "note": "孤儿反馈",
            "created_at": "2026-08-04T12:30:00",
        }],
    }, ensure_ascii=False),
    encoding="utf-8",
)

# ---- 夹具就绪，按路径加载迁移脚本（内部 import app，此时 env 已设置） ----
_spec = importlib.util.spec_from_file_location(
    "migrate_to_pg", _BACKEND_DIR / "scripts" / "migrate_to_pg.py"
)
migrate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(migrate)

from sqlalchemy import func, select  # noqa: E402

from app.db import new_session  # noqa: E402
from app.db.models import Chunk, Collection, Feedback, File, Session as DbSession  # noqa: E402


# ============================================================
#  ① 首轮迁移：行数 / 内容抽查 / 报告
# ============================================================

def test_first_run_migration():
    report = migrate.run_migration()
    assert report["zero_loss"] is True
    assert report["losses"] == []

    with new_session() as db:
        # collections：2 行，display_name 回填为 name
        cols = {c.name: c for c in db.scalars(select(Collection)).all()}
        assert set(cols) == {"kb_alpha", "kb_beta"}
        assert cols["kb_alpha"].display_name == "kb_alpha"

        # files：3 行；a.txt 有内容 hash，b.txt 原始文件缺失 → file_hash 置空
        files = {f.file_name: f for f in db.scalars(select(File)).all()}
        assert set(files) == {"a.txt", "b.txt", "c.txt"}
        assert files["a.txt"].file_hash == hashlib.sha256(
            "苹果文件原始内容".encode("utf-8")
        ).hexdigest()
        assert files["a.txt"].size_bytes > 0
        assert files["a.txt"].status == "done"
        assert files["b.txt"].file_hash is None
        assert files["b.txt"].storage_path == ""
        assert files["c.txt"].file_hash is not None

        # chunks：5 行；a.txt 两个切片 chunk_index 保序
        chunks = db.scalars(
            select(Chunk).where(Chunk.file_id == files["a.txt"].id)
            .order_by(Chunk.chunk_index)
        ).all()
        assert [c.chunk_index for c in chunks] == [0, 1]
        assert [c.content for c in chunks] == ["苹果 第一段", "苹果 第二段"]
        assert db.scalar(select(func.count()).select_from(Chunk)) == 5

        # embedding：与写入 FAISS 的向量逐位一致（reconstruct 导出，未重算）
        all_chunks = db.scalars(select(Chunk)).all()
        expected_by_content = {
            "苹果 第一段": _VECS[0], "苹果 第二段": _VECS[1],
            "香蕉 第一段": _VECS[2], "香蕉 第二段": _VECS[3],
            "橙子 第一段": _VECS[4],
        }
        assert {c.content for c in all_chunks} == set(expected_by_content)
        for c in all_chunks:
            assert c.embedding is not None
            assert len(c.embedding) == EMB_DIM
            assert np.allclose(
                np.asarray(c.embedding, dtype=np.float32),
                expected_by_content[c.content],
                atol=1e-6,
            ), f"{c.content} 的 embedding 与 FAISS 向量不一致"

        # sessions：2 行，JSON 列与时间原值 round-trip
        s1 = db.get(DbSession, SID1)
        assert s1 is not None
        assert s1.title == "后端面试"
        assert s1.mode == "examiner"
        assert s1.messages == _S1_MESSAGES
        assert s1.compressed_history == [{"summary": "已压缩一轮"}]
        assert s1.exam_state == _S1_EXAM_STATE
        assert s1.total_tokens == 1234
        assert s1.created_at.isoformat() == "2026-08-01T10:00:00.123456"
        assert s1.updated_at.isoformat() == "2026-08-02T11:22:33.654321"
        s2 = db.get(DbSession, SID2)
        assert s2 is not None and s2.total_tokens == 0

        # feedbacks：正常 1 行入库；孤儿被跳过
        fb = db.get(Feedback, FB_OK)
        assert fb is not None
        assert fb.session_id == SID1
        assert fb.type == "replace"
        assert fb.original_chunk_id == _ALPHA_IDS[0]
        assert fb.target_chunk_id == _ALPHA_IDS[2]
        assert fb.note == "引用错了"
        assert fb.created_at.isoformat() == "2026-08-04T12:00:00"
        assert db.get(Feedback, FB_ORPHAN) is None

    # 报告抽查：条数比对全 OK，孤儿反馈出现在报告，无 reconstruct 失败
    per_kb = {c["name"]: c for c in report["collections"]}
    assert per_kb["kb_alpha"]["expected_chunks"] == 4
    assert per_kb["kb_alpha"]["db_chunks"] == 4
    assert per_kb["kb_alpha"]["chunks_new"] == 4
    assert per_kb["kb_beta"]["db_chunks"] == 1
    assert report["reconstruct_failures"] == []
    assert report["sessions"]["new"] == 2
    assert report["feedbacks"]["new"] == 1
    assert [o["id"] for o in report["feedbacks"]["orphans"]] == [FB_ORPHAN]
    # b.txt 原始文件缺失应记入报告说明
    assert any("b.txt" in note and "file_hash" in note for note in report["notes"])


# ============================================================
#  ② 重复跑幂等：0 新增，行数不变
# ============================================================

def test_rerun_is_idempotent():
    report = migrate.run_migration()
    assert report["zero_loss"] is True
    assert sum(c["files_new"] for c in report["collections"]) == 0
    assert sum(c["chunks_new"] for c in report["collections"]) == 0
    assert sum(c["files_skipped"] for c in report["collections"]) == 3
    assert sum(c["chunks_skipped"] for c in report["collections"]) == 5
    assert report["sessions"]["new"] == 0
    assert report["sessions"]["skipped"] == 2
    assert report["feedbacks"]["new"] == 0
    assert report["feedbacks"]["skipped"] == 1
    # 孤儿反馈重跑仍然跳过且仍出现在报告
    assert [o["id"] for o in report["feedbacks"]["orphans"]] == [FB_ORPHAN]

    with new_session() as db:
        assert db.scalar(select(func.count()).select_from(Collection)) == 2
        assert db.scalar(select(func.count()).select_from(File)) == 3
        assert db.scalar(select(func.count()).select_from(Chunk)) == 5
        assert db.scalar(select(func.count()).select_from(DbSession)) == 2
        assert db.scalar(select(func.count()).select_from(Feedback)) == 1


# ============================================================
#  ③ CLI：--yes 直跑，零丢失时退出码 0
# ============================================================

def test_main_exit_code_zero(capsys):
    assert migrate.main(["--yes"]) == 0
    out = capsys.readouterr().out
    assert "迁移校验报告" in out
    assert "零丢失" in out
    # 幂等重跑，报告应显示 0 新增
    assert "新增0" in out


# ============================================================
#  ④ 丢失检测：人为删一行后校验必须报不一致、退出码非 0
#     （会破坏数据，放最后；前面的用例不依赖此后的库状态）
# ============================================================

def test_validation_detects_loss():
    with new_session() as db:
        chunk = db.scalar(select(Chunk).limit(1))
        db.delete(chunk)
        db.commit()

    # 文件行仍在 → 幂等跳过不会补回被删的 chunk → 校验必须抓到
    report = migrate.run_migration()
    assert report["zero_loss"] is False
    assert any("kb_alpha" in loss or "kb_beta" in loss for loss in report["losses"])
    assert migrate.main(["--yes"]) == 1
