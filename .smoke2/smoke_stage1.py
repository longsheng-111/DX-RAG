# -*- coding: utf-8 -*-
"""
阶段 1 波次 3 双后端端点冒烟：STORAGE_BACKEND=file / postgres 各跑一遍全链路。

用法（项目根目录）：
    backend/venv_pg/Scripts/python.exe .smoke2/smoke_stage1.py

结构：
- 父进程（默认）：依次以子进程跑 file、postgres 两套后端，两套全绿才退出码 0；
- 子进程（env SMOKE_CHILD=1 + SMOKE_BACKEND=xxx）：打桩重依赖 + 假 LLM/假嵌入，
  用 FastAPI TestClient 跑 7 组端点检查。

环境约束与对策：
- settings 是 import 期单例，两种后端无法在同进程内安全切换，故用子进程隔离；
- 本机无 torch / sentence-transformers / langchain_text_splitters：sys.modules 打桩
  （仅在真实 import 失败时才打桩；假切分器对无标题文本行为与真实实现一致——整体返回单 doc）；
- LLM 全部走 monkeypatch 的 TokenTracker.chat_completion（按 system prompt 分派罐头回复）；
- 嵌入为确定性 512 维 one-hot 向量：按文本前两个字符定轴，同前缀余弦相似度 1.0，
  保证冒烟问题必命中冒烟文档（0.65 过滤阈值 / 0.75 充分性阈值均通过）；
- postgres 后端用 SQLite（DATABASE_URL 指向 .smoke2/data/postgres/test.db），
  Base.metadata.create_all 建表（等价 alembic upgrade head 的 schema）；
- 项目路径含中文（RAG实干），FAISS C 扩展无法写非 ASCII 路径（config.py 的
  _safe_data_dir 注释记录的已知问题），故 file 后端的 VECTOR_PERSIST_DIR 指向
  tempfile.mkdtemp 的纯 ASCII 临时目录；其余数据（SQLite 库/会话/上传）仍按
  要求落在 .smoke2/data/<backend>/ 下。
"""
import atexit
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

SMOKE_ROOT = Path(__file__).resolve().parent          # .smoke2/
PROJECT_ROOT = SMOKE_ROOT.parent
BACKEND_DIR = PROJECT_ROOT / "backend"

EMB_DIM = 512
COLLECTION = "smoke_kb"                                # ASCII 名，Windows 下目录名最安全
DOC_TEXT = (
    "RAG 检索增强生成是一种结合检索与生成的技术。\n\n"
    "RAG 的核心流程包括索引、检索、生成三个步骤。\n\n"
    "冒烟测试用于验证系统主链路是否可用。"
)
QUESTION = "RAG 的核心流程是什么？"
EXAM_TOPIC = "RAG核心流程"
EXAM_ANSWER = "我认为包含索引、检索、生成三个步骤，先建索引再检索最后生成。"


# ============================================================
#  子进程：单后端全链路冒烟
# ============================================================

def _text_axis(text: str) -> int:
    """按文本前两个字符定轴：同前缀文本 one-hot 相同（余弦 1.0），保证冒烟检索必命中"""
    key = (text or "").strip()[:2] or "空"
    return int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % EMB_DIM


def _fake_encode_texts(texts):
    """假嵌入：确定性 512 维 one-hot 矩阵 (N, 512)"""
    vecs = np.zeros((len(texts), EMB_DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        vecs[i, _text_axis(t)] = 1.0
    return vecs


def _fake_encode_query(query):
    """假嵌入：单查询向量 (512,)"""
    return _fake_encode_texts([query])[0]


def _install_module_stubs() -> None:
    """sys.modules 打桩重依赖（仅在本机真实 import 失败时生效）"""
    try:
        import torch  # noqa: F401
    except ImportError:
        sys.modules["torch"] = types.ModuleType("torch")

    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        st_mod = types.ModuleType("sentence_transformers")

        class SentenceTransformer:
            """假嵌入模型：维度 512，encode 走确定性 one-hot（与 monkeypatch 的假嵌入同规则）"""

            def __init__(self, *args, **kwargs):
                pass

            def get_sentence_embedding_dimension(self):
                return EMB_DIM

            def encode(self, texts, normalize_embeddings=True):
                return _fake_encode_texts(list(texts))

        st_mod.SentenceTransformer = SentenceTransformer
        sys.modules["sentence_transformers"] = st_mod

    try:
        import langchain_text_splitters  # noqa: F401
    except ImportError:
        lts_mod = types.ModuleType("langchain_text_splitters")

        class MarkdownHeaderTextSplitter:
            """假切分器：文本整体返回单个 doc（与真实实现对无标题文本的行为一致）"""

            def __init__(self, headers_to_split_on=None, strip_headers=True):
                self.headers_to_split_on = headers_to_split_on or []

            def split_text(self, text):
                return [types.SimpleNamespace(page_content=text, metadata={})]

        lts_mod.MarkdownHeaderTextSplitter = MarkdownHeaderTextSplitter
        sys.modules["langchain_text_splitters"] = lts_mod


def _make_fake_chat_completion(TokenCost):
    """生成假 LLM：按 system prompt 分派罐头回复，覆盖 QA/标题/出题/评分/总结全部调用点"""

    def _fake_chat_completion(self, timeout=60.0, **kwargs):
        messages = kwargs.get("messages", [])
        system = next(
            (m.get("content", "") for m in messages if m.get("role") == "system"), ""
        )
        user = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        if "只输出题目和评分要点" in system:
            # examiner.generate_question（首题/追问/下一题共用）
            content = (
                "【题目】RAG 的核心流程包含哪些步骤？\n"
                "【类型】概念题\n"
                "【考察点】检索、生成\n"
                "【参考答案要点】\n1. 索引\n2. 检索\n3. 生成"
            )
        elif "必须输出合法 JSON" in system:
            # examiner.evaluate_answer（8 分：不触发追问，走下一题分支）
            content = json.dumps(
                {
                    "points": [{"point": "检索", "hit": True, "evidence": "冒烟证据"}],
                    "score": 8,
                    "comment": "回答正确",
                    "supplement": "无",
                    "correction": "无",
                    "has_next_question": True,
                },
                ensure_ascii=False,
            )
        elif "输出面试总结" in system:
            # examiner.generate_summary
            content = (
                "【总体评分】80\n"
                "【能力画像】基础扎实。\n"
                "【薄弱环节】无\n"
                "【复习建议】继续巩固。"
            )
        elif "只输出标题文本" in system:
            # title_generator.generate_session_title
            content = "RAG冒烟会话"
        elif "只输出查询词" in system:
            # agent.rewrite_query（冒烟检索一次命中，正常不会走到）
            content = user[:20] or "RAG"
        else:
            # qa.generate_answer（默认 QA 回答）
            content = "【冒烟回答】RAG 是检索增强生成技术，核心流程为索引、检索、生成。[^1^]"

        response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))],
            usage=types.SimpleNamespace(
                prompt_tokens=10, completion_tokens=10, total_tokens=20
            ),
        )
        return response, TokenCost(prompt_tokens=10, completion_tokens=10, total_tokens=20)

    return _fake_chat_completion


def run_child(backend: str) -> int:
    """子进程入口：在指定后端下跑 7 组端点检查，全绿返回 0"""
    data_root = SMOKE_ROOT / "data" / backend
    data_root.mkdir(parents=True, exist_ok=True)

    # ---- 环境变量：必须先于一切 app import（settings 是 import 期单例）----
    os.environ["DEEPSEEK_API_KEY"] = "dummy"
    os.environ["STORAGE_BACKEND"] = backend
    # postgres 轮的 DATABASE_URL 可用环境变量 SMOKE_PG_DATABASE_URL 覆盖为真实
    # PostgreSQL 连接串（目标库需已 alembic upgrade head，含 vector 扩展）；
    # 缺省回退 SQLite 文件库，保证无 PG 环境下冒烟仍可运行
    pg_url_override = os.environ.get("SMOKE_PG_DATABASE_URL", "")
    if backend == "postgres" and pg_url_override:
        os.environ["DATABASE_URL"] = pg_url_override
    else:
        os.environ["DATABASE_URL"] = f"sqlite:///{(data_root / 'test.db').as_posix()}"
    # FAISS C 扩展在 Windows 下无法写中文路径（config._safe_data_dir 记录的已知问题），
    # 向量库目录改用纯 ASCII 临时目录；进程退出时清理
    vector_dir = tempfile.mkdtemp(prefix=f"dxrag_smoke_vec_{backend}_")
    atexit.register(lambda: shutil.rmtree(vector_dir, ignore_errors=True))
    os.environ["VECTOR_PERSIST_DIR"] = vector_dir
    os.environ["SESSION_DIR"] = str(data_root / "sessions")
    os.environ["UPLOAD_DIR"] = str(data_root / "uploads")

    _install_module_stubs()
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))

    # ---- 假 LLM：monkeypatch TokenTracker.chat_completion（类属性，各调用点统一生效）----
    from app.services.token_tracker import TokenTracker, TokenCost

    TokenTracker.chat_completion = _make_fake_chat_completion(TokenCost)

    # ---- 导入应用（此刻 settings 定型，storage 按 STORAGE_BACKEND 完成分派）----
    from app.main import create_app

    # ---- postgres 后端：建表 + 确认引擎指向冒烟库 ----
    if backend == "postgres":
        from app.db import Base, get_engine

        engine = get_engine()
        Base.metadata.create_all(engine)
        if pg_url_override:
            # 覆盖为真实 PG 时，断言引擎确实指向目标库（防环境变量被吞）
            expected_db = pg_url_override.rsplit("/", 1)[-1].split("?")[0]
            assert engine.url.database == expected_db, f"DB 指向异常: {engine.url}"
        else:
            assert engine.url.database.endswith("test.db"), f"DB 指向异常: {engine.url}"

    # ---- 假嵌入：patch 当前后端实际生效的模块 + retrieval 的模块级引用 ----
    import app.services.retrieval as retrieval_mod

    retrieval_mod.encode_texts = _fake_encode_texts
    retrieval_mod.encode_query = _fake_encode_query
    if backend == "postgres":
        import app.db.vector_repository as active_vs
    else:
        import app.core.vector_store as active_vs
    active_vs.encode_texts = _fake_encode_texts
    active_vs.encode_query = _fake_encode_query

    from fastapi.testclient import TestClient

    client = TestClient(create_app())

    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" | {detail}" if detail else ""))
        if not cond:
            failures.append(name)

    print(f"\n===== 子进程冒烟：STORAGE_BACKEND={backend} =====")

    # ① 健康检查
    r = client.get("/api/health")
    check("① GET /api/health → 200", r.status_code == 200 and r.json().get("status") == "ok")

    # ② 建库 → 上传入库 → 文件列表
    r = client.post("/api/collections", json={"name": COLLECTION})
    check("②a POST /api/collections 建库 → 200", r.status_code == 200, r.text[:200])
    r = client.post(
        "/api/upload",
        files={"file": ("smoke_note.txt", DOC_TEXT.encode("utf-8"), "text/plain")},
        data={"collection_name": COLLECTION},
    )
    up = r.json() if r.status_code == 200 else {}
    check(
        "②b POST /api/upload 上传入库 → 200 且有切片",
        r.status_code == 200 and up.get("chunks", 0) >= 1,
        f"status={r.status_code} chunks={up.get('chunks')} body={r.text[:200]}",
    )
    r = client.get("/api/files", params={"collection_name": COLLECTION})
    names = [f["file_name"] for f in r.json().get("files", [])] if r.status_code == 200 else []
    check("②c GET /api/files 见到上传文件", r.status_code == 200 and "smoke_note.txt" in names,
          f"files={names}")

    # ③ 问答全链路（真检索 + 假 LLM）
    r = client.post(
        "/api/chat",
        json={"question": QUESTION, "collection_name": COLLECTION, "top_k": 3},
    )
    chat = r.json() if r.status_code == 200 else {}
    cost = chat.get("token_cost", {})
    check(
        "③a POST /api/chat → 200 且 answer/sources 非空",
        r.status_code == 200 and bool(chat.get("answer")) and len(chat.get("sources", [])) >= 1,
        f"status={r.status_code} answer={str(chat.get('answer'))[:40]!r} sources={len(chat.get('sources', []))}",
    )
    check(
        "③b 响应字段齐全（session_total/reasoning_steps/queries）",
        "session_total" in cost
        and isinstance(chat.get("reasoning_steps"), list)
        and len(chat.get("queries", [])) >= 1,
        f"token_cost={cost} queries={chat.get('queries')}",
    )
    sid = chat.get("session_id", "")

    # ④ 会话持久化：消息完整读回
    r = client.get(f"/api/sessions/{sid}")
    msgs = r.json().get("messages", []) if r.status_code == 200 else []
    roles = [m.get("role") for m in msgs]
    check(
        "④ GET /api/sessions/{id} 消息完整（user+assistant）",
        r.status_code == 200 and roles == ["user", "assistant"] and msgs[0].get("content") == QUESTION,
        f"status={r.status_code} roles={roles} sid={sid}",
    )

    # ⑤ 反馈闭环
    r = client.post(
        "/api/feedback",
        json={
            "session_id": sid,
            "original_chunk_id": "chunk_a",
            "target_chunk_id": "chunk_b",
            "type": "replace",
            "note": "冒烟反馈",
        },
    )
    fb = r.json() if r.status_code == 200 else {}
    check(
        "⑤ POST /api/feedback → 200 且回写 session_id",
        r.status_code == 200 and fb.get("status") == "ok"
        and fb.get("feedback", {}).get("session_id") == sid,
        f"status={r.status_code} body={r.text[:200]}",
    )

    # ⑥ 考官流：建考官会话 → start → next 作答 → end 收尾
    r = client.post(
        "/api/sessions",
        json={
            "persona": "default",
            "collection_name": COLLECTION,
            "title": "冒烟面试",
            "mode": "examiner",
            "target_position": "后端工程师",
            "topic": EXAM_TOPIC,
        },
    )
    exam_sid = r.json().get("session_id", "") if r.status_code == 200 else ""
    check("⑥a 创建考官会话 → 200", r.status_code == 200 and bool(exam_sid), f"sid={exam_sid}")

    r = client.post("/api/exam/start", json={"session_id": exam_sid})
    start = r.json() if r.status_code == 200 else {}
    check(
        "⑥b POST /api/exam/start → 200 且出第一题",
        r.status_code == 200 and start.get("status") == "asking" and bool(start.get("current_question")),
        f"status={r.status_code} q={str(start.get('current_question'))[:40]!r} body={r.text[:200]}",
    )

    r = client.post(f"/api/exam/{exam_sid}/next", json={"answer": EXAM_ANSWER})
    nxt = r.json() if r.status_code == 200 else {}
    ev = nxt.get("evaluation") or {}
    check(
        "⑥c POST /exam/next 作答 → 200 且有评分",
        r.status_code == 200 and ev.get("score") == 8.0 and len(nxt.get("scores", [])) == 1,
        f"status={r.status_code} score={ev.get('score')} body={r.text[:200]}",
    )

    r = client.post(f"/api/exam/{exam_sid}/next", json={"action": "end"})
    end = r.json() if r.status_code == 200 else {}
    check(
        "⑥d action=end → 200 且面试收尾（finished+summary）",
        r.status_code == 200 and end.get("status") == "finished"
        and isinstance(end.get("summary"), dict) and bool(end["summary"].get("raw")),
        f"status={r.status_code} exam_status={end.get('status')} total={ (end.get('summary') or {}).get('total_score') }",
    )
    # exam_state 落库/落盘后可恢复
    r = client.get(f"/api/exam/{exam_sid}")
    check(
        "⑥e GET /api/exam/{id} 恢复终态",
        r.status_code == 200 and r.json().get("status") == "finished",
        f"status={r.status_code} body={r.text[:160]}",
    )

    # ⑦ 删除知识库：不存在的库 → 404（裁决①，两种后端一致）；存在的库 → 200（回归）
    r = client.delete("/api/collections/不存在的库名")
    check(
        "⑦a DELETE /api/collections/{不存在} → 404",
        r.status_code == 404,
        f"status={r.status_code} body={r.text[:160]}",
    )
    r = client.delete(f"/api/collections/{COLLECTION}")
    check(
        "⑦b DELETE /api/collections/{存在} → 200（预检不误伤正常删除）",
        r.status_code == 200 and r.json().get("status") == "deleted",
        f"status={r.status_code} body={r.text[:160]}",
    )

    if failures:
        print(f"  → 后端 {backend} 冒烟失败 {len(failures)} 项: {failures}")
        return 1
    print(f"  → 后端 {backend} 冒烟全部通过")
    return 0


# ============================================================
#  父进程：依次调度两套后端子进程
# ============================================================

def main() -> int:
    # 统一 UTF-8 输出，避免 Windows GBK 控制台乱码
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if os.environ.get("SMOKE_CHILD") == "1":
        return run_child(os.environ["SMOKE_BACKEND"])

    print("===== 阶段 1 波次 3 双后端冒烟（file + postgres） =====")
    overall_ok = True
    for backend in ("file", "postgres"):
        # 每轮清空数据目录，保证可重复执行
        data_dir = SMOKE_ROOT / "data" / backend
        if data_dir.exists():
            shutil.rmtree(data_dir)
        env = dict(
            os.environ,
            SMOKE_CHILD="1",
            SMOKE_BACKEND=backend,
            PYTHONIOENCODING="utf-8",  # 子进程输出 UTF-8，避免 GBK 控制台乱码
        )
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve())],
            env=env,
            cwd=str(PROJECT_ROOT),
        )
        ok = proc.returncode == 0
        overall_ok = overall_ok and ok
        print(f"[父进程] 后端 {backend}: {'通过' if ok else '失败'}（退出码 {proc.returncode}）")

    print(f"\n===== 总体结果: {'全部通过' if overall_ok else '存在失败'} =====")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
