"""
Exam Graph（app/graphs/exam_graph.py）与 EXAM_ENGINE 接线测试

- 假 LLM：monkeypatch exam_graph 模块内的 generate_question / evaluate_answer /
  generate_summary，绝不发真实 API 请求
- 假检索：图驱动测试直接传入假 chunks；路由接线测试打桩 retrieval.AdvancedRetriever
- 每个用例重置编译单例（全新 InMemorySaver），session_id 互不串场

覆盖阶段 3 验收的审计 #5~#9 回归：
- #5 token 无双计（响应 token_cost 与逐消息 token_cost 同源，各计一次）
- #6 LLM 报错不落盘为题目（出题/评分失败返回 error，exam_state 为 None）
- #7 显式结束（action=end）不被当答案评分；give_up 按"不知道"正常评分
- #8 评分失败后可原样重试，进度不错位；无 checkpoint 时从 exam_state 落盘恢复
- #9 追问流程与面试进度一致性（question_index / follow_up_count / scores）
"""
import asyncio
import os
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

# ---- 测试环境准备：必须在 import app 之前完成 ----
_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# 单独运行本文件时，DATABASE_URL 指向 SQLite 临时库（同 test_vector_repo 约定）；
# 全量运行时本文件按字母序最先收集，随后 test_migrate_to_pg 会覆盖为自己的值
_TMP_DIR = tempfile.mkdtemp(prefix="exam_graph_test_")
os.environ["DATABASE_URL"] = f"sqlite:///{Path(_TMP_DIR, 'test.db').as_posix()}"

# venv_pg 无 sentence-transformers / torch：file 后端的 vector_store 在 import 期
# 硬依赖该包。本文件全程使用假检索/假 LLM，故打桩一个占位模块（同 test_qa_graph）
if "sentence_transformers" not in sys.modules:
    _st_stub = types.ModuleType("sentence_transformers")

    class SentenceTransformer:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("测试环境无真实嵌入模型，应使用假检索")

    _st_stub.SentenceTransformer = SentenceTransformer
    sys.modules["sentence_transformers"] = _st_stub

# ---- 惰性 import app 模块 ----
# settings 是 import 期单例：test_migrate_to_pg 等用例依赖"收集期首个 import app
# 的模块"注入 DATABASE_URL / VECTOR_PERSIST_DIR / SESSION_DIR 等环境变量。本文件
# 按字母序最先收集，若在模块级 import app 会抢建 settings 单例、挤掉它们的环境
# 变量，故一律推迟到 fixture 执行期（全部收集完成后）再 import。
exam_graph = None
routes = None
retrieval_mod = None
settings = None
Settings = None
TokenCost = None
set_cached_evaluation = None
MAX_QUESTIONS = 5
MAX_FOLLOW_UP = 2

# 出题 / 评分 / 总结三次 LLM 调用的固定假成本，便于精确断言无双计（fixture 期赋值）
Q_COST = None
E_COST = None
S_COST = None


@pytest.fixture(autouse=True)
def _lazy_app_imports():
    """延迟 import app 模块，并重置编译单例（全新 InMemorySaver，互不串场）"""
    global exam_graph, routes, retrieval_mod, settings, Settings
    global TokenCost, set_cached_evaluation, MAX_QUESTIONS, MAX_FOLLOW_UP
    global Q_COST, E_COST, S_COST

    from app.core.config import Settings as _Settings, settings as _settings
    from app.services import retrieval as _retrieval_mod
    from app.services.examiner import (
        MAX_FOLLOW_UP as _MAX_FOLLOW_UP,
        MAX_QUESTIONS as _MAX_QUESTIONS,
        set_cached_evaluation as _set_cached_evaluation,
    )
    from app.services.token_tracker import TokenCost as _TokenCost
    from app.api import routes as _routes
    from app.graphs import exam_graph as _exam_graph

    exam_graph = _exam_graph
    routes = _routes
    retrieval_mod = _retrieval_mod
    settings = _settings
    Settings = _Settings
    TokenCost = _TokenCost
    set_cached_evaluation = _set_cached_evaluation
    MAX_QUESTIONS = _MAX_QUESTIONS
    MAX_FOLLOW_UP = _MAX_FOLLOW_UP
    Q_COST = TokenCost(10, 5, 15)
    E_COST = TokenCost(20, 10, 30)
    S_COST = TokenCost(8, 4, 12)

    _exam_graph._compiled_graph = None
    yield
    _exam_graph._compiled_graph = None


FAKE_CHUNKS = [("file_a.txt", "RAG 是检索增强生成的缩写，先检索再生成。", 0.9)]


# ============================================================
#  假 LLM
# ============================================================

class _QuestionScript:
    """按调用次序返回预设题目的假出题器；follow_up_count>0 时自动加【追问】前缀。

    questions 队列中的元素为 None 时表示本次出题失败（返回 None，同真实
    generate_question 的失败约定）。
    """

    def __init__(self, questions=None, cost=None):
        self.questions = list(questions or [])
        self.calls = []
        self.cost = cost

    def __call__(self, follow_up_count=0, **kwargs):
        self.calls.append({"follow_up_count": follow_up_count, **kwargs})
        cost = self.cost or Q_COST
        question = self.questions.pop(0) if self.questions else f"第{len(self.calls)}题：什么是RAG？"
        if question is None:
            return None, [], [], cost
        if follow_up_count > 0:
            question = f"【追问】{question}"
        return question, ["考察点A"], ["要点1", "要点2"], cost


@pytest.fixture
def fake_llm(monkeypatch):
    """假 LLM 三件套，返回控制柄：score / eval_error 可调，calls 可断言"""
    ctrl = SimpleNamespace(
        score=8.0,
        eval_error=False,
        questions=_QuestionScript(),
        eval_calls=[],
        summary_calls=[],
    )

    def evaluate(answer="", **kwargs):
        ctrl.eval_calls.append({"answer": answer, **kwargs})
        if ctrl.eval_error:
            return {"error": "评分服务不可用"}, TokenCost()
        return {
            "raw": f"【评分】{ctrl.score}\n【点评】点评内容",
            "score": ctrl.score,
            "has_next_question": True,
            "points": [{"point": "要点1", "hit": ctrl.score >= 7, "evidence": "依据"}],
            "comment": "点评内容",
            "supplement": "",
            "correction": "",
        }, E_COST

    def summary(**kwargs):
        ctrl.summary_calls.append(kwargs)
        return {"raw": "【总体评分】80\n【能力画像】基础扎实", "total_score": 80.0}, S_COST

    monkeypatch.setattr(exam_graph, "generate_question", ctrl.questions)
    monkeypatch.setattr(exam_graph, "evaluate_answer", evaluate)
    monkeypatch.setattr(exam_graph, "generate_summary", summary)
    return ctrl


def _start(session_id: str, chunks=None):
    return exam_graph.exam_graph_start(
        session_id=session_id,
        target_position="后端工程师",
        topic="RAG",
        collection_name="kb_test",
        context_chunks=chunks if chunks is not None else FAKE_CHUNKS,
    )


def _next(session_id: str, exam_state, answer, action=None, chunks=None):
    return exam_graph.exam_graph_next(
        session_id=session_id,
        exam_state=exam_state,
        action=action,
        answer=answer,
        context_chunks=chunks if chunks is not None else FAKE_CHUNKS,
    )


# ============================================================
#  ① 开局：出题 + exam_state 落盘结构与 legacy 完全一致
# ============================================================

def test_start_returns_question_and_legacy_shaped_state(fake_llm):
    result = _start("sess_start_ok")

    assert result["error"] is None
    assert result["question"]
    state = result["exam_state"]
    # 键集合与 legacy exam_start 写入的 exam_state 完全一致
    assert set(state.keys()) == {
        "mode", "target_position", "topic", "collection_name",
        "status", "question_index", "current_question", "current_expectations",
        "reference_points", "current_thread", "follow_up_count", "max_follow_up",
        "scores", "weak_points", "summary", "answer_cache",
    }
    assert state["status"] == "asking"
    assert state["question_index"] == 1
    assert state["follow_up_count"] == 0
    assert state["max_follow_up"] == MAX_FOLLOW_UP
    assert state["current_question"] == result["question"]
    assert state["current_thread"] == [{"role": "assistant", "content": result["question"]}]
    assert state["scores"] == [] and state["summary"] is None
    # 消息与 token：题目消息一条，成本只计一次
    assert [m["role"] for m in result["new_messages"]] == ["assistant"]
    assert result["new_messages"][0]["token_cost"]["total_tokens"] == 15
    assert result["token_cost"]["total_tokens"] == 15


# ============================================================
#  ② 审计 #6：出题 LLM 故障不落盘为题目
# ============================================================

def test_start_llm_failure_not_persisted(fake_llm):
    fake_llm.questions.questions = [None]  # 本次出题失败

    result = _start("sess_start_fail")

    assert result["error"] == "生成题目失败，请重试"
    assert result["error_status"] == 502
    # error 时 exam_state 为 None，调用方无从落盘，报错文案不会变成题目
    assert result["exam_state"] is None
    assert result["new_messages"] == []


# ============================================================
#  ③ 正常作答：高分进下一题；审计 #5 token 无双计
# ============================================================

def test_high_score_goes_next_question_no_double_count(fake_llm):
    r0 = _start("sess_next")
    r1 = _next("sess_next", r0["exam_state"], "RAG 是检索增强生成")

    assert r1["error"] is None
    state = r1["exam_state"]
    assert state["status"] == "asking"
    assert state["question_index"] == 2
    assert state["follow_up_count"] == 0
    assert len(state["scores"]) == 1
    assert state["scores"][0]["score"] == 8.0
    assert state["current_question"] == state["current_thread"][-1]["content"]
    # 消息：用户作答 + 评分反馈 + 新题，顺序与 legacy 一致
    assert [m["role"] for m in r1["new_messages"]] == ["user", "assistant", "assistant"]
    # token：评分 30 + 出题 15，各计一次；逐消息成本之和 == 响应总成本（审计 #5）
    assert r1["token_cost"]["total_tokens"] == 45
    assert r1["token_cost"]["prompt_tokens"] == 30
    assert sum(
        m.get("token_cost", {}).get("total_tokens", 0) for m in r1["new_messages"]
    ) == 45
    assert r1["evaluation"]["score"] == 8.0
    assert r1["summary"] is None


# ============================================================
#  ④ 审计 #9：追问流程与追问上限
# ============================================================

def test_low_score_triggers_follow_up(fake_llm):
    fake_llm.score = 5.0
    r0 = _start("sess_fu")
    r1 = _next("sess_fu", r0["exam_state"], "我不太确定这个答案")

    state = r1["exam_state"]
    assert state["status"] == "follow_up"
    assert state["follow_up_count"] == 1
    assert state["question_index"] == 1  # 追问不换题
    assert state["current_question"].startswith("【追问】")
    # 追问出题收到 follow_up_count=1，且 thread 带上本轮问答（同 legacy）
    assert fake_llm.questions.calls[-1]["follow_up_count"] == 1
    assert len(fake_llm.questions.calls[-1]["thread"]) == 3
    assert [m["role"] for m in state["current_thread"]] == [
        "assistant", "user", "assistant", "assistant",
    ]
    assert len(state["scores"]) == 1


def test_follow_up_exhausted_then_next_question(fake_llm):
    fake_llm.score = 5.0
    r = _start("sess_fu2")
    for i in range(MAX_FOLLOW_UP):
        r = _next("sess_fu2", r["exam_state"], f"第{i + 1}次有漏洞的回答")
        assert r["exam_state"]["status"] == "follow_up"
        assert r["exam_state"]["follow_up_count"] == i + 1

    # 追问次数用完：即使低分也进下一题
    r = _next("sess_fu2", r["exam_state"], "第三次有漏洞的回答")
    state = r["exam_state"]
    assert state["status"] == "asking"
    assert state["question_index"] == 2
    assert state["follow_up_count"] == 0
    assert len(state["scores"]) == MAX_FOLLOW_UP + 1


# ============================================================
#  ⑤ 审计 #7：显式结束不被当答案评分；give_up 按"不知道"评分
# ============================================================

def test_action_end_skips_scoring(fake_llm):
    r0 = _start("sess_end")
    r1 = _next("sess_end", r0["exam_state"], "", action="end", chunks=[])

    assert r1["error"] is None
    # end 直接进入总结，不调用评分 LLM（action 不混在 answer 文本里）
    assert fake_llm.eval_calls == []
    state = r1["exam_state"]
    assert state["status"] == "finished"
    assert state["summary"]["total_score"] == 80.0
    assert state["scores"] == []
    assert r1["evaluation"] is None
    assert r1["summary"]["total_score"] == 80.0
    assert [m["role"] for m in r1["new_messages"]] == ["assistant"]
    # 只计总结一次 LLM 成本
    assert r1["token_cost"]["total_tokens"] == 12


def test_give_up_scored_as_unknown(fake_llm):
    r0 = _start("sess_gu")
    r1 = _next("sess_gu", r0["exam_state"], "", action="give_up")

    assert r1["error"] is None
    # give_up 按 legacy 语义记为"不知道"正常走评分
    assert fake_llm.eval_calls[0]["answer"] == "不知道"
    assert len(r1["exam_state"]["scores"]) == 1
    assert r1["exam_state"]["scores"][0]["answer"] == "不知道"


# ============================================================
#  ⑥ 答满 MAX_QUESTIONS 自动收尾（同时覆盖 restored 恢复模式）
# ============================================================

def _mid_exam_state(question_index, scores_count):
    """手工构造进行中的 exam_state（模拟服务重启后只剩落盘状态）"""
    return {
        "mode": "examiner",
        "target_position": "后端工程师",
        "topic": "RAG",
        "collection_name": "kb_test",
        "status": "asking",
        "question_index": question_index,
        "current_question": f"第{question_index}题：什么是RAG？",
        "current_expectations": ["考察点A"],
        "reference_points": ["要点1", "要点2"],
        "current_thread": [
            {"role": "assistant", "content": f"第{question_index}题：什么是RAG？"}
        ],
        "follow_up_count": 0,
        "max_follow_up": MAX_FOLLOW_UP,
        "scores": [{"score": 8.0, "question": f"第{i}题"} for i in range(scores_count)],
        "weak_points": [],
        "summary": None,
        "answer_cache": {},
    }


def test_finish_at_max_questions_restored_mode(fake_llm):
    # 无 checkpoint 的新 session_id：从 exam_state 落盘恢复（restored 模式）
    exam_state = _mid_exam_state(MAX_QUESTIONS, MAX_QUESTIONS - 1)
    r = _next("sess_finish", exam_state, "综合题的回答")

    assert r["error"] is None
    state = r["exam_state"]
    assert state["status"] == "finished"
    assert len(state["scores"]) == MAX_QUESTIONS
    assert state["summary"]["total_score"] == 80.0
    # 收尾后响应不再携带单题评分（同 legacy）
    assert r["evaluation"] is None
    # token = 评分 30 + 总结 12
    assert r["token_cost"]["total_tokens"] == 42
    assert [m["role"] for m in r["new_messages"]] == ["user", "assistant", "assistant"]


# ============================================================
#  ⑦ 审计 #6/#8：评分失败不落盘，原样重试不错位
# ============================================================

def test_evaluate_failure_then_retry(fake_llm):
    r0 = _start("sess_retry")
    fake_llm.eval_error = True
    r1 = _next("sess_retry", r0["exam_state"], "我的回答")

    assert r1["error"].startswith("评分失败")
    assert r1["error_status"] == 502
    # 评分失败：不记线程与得分，exam_state 为 None 无从落盘（同 legacy 的 502）
    assert r1["exam_state"] is None
    assert r1["new_messages"] == []

    # 原样重试：从 exam_state 落盘恢复，评分不重复、进度不错位（审计 #8）
    fake_llm.eval_error = False
    r2 = _next("sess_retry", r0["exam_state"], "我的回答")
    assert r2["error"] is None
    state = r2["exam_state"]
    assert len(state["scores"]) == 1
    assert state["question_index"] == 2
    assert state["status"] == "asking"


def test_follow_up_question_gen_failure_not_persisted(fake_llm):
    fake_llm.score = 5.0  # 触发追问分支
    r0 = _start("sess_qfail")
    fake_llm.questions.questions = [None]  # 追问出题失败

    r1 = _next("sess_qfail", r0["exam_state"], "有漏洞的回答")

    assert r1["error"] == "生成题目失败，请重试"
    assert r1["error_status"] == 502
    # 已写入的评分/消息均不落盘（同 legacy：raise 时会话尚未保存）
    assert r1["exam_state"] is None


# ============================================================
#  ⑧ 反作弊：复制参考资料原文 → 0 分、不调评分 LLM、状态不变
# ============================================================

def test_cheating_detected_zero_score(fake_llm):
    chunk_content = "检索增强生成是一种结合检索与生成的技术范式它先检索相关文档再交给大模型生成答案"
    chunks = [("file_a.txt", chunk_content, 0.9)]
    r0 = _start("sess_cheat", chunks)

    r1 = _next("sess_cheat", r0["exam_state"], chunk_content + "，就是这样。", chunks=chunks)

    assert r1["error"] is None
    assert r1["cheating_detected"] is True
    assert r1["evaluation"]["score"] == 0
    assert "请用自己的语言回答" in r1["evaluation"]["raw"] or "疑似直接复制" in r1["evaluation"]["raw"]
    # 反作弊命中不消耗评分 LLM，也不产生任何待落盘消息
    assert fake_llm.eval_calls == []
    assert r1["new_messages"] == []
    # 状态不变：仍是原题等待重新作答
    state = r1["exam_state"]
    assert state["status"] == "asking"
    assert state["question_index"] == 1
    assert state["scores"] == []
    assert state["current_question"] == r0["question"]


# ============================================================
#  ⑨ 评分缓存挂 state：命中时零 LLM 成本
# ============================================================

def test_evaluation_cache_hit_skips_llm(fake_llm):
    question = "第1题：什么是RAG？"
    answer = "缓存的答案"
    cached_eval = {
        "raw": "【评分】9\n【点评】缓存评价",
        "score": 9.0,
        "has_next_question": True,
        "points": [{"point": "要点1", "hit": True, "evidence": "缓存依据"}],
        "comment": "缓存评价",
        "supplement": "",
        "correction": "",
    }
    holder = {}
    set_cached_evaluation(holder, question, answer, cached_eval)

    exam_state = _mid_exam_state(1, 0)
    exam_state["current_question"] = question
    exam_state["current_thread"] = [{"role": "assistant", "content": question}]
    exam_state["answer_cache"] = holder["answer_cache"]

    r = _next("sess_cache", exam_state, answer)

    assert r["error"] is None
    assert fake_llm.eval_calls == []  # 缓存命中，未调评分 LLM
    assert r["evaluation"]["score"] == 9.0
    # 评分零成本，响应 token 只含出题一次
    assert r["token_cost"]["total_tokens"] == 15
    # 缓存仍挂在 exam_state 上随状态走
    assert r["exam_state"]["answer_cache"] == holder["answer_cache"]


# ============================================================
#  ⑩ 审计 #9：多轮后的面试进度一致性
# ============================================================

def test_progress_consistency_across_turns(fake_llm):
    r = _start("sess_prog")
    # 第一轮低分 → 追问
    fake_llm.score = 5.0
    r = _next("sess_prog", r["exam_state"], "回答一")
    assert r["exam_state"]["status"] == "follow_up"
    # 追问答得好 → 下一题
    fake_llm.score = 8.0
    r = _next("sess_prog", r["exam_state"], "回答二")

    state = r["exam_state"]
    assert state["status"] == "asking"
    assert state["question_index"] == 2
    assert state["follow_up_count"] == 0
    assert len(state["scores"]) == 2
    # 换题后线程重置为仅含当前题
    assert state["current_thread"] == [
        {"role": "assistant", "content": state["current_question"]}
    ]
    # 薄弱环节累计低分题的未命中要点
    assert state["weak_points"] == ["要点1"]
    # exam_get 端点读取的字段全部就位
    assert state["summary"] is None
    assert state["current_expectations"] == ["考察点A"]


# ============================================================
#  ⑪ EXAM_ENGINE 接线：路由分派 + 响应 schema 不变 + 默认 legacy
# ============================================================

class _FakeSession:
    """routes 接缝测试用的会话桩：属性与 add_message 签名同 core.session.Session"""

    def __init__(self, session_id, exam_state=None):
        self.session_id = session_id
        self.mode = "examiner"
        self.kb_id = "kb_test"
        self.title = "模拟面试"
        self.persona = "default"
        self.exam_state = exam_state
        self.messages = []
        self.total_tokens = 0

    def add_message(self, role, content, token_cost=None, sources=None, reasoning_steps=None):
        msg = {"role": role, "content": content}
        if token_cost:
            msg["token_cost"] = token_cost
            self.total_tokens += token_cost.get("total_tokens", 0)
        self.messages.append(msg)


class _FakeSessionManager:
    def __init__(self, session):
        self.session = session
        self.saved = 0

    def get_session(self, session_id):
        return self.session if session_id == self.session.session_id else None

    def save_session(self, session):
        self.saved += 1


class _FakeRetriever:
    def __init__(self, collection_name):
        self.collection_name = collection_name

    def hybrid_search(self, query, top_k=10):
        return list(FAKE_CHUNKS)


def _wire_route(monkeypatch, engine, session):
    """打桩路由依赖：引擎开关、会话管理器、检索器，返回会话管理器桩"""
    monkeypatch.setattr(settings, "exam_engine", engine)
    monkeypatch.setattr(retrieval_mod, "AdvancedRetriever", _FakeRetriever)
    sm = _FakeSessionManager(session)
    monkeypatch.setattr(routes, "session_manager", sm)
    return sm


def test_route_exam_start_graph_engine(fake_llm, monkeypatch):
    session = _FakeSession("sess_route_start")
    sm = _wire_route(monkeypatch, "graph", session)

    resp = asyncio.run(routes.exam_start(routes.ExamStartRequest(
        session_id="sess_route_start", target_position="后端工程师", topic="RAG")))

    # 响应 schema 与 legacy 完全一致，前端无感
    assert set(resp.keys()) == {
        "session_id", "status", "question_index", "current_question",
        "current_expectations", "follow_up_count", "scores", "token_cost",
    }
    assert resp["status"] == "asking" and resp["question_index"] == 1
    assert session.exam_state["current_question"] == resp["current_question"]
    # token 只经 _append_exam_messages 累加一次（审计 #5）
    assert session.total_tokens == 15 == resp["token_cost"]["total_tokens"]
    assert sm.saved == 1


def test_route_exam_next_graph_engine(fake_llm, monkeypatch):
    # 先经图驱动开好局，session.exam_state 与 checkpoint 就位
    r0 = _start("sess_route_next")
    session = _FakeSession("sess_route_next", exam_state=r0["exam_state"])
    sm = _wire_route(monkeypatch, "graph", session)
    session.add_message("assistant", r0["question"], token_cost=dict(Q_COST))

    resp = asyncio.run(routes.exam_next(
        "sess_route_next", routes.ExamNextRequest(answer="RAG 是检索增强生成")))

    # 响应 schema 与 legacy 完全一致
    assert set(resp.keys()) == {
        "session_id", "status", "question_index", "current_question",
        "current_expectations", "reference_points", "follow_up_count",
        "evaluation", "summary", "scores", "weak_points", "token_cost",
    }
    assert resp["status"] == "asking" and resp["question_index"] == 2
    assert resp["evaluation"]["score"] == 8.0
    assert resp["summary"] is None
    # 消息顺序：题目(开局) + 用户作答 + 评分 + 新题
    assert [m["role"] for m in session.messages] == [
        "assistant", "user", "assistant", "assistant",
    ]
    # 开局 15 + 本轮（评分 30 + 出题 15），全程无重复计数
    assert session.total_tokens == 15 + 45
    assert resp["token_cost"]["total_tokens"] == 45
    assert sm.saved == 1


def test_route_exam_next_end_graph_engine(fake_llm, monkeypatch):
    r0 = _start("sess_route_end")
    session = _FakeSession("sess_route_end", exam_state=r0["exam_state"])
    _wire_route(monkeypatch, "graph", session)

    resp = asyncio.run(routes.exam_next(
        "sess_route_end", routes.ExamNextRequest(answer="", action="end")))

    assert resp["status"] == "finished"
    assert resp["evaluation"] is None
    assert resp["summary"]["total_score"] == 80.0
    assert fake_llm.eval_calls == []  # end 不评分（审计 #7）
    assert session.total_tokens == 12  # 只计总结一次


def test_route_exam_start_legacy_engine_unchanged(monkeypatch):
    """legacy 路径一行未改：开关默认/显式 legacy 时仍走 routes 手写状态机"""
    monkeypatch.setattr(
        routes, "generate_question",
        lambda **kwargs: ("【概念题】什么是RAG？", ["考察点A"], ["要点1"], Q_COST),
    )
    session = _FakeSession("sess_route_legacy")
    sm = _wire_route(monkeypatch, "legacy", session)

    resp = asyncio.run(routes.exam_start(routes.ExamStartRequest(
        session_id="sess_route_legacy", target_position="后端工程师", topic="RAG")))

    assert resp["current_question"] == "【概念题】什么是RAG？"
    assert resp["status"] == "asking"
    assert session.exam_state["mode"] == "examiner"
    assert session.total_tokens == 15
    assert sm.saved == 1


def test_exam_engine_default_is_legacy():
    """开关默认值必须是 legacy：不加 EXAM_ENGINE 环境变量时行为完全不变"""
    assert Settings.model_fields["exam_engine"].default == "legacy"
