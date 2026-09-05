"""
Exam Graph：考官模式手写状态机的 LangGraph StateGraph 移植（设计文档 4.2）

节点与 routes.py 原 if/else 状态机的语义对照：
- start_exam        → 入口分发：正常开局走 generate_question；restored（服务重启
                      后无 checkpoint，从 exam_state 落盘恢复）直接进 wait_answer
- generate_question → 原 exam_start / exam_next 里的 generate_question 调用
                      （出题与追问同一节点，靠 follow_up_count 区分）；LLM 故障写
                      error 状态直接收尾，报错文案绝不落盘为题目（审计 #6）
- wait_answer       → 原"返回题目、等前端下一次请求"：interrupt 挂起，用户作答
                      经 Command(resume=...) 恢复；action 与 answer 分离传入
- classify_action   → 原 exam_next 顶部的 action 分支：answer / give_up / end /
                      空答案。action 由分支节点判断，不混在 answer 文本里，
                      结束面试不再被当答案评分（审计 #7）
- evaluate_answer   → 原 exam_next 的评分段：反作弊（detect_cheating）→ 评分缓存
                      （answer_cache 挂 state）→ evaluate_answer → 记录得分/线程 →
                      决定 追问 / 下一题 / 完成 分支（判定条件与 legacy 逐字一致）
- generate_summary  → 原两处 generate_summary 收尾；失败降级为占位文案（同 legacy）

token 记账与消息累加（审计 #5 双计消除）：
- 每次 LLM 调用的 TokenCost 只在节点内经 _accumulate 累加进 state["token_cost"]，
  作为本次请求的响应 token_cost，入口唯一；
- 会话消息统一由节点写入 state["new_messages"]，路由层只经 _append_exam_messages
  一个出口消费。wait_answer 在每次恢复时重置这两个"本请求"字段，保证跨请求
  不累旧账，同一次 LLM 调用不会被图与路由各记一次。

checkpointer：interrupt/resume 必须挂 checkpointer，阶段 3 用进程内
InMemorySaver；exam_state 仍由路由层落盘 sessions 表，无 checkpoint（如服务
重启）时据此走 restored 模式恢复，刷新/重启行为与 legacy 一致。
阶段 4 迁移 AsyncPostgresSaver（设计文档 4.3），thread_id 即 sess_xxx。
"""
from __future__ import annotations

from typing import Optional

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from app.services.examiner import (
    MAX_FOLLOW_UP,
    MAX_QUESTIONS,
    detect_cheating,
    evaluate_answer,
    generate_question,
    generate_summary,
    get_cached_evaluation,
    set_cached_evaluation,
)
from app.services.token_tracker import TokenCost
from app.graphs.state import ExamState


def _accumulate(cost: Optional[TokenCost], extra: TokenCost) -> TokenCost:
    """在已有 TokenCost 上累加一次 LLM 调用的消耗（统一记账 wrapper，审计 #5）"""
    total = cost or TokenCost()
    total["prompt_tokens"] += extra.prompt
    total["completion_tokens"] += extra.completion
    total["total_tokens"] += extra.total
    return total


# ============================================================
#  节点
# ============================================================

def _start_exam_node(state: ExamState) -> dict:
    """入口分发：正常开局先生成第一题；restored 恢复已有题目，直接等作答"""
    if state.get("restored"):
        return {"route": "wait"}
    return {"route": "question"}


def _generate_question_node(state: ExamState) -> dict:
    """出题/追问：LLM 故障写 error 状态收尾，报错文案绝不落盘为题目（审计 #6）"""
    follow_up_count = state.get("follow_up_count", 0)
    thread = state.get("current_thread", [])
    question, expectations, reference_points, cost = generate_question(
        target_position=state["target_position"],
        topic=state["topic"],
        context_chunks=state.get("context_chunks", []),
        thread=thread,
        question_index=state["question_index"],
        follow_up_count=follow_up_count,
    )
    token_cost = _accumulate(state.get("token_cost"), cost)
    if not question:
        return {
            "route": "error",
            "error": "生成题目失败，请重试",
            "error_status": 502,
            "token_cost": token_cost,
        }
    return {
        "route": "wait",
        "current_question": question,
        "current_expectations": expectations,
        "reference_points": reference_points,
        "current_thread": thread + [{"role": "assistant", "content": question}],
        "token_cost": token_cost,
        "new_messages": state.get("new_messages", []) + [
            {"role": "assistant", "content": question, "token_cost": dict(cost)}
        ],
        "error": None,
    }


def _wait_answer_node(state: ExamState) -> dict:
    """等待候选人作答：interrupt 挂起，Command(resume=...) 携作答恢复。

    每次恢复都是一次新 HTTP 请求的开始：在此重置 token_cost / new_messages /
    evaluation 等"本请求"字段，保证跨请求不累旧账（审计 #5 双计消除的前提）。
    restored 模式（重启后无 checkpoint）不挂起，直接消费 pending_input。
    """
    pending = state.get("pending_input")
    if pending is not None:
        payload = pending
    else:
        payload = interrupt({
            "type": "wait_answer",
            "question_index": state.get("question_index"),
            "current_question": state.get("current_question"),
            "follow_up_count": state.get("follow_up_count", 0),
        })
    return {
        "action": payload.get("action"),
        "answer": payload.get("answer") or "",
        # 恢复时路由层传入本次检索结果，覆盖上一请求留下的上下文
        "context_chunks": payload.get("context_chunks") or state.get("context_chunks", []),
        "pending_input": None,
        "new_messages": [],
        "token_cost": TokenCost(),
        "evaluation": None,
        "cheating_detected": False,
        "error": None,
    }


def _classify_action_node(state: ExamState) -> dict:
    """作答分类：action 由分支节点判断，不混在 answer 文本里（审计 #7）"""
    action = state.get("action")
    if action == "end":
        # 显式结束：跳过反作弊与评分，直接总结收尾
        return {"route": "summary"}
    if action == "give_up":
        # 放弃作答：按 legacy 语义记为"不知道"正常评分
        return {"route": "answer", "answer": "不知道"}
    answer = (state.get("answer") or "").strip()
    if not answer:
        return {"route": "invalid", "error": "回答不能为空", "error_status": 400}
    return {"route": "answer", "answer": answer}


def _evaluate_answer_node(state: ExamState) -> dict:
    """评分：反作弊 → 缓存 → LLM 评分 → 记录得分/线程 → 决定追问/下一题/完成"""
    answer = state["answer"]
    question = state["current_question"]
    thread = state.get("current_thread", [])
    chunks = state.get("context_chunks", [])

    # 反作弊：与 legacy 一致——0 分评价、不记线程/得分、回到 wait_answer 等重新作答
    previous_answers = [m["content"] for m in thread if m["role"] == "user"]
    is_cheating, cheat_reason = detect_cheating(answer, previous_answers, chunks)
    if is_cheating:
        return {
            "route": "wait",
            "cheating_detected": True,
            "evaluation": {
                "raw": f"【评分】0\n【点评】{cheat_reason}\n【补充】请独立作答，不要直接复制资料。\n【纠正】无",
                "score": 0,
                "points": [
                    {"point": p, "hit": False, "evidence": "反作弊判定未作答"}
                    for p in state.get("reference_points", [])
                ],
            },
        }

    # 评分缓存：answer_cache 挂 state，命中即省一次 LLM 调用（仅缓存成功结果）
    answer_cache = dict(state.get("answer_cache") or {})
    cache_holder = {"answer_cache": answer_cache}
    eval_result = get_cached_evaluation(cache_holder, question, answer)
    eval_cost = TokenCost()
    if not eval_result:
        eval_result, eval_cost = evaluate_answer(
            target_position=state["target_position"],
            topic=state["topic"],
            question=question,
            answer=answer,
            reference_points=state.get("reference_points", []),
            context_chunks=chunks,
            thread=thread,
            follow_up_count=state.get("follow_up_count", 0),
            question_index=state["question_index"],
        )
        if "error" not in eval_result:
            set_cached_evaluation(cache_holder, question, answer, eval_result)

    # 评分失败：不记线程与得分，error 收尾由路由层转 502，前端可原样重试（审计 #8）
    if "error" in eval_result:
        return {
            "route": "error",
            "error": f"评分失败：{eval_result['error']}",
            "error_status": 502,
        }

    token_cost = _accumulate(state.get("token_cost"), eval_cost)
    missed_points = [p["point"] for p in eval_result.get("points", []) if not p.get("hit")]
    new_thread = thread + [
        {"role": "user", "content": answer},
        {"role": "assistant", "content": eval_result["raw"]},
    ]
    updates = {
        "evaluation": eval_result,
        "cheating_detected": False,
        "answer_cache": answer_cache,
        "scores": state.get("scores", []) + [{
            "question": question,
            "answer": answer,
            "score": eval_result["score"],
            "feedback": eval_result["raw"],
            "points": eval_result.get("points", []),
            "missed_points": missed_points,
        }],
        "weak_points": (state.get("weak_points", []) + missed_points)[-20:],
        "token_cost": token_cost,
        "new_messages": state.get("new_messages", []) + [
            {"role": "user", "content": answer},
            {"role": "assistant", "content": eval_result["raw"], "token_cost": dict(eval_cost)},
        ],
    }

    # 分支：完成 / 追问 / 下一题（判定条件与 legacy 逐字一致）
    question_index = state["question_index"]
    follow_up_count = state.get("follow_up_count", 0)
    if question_index >= MAX_QUESTIONS:
        updates["route"] = "summary"
        updates["current_thread"] = new_thread
    elif eval_result["score"] < 7 and follow_up_count < state.get("max_follow_up", MAX_FOLLOW_UP):
        updates["route"] = "question"
        updates["status"] = "follow_up"
        updates["follow_up_count"] = follow_up_count + 1
        updates["current_thread"] = new_thread
    else:
        updates["route"] = "question"
        updates["status"] = "asking"
        updates["question_index"] = question_index + 1
        updates["follow_up_count"] = 0
        updates["current_thread"] = []
    return updates


def _generate_summary_node(state: ExamState) -> dict:
    """面试总结：模型失败时降级为占位文案（与 legacy 一致），保证流程闭环"""
    summary, summary_cost = generate_summary(
        target_position=state["target_position"],
        topic=state["topic"],
        scores=state.get("scores", []),
        weak_points=state.get("weak_points", []),
    )
    token_cost = _accumulate(state.get("token_cost"), summary_cost)
    if "error" in summary:
        # 总结生成失败时降级为占位文案，避免面试收尾直接崩溃（同 legacy）
        summary = {"raw": f"面试总结生成失败：{summary['error']}", "total_score": 0}
    return {
        "status": "finished",
        "summary": summary,
        # 收尾后响应不再携带单题评分（同 legacy 的 response_evaluation=None）
        "evaluation": None,
        "token_cost": token_cost,
        "new_messages": state.get("new_messages", []) + [
            {"role": "assistant", "content": summary["raw"], "token_cost": dict(summary_cost)}
        ],
    }


# ============================================================
#  图构建与入口
# ============================================================

def build_exam_graph() -> StateGraph:
    """构建 Exam StateGraph（未编译，调用方 .compile() 后使用）"""
    graph = StateGraph(ExamState)

    graph.add_node("start_exam", _start_exam_node)
    graph.add_node("generate_question", _generate_question_node)
    graph.add_node("wait_answer", _wait_answer_node)
    graph.add_node("classify_action", _classify_action_node)
    graph.add_node("evaluate_answer", _evaluate_answer_node)
    graph.add_node("generate_summary", _generate_summary_node)

    graph.set_entry_point("start_exam")
    graph.add_conditional_edges(
        "start_exam",
        lambda s: s["route"],
        {"question": "generate_question", "wait": "wait_answer"},
    )
    # 出题失败 → error 状态直接收尾，不出现在 wait_answer（审计 #6）
    graph.add_conditional_edges(
        "generate_question",
        lambda s: END if s.get("error") else "wait_answer",
    )
    graph.add_edge("wait_answer", "classify_action")
    graph.add_conditional_edges(
        "classify_action",
        lambda s: s["route"],
        {"answer": "evaluate_answer", "summary": "generate_summary", "invalid": END},
    )
    graph.add_conditional_edges(
        "evaluate_answer",
        lambda s: s["route"],
        {
            "wait": "wait_answer",  # 反作弊命中：等候选人重新作答
            "question": "generate_question",  # 追问 / 下一题（环）
            "summary": "generate_summary",  # 已答满 MAX_QUESTIONS：完成
            "error": END,
        },
    )
    graph.add_edge("generate_summary", END)
    return graph


_compiled_graph = None


def _get_compiled_graph():
    """编译后的图单例：注入的 LLM 函数走模块级引用，图本身可跨请求复用。

    interrupt/resume 必须挂 checkpointer：阶段 3 用进程内 InMemorySaver；
    阶段 4 迁移 AsyncPostgresSaver（设计文档 4.3），thread_id 即 sess_xxx。
    """
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_exam_graph().compile(checkpointer=InMemorySaver())
    return _compiled_graph


# exam_state 落盘字段（与 legacy routes 写入的键集合完全一致）
_EXAM_STATE_KEYS = (
    "status", "question_index", "current_question", "current_expectations",
    "reference_points", "current_thread", "follow_up_count", "max_follow_up",
    "scores", "weak_points", "summary", "answer_cache",
)


def _extract_exam_state(state: ExamState) -> dict:
    """从图状态提取 exam_state 落盘结构（键集合与 legacy 完全一致）"""
    exam_state = {
        "mode": "examiner",
        "target_position": state.get("target_position", ""),
        "topic": state.get("topic", ""),
        "collection_name": state.get("collection_name", ""),
    }
    for key in _EXAM_STATE_KEYS:
        exam_state[key] = state.get(key)
    return exam_state


def _initial_state(
    session_id: str,
    target_position: str,
    topic: str,
    collection_name: str,
    context_chunks: list,
) -> ExamState:
    """/exam/start 的初始状态：对应 legacy 中 exam_start 写入的 exam_state 初值"""
    return {
        "session_id": session_id,
        "target_position": target_position,
        "topic": topic,
        "collection_name": collection_name,
        "context_chunks": list(context_chunks or []),
        "restored": False,
        "status": "asking",
        "question_index": 1,
        "follow_up_count": 0,
        "max_follow_up": MAX_FOLLOW_UP,
        "current_question": None,
        "current_expectations": [],
        "reference_points": [],
        "current_thread": [],
        "scores": [],
        "weak_points": [],
        "summary": None,
        "answer_cache": {},
        "action": None,
        "answer": "",
        "evaluation": None,
        "cheating_detected": False,
        "new_messages": [],
        "token_cost": TokenCost(),
        "error": None,
        "error_status": 502,
        "route": "",
    }


def exam_graph_start(
    session_id: str,
    target_position: str,
    topic: str,
    collection_name: str,
    context_chunks: list,
) -> dict:
    """生成第一题（对应 POST /exam/start）。

    返回 {"error", "error_status", "exam_state", "question", "expectations",
    "reference_points", "new_messages", "token_cost"}；error 非空时 exam_state
    为 None，调用方不得落盘（审计 #6）。
    """
    config = {"configurable": {"thread_id": session_id}}
    final = _get_compiled_graph().invoke(
        _initial_state(session_id, target_position, topic, collection_name, context_chunks),
        config,
    )
    if final.get("error"):
        return {
            "error": final["error"],
            "error_status": final.get("error_status", 502),
            "exam_state": None,
            "new_messages": [],
            "token_cost": final.get("token_cost") or TokenCost(),
        }
    return {
        "error": None,
        "exam_state": _extract_exam_state(final),
        "question": final.get("current_question"),
        "expectations": final.get("current_expectations", []),
        "reference_points": final.get("reference_points", []),
        "new_messages": final.get("new_messages", []),
        "token_cost": final.get("token_cost") or TokenCost(),
    }


def exam_graph_next(
    session_id: str,
    exam_state: dict,
    action: Optional[str],
    answer: str,
    context_chunks: list,
) -> dict:
    """提交作答（对应 POST /exam/{session_id}/next）。

    有挂起的 interrupt 时走 Command(resume=...) 恢复；否则（服务重启或上一请求
    以错误收尾）从 exam_state 落盘恢复整段状态后重跑（restored 模式）。
    返回 {"error", "error_status", "cheating_detected", "exam_state",
    "evaluation", "summary", "new_messages", "token_cost"}；error 非空时
    exam_state 为 None，调用方不得落盘（与 legacy 的 502 语义一致）。
    """
    graph = _get_compiled_graph()
    config = {"configurable": {"thread_id": session_id}}
    resume_payload = {
        "action": action,
        "answer": answer,
        "context_chunks": list(context_chunks or []),
    }

    snapshot = graph.get_state(config)
    if snapshot and snapshot.next:
        # 图正挂在 wait_answer 的 interrupt 上：Command(resume=...) 恢复
        final = graph.invoke(Command(resume=resume_payload), config)
    else:
        # 无挂起中断：从 exam_state 落盘恢复，start_exam 直达 wait_answer
        restored = {
            "session_id": session_id,
            "target_position": exam_state.get("target_position", ""),
            "topic": exam_state.get("topic", ""),
            "collection_name": exam_state.get("collection_name", ""),
            "restored": True,
            "pending_input": resume_payload,
            "status": exam_state.get("status", "asking"),
            "question_index": exam_state.get("question_index", 1),
            "follow_up_count": exam_state.get("follow_up_count", 0),
            "max_follow_up": exam_state.get("max_follow_up", MAX_FOLLOW_UP),
            "current_question": exam_state.get("current_question"),
            "current_expectations": exam_state.get("current_expectations", []),
            "reference_points": exam_state.get("reference_points", []),
            "current_thread": list(exam_state.get("current_thread") or []),
            "scores": list(exam_state.get("scores") or []),
            "weak_points": list(exam_state.get("weak_points") or []),
            "summary": exam_state.get("summary"),
            "answer_cache": dict(exam_state.get("answer_cache") or {}),
            "new_messages": [],
            "token_cost": TokenCost(),
            "error": None,
            "error_status": 502,
            "route": "",
        }
        final = graph.invoke(restored, config)

    if final.get("error"):
        return {
            "error": final["error"],
            "error_status": final.get("error_status", 502),
            "cheating_detected": False,
            "exam_state": None,
            "evaluation": None,
            "summary": None,
            "new_messages": [],
            "token_cost": final.get("token_cost") or TokenCost(),
        }
    return {
        "error": None,
        "cheating_detected": bool(final.get("cheating_detected")),
        "exam_state": _extract_exam_state(final),
        "evaluation": final.get("evaluation"),
        "summary": final.get("summary"),
        "new_messages": final.get("new_messages", []),
        "token_cost": final.get("token_cost") or TokenCost(),
    }
