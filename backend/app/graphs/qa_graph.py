"""
QA Graph：agent.py 手写 ReAct 循环的 LangGraph StateGraph 移植（设计文档 4.1）

节点与原循环的语义对照：
- guard_input        → harness 输入校验（qa.py 调用前已有一道，图内按设计保留）
- retrieve_parallel  → 原 retrieve_func（hybrid_search）的双路召回环节：
                       vector ∥ bm25 并行执行，单路异常只降级为空分支、不拖死
                       另一路（审计 #3）
- merge_rrf          → 原 rrf_fuse + rerank 环节：按 chunk_id RRF 融合（修复
                       content[:300] 键碰撞）后接余弦精排，并就地记录
                       knowledge_search 推理步骤
- judge_sufficiency  → 原 is_sufficient 判定 + 循环检测 + 轮次上限三段 break
                       判断，充分性只信 score_type="cosine" 的分数（审计 #4）
- rewrite_query      → 原 Thought+Action 改写环节，直接复用 agent.rewrite_query
                       （prompt 与 token 记账逐字一致），含改写失败即终止判断
- generate           → 原循环尾部：跨轮结果去重取 top 5 → generate_func 生成 →
                       sources（src_N）与 confidence 分级组装

阶段 2 的 QA 是无状态单次问答，compile() 不挂 checkpointer（属阶段 3/4）。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

from langgraph.graph import END, StateGraph

from app.services.agent import (
    MAX_ITERATIONS,
    detect_loop,
    is_sufficient,
    rewrite_query,
)
from app.services.harness import fallback_error, guard_input
from app.services.retrieval import (
    SCORE_TYPE_COSINE,
    SUFFICIENCY_THRESHOLD,
    RetrievalResult,
    bm25_branch,
    merge_rrf,
    rerank_cosine,
    vector_branch,
)
from app.services.token_tracker import TokenCost
from app.graphs.state import QAState


def _as_tuples(results: list[RetrievalResult]) -> list[tuple]:
    """RetrievalResult 转 (file_name, content, score) 三元组，复用 agent 的判定函数"""
    return [(r.file_name, r.content, r.score) for r in results]


def _cosine_tuples(results: list[RetrievalResult]) -> list[tuple]:
    """只保留余弦分的三元组：充分性判定不采信 rrf / bm25 分（审计 #4）"""
    return [
        (r.file_name, r.content, r.score)
        for r in results
        if r.score_type == SCORE_TYPE_COSINE
    ]


def _accumulate(cost: Optional[TokenCost], extra: TokenCost) -> TokenCost:
    """在已有 TokenCost 上累加一次 LLM 调用的消耗（同 agent.py 的记账方式）"""
    total = cost or TokenCost()
    total["prompt_tokens"] += extra.prompt
    total["completion_tokens"] += extra.completion
    total["total_tokens"] += extra.total
    return total


# ============================================================
#  节点
# ============================================================

def _guard_input_node(state: QAState) -> dict:
    """harness 输入校验，失败则直接短路到 END 返回兜底回答"""
    error = guard_input(state["question"])
    if error:
        return {
            "input_error": error,
            "answer": fallback_error(error),
            "sources": [],
        }
    return {"input_error": None}


def _retrieve_parallel_node(state: QAState) -> dict:
    """并行双路召回：vector ∥ bm25，单路异常只降级为空分支（审计 #3）"""
    query = state["queries"][-1]
    collection_name = state["collection_name"]

    def _safe_branch(branch_func: Callable) -> list[RetrievalResult]:
        try:
            return branch_func(query, collection_name, top_k=20)
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=2) as pool:
        vector_future = pool.submit(_safe_branch, vector_branch)
        bm25_future = pool.submit(_safe_branch, bm25_branch)
        vector_results = vector_future.result()
        bm25_results = bm25_future.result()

    return {"branches": [vector_results, bm25_results]}


def _merge_rrf_node(state: QAState) -> dict:
    """RRF 融合（chunk_id 键）+ 余弦精排，并记录本轮 knowledge_search 步骤"""
    query = state["queries"][-1]
    merged = merge_rrf(state.get("branches", []), top_k=15)
    reranked = rerank_cosine(query, merged, top_k=state["top_k"] * 2)

    # 与 agent.py 一致：就地记录步骤，confidence 留空待图执行完毕后统一回填
    step = {
        "step": len(state.get("reasoning_steps", [])) + 1,
        "action": "knowledge_search",
        "query": query,
        "hits": len(reranked),
        "top_score": round(max((r.score for r in reranked), default=0.0), 4),
        "confidence": "",
    }
    return {
        "results": reranked,
        "all_results": reranked,  # operator.add 跨轮累加
        "iterations": state.get("iterations", 0) + 1,
        "reasoning_steps": [step],
    }


def _judge_sufficiency_node(state: QAState) -> dict:
    """充分性判定 + 循环检测 + 轮次上限，对应原循环的三段 break 判断"""
    results = state.get("results", [])
    iterations = state.get("iterations", 0)

    # 充分性只信余弦分（阈值 0.75 与原语义一致，复用 agent.is_sufficient）
    if is_sufficient(_cosine_tuples(results)):
        return {"route": "generate"}

    # 循环检测：第二轮起，连续两次检索结果 Jaccard > 0.8 则终止
    if iterations > 1 and detect_loop(
        _as_tuples(state.get("prev_results", [])),
        _as_tuples(results),
    ):
        return {"route": "generate"}

    # 轮次上限（原 step == MAX_ITERATIONS - 1 的 break）
    if iterations >= MAX_ITERATIONS:
        return {"route": "generate"}

    return {"route": "rewrite"}


def _rewrite_query_node(state: QAState) -> dict:
    """改写查询：复用 agent.rewrite_query（prompt 与 token 记账逐字一致）"""
    current_query = state["queries"][-1]
    rewritten, rewrite_cost = rewrite_query(
        state["question"],
        _as_tuples(state.get("results", [])),
        state.get("history", []),
    )
    token_cost = _accumulate(state.get("token_cost"), rewrite_cost)

    # 改写失败或与当前查询相同：直接转生成，避免白跑一轮重复检索
    if rewritten == current_query:
        return {"route": "generate", "token_cost": token_cost}

    step = {
        "step": len(state.get("reasoning_steps", [])) + 1,
        "action": "query_rewrite",
        "query": rewritten,
        "hits": 0,
        "top_score": 0.0,
        "confidence": "medium",
    }
    return {
        "route": "retrieve",
        "queries": [rewritten],  # operator.add 追加到改写链
        "rewritten_query": rewritten,
        "prev_results": state.get("results", []),
        "reasoning_steps": [step],
        "token_cost": token_cost,
    }


def _generate_node(state: QAState) -> dict:
    """生成答案：跨轮结果去重取 top 5 → generate_func → sources/confidence 组装"""
    # 去重并选择 top 结果（键 content[:200]，按分数降序，与 agent.py 逐字一致）
    seen_contents: set = set()
    unique_results: list[RetrievalResult] = []
    for r in sorted(state.get("all_results", []), key=lambda x: x.score, reverse=True):
        key = r.content[:200]
        if key not in seen_contents:
            seen_contents.add(key)
            unique_results.append(r)
        if len(unique_results) >= 5:
            break

    context_chunks = [r.content for r in unique_results]
    answer, gen_cost = state["generate_func"](
        state["question"], context_chunks, state.get("history", [])
    )
    token_cost = _accumulate(state.get("token_cost"), gen_cost)

    sources = [
        {
            "id": f"src_{idx + 1}",
            "content": r.content,
            "similarity": round(r.score, 4),
            "file_name": r.file_name,
        }
        for idx, r in enumerate(unique_results)
    ]

    max_source_score = unique_results[0].score if unique_results else 0.0
    confidence = (
        "high" if max_source_score >= SUFFICIENCY_THRESHOLD + 0.1
        else "medium" if max_source_score >= SUFFICIENCY_THRESHOLD
        else "low"
    )

    return {
        "answer": answer,
        "sources": sources,
        "token_cost": token_cost,
        "confidence": confidence,
    }


# ============================================================
#  图构建与入口
# ============================================================

def build_qa_graph() -> StateGraph:
    """构建 QA StateGraph（未编译，调用方 .compile() 后使用）"""
    graph = StateGraph(QAState)

    graph.add_node("guard_input", _guard_input_node)
    graph.add_node("retrieve_parallel", _retrieve_parallel_node)
    graph.add_node("merge_rrf", _merge_rrf_node)
    graph.add_node("judge_sufficiency", _judge_sufficiency_node)
    graph.add_node("rewrite_query", _rewrite_query_node)
    graph.add_node("generate", _generate_node)

    graph.set_entry_point("guard_input")
    graph.add_conditional_edges(
        "guard_input",
        lambda s: END if s.get("input_error") else "retrieve_parallel",
    )
    graph.add_edge("retrieve_parallel", "merge_rrf")
    graph.add_edge("merge_rrf", "judge_sufficiency")
    graph.add_conditional_edges(
        "judge_sufficiency",
        lambda s: s["route"],
        {"generate": "generate", "rewrite": "rewrite_query"},
    )
    graph.add_conditional_edges(
        "rewrite_query",
        lambda s: s["route"],
        {"retrieve": "retrieve_parallel", "generate": "generate"},
    )
    graph.add_edge("generate", END)
    return graph


_compiled_graph = None


def _get_compiled_graph():
    """编译后的图单例：generate_func 走 state 注入，图本身可跨请求复用。

    不挂 checkpointer——阶段 2 的 QA 是无状态单次问答，checkpoint 属阶段 3/4。
    """
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_qa_graph().compile()
    return _compiled_graph


def qa_graph_query(
    question: str,
    collection_name: str = "knowledge_chunks",
    history: Optional[list] = None,
    top_k: int = 5,
    generate_func: Optional[Callable] = None,
    system_prompt: str = "",
) -> dict:
    """QA Graph 入口，返回 dict 的键与 agent_query 完全一致：
    answer / sources / token_cost / iterations / rewritten_query /
    reasoning_steps / queries。

    generate_func 注入方式与 agent_query 相同：(question, context_chunks,
    history) -> (answer, TokenCost)；system_prompt 由 qa.py 在 generate_func
    闭包内捕获，图内不再单独使用。
    """
    if generate_func is None:
        raise ValueError("qa_graph_query 必须注入 generate_func")

    initial_state: QAState = {
        "question": question,
        "collection_name": collection_name,
        "history": history or [],
        "top_k": top_k,
        "generate_func": generate_func,
        "queries": [question],
        "branches": [],
        "results": [],
        "prev_results": [],
        "all_results": [],
        "reasoning_steps": [],
        "iterations": 0,
        "rewritten_query": None,
        "route": "",
        "input_error": None,
        "answer": "",
        "sources": [],
        "token_cost": TokenCost(),
        "confidence": "low",
    }

    final_state = _get_compiled_graph().invoke(initial_state)

    # 回填各检索步骤的置信度（执行检索时还无法预知最终结果，同 agent.py）
    confidence = final_state.get("confidence", "low")
    reasoning_steps = final_state.get("reasoning_steps", [])
    for rs in reasoning_steps:
        if rs["action"] == "knowledge_search":
            rs["confidence"] = confidence

    return {
        "answer": final_state.get("answer", ""),
        "sources": final_state.get("sources", []),
        "token_cost": final_state.get("token_cost") or TokenCost(),
        "iterations": final_state.get("iterations", 0),
        "rewritten_query": final_state.get("rewritten_query"),
        "reasoning_steps": reasoning_steps,
        "queries": final_state.get("queries", [question]),
    }
