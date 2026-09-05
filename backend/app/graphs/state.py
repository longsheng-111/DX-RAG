"""QA / Exam Graph 状态定义（设计文档 4.1 / 4.2）"""
from __future__ import annotations

import operator
from typing import Annotated, Callable, Optional, TypedDict

from app.services.retrieval import RetrievalResult
from app.services.token_tracker import TokenCost


class QAState(TypedDict, total=False):
    """QA Graph 状态（total=False：节点只返回需要更新的键）。

    queries / reasoning_steps / all_results 用 operator.add 累加，
    天然记录改写链与跨轮检索结果；其余字段为覆盖语义。
    """

    # ---- 输入（图入口一次性写入）----
    question: str
    collection_name: str
    history: list[dict]
    top_k: int
    # qa.py 注入的生成函数，签名同 agent_query：(question, context_chunks, history)
    # -> (answer, TokenCost)；prompt 构建与 token 记账全部在闭包内完成
    generate_func: Callable

    # ---- 循环过程 ----
    queries: Annotated[list[str], operator.add]  # 改写链，首轮为原始问题
    branches: list[list[RetrievalResult]]  # retrieve_parallel 的双路原始结果
    results: list[RetrievalResult]  # 本轮 融合+精排 后的结果
    prev_results: list[RetrievalResult]  # 上一轮结果（Jaccard 循环检测用）
    all_results: Annotated[list[RetrievalResult], operator.add]  # 跨轮累加
    reasoning_steps: Annotated[list[dict], operator.add]  # 审计 #2：随状态透传到底
    iterations: int
    rewritten_query: Optional[str]
    route: str  # 条件边路由标记："retrieve" / "rewrite" / "generate"
    input_error: Optional[str]

    # ---- 输出 ----
    answer: str
    sources: list[dict]
    token_cost: TokenCost
    confidence: str


class ExamState(TypedDict, total=False):
    """Exam Graph 状态（设计文档 4.2，total=False：节点只返回需要更新的键）。

    所有字段必须是 JSON 可序列化类型——图挂了 checkpointer（InMemorySaver，
    经 JsonPlusSerializer 落 checkpoint），因此 LLM 调用函数不挂 state，
    节点内直接引用模块级 examiner 函数，测试用 monkeypatch 打桩
    （与 test_qa_graph 的注入风格一致）。
    """

    # ---- 输入（图入口 / restored 恢复时写入）----
    session_id: str
    target_position: str
    topic: str
    collection_name: str
    context_chunks: list  # (file_name, content, score) 三元组，由路由层检索注入
    restored: bool  # True=从 exam_state 落盘恢复（无 checkpoint），入口跳过出题
    pending_input: Optional[dict]  # restored 模式注入的作答 {"action","answer","context_chunks"}

    # ---- 面试进度（与 exam_state 落盘字段一一对应）----
    status: str  # asking / follow_up / finished
    question_index: int
    follow_up_count: int
    max_follow_up: int
    current_question: Optional[str]
    current_expectations: list[str]
    reference_points: list[str]
    current_thread: list[dict]
    scores: list[dict]
    weak_points: list[str]
    summary: Optional[dict]
    answer_cache: dict  # 评分缓存挂 state（原 get_cached_evaluation 的 dict 入参）

    # ---- 本轮作答 ----
    action: Optional[str]  # None=作答 / give_up / end，与 answer 文本彻底分离（审计 #7）
    answer: str
    evaluation: Optional[dict]
    cheating_detected: bool

    # ---- 输出 ----
    new_messages: list[dict]  # 本次请求待落盘的消息，消息累加唯一出口（审计 #5）
    token_cost: TokenCost  # 本次请求累计 token，仅在节点内经 _accumulate 累加
    error: Optional[str]
    error_status: int
    route: str  # 条件边路由标记："question" / "wait" / "answer" / "summary" / "invalid" / "error"
