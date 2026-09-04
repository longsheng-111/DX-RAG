"""QA Graph 状态定义（设计文档 4.1）"""
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
