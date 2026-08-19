"""
Harness：Agent 执行环境的约束、调度和兜底

- 调度（Orchestration）：定义 RAG 执行 DAG
- 约束（Guardrails）：输入过滤、工具白名单、Token 预算、超时控制
- 兜底（Fallback）：检索为空、LLM 超时、连续出错时的降级策略

接线现状（如实说明）：
- guard_input / fallback_error：被 qa.py 调用
- check_token_budget：已在 routes.py 接线——/api/chat 按会话累计 Token、
  /api/query 按请求自带 history 的估算量判定，超预算直接返回 429
- LLM 超时控制不在本模块：统一由 token_tracker.chat_completion 注入 60s timeout
- guard_tool / run_with_timeout 已实现但未被外部调用
- orchestrate_rag 为预留的 DAG 调度实现，尚未接线
"""
import re
import time
from typing import Optional, Callable
from openai import OpenAI

from app.core.config import settings
from app.services.token_tracker import (
    TokenTracker,
    TokenCost,
    get_token_budget,
    estimate_tokens,
)


# 允许 Agent 调用的工具白名单
ALLOWED_TOOLS = {"knowledge_search"}

# 简单敏感词过滤（可按需扩展）
# 注意不能用 \b：Python re 中中文字符也是单词字符，\b 对中文边界判定会失效
SENSITIVE_PATTERNS = [
    re.compile(r"(?:密码|passwd|password)\s*[:=]", re.IGNORECASE),
]


class HarnessError(Exception):
    """Harness 控制异常"""

    def __init__(self, message: str, fallback_answer: Optional[str] = None):
        super().__init__(message)
        self.fallback_answer = fallback_answer


class HarnessResult:
    """Harness 执行结果"""

    def __init__(
        self,
        answer: str,
        sources: list[dict],
        token_cost: TokenCost,
        fallback_used: bool = False,
    ):
        self.answer = answer
        self.sources = sources
        self.token_cost = token_cost
        self.fallback_used = fallback_used


# ============================================================
#  约束（Guardrails）
# ============================================================

def guard_input(question: str) -> Optional[str]:
    """输入过滤，返回错误信息或 None"""
    if not question or not question.strip():
        return "问题不能为空"

    if len(question) > 2000:
        return "问题长度超过 2000 字符限制"

    for pattern in SENSITIVE_PATTERNS:
        if pattern.search(question):
            return "问题包含敏感信息，请重新输入"

    return None


def guard_tool(tool_name: str) -> bool:
    """工具白名单检查"""
    return tool_name in ALLOWED_TOOLS


def check_token_budget(
    session_total: int,
    estimated_prompt_tokens: int = 0,
) -> Optional[str]:
    """Token 预算检查"""
    budget = get_token_budget()
    max_total = budget["max_total_tokens_per_session"]
    if max_total <= 0:
        return None

    if session_total >= max_total:
        return (
            f"当前会话 Token 消耗已达上限（{session_total}/{max_total}），"
            "请新建会话继续提问。"
        )

    remaining = max_total - session_total
    if estimated_prompt_tokens > 0 and estimated_prompt_tokens > remaining:
        return "剩余 Token 预算不足，请精简问题或新建会话"

    return None


# ============================================================
#  兜底（Fallback）
# ============================================================

def fallback_empty_retrieval(question: str) -> str:
    """检索为空时的兜底回答"""
    return (
        f"知识库中暂无与「{question[:30]}」相关的资料。\n\n"
        "您可以：\n"
        "1. 检查当前选择的知识库是否正确\n"
        "2. 上传相关文档后重试\n"
        "3. 换用更通用的关键词提问"
    )


def fallback_timeout() -> str:
    """LLM 超时兜底"""
    return "模型响应超时，请稍后重试。如果多次超时，建议缩短问题或新建会话。"


def fallback_error(error_message: str) -> str:
    """通用错误兜底"""
    return f"系统处理出错：{error_message}。请重试或联系管理员。"


# ============================================================
#  调度（Orchestration）
# ============================================================

def run_with_timeout(
    func: Callable,
    timeout: float = 15.0,
    *args,
    **kwargs,
):
    """
    带超时的函数执行。
    注意：Python 标准库没有原生超时装饰器，这里使用简单轮询。
    对于 I/O 密集型任务（LLM API 调用），可通过线程实现超时。
    """
    import threading

    result = {"value": None, "error": None, "done": False}

    def target():
        try:
            result["value"] = func(*args, **kwargs)
        except Exception as e:
            result["error"] = e
        finally:
            result["done"] = True

    # daemon 线程：超时返回后即使工作线程仍挂起，也不阻塞进程退出
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if not result["done"]:
        raise TimeoutError(f"函数执行超过 {timeout} 秒")

    if result["error"]:
        raise result["error"]

    return result["value"]


def orchestrate_rag(
    question: str,
    retrieve_func: Callable[[str], list[tuple[str, str, float]]],
    generate_func: Callable,
    session_total: int = 0,
    timeout: float = 30.0,
) -> HarnessResult:
    """
    RAG 执行 DAG：
    输入校验 → 检索 → 判断充分性 →（改写重试）→ 生成答案 → 输出
    """
    # 1. 输入校验
    error = guard_input(question)
    if error:
        return HarnessResult(
            answer=fallback_error(error),
            sources=[],
            token_cost=TokenCost(),
            fallback_used=True,
        )

    # 2. Token 预算预检
    budget_error = check_token_budget(session_total, estimate_tokens(question))
    if budget_error:
        return HarnessResult(
            answer=budget_error,
            sources=[],
            token_cost=TokenCost(),
            fallback_used=True,
        )

    # 3. 检索（带超时）
    try:
        search_results = run_with_timeout(retrieve_func, timeout=10.0, query=question)
    except TimeoutError:
        return HarnessResult(
            answer=fallback_timeout(),
            sources=[],
            token_cost=TokenCost(),
            fallback_used=True,
        )
    except Exception as e:
        return HarnessResult(
            answer=fallback_error(f"检索失败: {e}"),
            sources=[],
            token_cost=TokenCost(),
            fallback_used=True,
        )

    # 4. 检索为空兜底
    if not search_results:
        return HarnessResult(
            answer=fallback_empty_retrieval(question),
            sources=[],
            token_cost=TokenCost(),
            fallback_used=True,
        )

    # 5. 生成答案（带超时）
    try:
        answer, token_cost = run_with_timeout(
            generate_func,
            timeout=timeout,
            question=question,
            search_results=search_results,
        )
    except TimeoutError:
        return HarnessResult(
            answer=fallback_timeout(),
            sources=[],
            token_cost=TokenCost(),
            fallback_used=True,
        )
    except Exception as e:
        return HarnessResult(
            answer=fallback_error(f"生成失败: {e}"),
            sources=[],
            token_cost=TokenCost(),
            fallback_used=True,
        )

    # 6. 格式化来源
    sources = [
        {
            "content": content[:300],
            "similarity": round(score, 4),
            "file_name": file_name,
        }
        for file_name, content, score in search_results[:5]
    ]

    return HarnessResult(
        answer=answer,
        sources=sources,
        token_cost=token_cost,
        fallback_used=False,
    )
