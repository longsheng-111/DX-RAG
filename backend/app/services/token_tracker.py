"""
Token 预算跟踪与上下文压缩模块

- 包装 LLM 调用，自动提取 usage 统计
- 提供 Token 估算兜底
- 当会话上下文占用超过预算阈值时，触发历史摘要压缩
"""
from typing import Optional
from openai import OpenAI

from app.core.config import settings
from app.services.llm_gate import call_with_gate

# LLM 调用默认超时（秒），防止 API 无响应时请求无限挂起
LLM_TIMEOUT = 60.0


class TokenCost(dict):
    """Token 消耗数据结构，便于直接序列化为 JSON"""

    def __init__(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        session_total: int = 0,
    ):
        super().__init__()
        self["prompt_tokens"] = prompt_tokens
        self["completion_tokens"] = completion_tokens
        self["total_tokens"] = total_tokens
        self["session_total"] = session_total

    @property
    def prompt(self) -> int:
        return self["prompt_tokens"]

    @property
    def completion(self) -> int:
        return self["completion_tokens"]

    @property
    def total(self) -> int:
        return self["total_tokens"]


class TokenTracker:
    """Token 跟踪器：真实 usage + 估算兜底"""

    def __init__(self, client: Optional[OpenAI] = None):
        self.client = client

    def chat_completion(self, timeout: float = LLM_TIMEOUT, **kwargs):
        """调用 LLM 并返回 (response, token_cost)，默认 60s 超时。

        统一经 llm_gate 的全局信号量与 429 退避重试（阶段 4），
        一处接入，问答/考官/压缩/标题全链路生效。
        """
        if self.client is None:
            raise RuntimeError("TokenTracker 未配置 OpenAI client")

        kwargs.setdefault("timeout", timeout)
        response = call_with_gate(lambda: self.client.chat.completions.create(**kwargs))

        usage = response.usage
        if usage:
            token_cost = TokenCost(
                prompt_tokens=usage.prompt_tokens or 0,
                completion_tokens=usage.completion_tokens or 0,
                total_tokens=usage.total_tokens or 0,
            )
        else:
            # API 未返回 usage 时兜底估算
            prompt_text = "\n".join(
                f"{m.get('role', '')}: {m.get('content', '')}"
                for m in kwargs.get("messages", [])
            )
            completion_text = response.choices[0].message.content or ""
            prompt_tokens = estimate_tokens(prompt_text)
            completion_tokens = estimate_tokens(completion_text)
            token_cost = TokenCost(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )

        return response, token_cost


def estimate_tokens(text: str) -> int:
    """
    粗略估算 Token 数。
    中文按 1 字 ≈ 1 token，英文按空格分词 ≈ 1 token。
    基础估算会漏算标点、数字与代码符号，系统性偏低，
    因此统一乘以 1.3 安全系数，宁可高估不可低估。
    """
    if not text:
        return 0

    # 中文字符数
    cn_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    # 英文单词数（按空格分）
    en_words = len([w for w in text.split() if any(c.isascii() for c in w)])

    return int((cn_chars + en_words) * 1.3)


def estimate_messages_tokens(messages: list[dict]) -> int:
    """估算一组消息（或压缩历史记录）的总 Token 数"""
    return sum(estimate_tokens(m.get("content", "")) for m in messages)


def get_token_budget() -> dict:
    """获取当前 Token 预算配置"""
    return {
        "max_input_tokens": settings.max_input_tokens,
        "max_output_tokens": settings.max_output_tokens,
        "max_total_tokens_per_session": settings.max_total_tokens_per_session,
        "compression_threshold": settings.context_compression_threshold,
    }


def should_compress_context(context_tokens: int) -> bool:
    """判断当前上下文估算 Token 数是否达到压缩阈值"""
    budget = get_token_budget()
    threshold = budget["compression_threshold"]
    max_total = budget["max_total_tokens_per_session"]
    if max_total <= 0:
        return False
    return context_tokens / max_total >= threshold


def compress_messages_with_llm(
    messages: list[dict],
    client: OpenAI,
    max_tokens: int = 300,
) -> tuple[str, TokenCost]:
    """
    调用 LLM 对历史消息生成摘要。
    返回 (summary, token_cost)
    """
    history_text = "\n".join(
        f"{'用户' if m['role'] == 'user' else '助手'}: {m['content']}"
        for m in messages
    )

    prompt = (
        "请用 200 字以内总结以下对话的关键信息，保留用户核心问题和系统关键回答，"
        "不要遗漏重要技术概念和结论。"
    )

    tracker = TokenTracker(client)
    response, token_cost = tracker.chat_completion(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": history_text},
        ],
        temperature=0.3,
        max_tokens=max_tokens,
    )

    summary = response.choices[0].message.content or "（摘要生成失败）"
    return summary, token_cost


def compress_session_context(
    session,
    client: Optional[OpenAI] = None,
) -> TokenCost:
    """
    压缩会话上下文。
    策略：保留最近 4 条完整消息，对更早消息生成摘要。
    如果 LLM 不可用，直接丢弃更早消息。
    """
    total_cost = TokenCost()

    if len(session.messages) <= 4:
        return total_cost

    # 保留最近 4 条消息
    recent_messages = session.messages[-4:]
    older_messages = session.messages[:-4]

    if client and settings.deepseek_api_key:
        try:
            summary, cost = compress_messages_with_llm(older_messages, client)
            total_cost = cost
            session.compressed_history.append({
                "type": "summary",
                "content": summary,
                "token_cost": dict(cost),
                "compressed_message_count": len(older_messages),
            })
            session.messages = [
                {
                    "role": "system",
                    "content": f"【历史对话摘要】\n{summary}",
                    "timestamp": session.updated_at,
                }
            ] + recent_messages
        except Exception:
            # LLM 摘要失败，降级为直接截断
            session.messages = recent_messages
    else:
        # 无 LLM 客户端，直接截断
        session.messages = recent_messages

    return total_cost
