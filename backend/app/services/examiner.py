"""
考官模式核心服务：出题 → 评分 → 追问/下一题 → 总结
包含：反作弊检测、答案缓存、追问隔离
"""
import hashlib
import json
import re
from typing import Optional
from openai import OpenAI

from app.core.config import settings
from app.services.token_tracker import TokenTracker, TokenCost, estimate_tokens


MAX_QUESTIONS = 5
MAX_FOLLOW_UP = 2
COPY_PASTE_MIN_CHUNK_LEN = 20
COPY_PASTE_SUBSTRING_LEN = 30
REPEAT_ANSWER_THRESHOLD = 0.85
SIMILARITY_SUBSTRING_MIN_LEN = 50


EXAMINER_SYSTEM_PROMPT = """你是一位经验丰富的{{target_position}}技术面试官，正在面试{{topic}}方向候选人。

## 核心定位
- 你代表企业在招聘「{{target_position}}」岗位，重点考察「{{topic}}」方向
- 所有题目必须紧扣「{{target_position}}」岗位职责和「{{topic}}」技术栈
- 题目难度要与候选人目标岗位级别匹配，由浅入深
- 不要暴露自己是 AI
- 所有题目、评分和反馈必须严格基于下面提供的参考资料
- 若参考资料不足以支撑该岗位/方向的面试，请明确告知候选人"当前资料不足以支撑该方向面试，请补充相关资料"

## 参考资料
{{context}}

## 当前面试状态
- 已回答题目数：{{question_index}}
- 当前追问次数：{{follow_up_count}}
- 本题最多追问次数：{{max_follow_up}}

## 历史面试记录
{{history}}

## 出题要求（关键）
- 每次只提一个问题
- 题目必须针对「{{target_position}}」岗位在「{{topic}}」方向的真实面试场景
- 题目类型按以下顺序轮换：
  1. 概念题（考察基础定义）
  2. 原理题（考察底层机制）
  3. 对比题（考察辨析能力）
  4. 场景题（考察实战经验）
  5. 综合题（考察综合运用与权衡能力）
- 难度由浅入深：第 1 题基础，第 5 题综合深入
- 每题后必须说明考察点
- 提出后停止输出，等待候选人回答，不要自问自答，不要泄露答案

## 评分反馈
当候选人回答后，按以下结构输出：

【评分】0-10 分

【点评】优点和不足

【补充】遗漏的关键点

【纠正】错误之处的正确解释

评分标准：
- 9-10 分：完整、准确、有深度，能结合实际
- 7-8 分：核心正确，但缺少细节或举例
- 5-6 分：部分正确，有明显遗漏或偏差
- 3-4 分：肤浅，只触及表面
- 0-2 分：完全错误或未回答

## 追问或下一题
根据回答质量决定：
- 如果回答有明显漏洞且当前追问次数 < {{max_follow_up}}：进行 1 轮更深入的追问，追问要针对漏洞，不要换题
- 如果回答充分或追问次数已用完：输出"【下一题】"，然后进入下一题

## 面试结束
当用户说"结束"或已回答至少 {{max_questions}} 题时，输出：

【总体评分】0-100 分

【能力画像】2-3 句话总结

【薄弱环节】2-3 个方向

【复习建议】具体可执行的建议

## 特殊处理
- 候选人说"不知道"时：先引导思考 1 次，给出启发性问题；若仍回答不出，再给出解释
- 候选人说"结束"时：立即停止出题，输出面试总结
- 不要一次性输出多道题
- 不要在出题阶段泄露答案
"""


def _create_client() -> Optional[OpenAI]:
    if not settings.deepseek_api_key:
        return None
    return OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)


def _normalize(text: str) -> str:
    """去除空白和标点，统一小写，用于相似度比对"""
    # 先移除中英文常用标点，再移除空白
    text = re.sub(
        r"[\s\n\r\t，。！？、；：\"\"''（）【】［］\[\]()!?;:]",
        "",
        text,
    )
    return text.lower()


def _text_similarity(a: str, b: str) -> float:
    """字符集合 Jaccard 相似度"""
    na, nb = _normalize(a), _normalize(b)
    if not na or not nb:
        return 0.0
    # 子串规则仅在较短串足够长时生效，避免短回答被误判为完全重复
    if min(len(na), len(nb)) >= SIMILARITY_SUBSTRING_MIN_LEN and (na in nb or nb in na):
        return 1.0
    set_a, set_b = set(na), set(nb)
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


def _build_context(chunks: list[tuple[str, str, float]]) -> str:
    """按 token 预算拼接参考资料"""
    selected = []
    tokens = 0
    max_tokens = settings.max_input_tokens - settings.max_output_tokens - 1000
    for file_name, content, score in chunks:
        text = f"[参考片段]\n来源：{file_name}\n相似度：{score:.3f}\n内容：{content}\n"
        t = estimate_tokens(text)
        if tokens + t > max_tokens and selected:
            break
        selected.append(text)
        tokens += t
    return "\n---\n".join(selected)


def _parse_summary(text: str) -> dict:
    total_match = re.search(r"【总体评分】\s*(\d+(?:\.\d+)?)", text)
    return {
        "raw": text,
        "total_score": float(total_match.group(1)) if total_match else 0,
    }


def _local_summary(scores: list[dict], weak_points: list[str]) -> dict:
    """模型调用失败时本地拼接的简易总结：各题均分（折算百分制）+ 薄弱环节列表"""
    valid = [s["score"] for s in scores if isinstance(s.get("score"), (int, float))]
    avg = sum(valid) / len(valid) if valid else 0.0
    total_score = round(avg * 10, 1)  # 单题满分 10 分，折算为百分制
    weak_text = "、".join(weak_points[:3]) if weak_points else "无明确记录"
    raw = (
        f"【总体评分】{total_score}\n"
        f"【能力画像】共作答 {len(scores)} 题，平均得分 {round(avg, 1)}/10。\n"
        f"【薄弱环节】{weak_text}\n"
        f"【复习建议】针对上述薄弱环节查漏补缺后重新参加面试。"
    )
    return {"raw": raw, "total_score": total_score}


def detect_cheating(
    answer: str,
    previous_answers: list[str],
    context_chunks: list[tuple[str, str, float]],
) -> tuple[bool, str]:
    """
    反作弊检测：
    1. 与同一题历史答案重复度过高
    2. 回答与参考资料原文高度重合（直接复制）
    返回 (是否作弊, 原因)
    """
    if not answer or len(answer.strip()) < 5:
        return False, ""

    # 1. 重复提交检测
    for prev in previous_answers:
        sim = _text_similarity(answer, prev)
        if sim >= REPEAT_ANSWER_THRESHOLD:
            return True, "检测到与此前回答高度重复，请重新组织语言作答"

    # 2. 复制参考资料检测
    normalized_answer = _normalize(answer)
    for _, content, _ in context_chunks:
        normalized_chunk = _normalize(content)
        if len(normalized_chunk) < COPY_PASTE_MIN_CHUNK_LEN:
            continue
        # 整段落入答案
        if normalized_chunk in normalized_answer:
            return True, "回答与参考资料原文高度重合，疑似直接复制，请用自己的语言回答"
        # 长连续子串落入答案
        step = COPY_PASTE_SUBSTRING_LEN
        for i in range(0, len(normalized_chunk) - step + 1, step // 2):
            substr = normalized_chunk[i:i + step]
            if substr in normalized_answer:
                return True, "回答包含参考资料原文片段，请用自己的语言回答"

    return False, ""


def _answer_cache_key(question: str, answer: str) -> str:
    """答案缓存 key：基于题目与回答的 MD5"""
    return hashlib.md5(f"{question}||{answer}".encode("utf-8")).hexdigest()


def get_cached_evaluation(state: dict, question: str, answer: str) -> Optional[dict]:
    key = _answer_cache_key(question, answer)
    return state.get("answer_cache", {}).get(key)


def set_cached_evaluation(state: dict, question: str, answer: str, result: dict) -> None:
    if "answer_cache" not in state:
        state["answer_cache"] = {}
    state["answer_cache"][_answer_cache_key(question, answer)] = result


def _format_thread(thread: list[dict]) -> str:
    """把当前题目线程格式化为历史记录文本"""
    if not thread:
        return "（无）"
    lines = []
    for m in thread:
        role = "考官" if m["role"] == "assistant" else "候选人"
        lines.append(f"{role}: {m['content'][:400]}")
    return "\n".join(lines)


def _extract_reference_points(text: str) -> list[str]:
    """从 LLM 输出中提取【参考答案要点】"""
    points = []
    match = re.search(r"【参考答案要点】\s*(.+?)(?=\n\n|【题目】|【追问】|【类型】|【考察点】|$)", text, re.S)
    if not match:
        return points
    content = match.group(1).strip()
    # 支持 "1. xxx\n2. xxx" 或 "- xxx\n- xxx" 或 "xxx、xxx"
    for line in content.split("\n"):
        line = re.sub(r"^\s*[\d\-\*•]+\.?\s*", "", line).strip()
        if line:
            points.append(line)
    return points[:6]  # 最多 6 个要点


def generate_question(
    target_position: str,
    topic: str,
    context_chunks: list[tuple[str, str, float]],
    thread: list[dict],
    question_index: int,
    follow_up_count: int = 0,
) -> tuple[Optional[str], list[str], list[str], TokenCost]:
    """
    生成下一道面试题或追问，同时输出参考答案要点。

    失败约定：LLM 未配置、调用异常或解析失败时返回 (None, [], [], TokenCost)，
    调用方需将 None 视为失败（返回 502），不得把报错文案当作题目落盘。
    """
    client = _create_client()
    if not client:
        return None, [], [], TokenCost()

    context = _build_context(context_chunks)
    history_text = _format_thread(thread)

    prompt = (
        EXAMINER_SYSTEM_PROMPT
        .replace("{{target_position}}", target_position)
        .replace("{{topic}}", topic)
        .replace("{{context}}", context)
        .replace("{{history}}", history_text)
        .replace("{{question_index}}", str(question_index))
        .replace("{{follow_up_count}}", str(follow_up_count))
        .replace("{{max_follow_up}}", str(MAX_FOLLOW_UP))
        .replace("{{max_questions}}", str(MAX_QUESTIONS))
    )

    if follow_up_count > 0:
        prompt += (
            "\n\n当前是追问环节。请针对候选人上一题回答中的漏洞，提出一个更深入的追问。"
            "不要换题，不要重复原题。格式：\n【追问】...\n【考察点】..."
        )
    else:
        question_types = ["概念题", "原理题", "对比题", "场景题", "综合题"]
        suggested_type = question_types[(question_index - 1) % len(question_types)]
        prompt += (
            f"\n\n当前是第 {question_index} 题，建议题型：{suggested_type}。"
            "请只输出当前要考察的题目、考察点和参考答案要点，不要输出答案。格式：\n"
            "【题目】...\n"
            "【类型】概念题/原理题/对比题/场景题/综合题\n"
            "【考察点】...\n"
            "【参考答案要点】\n1. ...\n2. ...\n3. ..."
        )

    try:
        tracker = TokenTracker(client)
        response, cost = tracker.chat_completion(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是严格的技术面试官，只输出题目和评分要点，不输出答案。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=settings.max_output_tokens,
        )
    except Exception as e:
        return None, [], [], TokenCost()

    text = response.choices[0].message.content or ""
    if not text.strip():
        return None, [], [], cost

    reference_points = _extract_reference_points(text)

    if follow_up_count > 0:
        q_match = re.search(r"【追问】\s*(.+?)(?=【考察点】|【参考答案要点】|$)", text, re.S)
        e_match = re.search(r"【考察点】\s*(.+?)(?=【参考答案要点】|$)", text, re.S)
        question = f"【追问】{q_match.group(1).strip()}" if q_match else text.strip()
    else:
        q_match = re.search(r"【题目】\s*(.+?)(?=【类型】|【考察点】|【参考答案要点】|$)", text, re.S)
        t_match = re.search(r"【类型】\s*(.+?)(?=【考察点】|【参考答案要点】|$)", text, re.S)
        e_match = re.search(r"【考察点】\s*(.+?)(?=【参考答案要点】|$)", text, re.S)
        question = q_match.group(1).strip() if q_match else text.strip()
        if t_match:
            question = f"【{t_match.group(1).strip()}】{question}"

    expectations = [s.strip() for s in e_match.group(1).split("、")] if e_match else []
    if not question:
        return None, [], [], cost
    return question, expectations, reference_points, cost


def _parse_structured_evaluation(text: str, reference_points: list[str]) -> dict:
    """解析结构化评分结果，优先 JSON，失败则回退正则"""
    result = {
        "raw": text,
        "score": 5.0,
        "has_next_question": "【下一题】" in text,
        "points": [],
        "comment": "",
        "supplement": "",
        "correction": "",
    }

    # 尝试解析 JSON
    try:
        # 找第一个 { 和最后一个 }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(text[start:end + 1])
            if isinstance(data.get("points"), list):
                result["points"] = data["points"]
            result["score"] = float(data.get("score", 5.0))
            result["comment"] = str(data.get("comment", ""))
            result["supplement"] = str(data.get("supplement", ""))
            result["correction"] = str(data.get("correction", ""))
            result["has_next_question"] = data.get("has_next_question", result["has_next_question"])
            return result
    except Exception:
        pass

    # 回退：正则解析旧格式
    score_match = re.search(r"【评分】\s*(\d+(?:\.\d+)?)", text)
    if score_match:
        result["score"] = min(max(float(score_match.group(1)), 0), 10)

    comment_match = re.search(r"【点评】\s*(.+?)(?=【补充】|【纠正】|$)", text, re.S)
    supplement_match = re.search(r"【补充】\s*(.+?)(?=【纠正】|$)", text, re.S)
    correction_match = re.search(r"【纠正】\s*(.+)", text, re.S)

    result["comment"] = (comment_match.group(1).strip() if comment_match else "").strip()
    result["supplement"] = (supplement_match.group(1).strip() if supplement_match else "").strip()
    result["correction"] = (correction_match.group(1).strip() if correction_match else "").strip()

    # 如果没有 points，按参考答案要点生成默认未命中
    if not result["points"] and reference_points:
        result["points"] = [
            {"point": p, "hit": False, "evidence": "未检测到明确命中"}
            for p in reference_points
        ]

    return result


def evaluate_answer(
    target_position: str,
    topic: str,
    question: str,
    answer: str,
    reference_points: list[str],
    context_chunks: list[tuple[str, str, float]],
    thread: list[dict],
    follow_up_count: int,
    question_index: int = 1,
) -> tuple[dict, TokenCost]:
    """
    基于参考答案要点对候选人回答做结构化评分。

    失败约定：API 未配置或调用异常时返回 ({"error": 错误信息}, TokenCost())，
    成功时返回 (_parse_structured_evaluation 结果 dict, TokenCost)。
    """
    client = _create_client()
    if not client:
        return {"error": "未配置 API KEY"}, TokenCost()

    context = _build_context(context_chunks)
    history_text = _format_thread(thread)
    points_text = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(reference_points)) if reference_points else "（无预设要点，请自行判断）"

    prompt = (
        EXAMINER_SYSTEM_PROMPT
        .replace("{{target_position}}", target_position)
        .replace("{{topic}}", topic)
        .replace("{{context}}", context)
        .replace("{{history}}", history_text)
        .replace("{{question_index}}", str(question_index))
        .replace("{{follow_up_count}}", str(follow_up_count))
        .replace("{{max_follow_up}}", str(MAX_FOLLOW_UP))
        .replace("{{max_questions}}", str(MAX_QUESTIONS))
    )
    prompt += (
        f"\n\n当前题目：{question}\n"
        f"参考答案要点：\n{points_text}\n\n"
        f"候选人回答：{answer}\n\n"
        "请严格按以下 JSON 格式输出评分结果（不要包含 markdown 代码块标记）：\n"
        "{\n"
        '  "points": [\n'
        '    {"point": "要点原文", "hit": true/false, "evidence": "用户回答中对应的依据，或说明为什么未命中"}\n'
        '  ],\n'
        '  "score": 0-10,\n'
        '  "comment": "总体点评",\n'
        '  "supplement": "遗漏的关键点",\n'
        '  "correction": "错误之处的正确解释",\n'
        '  "has_next_question": true/false\n'
        "}\n\n"
        "评分规则：\n"
        "- 每个要点命中得基础分，未命中扣分\n"
        "- 9-10 分：所有核心要点都命中且有深度\n"
        "- 7-8 分：大部分要点命中，但缺少细节\n"
        "- 5-6 分：半数要点命中\n"
        "- 3-4 分：只命中少量要点\n"
        "- 0-2 分：完全未命中或错误"
    )

    try:
        tracker = TokenTracker(client)
        response, cost = tracker.chat_completion(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是严格的技术面试官，必须输出合法 JSON，按要点判定得分。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=settings.max_output_tokens,
        )
    except Exception as e:
        return {"error": f"评分调用失败: {str(e)}"}, TokenCost()

    text = response.choices[0].message.content or ""
    return _parse_structured_evaluation(text, reference_points), cost


def generate_summary(
    target_position: str,
    topic: str,
    scores: list[dict],
    weak_points: list[str],
) -> tuple[dict, TokenCost]:
    """生成面试总结，结合高频遗漏点；模型调用失败时返回本地拼接的简易总结"""
    client = _create_client()
    if not client:
        return {"error": "未配置 API KEY"}, TokenCost()

    scores_text = "\n".join(
        f"第 {i + 1} 题：{s['score']} 分 - 命中 {sum(1 for p in s.get('points', []) if p.get('hit'))}/{len(s.get('points', []))} 个要点"
        for i, s in enumerate(scores)
    )
    weak_text = "\n".join(f"- {p}" for p in weak_points[:15]) if weak_points else "（无明显遗漏记录）"
    prompt = (
        f"你是一位技术面试官，请根据以下面试表现给出总结。\n\n"
        f"目标岗位：{target_position}\n"
        f"方向：{topic}\n\n"
        f"得分记录：\n{scores_text}\n\n"
        f"高频遗漏点（来自参考答案要点未命中）：\n{weak_text}\n\n"
        "请按以下结构输出：\n"
        "【总体评分】0-100 分\n"
        "【能力画像】2-3 句话\n"
        "【薄弱环节】2-3 个方向\n"
        "【复习建议】具体可执行的建议"
    )

    try:
        tracker = TokenTracker(client)
        response, cost = tracker.chat_completion(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是技术面试官，输出面试总结。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=settings.max_output_tokens,
        )
    except Exception as e:
        # 模型调用失败时返回本地拼接的简易总结，保证面试流程闭环
        print(f"[Examiner] 生成面试总结失败，改用本地简易总结: {e}")
        return _local_summary(scores, weak_points), TokenCost()

    text = response.choices[0].message.content or ""
    return _parse_summary(text), cost
