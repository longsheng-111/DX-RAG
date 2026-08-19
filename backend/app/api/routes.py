"""
API 路由定义 - 文件上传、知识问答、知识库管理、文件管理
"""
import asyncio
import logging
import os
import re

logger = logging.getLogger(__name__)
from pathlib import Path
from typing import Optional, Literal

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.persona import (
    list_personas,
    get_persona,
    get_persona_default_kb,
    get_persona_system_prompt,
    validate_persona,
)
from app.core.session import session_manager
from app.services.token_tracker import (
    compress_session_context,
    should_compress_context,
    estimate_messages_tokens,
    TokenCost,
)
from app.services.harness import check_token_budget
from app.services.feedback import feedback_store
from app.core.vector_store import (
    list_collections,
    create_collection,
    delete_collection,
    rename_collection,
    get_files,
    delete_file,
)
from app.services.ingest import process_file
from app.services.qa import qa_query
from app.services.title_generator import generate_session_title
from app.services.examiner import (
    generate_question,
    evaluate_answer,
    generate_summary,
    detect_cheating,
    get_cached_evaluation,
    set_cached_evaluation,
    MAX_QUESTIONS,
    MAX_FOLLOW_UP,
)

router = APIRouter(prefix="/api")

# 上传文件扩展名白名单（与 ingest 支持的格式一致）
ALLOWED_UPLOAD_EXTENSIONS = {
    ".txt", ".md", ".csv", ".json", ".log",
    ".pdf", ".docx",
    ".xlsx", ".xlsm", ".xltx", ".xltm",
}

# 上传文件大小上限（MB），可通过环境变量 MAX_UPLOAD_SIZE_MB 覆盖
MAX_UPLOAD_SIZE_MB = int(os.environ.get("MAX_UPLOAD_SIZE_MB", "50"))
MAX_UPLOAD_SIZE = MAX_UPLOAD_SIZE_MB * 1024 * 1024

# ---- 安全校验 ----

_COLLECTION_NAME_RE = re.compile(r'^[\w一-鿿-]+$')

def _validate_collection_name(name: str, label: str = "知识库名称") -> str:
    """校验知识库名称，防止注入"""
    name = name.strip()
    if not name or len(name) < 2 or len(name) > 50:
        raise HTTPException(status_code=400, detail=f"{label}需 2-50 个字符")
    if not _COLLECTION_NAME_RE.match(name):
        raise HTTPException(status_code=400, detail=f"{label}包含非法字符")
    return name


def _safe_file_path(file_name: str) -> Path:
    """安全解析文件路径，防止路径穿越"""
    # 去掉路径分隔符和 .. 穿越
    safe_name = os.path.basename(file_name)
    if safe_name != file_name or ".." in file_name or "/" in file_name.replace("\\", "/"):
        raise HTTPException(status_code=400, detail="文件名包含非法路径字符")
    full = (Path(settings.upload_dir) / safe_name).resolve()
    base = Path(settings.upload_dir).resolve()
    if not str(full).startswith(str(base)):
        raise HTTPException(status_code=403, detail="路径越权访问被拒绝")
    return full

# ============================================================
#  请求/响应模型
# ============================================================

class QueryRequest(BaseModel):
    question: str = Field(max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)
    collection_name: str = "knowledge_chunks"
    history: list[dict] = []


class ChatRequest(BaseModel):
    question: str = Field(max_length=2000)
    session_id: Optional[str] = None
    persona: str = "default"
    collection_name: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=50)


class CreateSessionRequest(BaseModel):
    persona: str = "default"
    collection_name: Optional[str] = None
    title: str = "新会话"
    mode: str = "qa"  # qa | examiner
    target_position: Optional[str] = None
    topic: Optional[str] = None


class SwitchPersonaRequest(BaseModel):
    persona: str
    clear_history: bool = False


class UpdateTitleRequest(BaseModel):
    title: str


class CreateCollectionRequest(BaseModel):
    name: str


class RenameCollectionRequest(BaseModel):
    new_name: str


class ExamStartRequest(BaseModel):
    session_id: str
    target_position: Optional[str] = None
    topic: Optional[str] = None
    collection_name: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=50)


class ExamNextRequest(BaseModel):
    answer: str = ""
    top_k: int = Field(default=5, ge=1, le=50)
    # 显式动作：give_up=放弃作答（按"不知道"计 0 分），end=结束面试（跳过评分直接收尾）
    # None 表示按 answer 正常作答
    action: Optional[Literal["give_up", "end"]] = None


class FeedbackRequest(BaseModel):
    session_id: str
    original_chunk_id: str
    target_chunk_id: str
    type: Literal["replace", "inaccurate"]
    original_content: str = ""
    target_content: str = ""
    note: str = ""


# ============================================================
#  健康检查
# ============================================================

@router.get("/health")
async def health():
    return {"status": "ok"}


# ============================================================
#  文件上传
# ============================================================

@router.post("/upload")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    collection_name: str = Form(default="knowledge_chunks"),
):
    """上传文件并自动入库"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="未选择文件")

    # 扩展名白名单预检（与 ingest 支持的格式一致）
    safe_name = Path(file.filename).name  # 去掉路径
    suffix = Path(safe_name).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {suffix or '（无扩展名）'}")

    # Content-Length 上限预检
    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit() and int(content_length) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail=f"文件超过大小限制（{MAX_UPLOAD_SIZE_MB}MB）")

    # 校验知识库名称
    _validate_collection_name(collection_name, "知识库名称")

    # 保存文件到 uploads 目录（同名文件直接覆盖，旧向量在入库前删除）
    settings.ensure_dirs()
    dest_path = Path(settings.upload_dir) / safe_name

    # 写盘循环内累计字节硬限制：Content-Length 预检可被 chunked 传输绕过，
    # 超限立即中断、删除残留文件并返回 413
    written = 0
    try:
        with open(dest_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                written += len(chunk)
                if written > MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail=f"文件超过大小限制（{MAX_UPLOAD_SIZE_MB}MB）",
                    )
                buffer.write(chunk)
    except HTTPException:
        dest_path.unlink(missing_ok=True)
        raise
    except Exception as e:
        logger.exception("Failed to save uploaded file")
        raise HTTPException(status_code=500, detail="Internal server error")

    # 处理入库；失败时清理已落盘的残留文件
    try:
        # 同知识库已存在同名文件：先删旧向量，避免重复入库
        existing_names = {f["file_name"] for f in get_files(collection_name)}
        if safe_name in existing_names:
            delete_file(safe_name, collection_name)

        result = process_file(str(dest_path), collection_name)
    except ValueError as e:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        dest_path.unlink(missing_ok=True)
        logger.exception("Failed to process uploaded file")
        raise HTTPException(status_code=500, detail="Internal server error")

    return {
        "message": "上传并入库成功",
        "file_name": result["file_name"],
        "chunks": result["chunks"],
        "collection_name": result["collection_name"],
    }


# ============================================================
#  知识问答
# ============================================================

@router.post("/query")
async def query(req: QueryRequest):
    """知识库问答（兼容旧接口）"""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    # Token 预算硬熔断：无会话接口按请求自带 history 的估算量判定
    budget_error = check_token_budget(estimate_messages_tokens(req.history))
    if budget_error:
        raise HTTPException(status_code=429, detail=budget_error)

    try:
        result = qa_query(
            question=req.question,
            collection_name=req.collection_name,
            top_k=req.top_k,
            history=req.history,
        )
    except Exception as e:
        logger.exception("QA query failed")
        raise HTTPException(status_code=500, detail="Internal server error")

    return {
        **result,
        "reasoning_steps": result.get("reasoning_steps", []),
        "queries": result.get("queries", [req.question]),
    }


# ============================================================
#  会话管理 + 新聊天接口
# ============================================================

def _session_to_dict(session) -> dict:
    """将会话对象转为 API 返回字典"""
    return {
        "session_id": session.session_id,
        "title": session.title,
        "persona": session.persona,
        "kb_id": session.kb_id,
        "mode": session.mode,
        "exam_state": session.exam_state,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "total_tokens": session.total_tokens,
        "message_count": len(session.messages),
    }


@router.post("/sessions")
async def create_session_api(req: CreateSessionRequest):
    """创建新会话"""
    persona = validate_persona(req.persona)
    kb_id = req.collection_name or get_persona_default_kb(persona)

    try:
        exam_state = None
        if req.mode == "examiner":
            exam_state = {
                "mode": "examiner",
                "target_position": req.target_position or "未指定岗位",
                "topic": req.topic or "未指定方向",
                "collection_name": kb_id,
                "status": "configuring",
                "question_index": 0,
                "current_question": None,
                "current_expectations": [],
                "current_thread": [],
                "follow_up_count": 0,
                "max_follow_up": 2,
                "scores": [],
                "weak_points": [],
                "summary": None,
                "answer_cache": {},
            }
        session = session_manager.create_session(
            persona=persona,
            kb_id=kb_id,
            title=req.title,
            mode=req.mode,
            exam_state=exam_state,
        )
        return _session_to_dict(session)
    except Exception as e:
        logger.exception("Failed to create session")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/sessions")
async def list_sessions_api():
    """列出所有会话"""
    try:
        sessions = session_manager.list_sessions()
        return {"sessions": [_session_to_dict(s) for s in sessions]}
    except Exception as e:
        logger.exception("Failed to list sessions")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/sessions/{session_id}")
async def get_session_api(session_id: str):
    """获取会话详情"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {
        **_session_to_dict(session),
        "messages": session.messages,
        "compressed_history": session.compressed_history,
    }


@router.delete("/sessions/{session_id}")
async def delete_session_api(session_id: str):
    """删除会话"""
    if session_manager.delete_session(session_id):
        return {"session_id": session_id, "status": "deleted"}
    raise HTTPException(status_code=404, detail="会话不存在")


@router.post("/sessions/{session_id}/clear")
async def clear_session_api(session_id: str):
    """清空会话的消息与压缩历史，保留会话本身及 persona / kb_id 配置"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    session.messages = []
    session.compressed_history = []
    session_manager.save_session(session)
    return {"status": "ok"}


@router.put("/sessions/{session_id}/persona")
async def switch_persona_api(session_id: str, req: SwitchPersonaRequest):
    """切换会话角色"""
    persona = validate_persona(req.persona)
    session = session_manager.switch_persona(
        session_id, persona, clear_history=req.clear_history
    )
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    return _session_to_dict(session)


@router.put("/sessions/{session_id}/title")
async def update_session_title_api(session_id: str, req: UpdateTitleRequest):
    """更新会话标题"""
    title = req.title.strip()
    if not title or len(title) > 100:
        raise HTTPException(status_code=400, detail="标题需 1-100 个字符")
    session = session_manager.update_title(session_id, title)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    return _session_to_dict(session)


@router.get("/sessions/{session_id}/export")
async def export_session_api(session_id: str):
    """导出会话历史为 Markdown"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    lines = [f"# {session.title or '未命名会话'}"]
    lines.append("")
    lines.append(f"- **会话 ID**: {session.session_id}")
    lines.append(f"- **角色**: {session.persona}")
    lines.append(f"- **知识库**: {session.kb_id}")
    lines.append(f"- **创建时间**: {session.created_at}")
    lines.append(f"- **累计 Token**: {session.total_tokens}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 对话历史")
    lines.append("")

    for msg in session.messages:
        role_label = "用户" if msg["role"] == "user" else "助手"
        lines.append(f"### {role_label}")
        lines.append("")
        lines.append(msg.get("content", ""))
        lines.append("")
        if msg.get("token_cost"):
            cost = msg["token_cost"]
            lines.append(
                f"_Tokens: {cost.get('total_tokens', 0)}_"
            )
            lines.append("")

    markdown = "\n".join(lines)

    from fastapi.responses import PlainTextResponse
    # 文件名只使用 session_id，避免中文标题导致 HTTP header 编码错误
    filename = f"session_{session_id}.md"
    return PlainTextResponse(
        markdown,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/personas")
async def list_personas_api():
    """获取角色列表"""
    return {"personas": list_personas()}


@router.post("/chat")
async def chat(req: ChatRequest):
    """基于会话的聊天接口"""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    # 1. 获取或创建会话
    session = None
    if req.session_id:
        session = session_manager.get_session(req.session_id)

    if session is None:
        persona = validate_persona(req.persona)
        kb_id = req.collection_name or get_persona_default_kb(persona)
        session = session_manager.create_session(
            persona=persona,
            kb_id=kb_id,
        )

    # 2. 更新 persona / kb_id（如果请求中显式指定）
    if req.persona:
        session.persona = validate_persona(req.persona)
    if req.collection_name:
        session.kb_id = req.collection_name

    # 3. Token 预算硬熔断：累计消耗超预算直接拒绝
    budget_error = check_token_budget(session.total_tokens)
    if budget_error:
        raise HTTPException(status_code=429, detail=budget_error)

    # 4. 上下文压缩：按当前上下文估算 Token 数判定（而非累计消耗）；
    #    压缩后记录水位，估算量未超过上次水位前不重复触发
    context_tokens = estimate_messages_tokens(session.messages) + estimate_messages_tokens(
        session.compressed_history
    )
    last_compress_level = next(
        (
            h.get("context_tokens", 0)
            for h in reversed(session.compressed_history)
            if h.get("type") == "compress_marker"
        ),
        0,
    )
    if should_compress_context(context_tokens) and context_tokens > last_compress_level:
        from openai import OpenAI
        client = None
        if settings.deepseek_api_key:
            client = OpenAI(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
                timeout=60.0,
            )
        compress_cost = compress_session_context(session, client)
        session.update_total_tokens(compress_cost.total)
        session.compressed_history.append({
            "type": "compress_marker",
            "context_tokens": context_tokens,
        })

    # 5. 添加用户消息
    session.add_message("user", req.question.strip())

    # 6. 构建对话历史（取最近 6 轮 / 12 条）
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in session.messages[:-1][-12:]
    ]

    # 7. 调用 QA
    system_prompt = get_persona_system_prompt(session.persona)
    try:
        # qa_query 是同步函数（内含 LLM 调用与向量检索，秒级耗时），
        # 直接调用会阻塞事件循环，扔进线程池执行让出调度权
        result = await asyncio.to_thread(
            qa_query,
            question=req.question,
            collection_name=session.kb_id,
            top_k=req.top_k,
            history=history,
            system_prompt=system_prompt,
            session_id=session.session_id,
        )
    except Exception as e:
        logger.exception("Chat QA failed")
        raise HTTPException(status_code=500, detail="Internal server error")

    # 8. 添加助手消息并保存（add_message 会把本轮 token_cost 自动累加进 session.total_tokens，
    #    同时保存来源和推理路径，便于页面刷新后恢复；此处无需再手动累加，否则会双倍计数）
    token_cost = result.get("token_cost", {})
    session.add_message(
        "assistant",
        result["answer"],
        token_cost=token_cost,
        sources=result.get("sources"),
        reasoning_steps=result.get("reasoning_steps"),
    )
    session_manager.save_session(session)

    # 9. 生成标题（首次用户提问后）
    if len(session.messages) == 2 and session.title == "新会话":
        new_title, title_cost = generate_session_title(req.question.strip())
        if new_title:
            session.title = new_title
            session.update_total_tokens(title_cost.total)
            session_manager.save_session(session)

    # 10. 返回真实 token_cost
    response_cost = TokenCost(
        prompt_tokens=token_cost.get("prompt_tokens", 0),
        completion_tokens=token_cost.get("completion_tokens", 0),
        total_tokens=token_cost.get("total_tokens", 0),
        session_total=session.total_tokens,
    )

    return {
        "session_id": session.session_id,
        "title": session.title,
        "persona": session.persona,
        "kb_id": session.kb_id,
        "answer": result["answer"],
        "sources": result["sources"],
        "token_cost": dict(response_cost),
        "query": req.question,
        "reasoning_steps": result.get("reasoning_steps", []),
        "queries": result.get("queries", [req.question]),
    }


@router.post("/feedback")
async def create_feedback(req: FeedbackRequest):
    """接收用户反馈并持久化，后续同一会话的问答会把它注入系统提示词"""
    try:
        feedback = feedback_store.add_feedback(
            session_id=req.session_id,
            original_chunk_id=req.original_chunk_id,
            target_chunk_id=req.target_chunk_id,
            feedback_type=req.type,
            original_content=req.original_content,
            target_content=req.target_content,
            note=req.note,
        )
        return {"status": "ok", "feedback": feedback}
    except Exception as e:
        logger.exception("Failed to save feedback")
        raise HTTPException(status_code=500, detail="保存反馈失败")


# ============================================================
#  考官模式（模拟面试）
# ============================================================

@router.post("/exam/start")
async def exam_start(req: ExamStartRequest):
    """开启一场考官模式面试，配置从会话中读取"""
    session = session_manager.get_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    if session.mode != "examiner":
        raise HTTPException(status_code=400, detail="该会话不是模拟面试模式")

    # 优先使用会话创建时指定的配置，请求中可覆盖
    state = session.exam_state or {}
    target_position = (req.target_position or state.get("target_position") or "").strip()
    topic = (req.topic or state.get("topic") or "").strip()
    collection_name = req.collection_name or state.get("collection_name") or session.kb_id

    if not target_position or not topic:
        raise HTTPException(status_code=400, detail="请先填写目标岗位和面试方向")

    # 检索参考资料
    from app.services.retrieval import AdvancedRetriever
    retriever = AdvancedRetriever(collection_name)
    chunks = retriever.hybrid_search(topic, top_k=req.top_k)

    if not chunks:
        raise HTTPException(status_code=400, detail="当前知识库没有相关资料，请补充后重试")

    # 生成第一题
    try:
        question, expectations, reference_points, cost = generate_question(
            target_position=target_position,
            topic=topic,
            context_chunks=chunks,
            thread=[],
            question_index=1,
            follow_up_count=0,
        )
    except Exception as e:
        logger.exception("Failed to generate first exam question")
        raise HTTPException(status_code=500, detail=f"生成题目失败: {str(e)}")

    # generate_question 失败约定返回 None：报错文案不得落盘为题目
    if not question:
        raise HTTPException(status_code=502, detail="生成题目失败，请重试")

    session.exam_state = {
        "mode": "examiner",
        "target_position": target_position,
        "topic": topic,
        "collection_name": collection_name,
        "status": "asking",
        "question_index": 1,
        "current_question": question,
        "current_expectations": expectations,
        "reference_points": reference_points,
        "current_thread": [{"role": "assistant", "content": question}],
        "follow_up_count": 0,
        "max_follow_up": MAX_FOLLOW_UP,
        "scores": [],
        "weak_points": [],
        "summary": None,
        "answer_cache": {},
    }
    session.add_message("assistant", question, token_cost=dict(cost))
    session_manager.save_session(session)

    return {
        "session_id": session.session_id,
        "status": "asking",
        "question_index": 1,
        "current_question": question,
        "current_expectations": expectations,
        "follow_up_count": 0,
        "scores": [],
        "token_cost": dict(cost),
    }


@router.post("/exam/{session_id}/next")
async def exam_next(session_id: str, req: ExamNextRequest):
    """提交答案并进入追问、下一题或总结"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    state = session.exam_state or {}
    if state.get("status") == "finished":
        raise HTTPException(status_code=400, detail="面试已结束")
    if not state.get("current_question"):
        raise HTTPException(status_code=400, detail="当前没有待回答的题目")

    # action == "end"：显式结束面试，跳过反作弊与评分，直接生成总结收尾
    if req.action == "end":
        summary, summary_cost = generate_summary(
            target_position=state["target_position"],
            topic=state["topic"],
            scores=state.get("scores", []),
            weak_points=state.get("weak_points", []),
        )
        if "error" in summary:
            # 总结生成失败时降级为占位文案，避免面试收尾直接崩溃
            summary = {"raw": f"面试总结生成失败：{summary['error']}", "total_score": 0}
        state["status"] = "finished"
        state["summary"] = summary
        session.add_message("assistant", summary["raw"], token_cost=dict(summary_cost))
        session.exam_state = state
        session_manager.save_session(session)
        return {
            "session_id": session_id,
            "status": state["status"],
            "question_index": state["question_index"],
            "current_question": state["current_question"],
            "current_expectations": state["current_expectations"],
            "reference_points": state.get("reference_points", []),
            "follow_up_count": state["follow_up_count"],
            "evaluation": None,
            "summary": summary,
            "scores": state["scores"],
            "weak_points": state.get("weak_points", []),
            "token_cost": dict(summary_cost),
        }

    # give_up 记为"不知道"按 0 分作答；其余按 answer 正常作答
    if req.action == "give_up":
        answer = "不知道"
    else:
        answer = req.answer.strip()
    if not answer:
        raise HTTPException(status_code=400, detail="回答不能为空")

    # 检索参考资料：query 由 topic 与当前题目拼接；结果按 query 缓存到 exam_state，
    # 同一道题的追问期间不重复全量检索
    retrieval_query = f"{state['topic']} {state['current_question']}"
    retrieval_cache = state.setdefault("retrieval_cache", {})
    chunks = retrieval_cache.get(retrieval_query)
    if chunks is None:
        from app.services.retrieval import AdvancedRetriever
        retriever = AdvancedRetriever(state["collection_name"])
        chunks = retriever.hybrid_search(retrieval_query, top_k=req.top_k)
        retrieval_cache[retrieval_query] = chunks

    # 反作弊检测（基于同一题历史答案 + 参考资料原文）
    previous_answers = [
        m["content"] for m in state.get("current_thread", [])
        if m["role"] == "user"
    ]
    is_cheating, cheat_reason = detect_cheating(answer, previous_answers, chunks)
    if is_cheating:
        return {
            "session_id": session_id,
            "status": state["status"],
            "question_index": state["question_index"],
            "current_question": state["current_question"],
            "current_expectations": state["current_expectations"],
            "reference_points": state.get("reference_points", []),
            "follow_up_count": state["follow_up_count"],
            "evaluation": {
                "raw": f"【评分】0\n【点评】{cheat_reason}\n【补充】请独立作答，不要直接复制资料。\n【纠正】无",
                "score": 0,
                "points": [
                    {"point": p, "hit": False, "evidence": "反作弊判定未作答"}
                    for p in state.get("reference_points", [])
                ],
            },
            "summary": None,
            "scores": state.get("scores", []),
            "weak_points": state.get("weak_points", []),
            "cheating_detected": True,
            "token_cost": dict(TokenCost()),
        }

    # 答案缓存命中则直接返回
    cached = get_cached_evaluation(state, state["current_question"], answer)
    eval_result = cached
    eval_cost = TokenCost()
    if not eval_result:
        eval_result, eval_cost = evaluate_answer(
            target_position=state["target_position"],
            topic=state["topic"],
            question=state["current_question"],
            answer=answer,
            reference_points=state.get("reference_points", []),
            context_chunks=chunks,
            thread=state.get("current_thread", []),
            follow_up_count=state["follow_up_count"],
            question_index=state["question_index"],
        )
        # 仅评分成功才写缓存，避免失败结果（{'error': ...}）投毒缓存
        if "error" not in eval_result:
            set_cached_evaluation(state, state["current_question"], answer, eval_result)

    # 评分失败：不记录线程与得分，直接返回 502，前端可提示重试
    if "error" in eval_result:
        raise HTTPException(status_code=502, detail=f"评分失败：{eval_result['error']}")

    # 记录当前线程
    state["current_thread"].append({"role": "user", "content": answer})
    state["current_thread"].append({"role": "assistant", "content": eval_result["raw"]})
    session.add_message("user", answer)
    session.add_message("assistant", eval_result["raw"], token_cost=dict(eval_cost))

    # 记录得分与要点命中情况
    missed_points = [p["point"] for p in eval_result.get("points", []) if not p.get("hit")]
    state["scores"].append({
        "question": state["current_question"],
        "answer": answer,
        "score": eval_result["score"],
        "feedback": eval_result["raw"],
        "points": eval_result.get("points", []),
        "missed_points": missed_points,
    })
    # 累计薄弱环节
    state["weak_points"] = (state.get("weak_points", []) + missed_points)[-20:]

    total_cost = TokenCost(
        prompt_tokens=eval_cost.prompt,
        completion_tokens=eval_cost.completion,
        total_tokens=eval_cost.total,
    )

    # 判断是否结束（action == "end" 已在上方提前收尾，不再匹配"结束"字面量）
    should_finish = state["question_index"] >= MAX_QUESTIONS

    response_evaluation = eval_result if state["status"] != "finished" else None
    response_summary = None

    if should_finish:
        summary, summary_cost = generate_summary(
            target_position=state["target_position"],
            topic=state["topic"],
            scores=state["scores"],
            weak_points=state.get("weak_points", []),
        )
        if "error" in summary:
            # 总结生成失败时降级为占位文案，避免面试收尾直接崩溃
            summary = {"raw": f"面试总结生成失败：{summary['error']}", "total_score": 0}
        total_cost = TokenCost(
            prompt_tokens=total_cost.prompt + summary_cost.prompt,
            completion_tokens=total_cost.completion + summary_cost.completion,
            total_tokens=total_cost.total + summary_cost.total,
        )
        state["status"] = "finished"
        state["summary"] = summary
        session.add_message("assistant", summary["raw"], token_cost=dict(summary_cost))
        response_summary = summary
        response_evaluation = None
    elif eval_result["score"] < 7 and state["follow_up_count"] < state["max_follow_up"]:
        # 追问
        state["status"] = "follow_up"
        state["follow_up_count"] += 1
        question, expectations, reference_points, q_cost = generate_question(
            target_position=state["target_position"],
            topic=state["topic"],
            context_chunks=chunks,
            thread=state["current_thread"],
            question_index=state["question_index"],
            follow_up_count=state["follow_up_count"],
        )
        if question is None:
            # 出题失败：此处 raise 时会话尚未保存，已写入的评分/消息均不落盘
            raise HTTPException(status_code=502, detail="生成题目失败，请重试")
        total_cost = TokenCost(
            prompt_tokens=total_cost.prompt + q_cost.prompt,
            completion_tokens=total_cost.completion + q_cost.completion,
            total_tokens=total_cost.total + q_cost.total,
        )
        state["current_question"] = question
        state["current_expectations"] = expectations
        state["reference_points"] = reference_points
        state["current_thread"].append({"role": "assistant", "content": question})
        session.add_message("assistant", question, token_cost=dict(q_cost))
    else:
        # 下一题
        state["status"] = "asking"
        state["question_index"] += 1
        state["follow_up_count"] = 0
        state["current_thread"] = []
        question, expectations, reference_points, q_cost = generate_question(
            target_position=state["target_position"],
            topic=state["topic"],
            context_chunks=chunks,
            thread=[],
            question_index=state["question_index"],
            follow_up_count=0,
        )
        if question is None:
            # 出题失败：此处 raise 时会话尚未保存，已写入的评分/消息均不落盘
            raise HTTPException(status_code=502, detail="生成题目失败，请重试")
        total_cost = TokenCost(
            prompt_tokens=total_cost.prompt + q_cost.prompt,
            completion_tokens=total_cost.completion + q_cost.completion,
            total_tokens=total_cost.total + q_cost.total,
        )
        state["current_question"] = question
        state["current_expectations"] = expectations
        state["reference_points"] = reference_points
        state["current_thread"] = [{"role": "assistant", "content": question}]
        session.add_message("assistant", question, token_cost=dict(q_cost))

    session.exam_state = state
    session_manager.save_session(session)

    return {
        "session_id": session_id,
        "status": state["status"],
        "question_index": state["question_index"],
        "current_question": state["current_question"],
        "current_expectations": state["current_expectations"],
        "reference_points": state.get("reference_points", []),
        "follow_up_count": state["follow_up_count"],
        "evaluation": response_evaluation,
        "summary": response_summary,
        "scores": state["scores"],
        "weak_points": state.get("weak_points", []),
        "token_cost": dict(total_cost),
    }


@router.get("/exam/{session_id}")
async def exam_get(session_id: str):
    """获取面试状态"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    state = session.exam_state or {}
    return {
        "session_id": session_id,
        "status": state.get("status", "asking"),
        "question_index": state.get("question_index", 1),
        "current_question": state.get("current_question"),
        "current_expectations": state.get("current_expectations", []),
        "reference_points": state.get("reference_points", []),
        "follow_up_count": state.get("follow_up_count", 0),
        "scores": state.get("scores", []),
        "summary": state.get("summary"),
        "weak_points": state.get("weak_points", []),
    }


# ============================================================
#  知识库管理
# ============================================================

@router.get("/collections")
async def get_collections():
    """获取所有知识库列表"""
    try:
        return {"collections": list_collections()}
    except Exception as e:
        logger.exception("Failed to list collections")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/collections")
async def create_collection_api(req: CreateCollectionRequest):
    """创建知识库"""
    name = _validate_collection_name(req.name, "知识库名称")

    try:
        return create_collection(name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Failed to create collection")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/collections/{name}")
async def rename_collection_api(name: str, req: RenameCollectionRequest):
    """重命名知识库"""
    new_name = _validate_collection_name(req.new_name, "新名称")

    try:
        return rename_collection(name, new_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Failed to rename collection")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/collections/{name}")
async def delete_collection_api(name: str):
    """删除知识库"""
    try:
        return delete_collection(name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Failed to delete collection")
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================================
#  文件管理
# ============================================================

@router.get("/files")
async def list_files(collection_name: str = Query(default="knowledge_chunks")):
    """获取知识库中的文件列表"""
    try:
        files = get_files(collection_name)
        return {"files": files, "collection_name": collection_name}
    except Exception as e:
        logger.exception("Failed to list files")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/files/{file_name:path}")
async def delete_file_api(
    file_name: str,
    collection_name: str = Query(default="knowledge_chunks"),
):
    """删除知识库中的文件及向量数据"""
    # 路径安全校验在 try 之外，HTTPException 原样向上抛
    local_path = _safe_file_path(file_name)

    try:
        result = delete_file(file_name, collection_name)
    except Exception as e:
        logger.exception("Failed to delete file vectors")
        raise HTTPException(status_code=500, detail="Internal server error")

    # 同时删除本地文件；向量与本地文件都不存在时返回 404
    if local_path.exists():
        os.remove(str(local_path))
    elif result.get("deleted_count", 0) == 0:
        raise HTTPException(status_code=404, detail="文件不存在")

    return result


@router.get("/files/{file_name:path}/preview")
async def preview_file(
    file_name: str,
    collection_name: str = Query(default="knowledge_chunks"),
):
    """预览文件内容（前 5000 字符）"""
    from app.services.ingest import read_text_from_file

    local_path = _safe_file_path(file_name)
    if not local_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")

    try:
        text = read_text_from_file(local_path)
        return {
            "file_name": file_name,
            "content": text[:5000],
            "total_length": len(text),
            "truncated": len(text) > 5000,
        }
    except Exception as e:
        logger.exception("Failed to preview file")
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================================
#  统计信息
# ============================================================

@router.get("/stats")
async def get_stats():
    """获取系统统计信息"""
    try:
        collections = list_collections()
        total_chunks = sum(c["chunk_count"] for c in collections)
        total_files = 0
        for col in collections:
            files = get_files(col["name"])
            total_files += len(files)

        return {
            "collections": len(collections),
            "total_chunks": total_chunks,
            "total_files": total_files,
        }
    except Exception as e:
        logger.exception("Failed to get stats")
        raise HTTPException(status_code=500, detail="Internal server error")
