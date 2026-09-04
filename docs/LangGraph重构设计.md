# DX-RAG 重构设计：LangGraph + PostgreSQL + 并发架构

> 版本：v1.0（2026-08-19）
> 前置文档：`docs/DX-RAG体检报告-2026-07.md`、`docs/TESTING.md`、27 路审计 bug 清单
> 目标读者：本项目的后续开发/重构执行者

---

## 1. 背景与目标

### 1.1 现状

| 层 | 现状 | 痛点（审计证据） |
|---|---|---|
| Agent 逻辑 | `agent.py` 手写 for 循环 ReAct（最多 3 轮检索/改写） | reasoning_steps 被 qa 层丢弃（P1 #2） |
| 考官模式 | routes 里 if/else 模拟状态机 | P1 #5~#9 五条连环 bug |
| 会话存储 | 每会话一个 JSON 文件 + 单例锁 | 读-改-写竞态丢消息（#14）、孤儿会话（#17） |
| 向量存储 | 磁盘 FAISS + metadata JSON，每库一目录 | 索引损坏永久封死（#11）、删除竞态（#12）、Windows 重命名失败（#10） |
| 反馈存储 | 单 JSON 文件，非原子写 | 瞬时故障清空全部历史反馈（#13） |
| 并发 | 同步 OpenAI SDK，事件循环被 LLM/摄入阻塞 | 上传卡死全站、exam_next 超前端超时 |

### 1.2 重构目标

1. **LangGraph 化**：QA 链路和考官模式改写为 StateGraph，状态显式化、可 checkpoint、可观测。
2. **数据库化**：PostgreSQL 一统业务数据、向量（pgvector）、LangGraph checkpoint，文件存储全部下线。
3. **并发安全**：全异步 IO + 任务队列 + 数据库事务，审计中所有竞态类 bug 结构性消除。

### 1.3 非目标（本期不做）

- 多租户/用户体系（仅加 API Key 鉴权，不做 RBAC）
- 分布式向量集群（pgvector 单实例足够支撑当前数据量级 < 100 万 chunk）
- 前端 UI 改版（仅配合 API 变化做最小适配 + SSE 接入）

---

## 2. 目标架构

```
frontend (Next.js)
      │ HTTP / SSE（考官与问答改为流式）
┌─────┴─────────────────────────────────┐
│ FastAPI（全异步，uvicorn 多 worker）    │
│  api/        路由：校验、鉴权、转发       │
│  deps.py     AsyncOpenAI / DB session  │
│  auth.py     API Key 校验（新增）        │
├───────────────────────────────────────┤
│ graphs/                               │
│  qa_graph.py    检索→充分性→改写→生成    │
│  exam_graph.py  出题→评分→分支→总结      │
│  state.py       QAState / ExamState    │
├───────────────────────────────────────┤
│ services/（保留并异步化）               │
│  retrieval.py   pgvector + BM25 混合    │
│  ingest.py      解析/切片（入队列执行）  │
│  llm.py         AsyncOpenAI 封装+记账   │
├───────────────────────────────────────┤
│ db/                                   │
│  models.py      SQLAlchemy 2.0 async   │
│  checkpointer   LangGraph PostgresSaver│
│  migrations/    Alembic               │
├───────────────────────────────────────┤
│ worker/（ARQ + Redis）                │
│  jobs.py        摄入、embedding、总结    │
└───────────────────────────────────────┘
         │                    │
   PostgreSQL 16 + pgvector   Redis（队列 + 缓存）
```

### 2.1 关键设计决策

**决策 1：PostgreSQL + pgvector，不经过 SQLite 过渡。**
理由：向量、业务数据、LangGraph checkpoint 同库，跨表事务一致（如「删除知识库」= 一个事务删 chunks + collection + 相关 checkpoint），开发环境用 docker-compose 起库，成本与 SQLite 相当。审计 #10~#14 五条数据类 bug 随文件存储下线自动消除。

**决策 2：会话状态 = LangGraph checkpoint（PostgresSaver）。**
不再自建 messages JSON。`thread_id = session_id`，对话历史、考官状态机进度都是图状态的一部分。「清空会话」「恢复会话」「中断续跑」「重试」统一为 checkpoint 操作——审计 #8（重试错位）在模型层面不可能发生。

**决策 3：考官模式是独立 graph，不是 QA graph 的分支。**
两者状态结构、生命周期、token 预算语义都不同。各自编译、各自 checkpoint，共享 retrieval 节点实现。

**决策 4：摄入/embedding 一律进任务队列。**
上传接口只做：存原始文件 → 入库一条 ingest_job → 返回 job_id。前端轮询/SSE 查询进度。同步阻塞事件循环（P2 高频项）从此消失。

**决策 5：BM25 保留，但与向量路解耦为并行分支。**
LangGraph 里并行节点分别召回，merge 节点做 RRF 融合。任一路失败只降级不拖死（修复审计 #3 的方式从 try/except 变为图结构上的分支容错）。

---

## 3. 数据库设计

### 3.1 业务表（SQLAlchemy 模型摘要）

```sql
-- 知识库
CREATE TABLE collections (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name          TEXT NOT NULL UNIQUE,          -- 保留现有 name 作为业务标识
    display_name  TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 上传文件（原始文件仍落盘 object storage/本地目录，DB 记元数据）
CREATE TABLE files (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    collection_id UUID NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    file_name     TEXT NOT NULL,
    storage_path  TEXT NOT NULL,                  -- 原始文件路径
    file_hash     TEXT NOT NULL,                  -- sha256，同名重传判重
    size_bytes    BIGINT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'pending', -- pending/processing/done/failed
    error         TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (collection_id, file_name)
);

-- 切片 + 向量
CREATE TABLE chunks (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_id       UUID NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    collection_id UUID NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    chunk_index   INT NOT NULL,
    content       TEXT NOT NULL,
    embedding     vector(1024),                   -- bge-large-zh 维度，按实际模型调整
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_chunks_collection ON chunks (collection_id);
CREATE INDEX idx_chunks_embedding ON chunks
    USING hnsw (embedding vector_cosine_ops);

-- 会话（消息进 LangGraph checkpoint，此处只存元数据）
CREATE TABLE sessions (
    id            TEXT PRIMARY KEY,               -- sess_xxx 兼容现有格式
    title         TEXT NOT NULL DEFAULT '新会话',
    persona       TEXT NOT NULL DEFAULT 'default',
    collection_id UUID REFERENCES collections(id),
    mode          TEXT NOT NULL DEFAULT 'qa',      -- qa / exam
    total_tokens  BIGINT NOT NULL DEFAULT 0,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 反馈（替代 feedback.json）
CREATE TABLE feedbacks (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id    TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    message_id    TEXT,                            -- checkpoint 中的消息标识
    rating        TEXT NOT NULL,                   -- like / dislike
    comment       TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 摄入任务
CREATE TABLE ingest_jobs (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_id       UUID NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    status        TEXT NOT NULL DEFAULT 'queued',  -- queued/running/done/failed
    progress      REAL NOT NULL DEFAULT 0,
    error         TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at   TIMESTAMPTZ
);
```

LangGraph 的 `PostgresSaver` 自建 `checkpoints` / `checkpoint_writes` 等表，不在此列出。

### 3.2 与现状的映射

| 现状（文件） | 迁移后 |
|---|---|
| `data/sessions/sess_*.json` | `sessions` 表（元数据）+ checkpoints 表（消息/状态） |
| `vector_db/<collection>/index.faiss + metadata.json` | `chunks.embedding` + 业务列 |
| `vector_db/collections.json` | `collections` 表 |
| `data/feedback.json` | `feedbacks` 表 |
| 上传目录散文件 | `files` 表 + 统一 `storage/` 目录 |

---

## 4. LangGraph 设计

### 4.1 QA Graph（`graphs/qa_graph.py`）

现有 `agent.py` 的 256 行循环 → 显式状态图：

```
                ┌──────────────┐
                │  guard_input │（harness 校验，保留）
                └──────┬───────┘
                       ▼
              ┌─────────────────┐
              │ retrieve_parallel│──┬── vector_search（pgvector）
              │  （并行分支）     │──┴── bm25_search（保留现有实现）
                └──────┬───────┘
                       ▼
              ┌─────────────────┐
              │   merge_rrf     │ 融合 + 去重（修复 RRF 键碰撞：
              │                 │ content[:300] → chunk id）
               └──────┬───────┘
                      ▼
             ┌──────────────────┐
             │ judge_sufficiency │── 充分 ──► ┌──────────┐
             │ （阈值+rerank 语义 │            │ generate │
             │  修复审计 #4）     │            └────┬─────┘
             └──────┬───────────┘                 ▼
                    │ 不充分且未达上限        ┌──────────┐
                    ▼                        │  END     │
             ┌─────────────┐                 └──────────┘
             │ rewrite_query│── 回到 retrieve_parallel
             │ （含循环检测） │
             └─────────────┘
```

**State 定义：**

```python
class QAState(TypedDict):
    question: str
    queries: Annotated[list[str], operator.add]      # 天然记录改写链
    results: list[RetrievalResult]                    # (chunk_id, file_name, content, score)
    reasoning_steps: Annotated[list[dict], operator.add]  # 修复审计 #2：图状态透传到底
    iterations: int
    answer: str
    sources: list[dict]
    token_cost: TokenCost
    confidence: Literal["high", "medium", "low"]
```

要点：

- 循环上限、循环检测（Jaccard 0.8）逻辑原样搬入条件边，语义不变。
- `reasoning_steps` 作为 state 字段随 checkpoint 持久化，前端「推理路径」直接可用。
- rerank 回退不再以 RRF 分冒充余弦分：RetrievalResult 增加 `score_type` 字段，充分性判断只认 `cosine`。

### 4.2 Exam Graph（`graphs/exam_graph.py`）

考官模式状态机显式化：

```
 start_exam ──► generate_question ──► wait_answer（interrupt，等待用户输入）
                                          │
                                          ▼
                               ┌──────────────────────┐
                               │ classify_action      │ answer / end / 无效输入
                               └───┬───────┬──────────┘
                            answer │       │ end
                                   ▼       ▼
                          ┌─────────────┐ ┌────────────┐
                          │evaluate_answer│ │generate_summary│──► END
                          └──────┬──────┘ └────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │ branch: 追问 / 下一题 / 完成 │──► generate_question（环）
                    └─────────────────────────┘
```

要点：

- `wait_answer` 用 LangGraph `interrupt()` 实现，用户作答即 `Command(resume=...)` 恢复——**不再有"结束面试被当答案评分"（#7）的可能**：action 由分支节点判断，不混在 answer 文本里。
- LLM 故障在 `generate_question` 节点内抛异常 → 图返回错误状态，**不会把报错文案落盘为题目**（#6）。
- token 记账收敛到图节点的统一 wrapper，`add_message` 自动累加入口只有一个，双倍计数（#5）消除。
- 评分缓存（`get_cached_evaluation`）保留，挂到 state 而非 dict。

### 4.3 Checkpointer

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

checkpointer = AsyncPostgresSaver(pool)  # asyncpg 连接池
qa_graph = build_qa_graph().compile(checkpointer=checkpointer)
```

- `thread_id` = 现有 `sess_xxx`，前端无感迁移。
- 会话列表、重命名、删除走 `sessions` 表；消息历史从 checkpoint 读。

---

## 5. 并发设计

| 问题 | 方案 |
|---|---|
| 同步 LLM 调用阻塞事件循环 | 全链路 `AsyncOpenAI`；CPU 密集（解析 PDF、embedding）进 worker |
| 上传/摄入长任务 | ARQ 队列；`/upload` 只入库 job 并返回 id；`/ingest-jobs/{id}` 查进度（SSE 可选） |
| 会话读-改-写竞态（#14） | 消息写入由 checkpointer 事务保证；元数据更新用 `SELECT ... FOR UPDATE` |
| 并发删文件/检索竞态（#12） | 删除 = 单事务 `DELETE FROM chunks WHERE file_id=...`，无文件系统中间态 |
| 上传限额绕过（#15） | Nginx/uvicorn 层 `client_max_body_size` + 应用层流式计数双重限制 + API Key 鉴权 |
| exam_next 超前端 90s | 改 SSE：`exam_next` 返回 stream，节点逐个推送（thinking/question/score） |
| LLM 并发限流 | `asyncio.Semaphore`（全局可配，默认 8）+ DeepSeek 429 指数退避重试 |

连接池基线：asyncpg pool `min=2, max=20`；uvicorn 2 workers 起步，压测后调。

---

## 6. 数据迁移方案

一次性迁移脚本 `backend/scripts/migrate_to_pg.py`（幂等，可重复跑）：

1. **collections.json** → `collections` 表；name 冲突时加后缀并记录映射表。
2. **每知识库 metadata.json + FAISS** → 逐条 insert `files` / `chunks`；**embedding 不重新计算**，直接从 FAISS `reconstruct` 导出（bge 向量可逆读出），失败才重 embed。
3. **sessions/*.json** → `sessions` 表 + 合成初始 checkpoint（messages 转为 LangGraph 消息序列）。
4. **feedback.json** → `feedbacks` 表。
5. 迁移后旧目录改名为 `data_legacy_YYYYMMDD/` 保留一个版本周期。

回滚策略：迁移期代码同时保留文件存储只读 fallback 一个版本，异常时可切回。

---

## 7. 分阶段路线与验收标准

### 阶段 0：止血（1~2 天）

修会挡重构路的审计 bug：#1（3.9 启动）、#5/#6/#7（考官三连）、#15（上传限额+鉴权占位）。

**验收**：Python 3.9 import 冒烟通过；考官模式演示路径无 P1；上传超限返回 413。

### 阶段 1：数据库化（~1 周）

- Alembic 建表 + docker-compose 加 postgres/redis 服务
- `SessionManager` → `SessionRepository`（接口签名不变，实现换 DB）
- `vector_store.py` → `PgVectorStore`（对外 `hybrid_search` 签名不变）
- 迁移脚本 + 数据校验报告（条数比对）

**验收**：全部现有 API 行为不变过现有测试；审计 #10~#14 对应回归用例通过（并发删文件、损坏恢复、重命名）；迁移脚本对现有数据跑出零丢失报告。

### 阶段 2：QA LangGraph 化（~1 周）

- `qa_graph` 实现 + 与旧 `agent_query` 的双跑对比（同问题两边结果 diff）
- reasoning_steps 透传到前端（#2 修复落地）
- retrieval 并行分支 + 单路降级（#3、#4 修复落地）

**验收**：双跑 50 个问题集，答案/sources 一致率 ≥ 90%（差异人工复核为可解释）；前端「推理路径」面板实际渲染；嵌入模型不可用时 BM25 路正常返回。

### 阶段 3：考官 LangGraph 化 + SSE（~1 周）

- `exam_graph` + interrupt/resume
- 前端 exam 面板接 SSE，废弃轮询
- token 预算统一进图节点 wrapper，考官接入预算检查（审计 P2 项）

**验收**：审计 #5~#9 五条全部有对应端到端测试且通过（模拟 LLM 报错、重试、追问、结束、进度一致性）；exam_next 全程无前端超时。

### 阶段 4：并发加固与部署（按需，~1 周）

- ARQ worker 上线，摄入全异步
- API Key 鉴权全端点
- 压测（locust）：20 并发问答 p95 延迟报告
- Dockerfile/compose 修复（审计 Docker 三项）

**验收**：20 并发下无消息丢失/串会话；上传大 PDF 期间 /chat 正常响应。

---

## 8. 风险与取舍

| 风险 | 应对 |
|---|---|
| LangGraph 版本迭代快，API 变动 | 锁定 langgraph 版本；graph 构建集中在一个 factory 文件，升级只改一处 |
| pgvector 性能不如 FAISS（大规模） | 当前量级（<100 万 chunk）HNSW 足够；留 `VectorStore` 抽象，超量后可插拔 Qdrant |
| 双跑期（阶段 2）token 双倍成本 | 双跑只在内测环境开，采样 50 题后即切 |
| checkpoint 表膨胀 | 定期清理已删除会话的 checkpoint；`sessions` 删除走 cascade 任务 |
| 前端 SSE 改造工作量 | QA 先保持一次性返回，仅考官上 SSE；QA 流式列为阶段 4 可选项 |

---

## 9. 审计 bug 消除对照（摘）

| 审计项 | 消除方式 |
|---|---|
| #2 reasoning_steps 断链 | 图 state 字段 + checkpoint 透传 |
| #3 嵌入故障拖死检索 | 并行分支降级 |
| #5~#9 考官五连 | exam_graph 状态机 + interrupt + 统一记账 |
| #10~#14 文件存储竞态 | 文件存储下线，DB 事务 |
| #15 上传绕过 | 双重限额 + 鉴权 |
| 上传阻塞事件循环 | ARQ 队列 |
