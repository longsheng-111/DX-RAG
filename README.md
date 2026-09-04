# DX-RAG · 作业本上的 RAG 冒险

> 仓库地址：https://github.com/longsheng-111/RAG-TEST

一个以「暖纸作业本 × 复古 RPG」为视觉世界观的知识库问答系统：把你的课程资料、面试八股文放进书包（知识库），然后向它提问、纠错、存档，甚至开一场 AI 考官的模拟面试。检索增强生成（RAG）负责找线索，DeepSeek 负责写答案。

## 主要玩法（已实现）

- **知识问答**：混合检索（BM25 + 向量双路召回 → RRF(k=60) 融合 → BGE 向量精排按 0.7×余弦 + 0.3×归一化 RRF 加权融合 → 0.65 阈值过滤），ReAct Agent 按 0.75 充分性阈值判定、最多 3 轮改写查询补检；回答附引用角标与推理路径，可点开查看原文片段。
- **会话存档**：多会话并行，首次提问自动生成标题，Token 用量逐条统计；会话累计超预算（默认 5 万）直接 429 熔断，上下文估算超 60% 自动压缩历史；一键导出 Markdown。
- **Persona 角色**：内置 5 位 NPC——通用面试官、前端面试官、后端面试官、面试辅导官、技术考官，各自绑定默认知识库与专属提示词。
- **引用替换与纠错反馈**：对回答引用的片段可「替换引用」或「标记不准确」，反馈落盘后会作为事实修正约束注入该会话的后续回答。
- **模拟面试（考官模式）**：AI 基于知识库出题（共 5 题）→ 逐题评分并指出得分点命中情况 → 低于 7 分自动追问（每题最多 2 次）→ 结束后生成总评；带复制粘贴反作弊检测。
- **知识库与文件管理**：多知识库创建/重命名/删除；支持 TXT/MD/CSV/JSON/LOG、PDF（图片型走视觉模型 OCR）、DOCX、XLSX 上传入库，支持预览与删除。

## 技术栈

| 层 | 技术（以代码为准） |
|---|------|
| 前端 | Next.js 14.2.24 + React 18 + Ant Design 5.22.7 + react-markdown 9 + framer-motion + axios |
| 后端 | FastAPI 0.104 + FAISS（faiss-cpu 1.7，向量检索）+ sentence-transformers 2.x + rank-bm25 |
| 文件处理 | PyMuPDF 1.26 / PyPDF2 / python-docx / openpyxl |
| 嵌入模型 | BAAI/bge-small-zh-v1.5（512 维，默认经 hf-mirror 或本地 ModelScope 缓存加载） |
| LLM | DeepSeek `deepseek-chat`（OpenAI 兼容接口） |
| 视觉模型 | 通义千问 `qwen-plus`（DashScope OpenAI 兼容接口，处理图片型 PDF） |
| 数据存储 | 本地 JSON 落盘 + FAISS 索引文件，无外部数据库 |

> 注：项目已从 ChromaDB 迁移至 FAISS，代码与 `docker-compose.yml` 均已切换为 `vector_*`/`VECTOR_*` 命名；仅个别历史文档中保留 `chroma_*` 字样作为演进记录。

## 开局指南（双端上手）

### 环境要求

| 依赖 | 版本 | 说明 |
|-----|------|------|
| Python | 3.10+ 推荐 | 当前 Mac 开发机用系统 3.9.6 也能跑，属临时方案（见 `docs/mac-dev-notes.md`） |
| Node.js | 18.17+（推荐 20 LTS） | Next.js 14 的最低要求 |
| npm | 9+ | 随 Node 安装 |

### 配置 API 密钥

```bash
cp .env.example .env   # Windows 用 copy .env.example .env
```

编辑 `.env`：

- `DEEPSEEK_API_KEY`：**必需**，知识问答与模拟面试的生成都依赖它。
- `DASHSCOPE_API_KEY`：仅当需要处理图片型 PDF（OCR）时配置。

> 提醒：`.env.example` 的向量库配置项为 `VECTOR_COLLECTION` / `VECTOR_PERSIST_DIR`（ChromaDB 时代的 `CHROMA_*` 键名已于 2026-07-20 退役，代码不再读取），一般保持默认即可。

### macOS

一键启动（推荐）：

```bash
bash start.sh
```

脚本会依次完成：检查 `.env` → 探测 Python（优先 `backend/venv_new`，其次 `backend/venv`，最后系统 Python）并安装依赖 → 安装前端依赖 → 检查/下载嵌入模型 → 同时拉起后端（8000）与前端（3000）。脚本开头已内置 `export OMP_NUM_THREADS=1`，用于规避 macOS 上 faiss-cpu 与 torch 的 OpenMP 冲突导致的段错误（细节见 `docs/mac-dev-notes.md`）。

手动启动：

```bash
# 后端
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 前端（新开一个终端）
cd frontend
npm install
npm run dev
```

### Windows

一键启动：双击 `start.bat`（或在 cmd 中执行）。脚本用 `py` 启动器查找 Python（优先 3.12），检查依赖后在两个新窗口分别拉起后端与前端。

手动启动：

```bat
:: 后端
cd backend
py -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

:: 前端（新开一个窗口）
cd frontend
npm install
npm run dev
```

### 关于仓库里的两个虚拟环境

`backend/venv/` 与 `backend/venv_new/` 是开发机本地环境，已被 `.gitignore` 排除，**不应依赖也不应提交**：

- `backend/venv/` 已损坏——它是从含中文路径的旧目录迁移来的，内部脚本 shebang 指向不存在的旧路径，直接调用会报 `bad interpreter`。
- `backend/venv_new/` 是当前 Mac 开发机实际使用的环境（Python 3.9.6 + faiss-cpu 1.7.4），`start.sh` 优先探测它。

新机器上的正确做法：**自己创建虚拟环境**（`python3 -m venv venv` 或 `py -m venv venv`），按上面的手动步骤安装 `backend/requirements.txt`。

### 启动验证

- 前端：<http://localhost:3000>
- 后端 API 文档（Swagger）：<http://localhost:8000/docs>
- 健康检查：<http://localhost:8000/api/health> 返回 `{"status": "ok"}`

首次问答前，嵌入模型（约百 MB）会自动下载；国内网络默认走 `HF_ENDPOINT=https://hf-mirror.com`，也会优先复用 `~/.cache/modelscope/BAAI/bge-small-zh-v1.5` 本地缓存。

> Docker 说明：`docker-compose.yml` 已完成 FAISS 化（卷名 `vector_data`、环境变量 `VECTOR_PERSIST_DIR` 等，并补挂了会话数据卷），`docker compose up -d` 可直接拉起前后端；首次启动需下载嵌入模型，健康检查已带 300s 宽限期。

## 冒险地图（项目结构）

```
DX-RAG/
├── backend/                        # FastAPI 后端
│   ├── app/
│   │   ├── main.py                 # 应用入口（CORS、路由注册、启动初始化）
│   │   ├── api/
│   │   │   └── routes.py           # API 路由（25 个端点，含请求校验）
│   │   ├── core/
│   │   │   ├── config.py           # 配置管理（pydantic-settings 读 .env）
│   │   │   ├── vector_store.py     # FAISS 索引 + JSON 元数据的读写
│   │   │   ├── session.py          # 会话模型与 JSON 落盘
│   │   │   └── persona.py          # 5 个内置角色及其提示词
│   │   ├── services/
│   │   │   ├── qa.py               # 问答主链路（prompt 构建、token 预算、反馈注入）
│   │   │   ├── agent.py            # ReAct 检索 Agent（思考→检索→改写，最多 3 轮）
│   │   │   ├── retrieval.py        # 混合检索（BM25 + 向量双路召回 + RRF 融合 + 向量精排加权融合）
│   │   │   ├── examiner.py         # 考官模式（出题/评分/追问/总结/反作弊）
│   │   │   ├── feedback.py         # 引用纠错反馈的存储与提示词注入
│   │   │   ├── ingest.py           # 文件解析、清洗、切片、图片型 PDF OCR
│   │   │   ├── harness.py          # 输入护栏、超时与兜底降级
│   │   │   ├── token_tracker.py    # Token 统计与上下文压缩
│   │   │   └── title_generator.py  # 会话自动标题
│   │   ├── data/                   # 运行时生成：sessions/ 与 feedback/
│   │   ├── uploads/                # 用户上传的原始文件
│   │   └── vector_db/              # FAISS 向量库（按知识库分目录）
│   ├── test_data/                  # 面试八股测试数据（frontend_bagu/backend_bagu/general）
│   └── requirements.txt
├── frontend/                       # Next.js 前端（App Router）
│   ├── app/
│   │   ├── layout.tsx              # 根布局（ConfigProvider 主题、整体框架）
│   │   ├── page.tsx                # 主页面
│   │   └── globals.css             # 设计 token 与全局样式（作业本风格基准）
│   └── components/
│       ├── QAPanel.tsx             # 知识问答面板（引用、反馈、检索来源）
│       ├── ExaminerPanel.tsx       # 模拟面试面板（黑板考场）
│       ├── SessionPanel.tsx        # 会话存档列表
│       ├── NewSessionModal.tsx     # 新建会话（开档）
│       ├── KnowledgeBaseManager.tsx# 知识库管理
│       ├── FileUpload.tsx          # 文件上传
│       ├── FileManager.tsx         # 文件管理
│       └── Sidebar.tsx             # 侧边导航
├── docs/                           # 体检报告、Mac 开发笔记、汇报材料
├── .env.example                    # 环境变量模板
├── start.sh / start.bat            # 双端一键启动脚本
└── docker-compose.yml              # 双端一键部署（已完成 FAISS 化）
```

## 存档结构（数据存储）

全部数据落在本地文件，无外部数据库；若项目路径含中文等非 ASCII 字符，数据目录会自动回退到 `~/.dx-rag/`（`backend/app/core/config.py` 的 `_safe_data_dir`）。

**1. 会话与反馈（JSON 落盘）**

- `backend/app/data/sessions/sess_<16位hex>.json`：每个会话一个文件，字段包括 `session_id / title / persona / kb_id / mode(qa|examiner) / messages / total_tokens / compressed_history / exam_state`。`messages` 中助手消息会携带 `token_cost`、`sources`、`reasoning_steps`，刷新页面后可完整恢复；`exam_state` 保存面试进度、逐题得分、`weak_points` 与总结。
- `backend/app/data/feedback/feedbacks.json`：以 `session_id` 分组的反馈数组，每条含 `id / type(replace|inaccurate) / original_chunk_id / target_chunk_id / original_content / target_content / note / created_at`。下次同会话问答时，`services/feedback.py` 会把它格式化为「事实修正约束」注入系统提示词。

**2. 向量库（FAISS + 元数据 JSON）**

```
backend/app/vector_db/
├── _collections.json               # 知识库注册表
└── <知识库名>/
    ├── index.faiss                 # FAISS 索引（IndexFlatIP，归一化向量内积 ≡ 余弦相似度）
    └── metadata.json               # 三个平行数组：documents / metadatas / ids
```

`metadata.json` 的 `metadatas[i]` 记录 `file_name / source / chunk_index / collection_name`，与 `documents[i]`、`ids[i]` 一一对应。FAISS 索引不支持按条删除，删除文件时会重建索引（`vector_store.py` 的 `delete_file`）。

**3. 原始文件**

- `backend/app/uploads/`：上传的原始文件，同名文件直接覆盖（旧向量在入库前删除）；删除文件时同步清理。

## API 一览

共 25 个端点，完整定义以 `backend/app/api/routes.py` 为准：

| 分组 | 端点 | 方法 | 功能 |
|------|------|------|------|
| 基础 | `/api/health` | GET | 健康检查 |
| 入库 | `/api/upload` | POST | 上传文件并切片入库 |
| 问答 | `/api/query` | POST | 无状态问答（兼容旧接口） |
| 问答 | `/api/chat` | POST | 会话化问答（推荐） |
| 会话 | `/api/sessions` | GET / POST | 列出 / 创建会话 |
| 会话 | `/api/sessions/{id}` | GET / DELETE | 会话详情 / 删除 |
| 会话 | `/api/sessions/{id}/clear` | POST | 清空会话消息与压缩历史（保留会话与配置） |
| 会话 | `/api/sessions/{id}/persona` | PUT | 切换角色 |
| 会话 | `/api/sessions/{id}/title` | PUT | 改标题 |
| 会话 | `/api/sessions/{id}/export` | GET | 导出 Markdown |
| 角色 | `/api/personas` | GET | 角色列表 |
| 反馈 | `/api/feedback` | POST | 提交引用纠错反馈 |
| 面试 | `/api/exam/start` | POST | 开始模拟面试 |
| 面试 | `/api/exam/{id}/next` | POST | 提交回答，进入评分/追问/下一题/总结（支持 `action: give_up\|end` 显式动作） |
| 面试 | `/api/exam/{id}` | GET | 获取面试状态 |
| 知识库 | `/api/collections` | GET / POST | 列出 / 创建知识库 |
| 知识库 | `/api/collections/{name}` | PUT / DELETE | 重命名 / 删除知识库 |
| 文件 | `/api/files` | GET | 文件列表 |
| 文件 | `/api/files/{name}` | DELETE | 删除文件及向量 |
| 文件 | `/api/files/{name}/preview` | GET | 预览文件（前 5000 字符） |
| 统计 | `/api/stats` | GET | 知识库/切片/文件计数 |

## 后续规划

### 当前里程碑 · UI 基准定型

以 `docs/STYLE_GUIDE.md`（风格基准文档，v1.0 已建档）的全面落地为本阶段完成标志。近期样式工作项来自 `docs/DX-RAG体检报告-2026-07.md` 第五章：

- **修补项（B 类）**：B1 antd 默认件露馅清零、B2 组件内重复 token 声明清剿、B3 工具类归并、B4 死代码大扫除、B5 文案修正包、B6 可访问性底线。
- **打磨项（C 类）**：C1 空态插画体系、C2 落笔动效、C3「本页完」页脚、C4 字体子集化。

目标是把「小而美」的作业本幻觉做扎实，再谈新功能。

### 功能新增

| 功能 | 阶段 | 出处 |
|------|------|------|
| 引用纠错记录（`GET /api/feedback` + 管理页，数据已落盘） | 近期（2–4 周） | 体检报告第六章 近期-1 |
| 面试评分反馈强化（得分点命中沉淀、追问对话化） | 近期 | 体检报告第六章 近期-2 |
| 面试报告（可回看/导出的成绩单） | 中期（1–2 月） | 体检报告第六章 中期-1 |
| 简历驱动的模拟面试（上传简历生成针对性提问） | 中期 | 新构想，与面试报告同线推进 |
| 错题本 / 抽认卡、学习进度仪表盘 | 中期 | 体检报告第六章 中期-2/3 |
| 桌面桌宠（快捷八股问答） | 远期构想 | 新构想，待评估形态 |
| 报告/知识库分享、协作知识库、多端 | 远期（3 月+） | 体检报告第六章 远期-1/2/3 |

### 工程化

- **完整测试流程建设**：分三层推进——先做后端 API 冒烟（24 个端点的健康检查/问答/上传主路径），再做检索质量回归（固定题库断言引用命中），最后补前端关键流程（建会话→提问→点引用→导出）。CI 以 Windows + macOS 双端 matrix 运行。流程骨架与端点用例清单已写入 `docs/TESTING.md`；框架选型已有推荐（后端 pytest + TestClient，前端 Playwright e2e 为主），引入新依赖仍在待议中。
- **数据访问层抽象与数据库迁移**：阶段 1 已完成——存储接口经 `backend/app/core/storage.py` 统一收口，新增 PostgreSQL + pgvector 后端（`backend/app/db/`，接口与文件版逐字兼容），`STORAGE_BACKEND` 环境变量整体切换（默认仍为 `file`，本机零负担）；配套 alembic 初始迁移与幂等数据迁移脚本（`backend/scripts/migrate_to_pg.py`，FAISS 向量 reconstruct 导出、零丢失校验）。设计与决策见 `docs/LangGraph重构设计.md`（决策 1：PG + pgvector，不经 SQLite 过渡）。
- **双端 CI 与其他欠账**：`start.bat` 无 venv 隔离、`.env.example` 历史键名清理，随工程化一并处理。（`docker-compose.yml` 的 FAISS 化、Token 预算熔断与 LLM 60s 超时、`scipy>=1.11` 的 Python 3.12 兼容均已完成。）

## 常见问题

- **Mac 上后端随机 `Segmentation fault: 11`**：faiss-cpu 的 OpenMP 与 torch 冲突，确保通过 `bash start.sh` 启动（已内置 `OMP_NUM_THREADS=1`），或手动 export 该变量。详见 `docs/mac-dev-notes.md`。
- **端口被占用**：`start.sh`/`start.bat` 启动前会自动清理 8000/3000 端口；手动启动时请先自行释放。
- **问答一直失败/提示未配置**：检查 `.env` 的 `DEEPSEEK_API_KEY` 是否有效；图片型 PDF 还需 `DASHSCOPE_API_KEY`。
- **想清空知识库**：在「知识库管理」页删除，或删除 `backend/app/vector_db/` 下对应子目录后重启后端。

## License

MIT
