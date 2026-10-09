# DX-RAG 测试流程化专项规划

> 规划日期：2026-07-17 · 代码基线：当前工作区（frontend Next.js 14.2.24 + antd 5.22.7 / backend FastAPI + FAISS）
> 本文档是**纯规划文档**：不含任何测试代码实现，只给出方案、清单与可直接复制的流程骨架。所有端点、路径、行号、依赖版本均已对照实际文件核实。编号引用（如 2.3、B4）指向《docs/DX-RAG体检报告-2026-07.md》。原定 6 项待定已于 **2026-07-17 全部拍板**，决策记录见文末附录，正文已同步定稿。

---

## 1. 现状（已全部核实）

| 事实 | 证据 |
|------|------|
| 后端无任何测试框架：`requirements.txt` 共 19 项依赖，无 pytest、无 httpx、无任何测试库 | `backend/requirements.txt:1-19` |
| 后端无测试目录：`backend/tests/` 不存在 | `ls backend/tests` → No such file or directory |
| 后端唯一的"类测试"文件 `scripts/ingest.py` 是**批量文档导入脚本**（docstring 自述"多源批量文档导入脚本……主要用于初始化知识库或批量更新八股文资料"），不是测试 | `backend/scripts/ingest.py:1-14` |
| 前端无 `test` 脚本：scripts 仅有 `dev` / `build` / `start` | `frontend/package.json:5-9` |
| 前端无任何测试框架：dependencies 与 devDependencies 中无 vitest / jest / @testing-library / playwright | `frontend/package.json:10-37` |
| 无 CI：仓库无 `.github/` 目录，无任何工作流配置 | `ls .github` → No such file or directory |
| 前端无 lint 配置：`package.json` 无 eslint 依赖、无 `lint` 脚本；`frontend/` 下无 `.eslintrc*` 文件；`next.config.js` 无 eslint 相关配置（Next.js 自带的 `next lint` 因缺 eslint 依赖无法直接运行） | `frontend/package.json:29-37`、`frontend/next.config.js:1-14` |
| 后端无 lint 工具：`requirements.txt` 中无 ruff / flake8 / pylint / black / mypy | `backend/requirements.txt:1-19` |

结论：项目当前**零自动化测试、零 CI、零 lint**。回归保障完全靠人工点页面。体检报告中列出的死代码（2.4）、静默吞错（3.4）、行为契约不清（如下文 2.1 的删除知识库案例）等问题，都是测试缺位下才会长期存活的问题。

---

## 2. 分层设计方案

### 2.1 后端 API 测试（pytest + FastAPI TestClient）

**选型**：`pytest` + `fastapi.testclient.TestClient`（基于 starlette，运行时需要 `httpx`，见第 4 章）。应用入口现成：`app/main.py:17-48` 的 `create_app()` 工厂可直接被测试导入，无需起 uvicorn 进程。路由集中于单文件 `backend/app/api/routes.py`（948 行，24 个端点，与体检报告附注一致），适合按端点分组组织测试文件。

**测试目录规划**（规划，未创建）：

```
backend/tests/
├── conftest.py            # 公共 fixture：隔离数据目录、TestClient、mock LLM/embedding
├── test_health.py
├── test_upload.py
├── test_sessions.py
├── test_chat.py
├── test_feedback.py
├── test_exam.py
├── test_collections.py
├── test_files.py
└── test_stats.py
```

**端点分组清单与关键 case**（24 个端点，行号已逐一核实）：

| 分组 | 端点（位置） | 应覆盖的关键 case |
|------|-------------|-------------------|
| health | `GET /health`（routes.py:154） | 正常路径：返回 `{"status": "ok"}`，作为冒烟基线 |
| upload | `POST /upload`（:163） | ① 正常 md/txt 上传入库（mock 入库）；② 不传文件 → 400"未选择文件"（:169-170）；③ 非法/不支持格式经 `process_file` 抛 `ValueError` → 400（:197-200，支持后缀集合参考 `scripts/ingest.py:30`）；④ 非法知识库名（<2 字符、特殊字符）→ 400（:60-67 校验链） |
| query（旧接口） | `POST /query`（:217） | ① 空问题/纯空白 → 400（:220-221）；② 正常提问（mock `qa_query`）；③ `qa_query` 抛异常 → 500（:230-232） |
| sessions（7 个） | `POST /sessions`（:261）、`GET /sessions`（:300）、`GET /sessions/{id}`（:311）、`DELETE /sessions/{id}`（:324）、`PUT /sessions/{id}/persona`（:332）、`PUT /sessions/{id}/title`（:344）、`GET /sessions/{id}/export`（:356） | ① 创建→列表→详情闭环；② 操作不存在的 session_id → 各端点的错误码契约；③ 非法 persona 值；④ 导出返回 Markdown 附件结构 |
| personas | `GET /personas`（:401） | 正常路径：返回人设列表且含 default |
| chat | `POST /chat`（:407） | ① 空问题 → 400（:410-411）；② **缺 `session_id` 是正常路径而非错误**：`ChatRequest.session_id` 默认为 `None`（:93-98），缺省或传入不存在的 id 都会自动新建会话（:413-424）——断言响应含新 `session_id`；③ mock `qa_query` 断言响应结构含 `answer/sources/token_cost/reasoning_steps`（:498-509）；④ 首轮提问触发标题生成（:483-488）；⑤ `qa_query` 抛异常 → 500（:467-469） |
| feedback | `POST /feedback`（:512） | ① 正常提交 `type=replace` / `inaccurate`（必填字段见 :140-147）；② 缺必填字段 → 422（pydantic 校验）；③ 落盘失败 → 500"保存反馈失败"（:526-528） |
| exam（3 个） | `POST /exam/start`（:535）、`POST /exam/{session_id}/next`（:612）、`GET /exam/{session_id}`（:799） | ① 缺 `session_id` → 422（`ExamStartRequest` 必填，:127-132）；② 对不存在会话 start；③ next 提空答/"不知道"/"结束"（体检报告 3.7 指出这是魔法字符串协议；**协议化已定案执行**：前后端同改，用例须与协议变更同 PR 同步提交，见附录裁决⑥；改造落地前按现行为契约固化）；④ `GET /exam/{id}` 恢复已持久化的 `exam_state` |
| collections（4 个） | `GET /collections`（:824）、`POST /collections`（:833）、`PUT /collections/{name}`（:846）、`DELETE /collections/{name}`（:859） | ① 创建→列表→重命名→删除闭环；② 名称非法（长度/字符）→ 400（:60-67）；③ 重名创建 → 400（:838-841）；④ 重命名不存在的库 → 400（服务层 `ValueError`，`vector_store.py:199-200` → routes.py:853-855）；⑤ **删除不存在的库 → 目标契约：404 + 统一错误体**（已定案，附录裁决①；前端按错误卡模式处理）：现状是 `delete_collection` 幂等、不存在的名字静默成功返回 200 `deleted`（`vector_store.py:182-193`）——本用例为**目标契约驱动的预期失败用例**，契约修复落地前先标预期失败（xfail），修复后转绿 |
| files（3 个） | `GET /files`（:874）、`DELETE /files/{file_name:path}`（:884）、`GET /files/{file_name:path}/preview`（:903） | ① 列表正常路径；② 路径穿越文件名（`../`、`a/b`）→ 400/403（`_safe_file_path`，:70-80）；③ 预览不存在文件 → 404（:911-913）；④ 预览截断逻辑（>5000 字符 `truncated=true`，:917-922） |
| stats | `GET /stats`（:931） | 正常路径：响应含 `collections/total_chunks/total_files` 计数结构（:942-946） |

**外部依赖 mock / 隔离方案**（按依赖实际调用点核实）：

1. **DeepSeek LLM**——`OpenAI` 客户端创建点共 6 处：`qa.py:91-93`、`agent.py:23-25`、`examiner.py:100`、`title_generator.py:21-23`、`token_tracker.py:139`，以及 `routes.py:437-443` 的上下文压缩处，模型均为 `deepseek-chat`。
   方案：**在路由边界 mock，不 mock OpenAI 客户端本身**。`routes.py` 顶部导入了 `qa_query` 等服务函数，测试用 `monkeypatch`/`unittest.mock.patch` 替换 `app.api.routes.qa_query`、`generate_session_title` 等符号为返回固定字典的假实现。这样测试覆盖路由层的参数校验、会话状态流转、响应组装，而不碰网络。上下文压缩分支（:436-445）默认不触发（需 token 超 60% 预算），单测可用低预算配置强制触发后单独验证。
2. **DashScope OCR**——仅 `services/ingest.py:81-87` 用于图片型 PDF。upload 测试用纯文本/md 样本即可绕开；PDF 路径单测直接 mock `process_file`。
3. **Embedding 模型**——`vector_store.py:31-47` 单例 `get_embed_model()` 加载 `BAAI/bge-small-zh-v1.5`（`config.py:86`），首次运行需联网下载（`vector_store.py:12` 注释已配 HF 镜像）。方案：API 测试层 mock `encode_texts`/`add_texts`，**不加载真模型**；真模型只出现在 2.2 的检索回归测试中。
4. **文件系统/数据目录**——`settings` 是模块级单例（`config.py:124`），数据目录为 `vector_persist_dir`/`session_dir`/`upload_dir`（`config.py:80-89`）。`conftest.py` 中用 pytest 的 `tmp_path` 为每个测试会话替换这三个目录，保证测试不写开发数据、用例间互不污染。

### 2.2 检索质量回归测试

**目的**：RAG 系统的核心质量不在 API 层，而在"改完 embedding 模型 / FAISS 索引参数 / 切分参数后，检索结果有没有悄悄变差"。这类退化 API 测试（全部 mock 检索）完全发现不了，需要一套**用真模型、真索引、固定问句集**的回归测试。

**语料与问句集**：仓库已有现成语料 `backend/test_data/`：`frontend_bagu/vue.md`、`backend_bagu/redis.md`、`general/note.md`（已逐一核实存在）。规划在此基础上为每个分类补充人工编写的问句集，每句标注"期望命中的来源文件/关键片段"。例如基于 `vue.md` 出 5-8 道 Vue 八股问句，基于 `redis.md` 出 5-8 道 Redis 问句。

**基线文件组织**（规划示例）：

```
backend/tests/regression/
├── corpus -> ../../test_data/        # 复用现成语料，不复制
├── questions.json                     # 问句集 + 期望
└── baseline.json                      # 首次运行生成的基线快照
```

`questions.json` 骨架（示例格式，可直接照此组织）：

```json
[
  {
    "id": "fe-001",
    "question": "Vue 的响应式原理是什么？",
    "expect_source_file": "vue.md",
    "expect_keywords": ["响应式"],
    "min_rank": 5
  },
  {
    "id": "be-001",
    "question": "Redis 持久化有哪几种方式？",
    "expect_source_file": "redis.md",
    "expect_keywords": ["RDB", "AOF"],
    "min_rank": 5
  }
]
```

`baseline.json` 记录首次（或基线更新时）每句问句的 top-k 命中文件与分数，供 diff 审查。

**断言策略——命中率阈值，而非精确排序**：

- 核心断言是 **Hit@k**：每句问句的 top-k（建议 k=5，与 `QueryRequest.top_k` 默认值一致，routes.py:88）命中结果中必须包含期望来源文件的 chunk。允许排名浮动，不允许掉出 top-k。
- 整体断言是**命中率下限**：如"全量问句 Hit@5 ≥ 90%，且任何单句不得从基线的命中变为未命中"。防止某次改动平均指标没变、但个别问句退化。
- **不断言精确排序与精确分数**：embedding 模型版本、FAISS 版本、numpy 版本的微小变化都会扰动分数，精确断言会产生大量误报。
- 测试入口走服务层 `AdvancedRetriever.hybrid_search`（`retrieval.py:151`）或 `qa_query` 的检索段，覆盖向量检索（`retrieval.py:85`）、BM25、RRF 融合（:91）、重排与阈值过滤（:119-149，阈值 `settings.retrieval_similarity_threshold=0.65`，`config.py:103`）这条真实链路。
- 基线更新流程：改动 embedding 模型 / 切分参数（`config.py:92-93` 的 `max_chunk_size=800`/`chunk_overlap=120`）/ 索引结构时，人工重新生成 `baseline.json` 并随 PR 提交，diff 即检索变化说明。
- 问句集需圈定固定 **10 句以内**的 `small` 子集（pytest mark），供 CI 的 PR 形态运行（决策③，见 3.1）；全量问句集由定时/手动形态运行。

**CI 注意事项**：该测试需要真实下载 embedding 模型（约百 MB 级），CI 中用 `actions/cache` 缓存 HuggingFace 模型目录。CI 形态已定案为**双形态**（附录裁决③）：PR 必跑小规模基线（固定 10 问句以内，保速度），`schedule` 定时 + `workflow_dispatch` 手动触发跑全量，骨架见 3.1。

### 2.3 前端关键流程测试

**方案对比与推荐**：

| 维度 | vitest + @testing-library/react（组件测试） | Playwright（e2e） |
|------|------------------------------------------|-------------------|
| 运行成本 | 快，秒级，PR 必跑可承受 | 慢，需起前后端，分钟级 |
| 对现有代码的侵入 | 大——组件需可独立渲染，依赖需可 mock | 零侵入，从浏览器视角驱动 |
| 覆盖"闭环"能力 | 弱，跨组件联动要拼 mock | 强，提问→回答→引用→反馈是真链路 |
| 与本项目契合度 | 受 QAPanel 巨型组件制约（见下） | 页面少、流程集中，用例数量可控 |

**推荐：Playwright e2e 为主，vitest 组件测试按需补小块纯逻辑**。理由：本项目要守住的是**跨组件的用户闭环**（提问→回答渲染→引用角标联动→反馈提交；新建会话），这恰是组件测试最弱、e2e 最强的部分；且前端无 `lint`/`typecheck` 门禁，e2e 顺带充当了最基本的构建冒烟。**引入节奏按决策④**：vitest 待 QAPanel 拆分后引入，playwright 暂缓、后续单独立项——立项前下文的 e2e 用例只停留在规划层，不落地。

**现实制约：QAPanel.tsx 2605 行单文件**（已 `wc -l` 核实，与体检报告 2.3 一致）。其中约 940 行 CSS 内联、5 个子组件、28 个 useState（体检报告 2.3）。这意味着：

- 想对"引用角标联动""CitationPopup 两态浮层"（`QAPanel.tsx:249-317`）做**组件级**单测，必须先做体检报告 2.3 的拆分，否则测试要渲染整个 2605 行组件并 mock 其全部依赖，维护成本高于收益。**该拆分前置已批准**（附录裁决⑤：按体检报告 2.3 纯结构拆分，不改逻辑）。
- 因此在拆分完成前，前端测试**只做 e2e 规划，不做组件单测**；拆分完成后随 vitest 引入再补组件测试（决策④），组件测试只挑天然独立的小件。

**先行测试清单**（e2e，按优先级）：

1. **提问闭环**：新建会话（NewSessionModal）→ 输入问题发送 → 回答渲染（含 Markdown）→ 引用角标出现 → 点击角标弹出 CitationPopup → 提交一条反馈（"不准确"）→ "已反馈"印章出现。这一条覆盖产品主链路，对应体检报告第 6 章近期-1 的反馈链路（`POST /api/feedback`，routes.py:512）。
2. **新建会话流程**：打开 NewSessionModal → 选 persona/知识库 → 创建 → 会话出现在 SessionPanel 列表。
3. **会话管理**：重命名、删除会话（含删除确认）。
4. **知识库管理**：创建知识库 → 上传文件（mock 或小样本）→ 文件出现在列表。
5. **后端故障态**：关掉后端，页面出现错误提示而非"永远的空态"——此用例直接守护体检报告 3.4（错误处理静默化）的修复成果。

vitest 组件测试的候选（**待 QAPanel 拆分后随 vitest 引入再做**，决策④）：`NewSessionModal` 的表单校验、`SessionPanel` 的列表渲染、以及任何新抽出的纯函数 hook（如体检报告 2.6 建议的 `useCollections`）。

---

## 3. 流程化

### 3.1 GitHub Actions 工作流骨架

以下为可直接复制到 `.github/workflows/ci.yml` 使用的骨架（目录当前不存在，属规划）。`backend-test` 与 `frontend-test` 双 job 并行，各自做 OS matrix；Node 20 与 Python 3.11 版本取自仓库 Dockerfile（`backend/Dockerfile:1` `python:3.11-slim`、`frontend/Dockerfile:1` `node:20-alpine`）。`retrieval-regression` 按决策③定为**双形态**：PR 触发跑小规模基线（固定 10 问句以内，保速度），`schedule` 定时（cron，每周一凌晨）+ `workflow_dispatch` 手动触发跑全量。

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
  schedule:
    - cron: "0 17 * * 0"   # UTC 周日 17:00 = 北京时间每周一 01:00，跑全量回归
  workflow_dispatch:        # 手动触发，跑全量回归

jobs:
  backend-test:
    name: backend (pytest) - ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [windows-latest, macos-latest]
    defaults:
      run:
        working-directory: backend
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"   # 与 backend/Dockerfile 的 python:3.11-slim 对齐
          cache: pip
          cache-dependency-path: backend/requirements.txt

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest httpx pytest-mock   # 测试专用依赖，见本文档第 4 章

      - name: Run API tests (LLM / embedding 全部 mock)
        run: pytest tests/ -v --ignore=tests/regression
        env:
          DEEPSEEK_API_KEY: ""    # 测试不联网，mock 掉全部外部调用
          DASHSCOPE_API_KEY: ""

  frontend-test:
    name: frontend - ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [windows-latest, macos-latest]
    defaults:
      run:
        working-directory: frontend
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: 20        # 与 frontend/Dockerfile 的 node:20-alpine 对齐
          cache: npm
          cache-dependency-path: frontend/package-lock.json

      - name: Install dependencies
        run: npm ci

      # 现状：无 lint/typecheck/test 脚本，先用 build 当冒烟门禁
      - name: Build (smoke)
        run: npm run build

      # 第三步落地后启用（见本文档第 5 章）：
      # - name: Component tests
      #   run: npx vitest run
      # - name: E2E tests
      #   run: npx playwright test

  # 检索回归（第 2.2 节）：需下载真实 embedding 模型，独立 job。
  # 双形态（决策③）：PR 必跑小规模基线（固定 10 问句以内，用 pytest mark 圈定 small 子集，保速度）；
  # schedule（每周一凌晨）/ workflow_dispatch 手动触发跑全量问句集。
  retrieval-regression:
    name: retrieval regression
    runs-on: macos-latest
    defaults:
      run:
        working-directory: backend
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Cache embedding model
        uses: actions/cache@v4
        with:
          path: ~/.cache/huggingface
          key: hf-model-bge-small-zh-v1.5
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest httpx pytest-mock
      # PR / main push 形态：小规模基线，固定 10 问句以内
      - name: Run small baseline (pull_request / push)
        if: github.event_name == 'pull_request' || github.event_name == 'push'
        run: pytest tests/regression/ -v -m small
      # 定时 / 手动形态：全量问句集
      - name: Run full regression (schedule / workflow_dispatch)
        if: github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'
        run: pytest tests/regression/ -v
```

### 3.2 提交前本地钩子（已定案：pre-commit）

| 方案 | 机制 | 优点 | 缺点 |
|------|------|------|------|
| husky + lint-staged | npm 包管理 git hooks，只对暂存文件跑命令 | 前端生态主流、增量检查快、配置 JSON 化 | 只管 Node 侧；后端 Python 检查要绕一道；本仓前后端双栈，需两处配置 |
| 原生 git hooks 脚本 | `.git/hooks/` 或 `core.hooksPath` 指向仓内脚本 | 零依赖、前后端命令随便编排 | 钩子不进版本库（需 `core.hooksPath` 技巧）、Windows 下 shell 兼容性要处理、无增量文件筛选需手写 |
| pre-commit 框架 | Python 包，`.pre-commit-config.yaml` 声明式管理 | 前后端通吃、钩子进版本库自动安装、自带增量文件传递 | 引入 Python 工具链依赖；前端开发者机器上需装 Python（本仓后端本来就要 Python，影响小） |

**已定案：pre-commit**（2026-07-17 拍板，附录裁决②）。**理由＝前后端混合仓库一套配置同时管 Python 与 JS**：本仓提交时需要同时跑 Python（pytest 快用例、ruff）与 Node（eslint、vitest）检查，pre-commit 是三者中唯一能声明式统一编排双栈、且钩子配置随仓库分发的方案；husky 双份配置与原生 hooks 的分发问题在本仓都更痛。上方对比表保留作选型依据存档。

注意：**前置条件是 lint 先落地**（见 3.3）——当前前后端均无 lint 可跑，钩子第一阶段只能挂 `pytest`（后端）和 `npm run build`（前端，慢，建议只挂 CI 不挂本地）。

### 3.3 lint 现状与补齐顺序

- 前端：无任何 eslint 配置（证据见第 1 章）。建议随测试第三步一并补 `eslint-config-next`（Next.js 官方配套，与 Next 14.2.24 匹配），先开 `next/core-web-vitals` 默认规则集，不追求一次过严。
- 后端：无 lint 工具。建议 `ruff`（单工具覆盖 flake8+isort，速度快，配置 5 行内）。其死代码检测可直接固化体检报告 2.4/2.7（未用 import、`shutil` 残留等）的修复成果。
- lint 依赖已随 pre-commit 定案一并【已批准】（见第 4 章批准表）。

---

## 4. 依赖分批批准表（2026-07-17 拍板）

以下全部是**工程必需依赖**而非业务依赖。理由共性：项目当前零测试零 CI（第 1 章），引入它们是落地本规划的最小集合，无替代存量方案。批准状态按 2026-07-17 拍板结果（附录裁决②④）分三批：**现在引入 / 待 QAPanel 拆分后引入 / 暂缓**。

| 包 | 侧 | 用途 | 批准状态 | 为什么是工程必需 |
|----|----|------|----------|------------------|
| `pytest` | 后端 | 测试运行器 | 【已批准·现在引入】 | 后端测试的事实标准；无它则第 2.1/2.2 节无从谈起 |
| `httpx` | 后端 | TestClient 运行时依赖 | 【已批准·现在引入】 | `fastapi.testclient.TestClient`（starlette 提供）内部基于 httpx；当前 `requirements.txt` 未显式声明，测试依赖应显式锁定而非依赖传递引入 |
| `pytest-mock` | 后端 | mock LLM/embedding 的语法糖 | 【已批准·现在引入】 | 第 2.1 节 mock 方案的核心工具；**可降级为标准库 `unittest.mock`，是否引入由实施时决定**——pytest-mock 与 fixture 体系更顺，但非硬性必需 |
| `vitest` | 前端 | 组件测试运行器 | 【待 QAPanel 拆分后引入】 | 与 React 18 + TS 5 生态契合、启动快；jest 在本仓无存量优势 |
| `@testing-library/react` + `jsdom` | 前端 | 组件渲染与 DOM 断言 | 【待 QAPanel 拆分后引入】 | vitest 跑 React 组件测试的标准组合，三者必须同时引入 |
| `playwright` | 前端 | e2e 测试 | 【暂缓·后续单独立项】 | 承载第 2.3 节全部先行用例（提问闭环等）；e2e 是本规划前端部分的推荐主力，立项后启动 |
| `pre-commit` | 双栈 | 本地提交钩子 | 【已批准】 | 3.2 节已定案：前后端混合仓库一套配置同时管 Python 与 JS（husky 方案弃选） |
| `ruff` / `eslint-config-next` | 双栈 | lint | 【已批准】 | 3.3 节；随 pre-commit 定案一并批准（钩子上需有 lint 可跑），在对应侧测试落地时同步引入，不单独占步 |

---

## 5. 推进顺序与验收标准

不要求一步到位，三步各自独立可交付、独立有验收。

| 步骤 | 内容 | 验收标准 | 用例数量级 | 工作量 |
|------|------|----------|-----------|--------|
| **第一步：后端 API 冒烟** | 第 2.1 节全量：9 组 24 端点的正常/异常路径 + mock 体系 + CI `backend-test` job | **pytest + httpx 已批准现在引入（第 4 章），可立即开工**；核心端点（health/upload/sessions/chat/collections）正常路径与关键异常路径用例，在 CI 的 **windows-latest 与 macos-latest 双端全绿**；测试不触网（LLM/embedding 全 mock）、不写开发数据目录 | 30–40 条 | **M** |
| **第二步：检索回归** | 第 2.2 节：问句集 + 基线快照 + Hit@k 断言 + CI `retrieval-regression` job（双形态：PR 小规模基线 + 定时/手动全量，决策③） | 基线用例集**可重复运行**（同一代码两次运行结果一致）；人为改动 embedding/索引/切分参数（如 `max_chunk_size`）时，断言按预期失败并指出退化问句——即"该疼的时候会疼" | 问句 15–20 条 | **M** |
| **第三步：前端流程** | 第 2.3 节：Playwright 先行用例 1–5 + CI `frontend-test` job 挂 e2e（playwright 暂缓、后续单独立项，决策④，开工以立项为前提） | **提问→回答渲染→引用角标→反馈提交闭环用例通过**，且 CI 双端可复现；其余流程用例陆续补齐 | e2e 5–8 条（后续再补组件测试） | **L**（e2e 基建 + QAPanel 拆分的前置纠缠） |

顺序理由：第一步无外部依赖、收益最确定（24 个端点的行为契约立即固化），且依赖已批、可立即开工；第二步依赖第一步的 pytest 基建，且 embedding 模型下载/缓存要单独处理；第三步最重（要起双端进程），受两项前置制约——体检报告 2.3 的 QAPanel 拆分（已批准，决策⑤）与 playwright 单独立项（决策④），放最后。

---

## 附：已裁决记录（2026-07-17）

原 6 项待定已于 **2026-07-17 由用户全部拍板**，正文相关章节已按决策同步定稿。各项决策与理由如下。

1. **删除不存在知识库的行为契约**：定为返回 **404 + 统一错误体**，前端按错误卡模式处理。现状为 200 `deleted`（幂等静默成功，`vector_store.py:182-193`），与重命名不存在库返回 400（`vector_store.py:199-200`）语义不一致。理由：同组端点语义应统一，删除失败要按统一错误体对前端可见，而非静默成功。
2. **本地钩子工具**：定为 **pre-commit**，前后端一套配置。理由：前后端混合仓库，一套配置同时管 Python 与 JS。
3. **检索回归的 CI 形态**：定为**双形态**——PR 必跑小规模基线（固定 10 问句以内，保速度）；`schedule` 定时（cron，每周一凌晨）+ `workflow_dispatch` 手动触发跑全量。理由：小规模基线保住 PR 速度，全量回归交给定时与手动，两者兼顾。
4. **新增依赖引入**：定为**分批批准**——`pytest` + `httpx` 现在引入；`vitest`（连同 `@testing-library/react` + `jsdom`）待 QAPanel 拆分后引入；`playwright` 暂缓、后续单独立项；`pytest-mock` 可降级为标准库 `unittest.mock`，由实施时决定。理由：引入节奏与前置条件（QAPanel 拆分、单独立项）对齐，不一次性扩大依赖面。
5. **QAPanel 组件级单测的前置**：同意以体检报告 2.3 的结构拆分为前置，**纯结构拆分、不改逻辑**。理由：不拆分则 2605 行组件的单测维护成本高于收益；纯结构拆分风险可控。
6. **面试魔法字符串协议化**（"不知道"/"结束"，体检报告 3.7）：定为**执行**，前后端同改，**测试用例与协议变更同 PR 提交**。理由：协议与用例原子化同改，避免契约切换中间态的测试漂移。

## 附：阶段 4 并发压测（2026-10-09）

摄入队列化的并发压测资产已入库：locust 脚本 `backend/tests/load/locustfile.py`
（S1 基线 / S2 上传干扰 / S3 考官混合三场景，用法见文件 docstring），
压测依赖在 `backend/requirements-dev.txt`（locust，不进 runtime 镜像），
结果与调优记录见 `docs/压测报告.md`。队列化单元测试在
`backend/tests/test_ingest_queue.py`（job store CRUD / 上传契约 / stale 兜底 / worker 分支）。
