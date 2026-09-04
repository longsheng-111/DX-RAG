# DX-RAG 知识库问答系统 - 项目说明书

---

## 目录

1. [项目概述](#1-项目概述)
2. [技术架构](#2-技术架构)
3. [核心功能模块](#3-核心功能模块)
4. [文件清洗与切片](#4-文件清洗与切片)
5. [检索优化详解](#5-检索优化详解)
6. [部署与运行](#6-部署与运行)
7. [API 接口说明](#7-api-接口说明)
8. [使用指南](#8-使用指南)
9. [配置说明](#9-配置说明)
10. [常见问题](#10-常见问题)

---

## 1. 项目概述

### 1.1 项目定位

**DX-RAG** 是一个基于 Retrieval-Augmented Generation（检索增强生成）技术的企业级知识库问答系统。该系统能够：

- 上传多种格式的文档（PDF、Word、Excel、Markdown 等）
- 自动进行文本清洗、切分和向量嵌入
- 支持多知识库管理（创建、重命名、删除）
- 基于混合检索（关键词 + 向量）进行智能问答
- 支持对话记忆功能

### 1.2 核心价值

| 特性 | 描述 |
|-----|------|
| **多格式支持** | 支持 PDF（含图片型）、DOCX、XLSX、TXT、MD 等格式 |
| **视觉模型集成** | 使用通义千问 qwen-plus（OpenAI 兼容接口）处理图片型 PDF |
| **混合检索** | 关键词检索 + 向量检索 + 加权融合 |
| **多知识库** | 支持创建多个独立知识库，按需切换 |
| **对话记忆** | 保留对话历史，支持上下文理解 |
| **Markdown 渲染** | 回答支持结构化展示（标题、列表、加粗等） |

### 1.3 技术栈

| 层级 | 技术 | 版本 | 说明 |
|-----|------|-----|------|
| **前端** | Next.js | 14.2.24 | React 服务端渲染框架 |
| **前端** | Ant Design | 5.22.7 | UI 组件库 |
| **前端** | React Markdown | ^9.0.1 | Markdown 渲染 |
| **后端** | FastAPI | ^0.104.1 | Python Web 框架 |
| **后端** | FAISS (faiss-cpu) | ^1.7.0 | 向量检索引擎（本地索引文件，已替代 ChromaDB） |
| **后端** | Sentence Transformers | ^2.2.2 | 文本嵌入模型 |
| **后端** | PyMuPDF | ^1.26.0 | PDF 处理 |
| **后端** | OpenAI Python | ^1.1.0 | LLM API 调用 |
| **嵌入模型** | bge-small-zh-v1.5 | - | 中文语义嵌入 |
| **LLM** | DeepSeek Chat | - | 问答生成 |
| **视觉模型** | Qwen-Plus (DashScope) | - | 图片型 PDF 处理 |

---

## 2. 技术架构

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        前端层 (Next.js)                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐ │
│  │ 知识库管理  │ │ 文件上传    │ │ 知识问答    │ │ 文件管理  │ │
│  └─────┬───────┘ └─────┬───────┘ └─────┬───────┘ └─────┬─────┘ │
└───────┼────────────────┼────────────────┼────────────────┼───────┘
        │                │                │                │
        ▼                ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    后端层 (FastAPI)                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    API Router                           │   │
│  │  /upload  /query  /files  /collections  /stats         │   │
│  └───────────────────┬─────────────────────────────────────┘   │
│                      │                                        │
│  ┌───────────────────▼─────────────────────────────────────┐   │
│  │                   Services                              │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────────┐ ┌─────────┐ │   │
│  │  │ Ingest   │ │ QA       │ │ VectorStore  │ │ Config  │ │   │
│  │  │ 文件处理 │ │ 问答服务 │ │ FAISS向量存储│ │ 配置管理 │ │   │
│  │  └──────────┘ └──────────┘ └──────────────┘ └─────────┘ │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       数据层                                   │
│  ┌─────────────────────────┐ ┌─────────────────────────────┐   │
│  │  FAISS + 元数据 JSON    │ │           Uploads           │   │
│  │  (vector_db 目录)       │ │  (文件存储目录)             │   │
│  └─────────────────────────┘ └─────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 数据流概览

```
                    ┌─────────────────┐
                    │   用户上传文件   │
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │   文件格式识别   │
                    └────────┬────────┘
                             ▼
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ 文本提取    │     │ PDF提取     │     │ Excel提取   │
│ (TXT/MD)   │     │ (PyPDF)     │     │(openpyxl)  │
└─────┬───────┘     └─────┬───────┘     └─────┬───────┘
      │                   │                   │
      └───────────────────┼───────────────────┘
                          ▼
              ┌───────────────────────┐
              │   图片型PDF检测       │
              │   (文本为空?)         │
              └───────────┬───────────┘
                    Yes   │   No
                    ▼     │     ▼
        ┌─────────────────┼─────────────────┐
        │  Qwen-VL视觉模型 │   直接清洗      │
        │  OCR提取         │                 │
        └────────┬─────────┴────────┬────────┘
                 │                  │
                 └────────┬─────────┘
                          ▼
              ┌───────────────────────┐
              │     文件清洗          │
              │  - 去除空行          │
              │  - 编码统一          │
              │  - 噪声过滤          │
              └───────────┬───────────┘
                          ▼
              ┌───────────────────────┐
              │     文本切片          │
              │  - 标题切分          │
              │  - 递归切分          │
              │  - 重叠保留          │
              └───────────┬───────────┘
                          ▼
              ┌───────────────────────┐
              │    向量嵌入           │
              │  BGE-small-zh-v1.5   │
              └───────────┬───────────┘
                          ▼
              ┌───────────────────────┐
              │  存入 FAISS 向量库    │
              │ (index.faiss + 元数据 │
              │  JSON，按知识库分目录)│
              └───────────────────────┘
```

---

## 3. 核心功能模块

### 3.1 文件处理模块 (ingest.py)

**功能**：负责文件上传、文本提取和切分

**支持的文件格式**：

| 格式 | 扩展名 | 处理方式 |
|-----|-------|---------|
| 文本 | .txt, .md, .csv, .json | 直接读取 |
| PDF | .pdf | 文本提取 + 视觉模型(OCR) |
| Word | .docx | python-docx 解析 |
| Excel | .xlsx, .xlsm | openpyxl 解析 |

### 3.2 向量存储模块 (vector_store.py)

**功能**：管理向量索引与元数据的增删改查

**向量引擎**：
- **FAISS（faiss-cpu）**：本地索引文件，无需额外部署。每个知识库一个目录（`backend/app/vector_db/<知识库名>/`），内含 `index.faiss`（IndexFlatIP 索引，归一化向量内积等价余弦相似度）与 `metadata.json`（`documents`/`metadatas`/`ids` 三个平行数组）；根目录 `_collections.json` 为知识库注册表。项目已从 ChromaDB 迁移至 FAISS（2026-07）。

**核心操作**：

| 方法 | 功能 |
|-----|------|
| `add_texts()` | 添加文本向量 |
| `vector_search()` | 向量相似度检索 |
| `delete_file()` | 删除文件相关数据（FAISS 不支持按条删除，重建索引） |
| `get_files()` | 获取文件列表 |
| `create/delete/rename_collection()` | 知识库创建、删除、重命名 |

### 3.3 问答模块 (qa.py)

**功能**：实现检索增强生成

**检索策略**：混合检索（Hybrid Retrieval），当前实现为 `services/retrieval.py` 的 `AdvancedRetriever`：

1. BM25 召回 top 20
2. 向量检索召回 top 20
3. RRF（倒数排名融合，k=60）得到 top 15
4. BGE 向量精排：余弦相似度（权重 0.7）与归一化 RRF 分数（权重 0.3）加权融合排序，取 top 5；余弦相似度低于 0.65 的结果被过滤

> 注：早期版本为「关键词 30% + 向量 70%」线性加权融合，现已升级为上述 BM25 + RRF 方案。

### 3.4 视觉模型模块

**功能**：处理图片型 PDF 的文字提取

**技术实现**：使用通义千问 qwen-plus 视觉模型（DashScope OpenAI 兼容接口，逐页渲染为图片后调用）

---

## 4. 文件清洗与切片

### 4.1 文件清洗 (File Cleaning)

#### 4.1.1 清洗流程

文件清洗是将原始文件转换为纯净文本的过程：

```
原始文件 → 格式识别 → 内容提取 → 噪声过滤 → 文本规范化
```

#### 4.1.2 核心清洗逻辑

```python
def read_text_from_file(file_path: Path) -> str:
    suffix = file_path.suffix.lower()

    # 1. 文本文件：编码处理
    if suffix in {".txt", ".md", ".csv", ".json", ".log"}:
        try:
            return file_path.read_text(encoding="utf-8")
        except:
            try:
                return file_path.read_text(encoding="utf-16")
            except:
                return file_path.read_text(encoding="gbk", errors="ignore")

    # 2. PDF：文本提取 + 视觉模型
    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)

        # 图片型 PDF：使用 Qwen-VL 提取
        if not text.strip():
            text = extract_text_with_qwen_vl(str(file_path))

        return text

    # 3. Word：提取段落
    if suffix == ".docx":
        doc = Document(str(file_path))
        return "\n".join(p.text for p in doc.paragraphs)

    # 4. Excel：表格转文本
    if suffix in {".xlsx", ".xlsm", ".xltx", ".xltm"}:
        wb = load_workbook(str(file_path), data_only=True)
        parts = []
        for sheet in wb.worksheets:
            for row in sheet.iter_rows(values_only=True):
                row_text = " ".join(str(cell) for cell in row if cell is not None)
                if row_text.strip():
                    parts.append(row_text)
        return "\n".join(parts)
```

#### 4.1.3 图片型 PDF 特殊处理

> 注：以下为早期实现示意（dashscope SDK + qwen-vl-plus）。当前实现经 OpenAI 兼容接口调用 qwen-plus，见 `backend/app/services/ingest.py` 的 `extract_text_with_qwen_vl()`。

当普通 PDF 提取失败时，自动调用通义千问视觉模型：

```python
def extract_text_with_qwen_vl(pdf_path: str) -> str:
    """使用通义千问视觉模型提取图片型PDF中的文字"""
    import fitz  # PyMuPDF
    import dashscope
    from dashscope import MultiModalConversation

    dashscope.api_key = settings.dashscope_api_key
    doc = fitz.open(pdf_path)
    full_text = ""

    for page_num in range(len(doc)):
        page = doc[page_num]

        # 将页面渲染为图片
        pix = page.get_pixmap()
        img_buffer = pix.tobytes("jpg")
        img_base64 = base64.b64encode(img_buffer).decode('utf-8')

        # 调用视觉模型
        response = MultiModalConversation.call(
            model='qwen-vl-plus',
            messages=[{
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': f"data:image/jpeg;base64,{img_base64}"},
                    {'type': 'text', 'text': '请提取图片中的所有文字，保持格式'}
                ]
            }]
        )

        if response.status_code == 200:
            content = response.output.choices[0].message.content
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and 'text' in item:
                        full_text += item['text'] + "\n\n"
            elif isinstance(content, str):
                full_text += content + "\n\n"

    doc.close()
    return full_text.strip()
```

### 4.2 文本切片 (Text Chunking)

#### 4.2.1 切片策略

文本切片的目标是将长文本切分为适合向量嵌入的小段（chunks）：

| 原则 | 说明 |
|-----|------|
| **语义完整性** | 保持段落、句子的完整性 |
| **固定大小** | 控制 chunk 长度（默认 800 字符） |
| **重叠窗口** | 相邻 chunk 重叠（默认 120 字符） |
| **层次切分** | Markdown 文件按标题层次切分 |

#### 4.2.2 切片流程

```
原始文本 → 空行过滤 → 标题识别 → 递归切分 → 结果合并
```

#### 4.2.3 核心切分逻辑

**第一步：基础清洗**

```python
def split_text(text: str, source_file: str = "") -> List[str]:
    # 去除空行和多余空格
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            cleaned_lines.append(stripped)
    cleaned = "\n".join(cleaned_lines)

    if not cleaned:
        return []

    return split_text_by_paragraphs(cleaned, source_file)
```

**第二步：Markdown 标题切分**

```python
def split_text_by_headers(text: str) -> List[str]:
    # 定义标题层次
    headers_to_split = [
        ("#", "大章节"),
        ("##", "小节"),
        ("###", "小点"),
        ("####", "段落"),
    ]

    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split)
    docs = header_splitter.split_text(text)

    chunks = []
    for doc in docs:
        chunk_content = doc.page_content.strip()
        if not chunk_content:
            continue

        # 将标题路径添加到内容前
        header_parts = []
        for level in ["大章节", "小节", "小点", "段落"]:
            if level in doc.metadata:
                header_parts.append(doc.metadata[level])

        if header_parts:
            header_line = " > ".join(header_parts)
            chunk_content = f"{header_line}\n\n{chunk_content}"

        chunks.append(chunk_content)

    return chunks
```

**第三步：递归字符切分**

```python
def split_text_by_paragraphs(text: str, source_file: str = "") -> List[str]:
    suffix = Path(source_file).suffix.lower() if source_file else ""

    # Markdown 文件优先按标题切分
    if suffix == ".md":
        return split_text_by_headers(text)

    # 其他格式：先标题切分，再按字符长度切分
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "大章节"),
            ("##", "小节"),
            ("###", "小点"),
            ("####", "段落"),
        ]
    )
    header_chunks = header_splitter.split_text(text)

    # 使用 RecursiveCharacterTextSplitter 处理长文本
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.max_chunk_size,      # 默认 800
        chunk_overlap=settings.chunk_overlap,    # 默认 120
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )

    final_chunks = []
    for chunk in header_chunks:
        content = chunk.page_content.strip()
        if not content:
            continue

        # 短文本直接保留，长文本递归切分
        if len(content) <= settings.max_chunk_size:
            final_chunks.append(content)
        else:
            sub_chunks = text_splitter.split_text(content)
            final_chunks.extend(sub_chunks)

    return final_chunks
```

#### 4.2.4 切分参数配置

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `max_chunk_size` | 800 | 每个 chunk 的最大字符数 |
| `chunk_overlap` | 120 | 相邻 chunk 的重叠字符数 |
| `separators` | `["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]` | 切分分隔符优先级 |

#### 4.2.5 切分示例

**输入文本**：
```
## 课程介绍

本节课主要讲解机器学习的基础知识，包括监督学习和无监督学习两种范式。

### 监督学习

监督学习是指利用标注数据进行模型训练的方法，常见算法包括线性回归、决策树等。

### 无监督学习

无监督学习则是在没有标注的情况下发现数据中的模式，例如聚类算法。
```

**输出 Chunks**：
```
Chunk 1: "课程介绍\n\n本节课主要讲解机器学习的基础知识..."
Chunk 2: "课程介绍 > 监督学习\n\n监督学习是指利用标注数据..."
Chunk 3: "课程介绍 > 无监督学习\n\n无监督学习则是在没有标注..."
```

---

## 5. 检索优化详解

> 注：本章 5.1–5.4 的代码与权重示例为**早期实现**（关键词 30% + 向量 70% 线性加权，基于 ChromaDB API）。当前实现以 `backend/app/services/retrieval.py` 为准：BM25 召回 + 向量召回 → RRF 融合 → BGE 重排序，向量引擎为 FAISS（见 3.2 节）。

### 5.1 混合检索架构

项目采用**混合检索策略**，结合关键词检索和向量检索的优势：

```
用户查询
    ↓
┌───────────────────────────────────────┐
│        HybridRetriever               │
├───────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────┐ │
│  │ 关键词检索      │  │ 向量检索   │ │
│  │ (30% 权重)     │  │ (70% 权重) │ │
│  └────────┬────────┘  └──────┬──────┘ │
│           │                  │        │
│           └────────┬─────────┘        │
│                    ↓                  │
│          加权融合 → 排序 → Top-K      │
└───────────────────────────────────────┘
```

### 5.2 关键词检索优化

#### 5.2.1 倒排索引构建

```python
class HybridRetriever:
    def _build_inverted_index(self):
        """构建倒排索引用于快速关键词查找"""
        if self.inverted_index is not None:
            return

        self.inverted_index = defaultdict(set)
        vector_store = get_vector_store(self.collection_name)

        results = vector_store._collection.get(include=['documents', 'metadatas'])
        docs = results.get('documents', [])
        metadatas = results.get('metadatas', [])

        for idx, doc in enumerate(docs):
            if doc:
                file_name = metadatas[idx].get('file_name', '')
                # 分词：提取中文、英文、数字
                words = re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', doc.lower())
                for word in words:
                    if len(word) >= 2:  # 过滤单字
                        self.inverted_index[word].add((file_name, idx))
                self.all_docs.append({
                    'content': doc,
                    'file_name': file_name
                })
```

#### 5.2.2 关键词检索实现

```python
def keyword_search(self, query: str, top_k: int = 10) -> List[tuple[str, str, float]]:
    self._build_inverted_index()

    if not self.inverted_index:
        return []

    # 分词查询
    query_words = re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', query.lower())
    scores = defaultdict(float)

    # 计算匹配分数
    for word in query_words:
        if len(word) >= 2 and word in self.inverted_index:
            for file_name, idx in self.inverted_index[word]:
                scores[(file_name, idx)] += 1  # 词频累加

    # 归一化分数
    results = []
    for (file_name, idx), score in scores.items():
        if idx < len(self.all_docs):
            results.append((file_name, self.all_docs[idx]['content'], score / len(query_words)))

    # 按分数排序
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:top_k]
```

### 5.3 向量检索优化

#### 5.3.1 语义嵌入与检索

```python
def vector_search(self, query: str, top_k: int = 10) -> List[tuple[str, str, float]]:
    # 1. 加载嵌入模型（BGE 中文模型）
    model = get_model()

    # 2. 将查询转为向量（512 维）
    query_vector = model.encode(query, normalize_embeddings=True).tolist()

    # 3. 在向量数据库中检索
    vector_store = get_vector_store(self.collection_name)
    results = vector_store.search(query_vector, top_k=top_k)

    return results
```

#### 5.3.2 向量数据库配置

| 配置项 | 值 | 说明 |
|-------|-----|------|
| 相似度度量 | Cosine | 归一化向量 + 内积，等价余弦相似度 |
| 索引类型 | FAISS IndexFlatIP | 精确内积检索（非 HNSW 近似索引） |
| 向量维度 | 512 | BGE 模型输出维度 |

### 5.4 加权融合策略

```python
def hybrid_search(self, query: str, weights: List[float] = [0.3, 0.7], top_k: int = 10) -> List[tuple[str, str, float]]:
    # 获取两种检索结果
    kw_results = dict()
    vec_results = dict()

    # 关键词检索（扩大范围）
    kw_search = self.keyword_search(query, top_k * 2)
    for file_name, content, score in kw_search:
        kw_results[content] = score * weights[0]  # 30% 权重

    # 向量检索（扩大范围）
    vec_search = self.vector_search(query, top_k * 2)
    for file_name, content, score in vec_search:
        vec_results[content] = (1 - score) * weights[1]  # 70% 权重

    # 融合分数
    all_contents = set(kw_results.keys()) | set(vec_results.keys())
    final_scores = {}

    for content in all_contents:
        kw_score = kw_results.get(content, 0)
        vec_score = vec_results.get(content, 0)
        final_scores[content] = kw_score + vec_score

    # 去重并排序
    results = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    final_results = []
    seen_contents = set()

    for file_name, content, score in kw_search + vec_search:
        if content not in seen_contents and len(final_results) < top_k:
            seen_contents.add(content)
            final_results.append((file_name, content, final_scores.get(content, 0)))

    return final_results
```

### 5.5 检索优化策略总结

| 优化维度 | 实现方式 | 效果 |
|---------|---------|------|
| **索引优化** | 倒排索引 + 向量索引 | 快速检索 |
| **召回率提升** | 混合检索（关键词 + 向量） | 兼顾精确与语义匹配 |
| **排序优化** | 加权融合 | 综合评分排序 |
| **性能优化** | 扩大召回范围后截断 | 保证质量的同时提升速度 |

### 5.6 当前方法 vs BM25 + RRF

> 注：当前检索已升级为右列方案（BM25 + 密集向量 + RRF，并增加 BGE 重排序，见 `retrieval.py`），本表保留作方案演进记录。

| 对比维度 | 当前方法 | BM25 + 密集向量 + RRF |
|---------|---------|---------------------|
| **关键词检索** | 简单词频匹配 | BM25（考虑词频和文档长度） |
| **向量检索** | BGE 嵌入 | 通常用更强大的模型 |
| **融合策略** | 线性加权（固定权重） | RRF（基于排名的非线性融合） |
| **参数敏感性** | 对权重敏感 | 相对鲁棒 |
| **召回率** | 中等 | 较高 |

---

## 6. 部署与运行

### 6.1 环境要求

| 依赖 | 版本 |
|-----|------|
| Python | 3.10+ |
| Node.js | 18+ |
| pip | 22+ |
| npm | 9+ |

### 6.2 后端部署

**步骤 1：安装依赖**
```bash
cd backend
pip install -r requirements.txt
```

**步骤 2：配置 API 密钥**

在项目根目录复制 `.env.example` 为 `.env` 并填入密钥（`config.py` 通过 pydantic-settings 自动读取）：
```bash
cp .env.example .env
```
```env
DEEPSEEK_API_KEY=your-deepseek-key
DASHSCOPE_API_KEY=your-dashscope-key
# ... 其他配置均有默认值，按需覆盖
```

**步骤 3：启动服务**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 6.3 前端部署

**步骤 1：安装依赖**
```bash
cd frontend
npm install
```

**步骤 2：启动开发服务器**
```bash
npm run dev
```

**步骤 3：构建生产版本**
```bash
npm run build
npm start
```

---

## 7. API 接口说明

### 7.1 健康检查

```
GET /api/health
```

**响应**：
```json
{"status": "ok"}
```

### 7.2 文件上传

```
POST /api/upload
Content-Type: multipart/form-data
```

**参数**：

| 参数 | 类型 | 必填 | 说明 |
|-----|------|-----|------|
| file | File | 是 | 上传的文件 |
| collection_name | String | 否 | 目标知识库名称 |

**响应**：
```json
{
  "message": "上传并入库成功",
  "file_name": "document.pdf",
  "chunks": 15,
  "collection_name": "test-db"
}
```

### 7.3 知识问答

```
POST /api/query
Content-Type: application/json
```

**参数**：
```json
{
  "question": "课后应该做什么",
  "top_k": 5,
  "collection_name": "test-db",
  "history": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "您好！"}
  ]
}
```

**响应**：
```json
{
  "answer": "### 课后学习建议\n\n1. 完成作业练习\n2. 复习当天知识点\n3. 做错题整理",
  "sources": [{"content": "...", "similarity": 0.85}],
  "query": "课后应该做什么",
  "collection_name": "test-db"
}
```

### 7.4 知识库管理

| 接口 | 方法 | 功能 |
|-----|------|------|
| `/api/collections` | GET | 获取知识库列表 |
| `/api/collections` | POST | 创建知识库 |
| `/api/collections/{name}` | PUT | 重命名知识库 |
| `/api/collections/{name}` | DELETE | 删除知识库 |

### 7.5 文件管理

| 接口 | 方法 | 功能 |
|-----|------|------|
| `/api/files?collection_name=xxx` | GET | 获取文件列表 |
| `/api/files/{file_name}?collection_name=xxx` | DELETE | 删除文件 |
| `/api/files/{file_name}/preview?collection_name=xxx` | GET | 预览文件内容 |

### 7.6 会话 / 聊天 / 反馈 / 模拟面试

> 本节为简表；全部 24 个端点的完整定义以 `backend/app/api/routes.py` 为准。

| 接口 | 方法 | 功能 |
|-----|------|------|
| `/api/sessions` | GET / POST | 列出 / 创建会话 |
| `/api/sessions/{session_id}` | GET / DELETE | 会话详情 / 删除会话 |
| `/api/sessions/{session_id}/persona` | PUT | 切换会话角色 |
| `/api/sessions/{session_id}/title` | PUT | 更新会话标题 |
| `/api/sessions/{session_id}/export` | GET | 导出会话为 Markdown |
| `/api/personas` | GET | 获取角色列表 |
| `/api/chat` | POST | 会话化问答（推荐，替代无状态 `/api/query`） |
| `/api/feedback` | POST | 提交引用纠错反馈（replace / inaccurate） |
| `/api/exam/start` | POST | 开始模拟面试 |
| `/api/exam/{session_id}/next` | POST | 提交回答，进入评分/追问/下一题/总结 |
| `/api/exam/{session_id}` | GET | 获取面试状态 |

---

## 8. 使用指南

### 8.1 创建知识库

1. 登录系统后，点击左侧菜单 **知识库管理**
2. 点击 **创建知识库** 按钮
3. 输入知识库名称（3-50字符，字母数字开头结尾）
4. 点击 **创建** 完成

### 8.2 上传文件

1. 点击左侧菜单 **上传文件**
2. 从下拉框选择目标知识库
3. 点击或拖拽文件到上传区域
4. 等待上传完成（大文件可能需要较长时间）

### 8.3 知识问答

1. 点击左侧菜单 **知识问答**
2. 选择目标知识库
3. 输入问题，按 `Ctrl+Enter` 或点击 **发送**
4. 系统将返回结构化的回答和参考来源

### 8.4 文件管理

1. 点击左侧菜单 **文件管理**
2. 选择目标知识库
3. 可进行 **预览** 或 **删除** 操作

---

## 9. 配置说明

### 9.1 后端配置 (backend/app/core/config.py)

| 配置项 | 默认值 | 说明 |
|-------|-------|------|
| `app_name` | dx-rag-demo | 应用名称 |
| `cors_origins` | ["http://localhost:3000", "http://127.0.0.1:3000"] | 允许的跨域来源 |
| `vector_collection` | knowledge_chunks | 默认知识库名称 |
| `vector_persist_dir` | backend/app/vector_db | 向量库存储目录（FAISS） |
| `session_dir` | backend/app/data/sessions | 会话数据目录（反馈数据存于其同级 feedback/ 目录） |
| `embed_model` | BAAI/bge-small-zh-v1.5 | 嵌入模型（优先使用本地 ModelScope 缓存） |
| `upload_dir` | backend/app/uploads | 文件上传目录 |
| `max_chunk_size` | 800 | 文本切分最大长度 |
| `chunk_overlap` | 120 | 切分重叠长度 |
| `retrieval_similarity_threshold` | 0.65 | 检索相关性阈值（余弦相似度） |
| `deepseek_api_key` | - | DeepSeek API 密钥 |
| `dashscope_api_key` | - | 通义千问 API 密钥 |

---

## 10. 常见问题

### Q1：文件上传后不显示？

**可能原因**：
1. 后端服务未运行或端口占用
2. 数据库连接异常
3. 文件内容为空（图片型 PDF 未正确处理）

**解决方法**：
```bash
# 检查后端服务
netstat -ano | findstr ":8000"

# 重启后端服务
taskkill /F /PID <pid>
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Q2：图片型 PDF 无法提取文字？

**要求**：需要配置有效的 `dashscope_api_key`

**验证方法**：检查后端日志是否有以下错误：
```
通义千问视觉模型提取失败: ...
```

### Q3：检索结果不准确？

**优化建议**：
1. 增加知识库文档数量
2. 调整 `top_k` 参数（默认 5）
3. 检查文档内容质量

### Q4：端口占用问题？

**解决方法**：
```bash
# 查找占用进程
netstat -ano | findstr ":8000"

# 终止进程
taskkill /F /PID <pid>
```

### Q5：如何清空知识库？

**方法**：
1. 在知识库管理页面删除知识库
2. 或删除 `backend/app/vector_db` 目录（或其中某个知识库子目录）后重启服务

---

## 附录：目录结构

```
dx-rag/
├── backend/                    # 后端服务
│   ├── app/
│   │   ├── api/                # API 路由
│   │   │   └── routes.py       # 接口定义（24 个端点）
│   │   ├── core/               # 核心模块
│   │   │   ├── config.py       # 配置管理（读取根目录 .env）
│   │   │   ├── vector_store.py # FAISS 向量存储 + JSON 元数据
│   │   │   ├── session.py      # 会话管理（JSON 落盘）
│   │   │   └── persona.py      # 内置角色模板
│   │   ├── services/           # 业务服务
│   │   │   ├── ingest.py       # 文件处理与切片
│   │   │   ├── qa.py           # 问答服务
│   │   │   ├── agent.py        # ReAct 检索 Agent
│   │   │   ├── retrieval.py    # 混合检索（BM25 + 向量 + RRF + 重排序）
│   │   │   ├── examiner.py     # 模拟面试（考官模式）
│   │   │   ├── feedback.py     # 引用纠错反馈存储
│   │   │   ├── token_tracker.py# Token 统计与上下文压缩
│   │   │   ├── title_generator.py # 会话自动标题
│   │   │   └── harness.py      # 输入护栏与兜底
│   │   ├── data/               # 运行时数据（sessions/、feedback/）
│   │   ├── uploads/            # 上传文件
│   │   ├── vector_db/          # FAISS 向量库（按知识库分目录）
│   │   └── main.py             # 入口文件
│   ├── test_data/              # 测试数据（八股文题库等）
│   └── requirements.txt
├── frontend/                   # 前端应用
│   ├── app/
│   │   ├── layout.tsx          # 根布局（主题与整体框架）
│   │   ├── page.tsx            # 主页面
│   │   └── globals.css         # 全局样式与设计 token
│   ├── components/             # 面板组件（问答/面试/会话/知识库/文件）
│   ├── next.config.js          # Next.js 配置（/api 代理到 8000）
│   └── package.json            # 依赖配置
├── .env.example                # 环境变量模板
├── start.sh / start.bat        # 一键启动脚本
└── README.md                   # 项目说明
```

---

**版本**: v1.0
**最后更新**: 2026年5月
**维护者**: DX-RAG Team

---

**修订说明（2026-07）**：向量引擎已从 ChromaDB 迁移至 FAISS（faiss-cpu），数据目录由 `chroma_db` 调整为 `backend/app/vector_db`；同步修订了技术栈表、3.2 节、5.3.2 节、Q5、附录目录结构，并补充 7.6 节（会话/聊天/反馈/模拟面试端点）。检索章节（5.1–5.4）与 4.1.3 节保留早期实现示意，文中已加注说明，当前实现以 `backend/app/services/retrieval.py`、`ingest.py` 为准。
