"""
检索层升级：BM25 + 密集向量 + RRF 融合 + 向量精排加权融合

召回流程：双路召回 → RRF 融合 → 向量精排加权融合
1. BM25 召回 top 20
2. 向量检索召回 top 20
3. RRF 融合得到 top 15
4. 向量精排：余弦相似度（0.7）与归一化 RRF 分数（0.3）加权融合排序，取 top 5
"""
import re
import numpy as np
from dataclasses import dataclass
from typing import Optional

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

from app.core.config import settings
from app.core.storage import (
    vector_search,
    get_vector_store,
    encode_query,
    encode_texts,
    get_store_version,
)

# 充分性判断阈值（供 agent.py import）：高于检索过滤阈值 0.65，
# 让“召回有结果但相关性平庸”的中间地带能触发 Agent 改写重试
SUFFICIENCY_THRESHOLD = 0.75

# BM25 索引模块级缓存：AdvancedRetriever 每次请求都会新建实例，
# 缓存必须放在模块级才能跨请求复用。
# key 为 (collection_name, 存储版本令牌)，文档增删导致令牌变化时才重建。
# 缓存值携带与 documents 平行的 chunk ids，供 bm25_branch 输出 RetrievalResult 用。
_bm25_cache: dict[
    tuple[str, tuple],
    tuple[list[str], list[dict], list[str], list[list[str]], Optional[BM25Okapi]],
] = {}


class AdvancedRetriever:
    """高级混合检索器"""

    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.documents: list[str] = []
        self.metadatas: list[dict] = []
        self.ids: list[str] = []
        self.tokenized_docs: list[list[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self._build_index()

    def _tokenize(self, text: str) -> list[str]:
        """
        轻量级分词：
        - 中文字符逐字切分
        - 英文、数字按连续片段切分
        - 单字符的英文/数字视为噪声过滤；单字中文是有效最小单位，保留
        """
        if not text:
            return []
        tokens = re.findall(r"[\u4e00-\u9fa5]|[a-zA-Z0-9]+", text.lower())
        return [t for t in tokens if len(t) >= 2 or "\u4e00" <= t <= "\u9fa5"]

    def _build_index(self) -> None:
        """从向量库构建 BM25 索引（模块级缓存，存储版本令牌未变化时直接复用）"""
        # 版本令牌统一走 storage.get_store_version：file 后端为 metadata.json 的
        # (mtime_ns, size)，postgres 后端为 (chunk 总数, 最新创建时间戳)，增删数据即变化
        version = get_store_version(self.collection_name)

        cache_key = (self.collection_name, version)
        cached = _bm25_cache.get(cache_key)
        if cached is None:
            try:
                data = get_vector_store(self.collection_name)
            except Exception:
                data = {"documents": [], "metadatas": [], "ids": []}

            documents = data.get("documents", []) or []
            metadatas = data.get("metadatas", []) or []
            ids = data.get("ids", []) or []
            tokenized_docs = [self._tokenize(d) for d in documents]
            bm25 = None
            if BM25Okapi is not None and tokenized_docs:
                bm25 = BM25Okapi(tokenized_docs)

            # 同一知识库只保留最新版本令牌的缓存条目，避免旧版本堆积
            for key in [k for k in _bm25_cache if k[0] == self.collection_name]:
                del _bm25_cache[key]
            _bm25_cache[cache_key] = (documents, metadatas, ids, tokenized_docs, bm25)
            cached = _bm25_cache[cache_key]

        (
            self.documents,
            self.metadatas,
            self.ids,
            self.tokenized_docs,
            self.bm25,
        ) = cached

    def bm25_search(
        self, query: str, top_k: int = 20
    ) -> list[tuple[str, str, float]]:
        """基于 BM25 的关键词检索"""
        if not self.bm25 or not self.documents:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            file_name = self.metadatas[idx].get("file_name", "") if self.metadatas[idx] else ""
            results.append((file_name, self.documents[idx], float(scores[idx])))
        return results

    def vector_search(
        self, query: str, top_k: int = 20
    ) -> list[tuple[str, str, float]]:
        """语义向量检索"""
        # 底层 vector_store.vector_search 仅接受文本查询并在内部自行编码，
        # 无法沿调用链传入预编码向量，query 会在此处与 rerank 中各编码一次
        return vector_search(query, self.collection_name, top_k)

    def rrf_fuse(
        self,
        bm25_results: list[tuple[str, str, float]],
        vec_results: list[tuple[str, str, float]],
        k: int = 60,
        top_k: int = 15,
    ) -> list[tuple[str, str, float]]:
        """Reciprocal Rank Fusion 融合 BM25 和向量检索结果"""
        scores: dict[str, float] = {}
        content_map: dict[str, tuple[str, str]] = {}

        for rank, (file_name, content, _) in enumerate(bm25_results):
            key = content[:300]
            scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
            content_map[key] = (file_name, content)

        for rank, (file_name, content, _) in enumerate(vec_results):
            key = content[:300]
            scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
            content_map[key] = (file_name, content)

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for key, score in sorted_items[:top_k]:
            file_name, content = content_map[key]
            results.append((file_name, content, score))
        return results

    def rerank(
        self,
        query: str,
        results: list[tuple[str, str, float]],
        top_k: int = 5,
    ) -> list[tuple[str, str, float]]:
        """
        向量精排：余弦相似度与 RRF 分数加权融合（0.7 / 0.3）后排序。

        - 排序使用融合分数，保留 RRF 融合给出的双路召回排序信息
        - 阈值过滤仍作用在余弦分数上，防止 RRF 加分误放不相关结果
        - 返回三元组中的 score 为余弦相似度，供下游充分性/置信度阈值判断
        """
        if not results:
            return []

        try:
            query_vec = encode_query(query)
            contents = [content for _, content, _ in results]
            content_vecs = encode_texts(contents)

            # 归一化后点积 = 余弦相似度
            similarities = (content_vecs @ query_vec).tolist()

            # RRF 分数 min-max 归一化到 [0, 1]，与余弦分数同量纲后加权
            rrf_scores = [score for _, _, score in results]
            rrf_min, rrf_max = min(rrf_scores), max(rrf_scores)
            rrf_range = rrf_max - rrf_min
            norm_rrf = [
                (s - rrf_min) / rrf_range if rrf_range > 0 else 1.0
                for s in rrf_scores
            ]

            fused = [
                (file_name, content, float(sim), 0.7 * float(sim) + 0.3 * nr)
                for (file_name, content, _), sim, nr in zip(results, similarities, norm_rrf)
            ]
            # 按加权融合分数排序
            fused.sort(key=lambda x: x[3], reverse=True)

            # 按配置阈值过滤不相关结果（作用在余弦分数上）
            threshold = settings.retrieval_similarity_threshold
            filtered = [
                (file_name, content, cos_sim)
                for file_name, content, cos_sim, _ in fused
                if cos_sim >= threshold
            ]
            return filtered[:top_k]
        except Exception:
            # 重排序失败，返回原始排序（不做阈值过滤，避免误伤）
            return results[:top_k]

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[str, str, float]]:
        """完整混合检索：双路召回 → RRF 融合 → 向量精排加权融合"""
        # 1. 多路召回
        bm25_results = self.bm25_search(query, top_k=20)
        vec_results = self.vector_search(query, top_k=20)

        # 2. RRF 融合
        fused = self.rrf_fuse(bm25_results, vec_results, top_k=15)

        # 3. 向量精排（余弦 0.7 + 归一化 RRF 0.3 加权融合）
        reranked = self.rerank(query, fused, top_k=top_k)

        return reranked


# ============================================================
#  分支化检索（阶段 2：QA LangGraph 化）
#
#  以下公开函数把 hybrid_search 的内部环节拆成可独立调用的分支，
#  供 graphs/qa_graph.py 的并行检索节点使用。hybrid_search 本身一个
#  字节不改，作为双跑对照组的 baseline。
# ============================================================

# score_type 取值（如实标注分数语义，充分性判定只认 "cosine"）：
# - "cosine"：余弦相似度（向量召回 / 精排正常路径）
# - "rrf"：RRF 融合分（融合结果 / 精排编码失败的回退路径）
# - "bm25"：BM25 原始分（BM25 召回分支），量纲与余弦不同，仅用于召回排序
SCORE_TYPE_COSINE = "cosine"
SCORE_TYPE_RRF = "rrf"
SCORE_TYPE_BM25 = "bm25"


@dataclass
class RetrievalResult:
    """单条检索结果，chunk_id 作为跨分支融合键（修复 content[:300] 键碰撞）"""

    chunk_id: str
    file_name: str
    content: str
    score: float
    score_type: str  # SCORE_TYPE_COSINE / SCORE_TYPE_RRF / SCORE_TYPE_BM25


def _content_id_map(collection_name: str) -> dict:
    """content -> chunk_id 反查表（documents 与 ids 是平行数组）。

    storage.vector_search 的返回不含 chunk id，这里用同一存储快照反查；
    内容完全重复的 chunk 会映射到同一 id，对融合而言等价于去重。
    """
    data = get_vector_store(collection_name)
    documents = data.get("documents", []) or []
    ids = data.get("ids", []) or []
    return {doc: cid for doc, cid in zip(documents, ids)}


def vector_branch(
    query: str,
    collection_name: str,
    top_k: int = 20,
) -> list[RetrievalResult]:
    """向量召回分支：score 为余弦相似度（score_type="cosine"）"""
    hits = vector_search(query, collection_name, top_k)
    if not hits:
        return []
    id_map = _content_id_map(collection_name)
    return [
        RetrievalResult(
            # 反查不到 id 属异常路径，回退 content 前缀作键（与旧行为一致）
            chunk_id=id_map.get(content) or f"noid:{content[:300]}",
            file_name=file_name,
            content=content,
            score=float(score),
            score_type=SCORE_TYPE_COSINE,
        )
        for file_name, content, score in hits
    ]


def bm25_branch(
    query: str,
    collection_name: str,
    top_k: int = 20,
) -> list[RetrievalResult]:
    """BM25 召回分支：复用 AdvancedRetriever 的索引构建与模块级缓存。

    score 为 BM25 原始分（score_type="bm25"），量纲与余弦不同，
    仅参与 RRF 排名，不参与充分性阈值判定。
    """
    retriever = AdvancedRetriever(collection_name)
    if not retriever.bm25 or not retriever.documents:
        return []

    query_tokens = retriever._tokenize(query)
    if not query_tokens:
        return []

    scores = retriever.bm25.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue
        metadata = retriever.metadatas[idx]
        file_name = metadata.get("file_name", "") if metadata else ""
        chunk_id = (
            retriever.ids[idx]
            if idx < len(retriever.ids)
            else f"noid:{retriever.documents[idx][:300]}"
        )
        results.append(
            RetrievalResult(
                chunk_id=chunk_id,
                file_name=file_name,
                content=retriever.documents[idx],
                score=float(scores[idx]),
                score_type=SCORE_TYPE_BM25,
            )
        )
    return results


def merge_rrf(
    branches: list[list[RetrievalResult]],
    top_k: int = 15,
    k: int = 60,
) -> list[RetrievalResult]:
    """RRF 融合多路召回结果，融合键为 chunk_id。

    旧 rrf_fuse 用 content[:300] 作键，不同 chunk 前 300 字相同会被错误合并；
    这里按 chunk_id 融合从根上避免该碰撞（设计文档 4.1）。融合结果的
    score_type 如实标为 "rrf"，等待 rerank_cosine 精排重写为余弦分。
    """
    scores: dict[str, float] = {}
    representatives: dict[str, RetrievalResult] = {}

    for branch in branches:
        for rank, item in enumerate(branch):
            scores[item.chunk_id] = scores.get(item.chunk_id, 0.0) + 1.0 / (k + rank + 1)
            representatives.setdefault(item.chunk_id, item)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        RetrievalResult(
            chunk_id=chunk_id,
            file_name=representatives[chunk_id].file_name,
            content=representatives[chunk_id].content,
            score=score,
            score_type=SCORE_TYPE_RRF,
        )
        for chunk_id, score in fused
    ]


def rerank_cosine(
    query: str,
    results: list[RetrievalResult],
    top_k: int = 5,
) -> list[RetrievalResult]:
    """向量精排：余弦 0.7 + 归一化 RRF 0.3 加权排序，逻辑与 AdvancedRetriever.rerank 一致。

    - 正常路径：score 重写为余弦相似度，score_type="cosine"
    - 编码失败回退：保持 RRF 原始排序，score_type 如实保持 "rrf"——不再拿
      RRF 分冒充余弦分（审计 #4），下游充分性判定只认 score_type="cosine"
    """
    if not results:
        return []

    try:
        query_vec = encode_query(query)
        content_vecs = encode_texts([r.content for r in results])

        # 归一化后点积 = 余弦相似度
        similarities = (content_vecs @ query_vec).tolist()

        # RRF 分数 min-max 归一化到 [0, 1]，与余弦分数同量纲后加权
        rrf_scores = [r.score for r in results]
        rrf_min, rrf_max = min(rrf_scores), max(rrf_scores)
        rrf_range = rrf_max - rrf_min
        norm_rrf = [
            (s - rrf_min) / rrf_range if rrf_range > 0 else 1.0
            for s in rrf_scores
        ]

        fused = [
            (r, float(sim), 0.7 * float(sim) + 0.3 * nr)
            for r, sim, nr in zip(results, similarities, norm_rrf)
        ]
        fused.sort(key=lambda x: x[2], reverse=True)

        # 阈值过滤作用在余弦分数上，防止 RRF 加分误放不相关结果
        threshold = settings.retrieval_similarity_threshold
        return [
            RetrievalResult(
                chunk_id=r.chunk_id,
                file_name=r.file_name,
                content=r.content,
                score=cos_sim,
                score_type=SCORE_TYPE_COSINE,
            )
            for r, cos_sim, _ in fused
            if cos_sim >= threshold
        ][:top_k]
    except Exception:
        # 编码失败：回退为 RRF 原始排序（不做阈值过滤，避免误伤），
        # score_type 保持 "rrf"，充分性判定不会采信这些分数
        return results[:top_k]
