"""
检索层升级：BM25 + 密集向量 + RRF 融合 + 向量精排加权融合

召回流程：双路召回 → RRF 融合 → 向量精排加权融合
1. BM25 召回 top 20
2. 向量检索召回 top 20
3. RRF 融合得到 top 15
4. 向量精排：余弦相似度（0.7）与归一化 RRF 分数（0.3）加权融合排序，取 top 5
"""
import os
import re
import numpy as np
from typing import Optional

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

from app.core.config import settings
from app.core.vector_store import (
    vector_search,
    get_vector_store,
    encode_query,
    encode_texts,
    _meta_path,
)

# 充分性判断阈值（供 agent.py import）：高于检索过滤阈值 0.65，
# 让“召回有结果但相关性平庸”的中间地带能触发 Agent 改写重试
SUFFICIENCY_THRESHOLD = 0.75

# BM25 索引模块级缓存：AdvancedRetriever 每次请求都会新建实例，
# 缓存必须放在模块级才能跨请求复用。
# key 为 (collection_name, metadata.json 的 mtime)，文档增删导致 mtime 变化时才重建。
_bm25_cache: dict[
    tuple[str, float],
    tuple[list[str], list[dict], list[list[str]], Optional[BM25Okapi]],
] = {}


class AdvancedRetriever:
    """高级混合检索器"""

    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.documents: list[str] = []
        self.metadatas: list[dict] = []
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
        """从向量库构建 BM25 索引（模块级缓存，metadata.json 未变化时直接复用）"""
        try:
            mtime = os.path.getmtime(_meta_path(self.collection_name))
        except OSError:
            mtime = 0.0

        cache_key = (self.collection_name, mtime)
        cached = _bm25_cache.get(cache_key)
        if cached is None:
            try:
                data = get_vector_store(self.collection_name)
            except Exception:
                data = {"documents": [], "metadatas": [], "ids": []}

            documents = data.get("documents", []) or []
            metadatas = data.get("metadatas", []) or []
            tokenized_docs = [self._tokenize(d) for d in documents]
            bm25 = None
            if BM25Okapi is not None and tokenized_docs:
                bm25 = BM25Okapi(tokenized_docs)

            # 同一知识库只保留最新 mtime 的缓存条目，避免旧版本堆积
            for key in [k for k in _bm25_cache if k[0] == self.collection_name]:
                del _bm25_cache[key]
            _bm25_cache[cache_key] = (documents, metadatas, tokenized_docs, bm25)
            cached = _bm25_cache[cache_key]

        self.documents, self.metadatas, self.tokenized_docs, self.bm25 = cached

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
