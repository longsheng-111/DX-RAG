"""
QA Graph（app/graphs/qa_graph.py）与 QA_ENGINE 接线测试

- 假 LLM：monkeypatch generate_func 与 rewrite 的 LLM 调用，绝不发真实 API 请求
- 假检索：monkeypatch qa_graph 模块内的 vector_branch / bm25_branch
- 假嵌入：monkeypatch retrieval 的 encode_query / encode_texts（512 维确定性向量，
  通过构造向量方向精确控制余弦分数，触发/不触发充分性阈值 0.75）

覆盖审计修复：#2（reasoning_steps/queries 透传）、#3（单路检索异常降级）、
#4（rerank 回退不拿 RRF 分冒充余弦分，充分性只信 cosine）。
"""
import sys
import types
from pathlib import Path

import numpy as np
import pytest

# ---- 测试环境准备：必须在 import app 之前完成 ----
_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# venv_pg 无 sentence-transformers / torch：file 后端的 vector_store 在 import 期
# 硬依赖该包（db 版是延迟导入）。本文件全程使用假嵌入，故打桩一个占位模块，
# 真实模型一旦意外被实例化会立即报错暴露
if "sentence_transformers" not in sys.modules:
    _st_stub = types.ModuleType("sentence_transformers")

    class SentenceTransformer:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("测试环境无真实嵌入模型，应使用假嵌入")

    _st_stub.SentenceTransformer = SentenceTransformer
    sys.modules["sentence_transformers"] = _st_stub

from app.core.config import settings  # noqa: E402
from app.services import qa as qa_service  # noqa: E402
from app.services import retrieval  # noqa: E402
from app.services.retrieval import (  # noqa: E402
    RetrievalResult,
    merge_rrf,
    rerank_cosine,
)
from app.services.token_tracker import TokenCost  # noqa: E402
from app.graphs import qa_graph  # noqa: E402
from app.graphs.qa_graph import qa_graph_query  # noqa: E402

EMB_DIM = 512


# ============================================================
#  假嵌入 / 假检索 / 假 LLM
# ============================================================

def _unit_with_cos(cos_target: float) -> np.ndarray:
    """构造与查询向量（e0）余弦相似度恰为 cos_target 的 512 维单位向量"""
    v = np.zeros(EMB_DIM, dtype=np.float32)
    v[0] = cos_target
    v[1] = np.sqrt(max(0.0, 1.0 - cos_target ** 2))
    return v


@pytest.fixture
def fake_embeddings(monkeypatch):
    """按内容控制余弦分数的假嵌入，返回 {content: 目标余弦分} 字典供用例填写"""
    scores: dict = {}

    def encode_query(_query: str) -> np.ndarray:
        v = np.zeros(EMB_DIM, dtype=np.float32)
        v[0] = 1.0
        return v

    def encode_texts(texts: list) -> np.ndarray:
        return np.stack([_unit_with_cos(scores.get(t, 0.0)) for t in texts])

    monkeypatch.setattr(retrieval, "encode_query", encode_query)
    monkeypatch.setattr(retrieval, "encode_texts", encode_texts)
    return scores


def _rr(idx: int, content: str, score: float = 1.0, score_type: str = "cosine"):
    return RetrievalResult(
        chunk_id=f"chunk_{idx}",
        file_name=f"file_{idx}.txt",
        content=content,
        score=score,
        score_type=score_type,
    )


@pytest.fixture
def fake_branches(monkeypatch):
    """按 query 返回预设结果的假检索分支，返回 (vector_data, bm25_data) 供用例填写"""
    vector_data: dict = {}
    bm25_data: dict = {}
    monkeypatch.setattr(
        qa_graph, "vector_branch",
        lambda q, c, top_k=20: list(vector_data.get(q, [])),
    )
    monkeypatch.setattr(
        qa_graph, "bm25_branch",
        lambda q, c, top_k=20: list(bm25_data.get(q, [])),
    )
    return vector_data, bm25_data


@pytest.fixture(autouse=True)
def _no_real_rewrite(monkeypatch):
    """兜底：默认 rewrite 原样返回（等价改写失败即终止），绝不触发真实 LLM 调用"""
    monkeypatch.setattr(
        qa_graph, "rewrite_query", lambda q, results, h: (q, TokenCost())
    )


def _fake_generate(answer="测试答案", cost=None):
    def generate(question, context_chunks, history):
        return answer, (cost or TokenCost(10, 5, 15))

    return generate


# ============================================================
#  ① 首轮充分直接生成
# ============================================================

def test_first_round_sufficient(fake_embeddings, fake_branches):
    vector_data, bm25_data = fake_branches
    vector_data["什么是RAG"] = [_rr(1, "RAG 是检索增强生成")]
    bm25_data["什么是RAG"] = [_rr(2, "RAG 结合检索与生成")]
    fake_embeddings["RAG 是检索增强生成"] = 0.90
    fake_embeddings["RAG 结合检索与生成"] = 0.85

    result = qa_graph_query("什么是RAG", generate_func=_fake_generate())

    assert result["answer"] == "测试答案"
    assert result["iterations"] == 1
    assert result["queries"] == ["什么是RAG"]
    assert result["rewritten_query"] is None
    assert [s["action"] for s in result["reasoning_steps"]] == ["knowledge_search"]
    # 0.90 >= 0.75 + 0.1 → high，且已回填到检索步骤
    assert result["reasoning_steps"][0]["confidence"] == "high"
    assert result["sources"][0]["id"] == "src_1"
    assert result["sources"][0]["similarity"] == 0.90
    assert result["token_cost"]["total_tokens"] == 15


# ============================================================
#  ② 不充分 → 改写 → 第二轮充分，reasoning_steps/queries 累加正确
# ============================================================

def test_rewrite_then_sufficient(fake_embeddings, fake_branches, monkeypatch):
    vector_data, _ = fake_branches
    vector_data["原始问题"] = [_rr(1, "沾边内容A")]
    vector_data["改写查询B"] = [_rr(2, "精准内容B")]
    # 0.70 过检索过滤阈值 0.65，但低于充分性阈值 0.75
    fake_embeddings["沾边内容A"] = 0.70
    fake_embeddings["精准内容B"] = 0.90
    monkeypatch.setattr(
        qa_graph, "rewrite_query",
        lambda q, results, h: ("改写查询B", TokenCost(5, 2, 7)),
    )

    result = qa_graph_query("原始问题", generate_func=_fake_generate())

    assert result["iterations"] == 2
    assert result["queries"] == ["原始问题", "改写查询B"]
    assert result["rewritten_query"] == "改写查询B"
    steps = result["reasoning_steps"]
    assert [s["action"] for s in steps] == [
        "knowledge_search", "query_rewrite", "knowledge_search",
    ]
    assert [s["step"] for s in steps] == [1, 2, 3]
    # query_rewrite 步骤位于两次 knowledge_search 之间，confidence 固定 medium
    assert steps[1]["confidence"] == "medium"
    # 最终 0.90 → high，两个 knowledge_search 步骤统一回填
    assert steps[0]["confidence"] == "high"
    assert steps[2]["confidence"] == "high"


# ============================================================
#  ③ 3 轮上限停止
# ============================================================

def test_max_iterations_stop(fake_embeddings, fake_branches, monkeypatch):
    vector_data, _ = fake_branches
    for query, idx in [("Q0", 1), ("Q1", 2), ("Q2", 3)]:
        # 各轮内容完全不同，避免提前触发 Jaccard 循环检测
        content = f"第{idx}轮完全不同的内容" * (idx + 1)
        vector_data[query] = [_rr(idx, content)]
        fake_embeddings[content] = 0.70
    outputs = iter(["Q1", "Q2"])
    monkeypatch.setattr(
        qa_graph, "rewrite_query",
        lambda q, results, h: (next(outputs), TokenCost(5, 2, 7)),
    )

    result = qa_graph_query("Q0", generate_func=_fake_generate())

    assert result["iterations"] == 3
    assert result["queries"] == ["Q0", "Q1", "Q2"]
    assert result["rewritten_query"] == "Q2"
    assert [s["action"] for s in result["reasoning_steps"]] == [
        "knowledge_search", "query_rewrite",
        "knowledge_search", "query_rewrite",
        "knowledge_search",
    ]


# ============================================================
#  ④ Jaccard 0.8 循环检测停止
# ============================================================

def test_loop_detection_stop(fake_embeddings, fake_branches, monkeypatch):
    vector_data, _ = fake_branches
    # 改写后检索结果与上一轮完全一致 → Jaccard = 1.0 > 0.8 → 终止
    shared = [_rr(1, "相同内容甲乙丙"), _rr(2, "相同内容丁戊己")]
    vector_data["Q0"] = shared
    vector_data["Q1"] = list(shared)
    fake_embeddings["相同内容甲乙丙"] = 0.70
    fake_embeddings["相同内容丁戊己"] = 0.70
    monkeypatch.setattr(
        qa_graph, "rewrite_query",
        lambda q, results, h: ("Q1", TokenCost(5, 2, 7)),
    )

    result = qa_graph_query("Q0", generate_func=_fake_generate())

    assert result["iterations"] == 2  # 第二轮触发循环检测，不打满 3 轮
    assert result["queries"] == ["Q0", "Q1"]


# ============================================================
#  ⑤ vector 分支抛异常 → BM25 单路照样出答案（审计 #3）
# ============================================================

def test_vector_branch_failure_degrades(fake_embeddings, fake_branches, monkeypatch):
    def boom(query, collection_name, top_k=20):
        raise RuntimeError("向量库不可用")

    monkeypatch.setattr(qa_graph, "vector_branch", boom)
    _, bm25_data = fake_branches
    bm25_data["Q"] = [_rr(1, "BM25 单路命中的内容", score=3.5, score_type="bm25")]
    fake_embeddings["BM25 单路命中的内容"] = 0.90

    result = qa_graph_query("Q", generate_func=_fake_generate())

    assert result["answer"] == "测试答案"
    assert result["iterations"] == 1
    assert len(result["sources"]) == 1
    assert result["sources"][0]["content"] == "BM25 单路命中的内容"


# ============================================================
#  ⑥ rerank 编码失败回退：score_type="rrf"，充分性判定不采信（审计 #4）
# ============================================================

def _raise_encode(_texts):
    raise RuntimeError("嵌入模型不可用")


def test_rerank_fallback_keeps_rrf_score_type(monkeypatch):
    """rerank_cosine 编码失败：保持 RRF 原始排序，score_type 如实为 "rrf\""""
    monkeypatch.setattr(retrieval, "encode_query", _raise_encode)
    results = [
        _rr(1, "内容A", score=0.033, score_type="rrf"),
        _rr(2, "内容B", score=0.016, score_type="rrf"),
    ]

    reranked = rerank_cosine("Q", results, top_k=5)

    assert [r.score_type for r in reranked] == ["rrf", "rrf"]
    assert reranked[0].score == 0.033  # 原 RRF 分，不冒充余弦


def test_rrf_only_scores_never_sufficient(fake_branches, monkeypatch):
    """嵌入整体不可用 → 精排全部回退为 rrf 分 → 充分性永不通过，打满 3 轮"""
    monkeypatch.setattr(retrieval, "encode_query", _raise_encode)
    monkeypatch.setattr(retrieval, "encode_texts", _raise_encode)
    vector_data, _ = fake_branches
    # 分支自带的分数再高也不作数：充分性只信精排后的 cosine 分
    vector_data["Q0"] = [_rr(1, "内容X", score=0.99)]
    vector_data["Q1"] = [_rr(2, "内容Y", score=0.99)]
    vector_data["Q2"] = [_rr(3, "内容Z", score=0.99)]
    outputs = iter(["Q1", "Q2"])
    monkeypatch.setattr(
        qa_graph, "rewrite_query",
        lambda q, results, h: (next(outputs), TokenCost(5, 2, 7)),
    )

    result = qa_graph_query("Q0", generate_func=_fake_generate())

    assert result["iterations"] == 3
    # RRF 分（~0.03）远低于 0.75，整体置信度 low
    assert result["reasoning_steps"][0]["confidence"] == "low"


# ============================================================
#  ⑦ qa_query 在 legacy / graph 两种 QA_ENGINE 下均透传
#     非空 reasoning_steps / queries（审计 #2）
# ============================================================

@pytest.mark.parametrize("engine", ["legacy", "graph"])
def test_qa_query_passes_through_reasoning(
    engine, fake_embeddings, fake_branches, monkeypatch
):
    monkeypatch.setattr(settings, "qa_engine", engine)
    # 两条路径共用同一个 generate_answer 注入点
    monkeypatch.setattr(
        qa_service, "generate_answer",
        lambda q, chunks, hist, sp="", sid=None: ("答案", TokenCost(10, 5, 15)),
    )
    # legacy 路径：打桩 hybrid_search（baseline，一行未改）
    monkeypatch.setattr(
        qa_service.AdvancedRetriever, "hybrid_search",
        lambda self, query, top_k=10: [("a.txt", "RAG 是检索增强生成", 0.9)],
    )
    # graph 路径：打桩检索分支
    vector_data, _ = fake_branches
    vector_data["什么是RAG"] = [_rr(1, "RAG 是检索增强生成")]
    fake_embeddings["RAG 是检索增强生成"] = 0.90

    # 用不存在的知识库名，避免 AdvancedRetriever 初始化时读取真实向量库
    result = qa_service.qa_query("什么是RAG", collection_name="test_empty_kb_qa_graph")

    assert result["answer"] == "答案"
    assert result["reasoning_steps"]
    assert result["reasoning_steps"][0]["action"] == "knowledge_search"
    assert result["queries"] == ["什么是RAG"]


# ============================================================
#  ⑧ token_cost 汇总：改写轮次 + 生成轮次逐笔累加
# ============================================================

def test_token_cost_aggregation(fake_embeddings, fake_branches, monkeypatch):
    vector_data, _ = fake_branches
    vector_data["Q0"] = [_rr(1, "沾边内容")]
    vector_data["Q1"] = [_rr(2, "精准内容")]
    fake_embeddings["沾边内容"] = 0.70
    fake_embeddings["精准内容"] = 0.90
    monkeypatch.setattr(
        qa_graph, "rewrite_query",
        lambda q, results, h: ("Q1", TokenCost(5, 2, 7)),
    )

    result = qa_graph_query(
        "Q0", generate_func=_fake_generate(cost=TokenCost(10, 5, 15))
    )

    # 改写 (5/2/7) + 生成 (10/5/15)
    assert result["token_cost"]["prompt_tokens"] == 15
    assert result["token_cost"]["completion_tokens"] == 7
    assert result["token_cost"]["total_tokens"] == 22


# ============================================================
#  补充：merge_rrf 融合键为 chunk_id（修复 content[:300] 碰撞）
# ============================================================

def test_merge_rrf_uses_chunk_id_not_content_prefix():
    same_prefix = "相同前缀" * 100  # 前 300 字完全相同，旧键会错误合并
    branch_a = [_rr(1, same_prefix + "甲", score_type="bm25")]
    branch_b = [_rr(2, same_prefix + "乙", score_type="cosine")]

    merged = merge_rrf([branch_a, branch_b])

    assert len(merged) == 2
    assert {r.chunk_id for r in merged} == {"chunk_1", "chunk_2"}
    assert all(r.score_type == "rrf" for r in merged)


def test_merge_rrf_same_chunk_boosts_score():
    hit_bm25 = [_rr(1, "内容", score_type="bm25")]
    hit_cosine = [_rr(1, "内容", score_type="cosine")]

    merged = merge_rrf([hit_bm25, hit_cosine])

    assert len(merged) == 1
    # 同一 chunk 两路各排第 1：1/61 + 1/61
    assert abs(merged[0].score - 2.0 / 61) < 1e-9
