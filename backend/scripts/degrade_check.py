"""
阶段 2 验收 #3：嵌入模型不可用时的降级行为端到端验证

通过 monkeypatch app.core.vector_store.get_embed_model 抛异常模拟嵌入模型
不可用（所有编码路径——向量召回 / 精排——都会随之失败）：

- graph 引擎：并行双路召回中 vector 分支降级为空，BM25 单路照常出答案，
  且 HTTP 层 /api/chat 返回 200
- legacy 引擎：hybrid_search 串行链路被向量检索异常拖死，qa_query 走 harness
  兜底（sources / reasoning_steps 为空）——证明 #3 是图侧修复

用法：python backend/scripts/degrade_check.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _dual_run_env import COLLECTION, SMOKE_DIR, setup_env

setup_env()

# ---- 在 import app 业务模块后、任何编码调用前，打爆嵌入模型 ----
import app.core.vector_store as vector_store  # noqa: E402


def _unavailable(*args, **kwargs):
    raise RuntimeError("嵌入模型不可用（验收模拟：get_embed_model 被 monkeypatch）")


vector_store.get_embed_model = _unavailable

from app.core.config import settings  # noqa: E402
from app.services.qa import qa_query  # noqa: E402

QUESTION = "Redis 支持哪些数据类型？"
OUT_PATH = SMOKE_DIR / "dual_run_degrade.json"


def main() -> None:
    report: dict = {"question": QUESTION, "collection": COLLECTION}

    # ---- graph 引擎：BM25 单路降级，必须正常出答案 ----
    settings.qa_engine = "graph"
    g = qa_query(QUESTION, collection_name=COLLECTION)
    g_ok = (
        bool(g["answer"])
        and not g["answer"].startswith("系统处理出错")
        and len(g["sources"]) > 0
        and len(g["reasoning_steps"]) > 0
    )
    report["graph"] = {
        "ok": g_ok,
        "answer_head": g["answer"][:200],
        "source_files": sorted({s.get("file_name", "") for s in g["sources"]}),
        "reasoning_steps": g["reasoning_steps"],
        "queries": g["queries"],
    }
    print(f"[graph] ok={g_ok} sources={report['graph']['source_files']}")
    print(f"[graph] steps={json.dumps(g['reasoning_steps'], ensure_ascii=False)}")

    # ---- legacy 引擎：预期整体失败（harness 兜底） ----
    settings.qa_engine = "legacy"
    l = qa_query(QUESTION, collection_name=COLLECTION)
    l_failed = (
        l["answer"].startswith("系统处理出错")
        and l["sources"] == []
        and l["reasoning_steps"] == []
    )
    report["legacy"] = {
        "failed_as_expected": l_failed,
        "answer_head": l["answer"][:200],
        "sources": l["sources"],
        "reasoning_steps": l["reasoning_steps"],
    }
    print(f"[legacy] failed_as_expected={l_failed} answer={l['answer'][:80]}")

    # ---- HTTP 层：graph 引擎 /api/chat 必须 200 ----
    settings.qa_engine = "graph"
    from fastapi.testclient import TestClient  # noqa: E402

    from app.main import app  # noqa: E402

    with TestClient(app) as client:
        r = client.post(
            "/api/chat",
            json={"question": QUESTION, "collection_name": COLLECTION},
        )
    http_ok = (
        r.status_code == 200
        and bool(r.json().get("reasoning_steps"))
        and bool(r.json().get("sources"))
    )
    report["http_graph"] = {
        "status_code": r.status_code,
        "ok": http_ok,
        "steps_count": len(r.json().get("reasoning_steps", [])) if r.status_code == 200 else 0,
    }
    print(f"[http graph] status={r.status_code} ok={http_ok}")

    report["pass"] = g_ok and l_failed and http_ok
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n降级验收 {'通过' if report['pass'] else '未通过'}，明细: {OUT_PATH}")
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
