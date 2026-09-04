"""
阶段 2 验收 #2：reasoning_steps 前端契约端到端验证

TestClient 起真实 app（STORAGE_BACKEND=file，QA_ENGINE=graph，真实 DeepSeek），
POST /api/chat 对 dual_run_kb 真实问一题，断言：
- HTTP 200
- reasoning_steps 非空、queries 非空
- reasoning_steps 元素结构与前端 QAPanel.tsx 的 ReasoningStep 类型
  （step/action/query/hits/top_score/confidence）逐字段匹配

用法：python backend/scripts/contract_check.py
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _dual_run_env import COLLECTION, SMOKE_DIR, setup_env

# QA_ENGINE 必须在 import app 前设置
os.environ["QA_ENGINE"] = "graph"
setup_env()

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

QUESTION = "Vue 3 用什么实现响应式？"
OUT_PATH = SMOKE_DIR / "dual_run_contract.json"

# 前端 ReasoningStep 类型（frontend/components/QAPanel.tsx:54-61）：
#   step: number; action: string; query: string; hits: number;
#   top_score: number; confidence: 'high' | 'medium' | 'low'
FIELD_TYPES = {
    "step": int,
    "action": str,
    "query": str,
    "hits": int,
    "top_score": (int, float),
    "confidence": str,
}
CONFIDENCE_ENUM = {"high", "medium", "low"}


def main() -> None:
    with TestClient(app) as client:
        r = client.post(
            "/api/chat",
            json={"question": QUESTION, "collection_name": COLLECTION},
        )

    report: dict = {"question": QUESTION, "status_code": r.status_code}
    if r.status_code != 200:
        report["pass"] = False
        report["error"] = r.text[:500]
        _finish(report)
        return

    data = r.json()
    steps = data.get("reasoning_steps", [])
    queries = data.get("queries", [])

    field_checks = []
    steps_ok = bool(steps)
    for s in steps:
        for field, types in FIELD_TYPES.items():
            ok = field in s and isinstance(s[field], types)
            # bool 是 int 子类，排除
            if ok and types is int and isinstance(s[field], bool):
                ok = False
            field_checks.append({"field": field, "ok": ok})
            steps_ok = steps_ok and ok
        steps_ok = steps_ok and s.get("confidence") in CONFIDENCE_ENUM

    report.update(
        reasoning_steps=steps,
        queries=queries,
        sources_count=len(data.get("sources", [])),
        answer_head=data.get("answer", "")[:200],
        checks={
            "http_200": True,
            "reasoning_steps_non_empty": bool(steps),
            "queries_non_empty": bool(queries),
            "fields_match_frontend_type": steps_ok,
        },
    )
    report["pass"] = all(report["checks"].values())
    _finish(report)


def _finish(report: dict) -> None:
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report.get("checks", {}), ensure_ascii=False, indent=2))
    print(f"steps={json.dumps(report.get('reasoning_steps', []), ensure_ascii=False)}")
    print(f"queries={report.get('queries')}")
    print(f"契约验收 {'通过' if report['pass'] else '未通过'}，明细: {OUT_PATH}")
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
