"""
阶段 2 验收：双跑对比（legacy 手写 ReAct vs LangGraph qa_graph）

- 题库 50 题（取自 test_data 八股内容，分 frontend_bagu / backend_bagu / general 三类），
  同步落盘 .smoke2/dual_run_questions.json
- 每题分别用 legacy / graph 两引擎各跑一遍 qa_query（分阶段设置
  settings.qa_engine，阶段内为常量，无并发竞争），ThreadPoolExecutor 4 路并发
- 结果逐条写入 .smoke2/dual_run_results.jsonl 并即写即 flush，中断后重跑自动
  跳过已完成记录（带 error 的记录视为未完成，重跑会重试）
- 对比指标：sources 的 file_name 集合 Jaccard + 答案 difflib.SequenceMatcher ratio；
  一致判定 = Jaccard >= 0.6 且 ratio >= 0.5

用法：
    python backend/scripts/dual_run_compare.py            # 续跑缺失记录 + 输出对比
    python backend/scripts/dual_run_compare.py --limit 5  # 只跑前 5 题（冒烟用）
    python backend/scripts/dual_run_compare.py --compare-only  # 只对比，不调 LLM
"""
import difflib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _dual_run_env import COLLECTION, SMOKE_DIR, setup_env

setup_env()

from app.core.config import settings  # noqa: E402
from app.services.qa import qa_query  # noqa: E402

# ============================================================
#  题库：50 题，按知识库三个来源目录分类（八股内容的事实性问法 + 改写）
# ============================================================
QUESTIONS: list[dict] = [
    # ---- frontend_bagu / vue.md（18 题：响应式原理 + 生命周期） ----
    {"qid": 1, "category": "frontend_bagu", "question": "Vue 2 是如何实现响应式的？"},
    {"qid": 2, "category": "frontend_bagu", "question": "Vue2 的响应式原理是什么？"},
    {"qid": 3, "category": "frontend_bagu", "question": "Vue 2 用哪个 API 实现数据响应式？"},
    {"qid": 4, "category": "frontend_bagu", "question": "Object.defineProperty 在 Vue 2 中起什么作用？"},
    {"qid": 5, "category": "frontend_bagu", "question": "Vue 3 的响应式是怎么实现的？"},
    {"qid": 6, "category": "frontend_bagu", "question": "Vue3 用什么实现响应式？"},
    {"qid": 7, "category": "frontend_bagu", "question": "Vue 3 响应式原理里的 Proxy 是做什么的？"},
    {"qid": 8, "category": "frontend_bagu", "question": "Vue 2 和 Vue 3 的响应式实现有什么区别？"},
    {"qid": 9, "category": "frontend_bagu", "question": "请介绍一下 Vue 的响应式原理。"},
    {"qid": 10, "category": "frontend_bagu", "question": "Vue 组件有哪些生命周期钩子？"},
    {"qid": 11, "category": "frontend_bagu", "question": "Vue 的生命周期包括哪些阶段？"},
    {"qid": 12, "category": "frontend_bagu", "question": "Vue 组件的 created 和 mounted 钩子分别是什么？"},
    {"qid": 13, "category": "frontend_bagu", "question": "Vue 生命周期中的 updated 和 destroyed 有什么用？"},
    {"qid": 14, "category": "frontend_bagu", "question": "列举 Vue 组件的生命周期钩子函数。"},
    {"qid": 15, "category": "frontend_bagu", "question": "Vue 组件在什么时候会触发 mounted 钩子？"},
    {"qid": 16, "category": "frontend_bagu", "question": "Vue 面试中响应式原理常考哪些点？"},
    {"qid": 17, "category": "frontend_bagu", "question": "Vue 3 为什么用 Proxy 而不是 defineProperty？"},
    {"qid": 18, "category": "frontend_bagu", "question": "说说 Vue 生命周期的主要内容。"},
    # ---- backend_bagu / redis.md（18 题：数据类型 + 持久化） ----
    {"qid": 19, "category": "backend_bagu", "question": "Redis 支持哪些数据类型？"},
    {"qid": 20, "category": "backend_bagu", "question": "Redis 有哪些基本数据结构？"},
    {"qid": 21, "category": "backend_bagu", "question": "Redis 的 String 和 Hash 类型分别是什么？"},
    {"qid": 22, "category": "backend_bagu", "question": "Redis 的 List、Set 和 Sorted Set 分别是什么？"},
    {"qid": 23, "category": "backend_bagu", "question": "列举 Redis 的五种数据类型。"},
    {"qid": 24, "category": "backend_bagu", "question": "Redis 数据类型面试常考什么？"},
    {"qid": 25, "category": "backend_bagu", "question": "Redis 支持哪几种持久化方式？"},
    {"qid": 26, "category": "backend_bagu", "question": "Redis 的持久化机制有哪些？"},
    {"qid": 27, "category": "backend_bagu", "question": "什么是 Redis 的 RDB 持久化？"},
    {"qid": 28, "category": "backend_bagu", "question": "什么是 Redis 的 AOF 持久化？"},
    {"qid": 29, "category": "backend_bagu", "question": "Redis 的 RDB 和 AOF 有什么区别？"},
    {"qid": 30, "category": "backend_bagu", "question": "请介绍 Redis 的持久化方案。"},
    {"qid": 31, "category": "backend_bagu", "question": "Redis 面试中持久化部分常问什么？"},
    {"qid": 32, "category": "backend_bagu", "question": "Redis 的 Sorted Set 属于哪种数据类型？"},
    {"qid": 33, "category": "backend_bagu", "question": "Redis 支持 Hash 类型吗？"},
    {"qid": 34, "category": "backend_bagu", "question": "Redis 支持 Set 集合类型吗？"},
    {"qid": 35, "category": "backend_bagu", "question": "Redis 有几种数据类型，分别叫什么？"},
    {"qid": 36, "category": "backend_bagu", "question": "Redis 持久化的两种方式分别叫什么？"},
    # ---- general / note.md（14 题：通用笔记 / 批量导入特性） ----
    {"qid": 37, "category": "general", "question": "这份通用笔记是用来测试什么功能的？"},
    {"qid": 38, "category": "general", "question": "通用笔记里提到批量导入支持哪些特性？"},
    {"qid": 39, "category": "general", "question": "批量导入支持多源路径吗？"},
    {"qid": 40, "category": "general", "question": "批量导入支持分类导入吗？"},
    {"qid": 41, "category": "general", "question": "文本处理支持 Markdown 标题切分吗？"},
    {"qid": 42, "category": "general", "question": "通用笔记的细节部分提到了哪些功能点？"},
    {"qid": 43, "category": "general", "question": "笔记中说的多源路径是什么意思？"},
    {"qid": 44, "category": "general", "question": "什么是分类导入？"},
    {"qid": 45, "category": "general", "question": "Markdown 标题切分是做什么的？"},
    {"qid": 46, "category": "general", "question": "通用笔记的介绍部分说了什么？"},
    {"qid": 47, "category": "general", "question": "这个测试笔记支持哪些导入方式？"},
    {"qid": 48, "category": "general", "question": "批量导入功能支持哪几个特性？"},
    {"qid": 49, "category": "general", "question": "通用笔记中列出的三个细节是什么？"},
    {"qid": 50, "category": "general", "question": "项目里的通用测试笔记包含哪些内容？"},
]

# 一致判定阈值（报告需注明）：sources file_name 集合 Jaccard >= 0.6 且
# 答案 SequenceMatcher ratio >= 0.5 视为两引擎结果一致
JACCARD_MIN = 0.6
ANSWER_RATIO_MIN = 0.5

QUESTIONS_PATH = SMOKE_DIR / "dual_run_questions.json"
RESULTS_PATH = SMOKE_DIR / "dual_run_results.jsonl"
SUMMARY_PATH = SMOKE_DIR / "dual_run_compare_summary.json"


def _load_done() -> set[tuple[int, str]]:
    """读取已完成记录（带 error 的视为未完成，允许续跑重试）"""
    done: set[tuple[int, str]] = set()
    if not RESULTS_PATH.exists():
        return done
    with open(RESULTS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # 中断造成的半行，忽略
            if not rec.get("error"):
                done.add((rec["qid"], rec["engine"]))
    return done


def _run_one(q: dict, engine: str) -> dict:
    """单题单引擎执行，异常落 error 字段不中断整批"""
    start = time.time()
    rec = {
        "qid": q["qid"],
        "category": q["category"],
        "question": q["question"],
        "engine": engine,
    }
    try:
        result = qa_query(q["question"], collection_name=COLLECTION)
        answer = result.get("answer", "")
        rec.update(
            answer=answer,
            sources=result.get("sources", []),
            queries=result.get("queries", []),
            reasoning_steps=result.get("reasoning_steps", []),
            token_cost=result.get("token_cost", {}),
            error=None,
        )
        # qa_query 内部兜底（harness fallback_error）说明该次执行撞上检索/编码
        # 基础设施异常（如并发首次编码的 meta tensor 抖动），不是引擎语义差异：
        # 标记 error 让续跑重试，且对比时剔除
        if answer.startswith("系统处理出错"):
            rec["error"] = f"harness_fallback: {answer[:120]}"
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
    rec["elapsed_s"] = round(time.time() - start, 2)
    return rec


def run_phase(engine: str, questions: list[dict], done: set[tuple[int, str]]) -> None:
    """单引擎整批执行：阶段内 settings.qa_engine 为常量，4 路并发无竞争"""
    todo = [q for q in questions if (q["qid"], engine) not in done]
    if not todo:
        print(f"[{engine}] 全部完成，跳过")
        return
    settings.qa_engine = engine
    print(f"[{engine}] 待跑 {len(todo)} 题（qa_engine={settings.qa_engine}）")
    finished = 0
    with ThreadPoolExecutor(max_workers=4) as pool, open(
        RESULTS_PATH, "a", encoding="utf-8"
    ) as f:
        futures = {pool.submit(_run_one, q, engine): q for q in todo}
        for fut in as_completed(futures):
            rec = fut.result()
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()  # 即写即 flush，中断可续跑
            finished += 1
            tag = "ERR" if rec.get("error") else "ok"
            print(
                f"[{engine}] {finished}/{len(todo)} qid={rec['qid']} {tag} "
                f"{rec['elapsed_s']}s",
                flush=True,
            )


# ============================================================
#  对比
# ============================================================

def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def compare() -> dict:
    records: dict[int, dict[str, dict]] = {}
    with open(RESULTS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("error"):
                continue
            records.setdefault(rec["qid"], {})[rec["engine"]] = rec

    qmeta = {q["qid"]: q for q in QUESTIONS}
    pairs = []
    for qid, engines in sorted(records.items()):
        if "legacy" not in engines or "graph" not in engines:
            continue
        leg, gra = engines["legacy"], engines["graph"]
        leg_files = {s.get("file_name", "") for s in leg.get("sources", [])}
        gra_files = {s.get("file_name", "") for s in gra.get("sources", [])}
        j = _jaccard(leg_files, gra_files)
        ratio = difflib.SequenceMatcher(
            None, leg.get("answer", ""), gra.get("answer", "")
        ).ratio()
        consistent = j >= JACCARD_MIN and ratio >= ANSWER_RATIO_MIN
        pairs.append(
            {
                "qid": qid,
                "category": qmeta.get(qid, {}).get("category", leg.get("category", "")),
                "question": leg.get("question", ""),
                "sources_jaccard": round(j, 4),
                "answer_ratio": round(ratio, 4),
                "consistent": consistent,
                "legacy_files": sorted(leg_files),
                "graph_files": sorted(gra_files),
                "legacy_steps": [
                    {k: s.get(k) for k in ("step", "action", "query", "hits", "top_score", "confidence")}
                    for s in leg.get("reasoning_steps", [])
                ],
                "graph_steps": [
                    {k: s.get(k) for k in ("step", "action", "query", "hits", "top_score", "confidence")}
                    for s in gra.get("reasoning_steps", [])
                ],
                "legacy_answer_head": leg.get("answer", "")[:300],
                "graph_answer_head": gra.get("answer", "")[:300],
            }
        )

    def _rate(items: list[dict]) -> float:
        return sum(1 for p in items if p["consistent"]) / len(items) if items else 0.0

    categories = sorted({p["category"] for p in pairs})
    summary = {
        "thresholds": {"sources_jaccard_min": JACCARD_MIN, "answer_ratio_min": ANSWER_RATIO_MIN},
        "paired": len(pairs),
        "consistent": sum(1 for p in pairs if p["consistent"]),
        "overall_rate": round(_rate(pairs), 4),
        "by_category": {
            c: {
                "total": len([p for p in pairs if p["category"] == c]),
                "consistent": sum(1 for p in pairs if p["category"] == c and p["consistent"]),
                "rate": round(_rate([p for p in pairs if p["category"] == c]), 4),
            }
            for c in categories
        },
        "inconsistent_qids": [p["qid"] for p in pairs if not p["consistent"]],
        "pairs": pairs,
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===== 双跑对比汇总 =====")
    print(f"成对题目: {summary['paired']}，一致: {summary['consistent']}，"
          f"总一致率: {summary['overall_rate'] * 100:.1f}%")
    for c, stat in summary["by_category"].items():
        print(f"  {c}: {stat['consistent']}/{stat['total']} = {stat['rate'] * 100:.1f}%")
    print(f"不一致题 qid: {summary['inconsistent_qids']}")

    # 差异复核素材：前 5 道不一致题的两侧答案摘要 + sources + 推理步骤
    print("\n===== 不一致题复核素材（前 5 道） =====")
    for p in [p for p in pairs if not p["consistent"]][:5]:
        print(f"\n--- qid={p['qid']} [{p['category']}] {p['question']}")
        print(f"  Jaccard={p['sources_jaccard']} ratio={p['answer_ratio']}")
        print(f"  legacy sources: {p['legacy_files']}")
        print(f"  graph  sources: {p['graph_files']}")
        print(f"  legacy steps: {json.dumps(p['legacy_steps'], ensure_ascii=False)}")
        print(f"  graph  steps: {json.dumps(p['graph_steps'], ensure_ascii=False)}")
        print(f"  legacy 答案摘要: {p['legacy_answer_head'][:200]}")
        print(f"  graph  答案摘要: {p['graph_answer_head'][:200]}")

    return summary


def main() -> None:
    args = sys.argv[1:]
    compare_only = "--compare-only" in args
    limit = None
    if "--limit" in args:
        limit = int(args[args.index("--limit") + 1])

    questions = QUESTIONS[:limit] if limit else QUESTIONS
    with open(QUESTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(QUESTIONS, f, ensure_ascii=False, indent=2)

    if not compare_only:
        # 预热嵌入模型：并发首次编码可能撞 torch meta tensor 抖动，先串行编码一次
        from app.core.storage import encode_query

        encode_query("warmup")
        done = _load_done()
        print(f"已完成记录 {len(done)} 条，断点续跑")
        for engine in ("legacy", "graph"):
            run_phase(engine, questions, done)
            done = _load_done()

    if RESULTS_PATH.exists():
        compare()
    else:
        print("无结果文件，跳过对比")


if __name__ == "__main__":
    main()
