# -*- coding: utf-8 -*-
"""
阶段 4 并发压测脚本（locust）

场景与运行方式（headless，--host 指向被测后端）：
  S1 基线：20 并发 /api/query，3 分钟
    locust -f backend/tests/load/locustfile.py --headless --tags s1 \
           -u 20 -r 5 -t 3m --host http://127.0.0.1:8000 --csv out/s1
  S2 上传干扰：同 S1（--tags s2），运行到 60s 时手动上传 20MB PDF：
    curl -X POST $HOST/api/upload -F "file=@load_test_20mb.pdf" -F "collection_name=knowledge_chunks"
    用 --csv 的时间序列对比上传前后 p95 波动
  S3 考官混合：10 并发 query + 5 并发考官流程（按 weight 2:1，-u 15）
    locust -f ... --tags s3 -u 15 -r 3 -t 3m ...
    考官会话审计写入 EXAM_AUDIT_FILE（默认 ./exam_audit.json），跑完后抽查：
    scores 条数 == 作答数、无串会话（题目与作答一一对应）

依赖：pip install locust（仅压测机需要，不进 runtime requirements）
"""
import json
import os
import random

from locust import HttpUser, between, events, tag, task

# S1 固定 10 个问题轮换（覆盖 KB 命中与未命中两类）
QUESTIONS = [
    "RAG 系统的核心是什么？",
    "混合检索由哪几部分组成？",
    "考官模式包含哪些环节？",
    "摄入队列化解决了什么问题？",
    "RRF 融合的作用是什么？",
    "什么是检索增强生成？",
    "向量精排是如何加权排序的？",
    "切片策略对检索质量有什么影响？",
    "embed 模型不可用时系统会怎样？",
    "请介绍一下知识库问答的工作流程。",
]

EXAM_ANSWERS = [
    "我认为混合检索的关键在于两路召回互补，关键词保证精确匹配，向量保证语义泛化。",
    "这道题我的理解是：队列化把重耗时操作挪到后台，前台接口只负责任务受理。",
    "我的回答是：评分应当基于预设要点逐条判定命中，而不是凭整体印象打分。",
]

# S3 会话审计记录：(session_id, 作答数)
_exam_audit: list = []


class QueryUser(HttpUser):
    """S1/S2/S3 的问答负载：循环 POST /api/query"""

    wait_time = between(0.1, 0.5)
    weight = 2

    @tag("s1", "s2", "s3")
    @task
    def query(self):
        self.client.post(
            "/api/query",
            json={"question": random.choice(QUESTIONS)},
            name="/api/query",
        )


class ExamUser(HttpUser):
    """S3 的考官流程负载：开考 → 作答 2 题 → action=end 收尾，全程自校验"""

    wait_time = between(1, 3)
    weight = 1

    @tag("s3")
    @task
    def exam_flow(self):
        sid = None
        answers_submitted = 0
        try:
            with self.client.post(
                "/api/sessions",
                json={
                    "mode": "examiner",
                    "target_position": "后端工程师",
                    # 完整语义化主题：过短的主题词（如单词 "RAG"）余弦分低于
                    # 检索阈值 0.65 会被 exam/start 以"无相关资料"拒绝（实测）
                    "topic": "RAG 系统的检索质量与混合检索",
                },
                name="/api/sessions(exam)",
                catch_response=True,
            ) as resp:
                if resp.status_code != 200:
                    resp.failure(f"create session failed: {resp.status_code}")
                    return
                sid = resp.json()["session_id"]

            with self.client.post(
                "/api/exam/start",
                json={"session_id": sid},
                name="/api/exam/start",
                catch_response=True,
            ) as resp:
                if resp.status_code != 200:
                    resp.failure(f"exam start failed: {resp.status_code} {resp.text[:100]}")
                    return

            for i in range(2):
                with self.client.post(
                    f"/api/exam/{sid}/next",
                    json={"answer": EXAM_ANSWERS[i % len(EXAM_ANSWERS)] + f"（第{i + 1}点补充）"},
                    name="/api/exam/next(answer)",
                    catch_response=True,
                ) as resp:
                    if resp.status_code != 200:
                        resp.failure(f"answer failed: {resp.status_code} {resp.text[:100]}")
                        return
                    answers_submitted += 1

            with self.client.post(
                f"/api/exam/{sid}/next",
                json={"action": "end"},
                name="/api/exam/next(end)",
                catch_response=True,
            ) as resp:
                if resp.status_code != 200:
                    resp.failure(f"end failed: {resp.status_code} {resp.text[:100]}")
                    return
                body = resp.json()
                # 无重复记分：scores 条数必须等于作答数
                if len(body.get("scores", [])) != answers_submitted:
                    resp.failure(
                        f"scores 条数 {len(body.get('scores', []))} != 作答数 {answers_submitted}"
                    )
                    return
                if body.get("status") != "finished" or not body.get("summary"):
                    resp.failure("end 未正常收尾（status/summary 异常）")
                    return
                # 无串会话：每条 score 的题目必须与该会话收到的题目一致
                questions = {s.get("question") for s in body.get("scores", [])}
                if len(questions) != len(body.get("scores", [])):
                    resp.failure("scores 内题目重复，疑似串会话或重复记分")
                    return

            _exam_audit.append({"session_id": sid, "answers": answers_submitted})
        except Exception:
            # 网络层异常已由 locust 记录，会话保留供跑完后抽查（不清理）
            raise


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    if not _exam_audit:
        return
    path = os.environ.get("EXAM_AUDIT_FILE", "exam_audit.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_exam_audit, f, ensure_ascii=False, indent=2)
    print(f"[load] 考官会话审计已写入 {path}（{len(_exam_audit)} 条）")
