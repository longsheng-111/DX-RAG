# -*- coding: utf-8 -*-
"""阶段 4：摄入队列化单元测试

覆盖：job store CRUD / 同名重传撤单 / 上传契约（立即返回 job_id）/
job 查询端点（含 stale-running 兜底）/ worker 撤单与重试分支。
队列投递统一 mock 掉，不依赖真实 Redis；worker 逻辑直接调函数验证。
"""
import asyncio
import os
import tempfile
from datetime import datetime, timedelta
from unittest import mock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def env_dirs(monkeypatch, tmp_path):
    """隔离全部数据目录（settings 是 import 期单例，用 monkeypatch 改实例字段）"""
    from app.core.config import settings

    monkeypatch.setattr(settings, "session_dir", str(tmp_path / "sessions"))
    monkeypatch.setattr(settings, "job_dir", str(tmp_path / "jobs"))
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path / "uploads"))
    monkeypatch.setattr(settings, "vector_persist_dir", str(tmp_path / "vdb"))

    # job_store 是 import 期单例，其目录在构造时固化；app 已被先前用例 import 过时，
    # 需把单例的 job_dir 一并指向本用例的临时目录，避免跨用例串目录
    from pathlib import Path

    from app.core.job_store import job_store

    job_store.job_dir = Path(settings.job_dir)
    job_store.job_dir.mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture()
def fresh_job_store(env_dirs):
    from app.core.job_store import JobStore

    return JobStore()


@pytest.fixture()
def client(env_dirs):
    from app.main import app

    return TestClient(app)


# ============ job store（file 后端） ============

def test_job_store_crud(fresh_job_store):
    job = fresh_job_store.create_job("a.txt", "kb1", "/tmp/a.txt")
    assert job["status"] == "queued" and job["progress"] == 0.0

    updated = fresh_job_store.update_job(job["job_id"], status="running", progress=0.4)
    assert updated["status"] == "running"
    assert updated["updated_at"] >= job["updated_at"]

    assert fresh_job_store.find_active_job("a.txt", "kb1")["job_id"] == job["job_id"]
    assert fresh_job_store.find_active_job("a.txt", "kb2") is None

    fresh_job_store.update_job(job["job_id"], status="done", progress=1.0, chunk_count=7)
    assert fresh_job_store.find_active_job("a.txt", "kb1") is None
    got = fresh_job_store.get_job(job["job_id"])
    assert got["status"] == "done" and got["chunk_count"] == 7


def test_job_store_rejects_bad_id(fresh_job_store):
    with pytest.raises(ValueError):
        fresh_job_store.get_job("../../etc/passwd")


# ============ 上传契约与 job 查询端点 ============

def test_upload_returns_job_immediately(client):
    with mock.patch("app.api.routes.enqueue_ingest", new=mock.AsyncMock()):
        resp = client.post(
            "/api/upload",
            files={"file": ("hello.txt", b"hello rag queue " * 10)},
            data={"collection_name": "knowledge_chunks"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "queued" and body["job_id"].startswith("job_")
    assert "chunks" not in body  # 阶段 4 契约：切片数挪到 job 查询结果

    job = client.get(f"/api/ingest-jobs/{body['job_id']}").json()
    assert job["status"] == "queued"
    assert job["file_name"] == "hello.txt"


def test_upload_enqueue_failure_returns_503(client):
    with mock.patch(
        "app.api.routes.enqueue_ingest", side_effect=RuntimeError("redis down")
    ):
        resp = client.post(
            "/api/upload",
            files={"file": ("hello.txt", b"hello " * 10)},
            data={"collection_name": "knowledge_chunks"},
        )
    assert resp.status_code == 503


def test_ingest_job_404(client):
    assert client.get("/api/ingest-jobs/job_nonexistent0").status_code == 404


def test_stale_running_marked_failed(client):
    """running 超过阈值分钟数无心跳 → 读取时标记 failed"""
    from app.core.config import settings
    from app.api import routes

    with mock.patch("app.api.routes.enqueue_ingest", new=mock.AsyncMock()):
        resp = client.post(
            "/api/upload",
            files={"file": ("stale.txt", b"stale " * 10)},
            data={"collection_name": "knowledge_chunks"},
        )
    job_id = resp.json()["job_id"]
    stale_since = (
        datetime.now() - timedelta(minutes=settings.ingest_job_stale_minutes + 1)
    ).isoformat()
    routes.job_store.update_job(job_id, status="running", updated_at=stale_since)
    # update_job 会自动刷新 updated_at，直接改文件绕过心跳刷新
    from pathlib import Path
    import json

    path = Path(settings.job_dir) / f"{job_id}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["updated_at"] = stale_since
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    job = client.get(f"/api/ingest-jobs/{job_id}").json()
    assert job["status"] == "failed"
    assert "无心跳" in (job["error"] or "")


def test_reupload_cancels_active_job(client):
    with mock.patch("app.api.routes.enqueue_ingest", new=mock.AsyncMock()):
        for name in ("first.txt", "first.txt"):
            resp = client.post(
                "/api/upload",
                files={"file": (name, b"reupload " * 10)},
                data={"collection_name": "knowledge_chunks"},
            )
        assert resp.status_code == 200
        # 第一次上传的任务应已被撤单；文件 job 记录里最新的是第二次的 queued
        from app.api import routes

        active = routes.job_store.find_active_job("first.txt", "knowledge_chunks")
        assert active is not None and active["job_id"] == resp.json()["job_id"]


# ============ worker 分支逻辑（直接调函数，不经 broker） ============

def test_worker_skips_cancelled_job(fresh_job_store):
    from app.worker import ingest_file

    job = fresh_job_store.create_job("b.txt", "kb1", "/tmp/b.txt")
    fresh_job_store.update_job(job["job_id"], status="cancelled")

    with mock.patch("app.worker.job_store", fresh_job_store):
        asyncio.run(ingest_file({}, job["job_id"]))
    assert fresh_job_store.get_job(job["job_id"])["status"] == "cancelled"


def test_worker_marks_failed_after_retry_exhausted(fresh_job_store, tmp_path):
    from app.worker import ingest_file, MAX_TRIES

    target = tmp_path / "empty.txt"
    target.write_text("x", encoding="utf-8")
    job = fresh_job_store.create_job("empty.txt", "kb1", str(target))

    with mock.patch("app.worker.job_store", fresh_job_store), \
         mock.patch(
             "app.worker.process_file", side_effect=ValueError("文件内容为空")
         ):
        with pytest.raises(ValueError):
            asyncio.run(ingest_file({"job_try": MAX_TRIES}, job["job_id"]))
    got = fresh_job_store.get_job(job["job_id"])
    assert got["status"] == "failed" and "文件内容为空" in got["error"]
