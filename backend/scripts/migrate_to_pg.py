"""
一次性迁移脚本：文件存储 → 数据库（阶段 1 波次 3，对应设计文档第 6 节）

迁移内容（幂等，可重复跑）：
1. vector_db/_collections.json          → collections 表（name 冲突加后缀并记录映射表）
2. 每知识库 metadata.json + index.faiss → files / chunks 表
   - embedding 不重新计算：faiss.read_index + index.reconstruct(i) 逐条导出
     （本项目索引为 IndexFlatIP，支持 reconstruct；失败的条目 embedding 置空并记入报告）
   - file 粒度按 metadata.file_name 聚合，chunk_index 保序
   - file_hash 取原始文件内容 sha256（优先 metadata.source 路径，其次 upload_dir/file_name；
     原始文件缺失则置空并记入报告）
3. data/sessions/sess_*.json            → sessions 表（messages/compressed_history/exam_state
   整存 JSON 列，total_tokens/created_at/updated_at 保原值，naive 本地时间口径与
   app/db/session_repository.py 的 _fill_row 一致）
4. data/feedback/feedbacks.json         → feedbacks 表（按 session_id 分组拍平；
   孤儿反馈——session 不存在——跳过并记入报告，PG 有 FK CASCADE，sqlite 同理跳过）
5. 校验报告：每知识库 chunks 数比对（metadata ids 数 vs DB 行数）、sessions/feedbacks
   条数比对、reconstruct 失败清单、跳过清单；结尾打印零丢失结论，非零丢失退出码为 1

幂等键：主键由源数据身份确定性生成（uuid5），目标表已有同 id 记录即跳过；
重复跑报告应显示 0 新增。

用法：
    py backend/scripts/migrate_to_pg.py [--source-dir DIR] [--session-dir DIR]
        [--feedback-file FILE] [--upload-dir DIR] [--yes]
源目录默认复用 settings（vector_persist_dir / session_dir / upload_dir，
已含中文路径回退 ~/.dx-rag 的逻辑，不要自己拼路径）；目标库由 DATABASE_URL 决定。
"""
import argparse
import hashlib
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

# 将 backend/ 加入 sys.path，以便导入 app 模块
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import faiss  # noqa: E402
from sqlalchemy import func, select  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.db.base import Base, get_engine, new_session  # noqa: E402
from app.db.models import (  # noqa: E402
    EMBEDDING_DIM,
    Chunk,
    Collection,
    Feedback,
    File,
    Session as DbSession,
)

# 迁移专用 uuid5 命名空间：主键由源数据身份确定性生成，重复跑命中同 id 即跳过
_NS = uuid.uuid5(uuid.NAMESPACE_URL, "https://dx-rag.local/migrate-to-pg")


def _parse_dt(value) -> datetime:
    """
    JSON 里的时间字符串 → naive datetime。
    与 SessionRepository._fill_row 的写入口径一致（datetime.fromisoformat，
    剥掉时区保留 naive 本地语义）；缺失/非法时回退当前时间。
    """
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value)
            return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt
        except ValueError:
            pass
    return datetime.now()


def _sha256_file(path: Path) -> str:
    """分块读取计算文件内容 sha256（避免大文件一次性读入内存）"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _new_report() -> dict:
    """迁移/校验报告结构（main 打印 + 测试断言用）"""
    return {
        # 每知识库明细：[{name, db_name, files_new, files_skipped, chunks_new,
        #                chunks_skipped, expected_chunks, db_chunks, ok}]
        "collections": [],
        "name_mappings": [],        # name 冲突映射 [{source, db_name}]
        "reconstruct_failures": [],  # [{collection, chunk_id, faiss_index, error}]
        "notes": [],                # 索引缺失/维度不符/数量不一致/原始文件缺失等说明
        "sessions": {"expected": 0, "new": 0, "skipped": 0,
                     "failed": [], "ids": [], "db_present": 0},
        "feedbacks": {"expected": 0, "new": 0, "skipped": 0,
                      "orphans": [], "ids": [], "db_present": 0, "read_error": None},
        "zero_loss": True,          # 校验结论
        "losses": [],               # 丢失/不一致明细
    }


# ============================================================
#  步骤 1：_collections.json → collections 表
# ============================================================

def _ensure_collection(db, name: str, report: dict):
    """
    插入知识库并返回 (collection_id, DB 实际 name)。
    - 同 id（= 本脚本之前迁移过）→ 直接返回，幂等跳过
    - 同名不同 id（迁移前库里已有数据）→ 加 _migrated 后缀并记录映射表
    """
    cid = uuid.uuid5(_NS, f"collection:{name}")
    row = db.get(Collection, cid)
    if row is not None:
        return row.id, row.name

    actual = name
    if db.scalar(select(Collection.id).where(Collection.name == name)) is not None:
        candidate = f"{name}_migrated"
        k = 2
        while db.scalar(select(Collection.id).where(Collection.name == candidate)) is not None:
            candidate = f"{name}_migrated{k}"
            k += 1
        actual = candidate
        report["name_mappings"].append({"source": name, "db_name": actual})

    db.add(Collection(id=cid, name=actual, display_name=actual))
    db.commit()
    return cid, actual


def _migrate_collections(db, vector_dir: Path, report: dict) -> dict:
    """返回 {源 name: (collection_id, DB 实际 name)}"""
    registry = {}
    index_path = vector_dir / "_collections.json"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            names = json.load(f)
        for name in names:
            registry[name] = _ensure_collection(db, name, report)
    else:
        report["notes"].append(f"未找到 {index_path}，跳过注册表，仅扫描目录")

    # 目录存在但未注册进 _collections.json 的知识库一并迁移，防静默丢失
    if vector_dir.is_dir():
        for col_dir in sorted(p for p in vector_dir.iterdir() if p.is_dir()):
            if col_dir.name in registry:
                continue
            if (col_dir / "metadata.json").exists():
                report["notes"].append(
                    f"{col_dir.name}: 未在 _collections.json 注册，按目录数据一并迁移"
                )
                registry[col_dir.name] = _ensure_collection(db, col_dir.name, report)
    return registry


# ============================================================
#  步骤 2：metadata.json + index.faiss → files / chunks 表
# ============================================================

def _reconstruct(index, i: int, name: str, chunk_id: str, report: dict):
    """
    从 FAISS 索引导出第 i 条向量（不重新计算 embedding）。
    索引缺失、越界或 reconstruct 不支持/失败时返回 None（embedding 置空），记入报告。
    """
    if index is None or i >= index.ntotal:
        return None
    try:
        return index.reconstruct(i).astype("float32").tolist()
    except (NotImplementedError, RuntimeError) as e:
        report["reconstruct_failures"].append({
            "collection": name, "chunk_id": chunk_id, "faiss_index": i, "error": str(e),
        })
        return None


def _migrate_collection_data(db, vector_dir: Path, name: str, cid, db_name: str,
                             upload_dir: Path, report: dict) -> None:
    """单个知识库的 metadata.json + index.faiss → files + chunks"""
    entry = {"name": name, "db_name": db_name,
             "files_new": 0, "files_skipped": 0,
             "chunks_new": 0, "chunks_skipped": 0,
             "expected_chunks": 0, "db_chunks": 0, "ok": True}
    report["collections"].append(entry)

    col_dir = vector_dir / name
    meta_path = col_dir / "metadata.json"
    if not meta_path.exists():
        report["notes"].append(f"{name}: 无 metadata.json，按空知识库处理")
        return
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    documents = meta.get("documents", [])
    metadatas = meta.get("metadatas", [])
    ids = meta.get("ids", [])
    # 校验口径以 ids 数为准；平行数组不一致时只迁移有完整数据的头部，差额算丢失
    entry["expected_chunks"] = len(ids)
    n = len(ids)
    if not (len(documents) == len(metadatas) == n):
        report["notes"].append(
            f"{name}: 平行数组长度不一致 documents={len(documents)} "
            f"metadatas={len(metadatas)} ids={len(ids)}，仅迁移前 {min(len(documents), len(metadatas), n)} 条"
        )
        n = min(len(documents), len(metadatas), n)

    # 加载 FAISS 索引（只读 reconstruct，不做一致性重建——重建需要 embedding 模型）
    index = None
    faiss_path = col_dir / "index.faiss"
    if faiss_path.exists():
        try:
            index = faiss.read_index(str(faiss_path))
        except Exception as e:
            report["notes"].append(f"{name}: index.faiss 读取失败（{e}），embedding 全部置空")
    else:
        report["notes"].append(f"{name}: 无 index.faiss，embedding 全部置空")
    if index is not None:
        if index.d != EMBEDDING_DIM:
            report["notes"].append(
                f"{name}: 索引维度 {index.d} ≠ {EMBEDDING_DIM}，embedding 全部置空"
            )
            index = None
        elif index.ntotal != len(ids):
            report["notes"].append(
                f"{name}: 索引与元数据数量不一致 ntotal={index.ntotal} ids={len(ids)}，"
                "超出部分 embedding 置空"
            )

    # 按 file_name 聚合 chunks（保持首次出现顺序）
    files_order = []
    file_chunks = {}
    for i in range(n):
        md = metadatas[i] or {}
        fname = md.get("file_name", "unknown")
        if fname not in file_chunks:
            file_chunks[fname] = []
            files_order.append(fname)
        file_chunks[fname].append((i, md))

    for fname in files_order:
        items = file_chunks[fname]
        fid = uuid.uuid5(_NS, f"file:{name}/{fname}")
        if db.get(File, fid) is not None:
            # 幂等：文件连同其 chunks 是同一事务写入的，整体跳过
            entry["files_skipped"] += 1
            entry["chunks_skipped"] += len(items)
            continue

        # 原始文件定位：优先 metadata.source（摄入时的原路径），其次 upload_dir/file_name
        source_field = next((md.get("source") for _, md in items if md.get("source")), "")
        storage_path, file_hash, size_bytes = "", None, 0
        candidates = ([Path(source_field)] if source_field else []) + [upload_dir / fname]
        for cand in candidates:
            if cand.is_file():
                storage_path = str(cand)
                file_hash = _sha256_file(cand)
                size_bytes = cand.stat().st_size
                break
        if file_hash is None:
            report["notes"].append(f"{name}/{fname}: 原始文件缺失，file_hash 置空")

        db.add(File(
            id=fid, collection_id=cid, file_name=fname,
            storage_path=storage_path, file_hash=file_hash,
            size_bytes=size_bytes, status="done",
        ))

        # chunk_index 保序：metadata 里有 chunk_index 按其排序，否则按出现顺序
        def _sort_key(item):
            pos, md = item
            ci = md.get("chunk_index")
            return (0, ci) if isinstance(ci, int) else (1, pos)

        for new_idx, (i, md) in enumerate(sorted(items, key=_sort_key)):
            old_id = str(ids[i])
            chunk_pk = uuid.uuid5(_NS, f"chunk:{name}/{old_id}")
            if db.get(Chunk, chunk_pk) is not None:
                entry["chunks_skipped"] += 1
                continue
            ci = md.get("chunk_index")
            db.add(Chunk(
                id=chunk_pk, file_id=fid, collection_id=cid,
                chunk_index=ci if isinstance(ci, int) else new_idx,
                content=str(documents[i]),
                embedding=_reconstruct(index, i, name, old_id, report),
            ))
            entry["chunks_new"] += 1
        entry["files_new"] += 1
        # 文件 + 其 chunks 同事务提交，保证"文件已存在 → chunks 已完整"的幂等语义
        db.commit()


# ============================================================
#  步骤 3：sessions/*.json → sessions 表
# ============================================================

def _migrate_sessions(db, session_dir: Path, report: dict) -> None:
    stats = report["sessions"]
    if not session_dir.is_dir():
        report["notes"].append(f"会话目录不存在: {session_dir}")
        return
    for path in sorted(session_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            stats["failed"].append({"file": path.name, "error": str(e)})
            continue
        if not isinstance(data, dict):
            stats["failed"].append({"file": path.name, "error": "内容不是 JSON 对象"})
            continue
        sid = data.get("session_id") or path.stem
        stats["expected"] += 1
        stats["ids"].append(sid)
        if db.get(DbSession, sid) is not None:
            stats["skipped"] += 1
            continue
        db.add(DbSession(
            id=sid,
            title=data.get("title") or "新会话",
            persona=data.get("persona") or "default",
            kb_id=data.get("kb_id") or "",
            mode=data.get("mode") or "qa",
            # JSON 列整存，保留原结构
            messages=data.get("messages") or [],
            compressed_history=data.get("compressed_history") or [],
            exam_state=data.get("exam_state") or {},
            total_tokens=int(data.get("total_tokens") or 0),
            # 时间保原值，naive 本地口径与 SessionRepository 一致
            created_at=_parse_dt(data.get("created_at")),
            updated_at=_parse_dt(data.get("updated_at")),
        ))
        db.commit()
        stats["new"] += 1


# ============================================================
#  步骤 4：feedbacks.json → feedbacks 表
# ============================================================

def _migrate_feedbacks(db, feedback_file: Path, report: dict) -> None:
    stats = report["feedbacks"]
    if not feedback_file.exists():
        report["notes"].append(f"反馈文件不存在: {feedback_file}")
        return
    try:
        data = json.loads(feedback_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        stats["read_error"] = str(e)
        return
    if not isinstance(data, dict):
        stats["read_error"] = "内容不是 {session_id: [反馈...]} 结构"
        return

    for group_sid, items in data.items():
        if not isinstance(items, list):
            continue
        for fb in items:
            if not isinstance(fb, dict):
                continue
            sid = fb.get("session_id") or group_sid
            fid = fb.get("id")
            if not fid:
                report["notes"].append(f"反馈缺少 id（session={sid}），已跳过")
                continue
            # 孤儿反馈：session 不存在 → 跳过并记报告（PG FK CASCADE / sqlite 同理不落库）
            if db.get(DbSession, sid) is None:
                stats["orphans"].append({"id": fid, "session_id": sid})
                continue
            stats["expected"] += 1
            stats["ids"].append(fid)
            if db.get(Feedback, fid) is not None:
                stats["skipped"] += 1
                continue
            db.add(Feedback(
                id=fid,
                session_id=sid,
                type=fb.get("type") or "",
                original_chunk_id=fb.get("original_chunk_id") or "",
                target_chunk_id=fb.get("target_chunk_id") or "",
                original_content=fb.get("original_content") or "",
                target_content=fb.get("target_content") or "",
                note=fb.get("note") or "",
                created_at=_parse_dt(fb.get("created_at")),
            ))
            db.commit()
            stats["new"] += 1


# ============================================================
#  步骤 5：校验报告
# ============================================================

def _validate(db, report: dict, registry: dict) -> None:
    """条数比对：metadata ids 数 vs DB 行数；sessions/feedbacks 源数 vs DB 存在数"""
    losses = report["losses"]
    for entry in report["collections"]:
        cid = registry[entry["name"]][0]
        db_count = db.scalar(
            select(func.count()).select_from(Chunk).where(Chunk.collection_id == cid)
        ) or 0
        entry["db_chunks"] = db_count
        entry["ok"] = db_count == entry["expected_chunks"]
        if not entry["ok"]:
            losses.append(
                f"知识库 {entry['name']}: metadata ids={entry['expected_chunks']} "
                f"vs DB chunks={db_count}"
            )

    s = report["sessions"]
    missing = [sid for sid in s["ids"] if db.get(DbSession, sid) is None]
    s["db_present"] = len(s["ids"]) - len(missing)
    for sid in missing:
        losses.append(f"会话未入库: {sid}")
    for f in s["failed"]:
        losses.append(f"会话文件解析失败: {f['file']}（{f['error']}）")

    f = report["feedbacks"]
    missing_fb = [fid for fid in f["ids"] if db.get(Feedback, fid) is None]
    f["db_present"] = len(f["ids"]) - len(missing_fb)
    for fid in missing_fb:
        losses.append(f"反馈未入库: {fid}")
    if f["read_error"]:
        losses.append(f"反馈文件解析失败: {f['read_error']}")

    report["zero_loss"] = not losses


def _print_report(report: dict) -> None:
    print("=" * 60)
    print("迁移校验报告")
    print("=" * 60)
    for c in report["collections"]:
        mark = "OK" if c["ok"] else "不一致"
        print(
            f"[知识库] {c['name']} -> {c['db_name']}: "
            f"文件 新增{c['files_new']}/跳过{c['files_skipped']}，"
            f"切片 新增{c['chunks_new']}/跳过{c['chunks_skipped']}，"
            f"校验 ids={c['expected_chunks']} vs DB={c['db_chunks']} [{mark}]"
        )
    if report["name_mappings"]:
        print("[name 冲突映射]")
        for m in report["name_mappings"]:
            print(f"  {m['source']} -> {m['db_name']}")
    s = report["sessions"]
    print(
        f"[会话] 新增{s['new']}/跳过{s['skipped']}/解析失败{len(s['failed'])}，"
        f"校验 文件数={s['expected']} vs DB={s['db_present']}"
    )
    f = report["feedbacks"]
    print(
        f"[反馈] 新增{f['new']}/跳过{f['skipped']}/孤儿{len(f['orphans'])}，"
        f"校验 非孤儿数={f['expected']} vs DB={f['db_present']}"
    )
    if report["reconstruct_failures"]:
        print(f"[reconstruct 失败 {len(report['reconstruct_failures'])} 条，embedding 已置空]")
        for r in report["reconstruct_failures"]:
            print(f"  {r['collection']} chunk={r['chunk_id']} faiss#{r['faiss_index']}: {r['error']}")
    if f["orphans"]:
        print("[孤儿反馈，已跳过]")
        for o in f["orphans"]:
            print(f"  {o['id']} (session={o['session_id']})")
    if report["notes"]:
        print("[说明]")
        for note in report["notes"]:
            print(f"  - {note}")
    print("-" * 60)
    if report["zero_loss"]:
        print("结论：零丢失，迁移校验通过")
    else:
        print("结论：发现数据丢失/不一致！")
        for loss in report["losses"]:
            print(f"  ! {loss}")


# ============================================================
#  入口
# ============================================================

def run_migration(source_dir=None, session_dir=None, feedback_file=None,
                  upload_dir=None) -> dict:
    """执行迁移 + 校验，返回报告 dict。源目录默认复用 settings 解析结果"""
    vector_dir = Path(source_dir) if source_dir else Path(settings.vector_persist_dir)
    sess_dir = Path(session_dir) if session_dir else Path(settings.session_dir)
    # 反馈文件与会话目录的相对位置沿用 FeedbackStore._init 的约定
    fb_file = (Path(feedback_file) if feedback_file
               else sess_dir.parent / "feedback" / "feedbacks.json")
    up_dir = Path(upload_dir) if upload_dir else Path(settings.upload_dir)

    # 幂等建表兜底；正式 PostgreSQL 环境由 alembic 建表，create_all 不会重复建
    Base.metadata.create_all(get_engine())

    report = _new_report()
    with new_session() as db:
        registry = _migrate_collections(db, vector_dir, report)
        for name, (cid, db_name) in registry.items():
            _migrate_collection_data(db, vector_dir, name, cid, db_name, up_dir, report)
        _migrate_sessions(db, sess_dir, report)
        _migrate_feedbacks(db, fb_file, report)
        _validate(db, report, registry)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="文件存储 → PostgreSQL 一次性迁移（幂等，可重复跑）"
    )
    parser.add_argument("--source-dir", help="向量库源目录（默认 settings.vector_persist_dir）")
    parser.add_argument("--session-dir", help="会话 JSON 目录（默认 settings.session_dir）")
    parser.add_argument("--feedback-file",
                        help="feedbacks.json 路径（默认会话目录同级的 feedback/feedbacks.json）")
    parser.add_argument("--upload-dir", help="原始上传文件目录（默认 settings.upload_dir）")
    parser.add_argument("--yes", action="store_true", help="跳过交互确认（脚本/测试用）")
    args = parser.parse_args(argv)

    if not args.yes:
        print("即将执行迁移：")
        print(f"  向量库源目录: {args.source_dir or settings.vector_persist_dir}")
        print(f"  会话目录:     {args.session_dir or settings.session_dir}")
        print(f"  目标数据库:   {settings.database_url}")
        if input("确认开始迁移？[y/N] ").strip().lower() not in ("y", "yes"):
            print("已取消")
            return 2

    try:
        report = run_migration(args.source_dir, args.session_dir,
                               args.feedback_file, args.upload_dir)
    except Exception as e:
        print(f"迁移失败: {e}")
        return 2
    _print_report(report)
    return 0 if report["zero_loss"] else 1


if __name__ == "__main__":
    sys.exit(main())
