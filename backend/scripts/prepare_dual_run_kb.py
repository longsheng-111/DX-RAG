"""
阶段 2 验收：构建双跑知识库 dual_run_kb（file 后端，真实嵌入入库）

用法：
    python backend/scripts/prepare_dual_run_kb.py          # 已入库则跳过
    python backend/scripts/prepare_dual_run_kb.py --force  # 清空重灌
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _dual_run_env import BACKEND_DIR, COLLECTION, DUAL_RUN_DIR, setup_env

setup_env()

from app.core.storage import get_vector_store  # noqa: E402
from app.services.ingest import process_file  # noqa: E402

FILES = [
    BACKEND_DIR / "test_data" / "frontend_bagu" / "vue.md",
    BACKEND_DIR / "test_data" / "backend_bagu" / "redis.md",
    BACKEND_DIR / "test_data" / "general" / "note.md",
]


def main() -> None:
    if "--force" in sys.argv:
        shutil.rmtree(DUAL_RUN_DIR / "vector_db" / COLLECTION, ignore_errors=True)
        print(f"[Force] 已清空知识库目录: {COLLECTION}")

    existing = get_vector_store(COLLECTION).get("documents", [])
    if existing:
        print(f"[Skip] {COLLECTION} 已有 {len(existing)} 个 chunk，加 --force 重灌")
        return

    for f in FILES:
        info = process_file(str(f), COLLECTION)
        print(
            f"[Ingest] {info['file_name']}: {info['chunks']} chunks "
            f"-> {info['collection_name']}"
        )

    total = len(get_vector_store(COLLECTION).get("documents", []))
    print(f"[Done] {COLLECTION} 共 {total} 个 chunk，目录: {DUAL_RUN_DIR}")


if __name__ == "__main__":
    main()
