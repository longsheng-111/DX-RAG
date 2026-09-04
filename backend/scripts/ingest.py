"""
多源批量文档导入脚本。

支持从多个本地路径批量导入 Markdown / TXT / PDF / DOCX / XLSX 等文件，
并按来源分类写入向量库。主要用于初始化知识库或批量更新八股文资料。

用法示例:
    cd backend
    python scripts/ingest.py \
        --sources "C:/Users/dell/Desktop/notes:general" \
                "D:/八股文/前端:frontend_bagu" \
                "D:/八股文/后端:backend_bagu" \
        --collection knowledge_chunks
"""
import argparse
import sys
import time
from pathlib import Path
from collections import defaultdict

# 将 backend/ 加入 sys.path，以便导入 app 模块
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings
# 走存储开关（app.core.storage），与 API/服务层同一入口：
# STORAGE_BACKEND=postgres 时写入 DB，默认 file 时写入 FAISS 文件
from app.core.storage import add_texts, create_collection, list_collections, get_vector_store
from app.services.ingest import read_text_from_file, split_text


SUPPORTED_SUFFIXES = {".md", ".txt", ".csv", ".json", ".log", ".pdf", ".docx", ".xlsx", ".xlsm", ".xltx", ".xltm"}


def parse_sources(raw_sources: list[str]) -> list[tuple[Path, str]]:
    """解析 --sources 参数，格式为 path:category"""
    result = []
    for item in raw_sources:
        if ":" not in item:
            raise ValueError(f"--sources 参数格式错误，应为 path:category，得到: {item}")
        path_str, category = item.rsplit(":", 1)
        path = Path(path_str).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"路径不存在: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"必须是目录: {path}")
        result.append((path, category.strip() or "general"))
    return result


def collect_files(source_dir: Path) -> list[Path]:
    """递归收集目录下所有支持的文件"""
    files = []
    for path in source_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(path)
    return sorted(files)


def ingest_file(file_path: Path, category: str, collection_name: str) -> dict:
    """处理单个文件并入库"""
    raw_text = read_text_from_file(file_path)
    if not raw_text.strip():
        raise ValueError(f"文件内容为空: {file_path}")

    chunks = split_text(raw_text, file_path.name)
    if not chunks:
        raise ValueError(f"切分结果为空: {file_path}")

    metadatas = [
        {
            "file_name": file_path.name,
            # source 为应用内约定键（services/ingest.py 同款），postgres 后端会写入 File.storage_path；
            # source_path 为脚本历史键，保留以兼容 file 后端已有数据的增量去重
            "source": str(file_path),
            "source_path": str(file_path),
            "category": category,
            "chunk_index": i,
            "collection_name": collection_name,
        }
        for i in range(len(chunks))
    ]

    count = add_texts(chunks, metadatas, collection_name)
    return {
        "file_name": file_path.name,
        "category": category,
        "chunks": count,
    }


def ensure_collection(collection_name: str) -> None:
    """确保目标知识库存在"""
    existing = {c["name"] for c in list_collections()}
    if collection_name not in existing:
        create_collection(collection_name)
        print(f"[Ingest] 已自动创建知识库: {collection_name}")


def get_existing_entries(collection_name: str) -> tuple[set[str], set[str]]:
    """获取知识库中已存在文件的 (路径集合, 文件名集合)，用于增量导入去重。

    - 路径取自 metadatas 的 source_path（脚本历史键）与 source（应用约定键），
      file 后端两者都可能存在，postgres 后端只有 source（File.storage_path）
    - 文件名作为兜底：postgres 后端对同库同名文件有 UNIQUE 约束，重入库会直接报错，
      提前按名跳过可避免整批导入因此失败
    """
    try:
        data = get_vector_store(collection_name)
        paths: set[str] = set()
        names: set[str] = set()
        for m in data.get("metadatas", []) or []:
            for key in ("source_path", "source"):
                value = m.get(key)
                if value:
                    paths.add(value)
            file_name = m.get("file_name")
            if file_name:
                names.add(file_name)
        return paths, names
    except Exception:
        return set(), set()


def main():
    parser = argparse.ArgumentParser(description="批量导入文档到 RAG 知识库")
    parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help='数据源路径和分类，格式: "path:category"，可多个',
    )
    parser.add_argument(
        "--collection",
        default=settings.vector_collection,
        help=f"目标知识库名称，默认: {settings.vector_collection}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新导入，跳过 source_path 去重检查",
    )
    args = parser.parse_args()

    settings.ensure_dirs()
    ensure_collection(args.collection)

    sources = parse_sources(args.sources)

    print(f"[Ingest] 目标知识库: {args.collection}")
    print(f"[Ingest] 数据源: {len(sources)} 个")
    for path, category in sources:
        print(f"  - {path} -> category={category}")
    print()

    # 获取已存在的文件路径与文件名（用于增量导入去重）
    existing_paths: set[str] = set()
    existing_names: set[str] = set()
    if not args.force:
        existing_paths, existing_names = get_existing_entries(args.collection)
        if existing_paths or existing_names:
            print(f"[Ingest] 知识库中已有 {len(existing_names)} 个文件，将自动跳过")

    start_time = time.time()
    stats = defaultdict(lambda: {"files": 0, "chunks": 0})
    failed_files: list[tuple[str, str]] = []
    skipped_files: list[str] = []

    for source_dir, category in sources:
        files = collect_files(source_dir)
        print(f"[Ingest] 扫描 {source_dir} ({category}): 发现 {len(files)} 个文件")

        for file_path in files:
            file_key = str(file_path)

            # 去重检查：路径命中（精确增量）或文件名命中（postgres 同名约束兜底）
            if not args.force and (file_key in existing_paths or file_path.name in existing_names):
                skipped_files.append(file_key)
                print(f"  [SKIP] {file_path.name} -> 已存在")
                continue

            try:
                result = ingest_file(file_path, category, args.collection)
                stats[category]["files"] += 1
                stats[category]["chunks"] += result["chunks"]
                existing_paths.add(file_key)
                print(f"  [OK] {file_path.name} -> {result['chunks']} chunks")
            except Exception as e:
                failed_files.append((str(file_path), str(e)))
                print(f"  [FAIL] {file_path.name} -> 失败: {e}")

    elapsed = time.time() - start_time

    print("\n" + "=" * 50)
    print("[Ingest] 导入完成")
    print(f"总耗时: {elapsed:.2f}s")
    print("\n分类统计:")
    total_files = 0
    total_chunks = 0
    for category, data in sorted(stats.items()):
        print(f"  {category}: {data['files']} 个文件, {data['chunks']} 个 chunks")
        total_files += data["files"]
        total_chunks += data["chunks"]
    print(f"  总计: {total_files} 个文件, {total_chunks} 个 chunks")

    if skipped_files:
        print(f"\n跳过文件 ({len(skipped_files)} 个，已存在):")
        for path in skipped_files[:10]:
            print(f"  - {path}")
        if len(skipped_files) > 10:
            print(f"  ... 还有 {len(skipped_files) - 10} 个")

    if failed_files:
        print(f"\n失败文件 ({len(failed_files)} 个):")
        for path, err in failed_files:
            print(f"  - {path}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
