"""
应用配置管理 - 基于 pydantic-settings 从 .env 文件加载配置
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings


# 项目根目录 (backend/)
# config.py 位于 backend/app/core/，向上回退两级即为 backend/app 目录
BACKEND_DIR = Path(__file__).resolve().parent.parent

# .env 位于项目根目录（从 config.py 向上回退四级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# 如果项目路径含非 ASCII 字符（如中文），底层向量库 C 扩展可能崩溃，
# 此时使用用户目录下的安全路径
def _safe_data_dir(dirname: str) -> str:
    raw = str(BACKEND_DIR / dirname)
    if raw.isascii():
        return raw
    # 回退到用户目录，并显式告知实际数据位置
    safe = Path.home() / ".dx-rag" / dirname
    print(f"[Config] 项目路径包含非 ASCII 字符，数据目录回退到: {safe}")
    return str(safe)


def _migrate_legacy_vector_dir() -> None:
    """
    历史兼容性处理：
    1. 早期版本目录名使用 chroma_db，现在统一为 vector_db
    2. 早期 BACKEND_DIR 计算多跳了一层，导致数据实际存放在 backend/backend/chroma_db
    启动时自动迁移旧目录到新的正确位置，避免数据丢失。
    """
    new_dir = Path(_safe_data_dir("vector_db"))
    if new_dir.exists():
        return

    # 可能的历史旧目录，按优先级依次尝试
    legacy_candidates = [
        # 旧命名 + 旧 BACKEND_DIR（多跳一层）
        BACKEND_DIR / "backend" / "chroma_db",
        # 旧命名 + 正确 BACKEND_DIR
        BACKEND_DIR / "chroma_db",
        # 安全路径下的旧命名
        Path.home() / ".dx-rag" / "chroma_db",
    ]

    for legacy_dir in legacy_candidates:
        if legacy_dir.exists():
            try:
                new_dir.parent.mkdir(parents=True, exist_ok=True)
                # 如果旧目录和目标目录都在同一文件系统，直接重命名；否则复制
                if legacy_dir.parent == new_dir.parent:
                    legacy_dir.rename(new_dir)
                else:
                    import shutil
                    shutil.copytree(legacy_dir, new_dir)
                    shutil.rmtree(legacy_dir)
                print(f"[Config] 已迁移旧向量库目录: {legacy_dir} -> {new_dir}")
                return
            except Exception as e:
                print(f"[Config] 迁移向量库目录失败 ({legacy_dir}): {e}")


class Settings(BaseSettings):
    """应用配置，所有值优先从 .env 读取"""

    # ---- 应用 ----
    app_name: str = "dx-rag-demo"

    # ---- API Keys ----
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # ---- 向量数据库 ----
    # 项目实际使用 FAISS 作为向量检索引擎，命名统一为 vector 避免歧义
    vector_collection: str = "knowledge_chunks"
    vector_persist_dir: str = _safe_data_dir("vector_db")

    # ---- 会话数据 ----
    session_dir: str = _safe_data_dir("data/sessions")

    # ---- 嵌入模型 ----
    embed_model: str = "BAAI/bge-small-zh-v1.5"

    # ---- 文件上传 ----
    upload_dir: str = _safe_data_dir("uploads")

    # ---- 文本切分 ----
    max_chunk_size: int = 800
    chunk_overlap: int = 120

    # ---- Token 预算 ----
    max_input_tokens: int = 4000
    max_output_tokens: int = 1024
    max_total_tokens_per_session: int = 50000
    context_compression_threshold: float = 0.6  # 超过 60% 触发上下文压缩

    # ---- 检索过滤 ----
    # BGE / 向量检索的余弦相似度阈值，低于此值认为检索结果不相关
    retrieval_similarity_threshold: float = 0.65

    # ---- CORS ----
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    model_config = {
        "env_file": str(PROJECT_ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    def ensure_dirs(self) -> None:
        """确保必要的目录存在"""
        os.makedirs(self.vector_persist_dir, exist_ok=True)
        os.makedirs(self.upload_dir, exist_ok=True)


# 启动时执行一次性迁移
_migrate_legacy_vector_dir()

# 单例
settings = Settings()
