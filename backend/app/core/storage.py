"""
存储后端开关（阶段 1：数据库化，波次 3）

按 settings.storage_backend 把路由/服务层消费的存储接口统一转发到具体实现：
- "file"（默认）：现有文件存储——core/session.py 的 session_manager、
  services/feedback.py 的 feedback_store、core/vector_store.py 的全部向量函数
- "postgres"：DB Repository——db/session_repository.py、db/feedback_repository.py、
  db/vector_repository.py（接口与文件版逐字兼容）

调用方（api/routes.py、services/qa.py、services/retrieval.py、services/ingest.py）
一律从本模块 import，不直接引用具体实现；切换后端只改 STORAGE_BACKEND 环境变量。

注意：settings 是 import 期单例，本模块同样在 import 期完成分派，
STORAGE_BACKEND 必须在 import app 任何模块前设置，运行期切换后端需重启进程。
"""
from app.core.config import settings

if settings.storage_backend == "postgres":
    # ========================================================
    #  DB 存储分支（PostgreSQL + pgvector；本机测试可用 SQLite）
    # ========================================================
    from app.db.session_repository import session_repository as session_manager
    from app.db.feedback_repository import feedback_repository as feedback_store
    from app.db.vector_repository import (
        list_collections,
        create_collection,
        rename_collection,
        delete_collection,
        add_texts,
        vector_search,
        get_files,
        delete_file,
        get_vector_store,
        encode_texts,
        encode_query,
        get_embed_model,
        # 检索缓存版本令牌：(chunk 总数, 最新 chunk 创建时间戳)，增删数据即变化
        get_store_version,
    )
elif settings.storage_backend == "file":
    # ========================================================
    #  文件存储分支（现状，默认，API 行为不变）
    # ========================================================
    from app.core.session import session_manager
    from app.services.feedback import feedback_store
    from app.core.vector_store import (
        list_collections,
        create_collection,
        rename_collection,
        delete_collection,
        add_texts,
        vector_search,
        get_files,
        delete_file,
        get_vector_store,
        encode_texts,
        encode_query,
        get_embed_model,
        # _meta_path 仅用于推导缓存版本令牌，不向消费方 re-export
        _meta_path,
    )

    def get_store_version(collection_name: str) -> tuple[int, int]:
        """
        file 后端的检索缓存版本令牌：metadata.json 的 (mtime_ns, size)。
        增删数据重写该文件即变化，与 DB 版 (chunk 总数, 最新创建时间戳) 语义等价；
        文件不存在返回 (0, 0)（对应空库，与 DB 版空库返回值一致）。
        retrieval.py 的 BM25 缓存只认本函数，不关心后端差异。
        """
        try:
            st = _meta_path(collection_name).stat()
            return (st.st_mtime_ns, st.st_size)
        except OSError:
            return (0, 0)
else:
    raise ValueError(
        f"未知 STORAGE_BACKEND: {settings.storage_backend!r}（可选值: file / postgres）"
    )
