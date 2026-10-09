"""
DX-RAG FastAPI 应用入口
"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# 将 backend/ 加入 sys.path（支持直接 `python app/main.py` 运行）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化目录，关停时释放 ARQ 连接池"""
    settings.ensure_dirs()
    print(f"[Startup] 应用启动: {settings.app_name}")
    print(f"[Startup] 上传目录: {settings.upload_dir}")
    print(f"[Startup] 向量数据库: {settings.vector_persist_dir}")
    yield
    from app.services.ingest_queue import close_pool
    await close_pool()


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用"""
    app = FastAPI(
        title=settings.app_name,
        description="DX-RAG 知识库问答系统 API",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS 中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册路由
    app.include_router(router)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
