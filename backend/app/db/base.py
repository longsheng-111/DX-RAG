"""
数据库基础层（阶段 1：数据库化）

同步 SQLAlchemy 2.0 + psycopg2-binary（异步化属阶段 4，本阶段不引入）。
Engine 惰性创建：首次访问时才读取 settings.database_url 并建连接池，
便于测试通过环境变量 DATABASE_URL 切换到 SQLite（如本机无 PostgreSQL 的场景）。
"""
from typing import Generator, Optional

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import settings


class Base(DeclarativeBase):
    """全部 ORM 模型的声明式基类"""


_engine: Optional[Engine] = None

# 常规 sessionmaker 工厂，bind 在 get_engine() 首次调用时配置
SessionLocal = sessionmaker(autoflush=False, expire_on_commit=False)


def get_engine() -> Engine:
    """惰性创建全局 Engine（首次调用时按 settings.database_url 建连）"""
    global _engine
    if _engine is None:
        connect_args = {}
        if settings.database_url.startswith("sqlite"):
            # SQLite 默认不允许跨线程使用连接，FastAPI/测试多线程场景需要关闭
            connect_args["check_same_thread"] = False
        _engine = create_engine(
            settings.database_url,
            connect_args=connect_args,
            pool_pre_ping=True,
        )
        SessionLocal.configure(bind=_engine)
    return _engine


def new_session() -> Session:
    """新建一个数据库 Session（内部确保 Engine 已初始化），用完需自行 close()"""
    get_engine()
    return SessionLocal()


def get_db() -> Generator[Session, None, None]:
    """FastAPI Depends 入口：每请求一个 Session，结束自动关闭"""
    db = new_session()
    try:
        yield db
    finally:
        db.close()


def dispose_engine() -> None:
    """销毁缓存的 Engine（测试切换 DATABASE_URL 后调用）"""
    global _engine
    if _engine is not None:
        _engine.dispose()
        _engine = None
