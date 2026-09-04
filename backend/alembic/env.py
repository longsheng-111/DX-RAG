"""
Alembic 环境配置

连接串与 metadata 均以应用侧为单一事实来源：
- URL 取 app.core.config.settings.database_url（即环境变量 DATABASE_URL / .env）
- target_metadata 取 app.db.base.Base.metadata（导入 app.db.models 完成注册）
"""
import os
import sys
from logging.config import fileConfig

from alembic import context

# 确保可导入 app 包（alembic/ 的上级即 backend/）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings  # noqa: E402
from app.db.base import Base  # noqa: E402
from app.db import models  # noqa: F401,E402  导入即注册全部模型到 Base.metadata

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 以应用配置为准，覆盖 alembic.ini 中的占位值
config.set_main_option("sqlalchemy.url", settings.database_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """离线模式：只生成 SQL，不连接数据库（alembic upgrade head --sql）"""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """在线模式：实际连接数据库执行迁移"""
    from sqlalchemy import engine_from_config, pool

    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
