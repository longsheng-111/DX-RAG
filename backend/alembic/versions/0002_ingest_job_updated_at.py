"""stage4: ingest_jobs 增加 updated_at（worker 心跳列）

阶段 4 摄入队列化：worker 推进进度时刷新该列，API 读取时据它判定
running 任务是否因 worker 宕机而僵死（超时标记 failed）。
"""
from alembic import op
import sqlalchemy as sa

revision = "0002_ingest_job_updated_at"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "ingest_jobs",
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("ingest_jobs", "updated_at")
