FROM python:3.11-slim

WORKDIR /app

# 系统依赖 (PyMuPDF 需要)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 应用代码
COPY . .

# 创建数据目录
RUN mkdir -p vector_db models uploads

EXPOSE 8000

# 启动时自动下载嵌入模型（如不存在）；
# STORAGE_BACKEND=postgres 时先跑 alembic 建表（幂等），避免首次启动在无表库上直接写入
CMD ["sh", "-c", "python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer(\"BAAI/bge-small-zh-v1.5\")' 2>/dev/null; if [ \"$STORAGE_BACKEND\" = \"postgres\" ]; then alembic upgrade head; fi; uvicorn app.main:app --host 0.0.0.0 --port 8000"]
