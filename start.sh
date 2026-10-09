#!/bin/bash
# DX-RAG 一键启动脚本 (Git Bash / Linux / macOS)

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# Mac 上 faiss-cpu 的 Intel OpenMP 与 torch/sentence-transformers 多线程冲突，
# 限制为单线程可避免 Segmentation fault。Windows 上无副作用。
export OMP_NUM_THREADS=1

echo ""
echo "╔══════════════════════════════════════╗"
echo "║   DX-RAG 知识库问答系统 v1.0        ║"
echo "╚══════════════════════════════════════╝"
echo ""

# ---- 1. 检查 .env ----
echo "[1/5] 检查配置文件..."
if [ ! -f ".env" ]; then
    echo "  ⚠ .env 不存在，正在从 .env.example 创建..."
    cp .env.example .env
    echo "  请编辑 .env 填入 API 密钥后重新运行"
    exit 1
fi
echo "  .env ✓"

# ---- 2. 后端虚拟环境 & 依赖 ----
echo "[2/5] 检查后端依赖..."
# 优先使用项目内的虚拟环境（Mac 临时开发机已重建）
if [ -x "$DIR/backend/venv_new/bin/python" ]; then
    PYTHON="$DIR/backend/venv_new/bin/python"
elif [ -x "$DIR/backend/venv/bin/python" ]; then
    PYTHON="$DIR/backend/venv/bin/python"
else
    PYTHON=""
    for py in python3 python; do
        if command -v $py &>/dev/null; then
            PYTHON=$py
            break
        fi
    done
fi
if [ -z "$PYTHON" ]; then
    echo "  ✗ 未找到 Python，请安装 Python 3.10+"
    exit 1
fi
echo "  Python: $($PYTHON --version) ✓"

$PYTHON -c "import fastapi, faiss, sentence_transformers" 2>/dev/null || {
    echo "  正在安装后端依赖..."
    cd backend
    $PYTHON -m pip install -r requirements.txt -q
    cd ..
}
echo "  后端依赖 ✓"

# ---- 3. 前端依赖 ----
echo "[3/5] 检查前端依赖..."
if [ ! -d "frontend/node_modules" ]; then
    echo "  正在安装前端依赖..."
    cd frontend
    npm install --silent
    cd ..
fi
echo "  前端依赖 ✓"

# ---- 4. 检查嵌入模型 ----
echo "[4/5] 检查嵌入模型..."
MODEL_DIR="$HOME/.cache/modelscope/BAAI/bge-small-zh-v1.5"
if [ ! -d "$MODEL_DIR" ]; then
    echo "  嵌入模型未下载，正在通过 ModelScope 下载..."
    $PYTHON -c "
from modelscope import snapshot_download
snapshot_download('BAAI/bge-small-zh-v1.5', cache_dir='$HOME/.cache/modelscope')
" 2>/dev/null || echo "  ⚠ 模型下载失败，后端启动时会自动尝试"
fi
echo "  嵌入模型 ✓"

# ---- 5. 启动服务 ----
echo "[5/5] 启动服务..."
echo ""

# 清理旧进程
kill $(lsof -ti:8000) 2>/dev/null || true
kill $(lsof -ti:3000) 2>/dev/null || true
sleep 1

# 启动后端 (后台)
echo "  启动后端 (port 8000)..."
cd "$DIR/backend"
$PYTHON -m uvicorn app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd "$DIR"

# 启动摄入 worker (后台，阶段 4：上传队列消费者)
# 需要 Redis：docker compose up -d redis，或本地 redis-server；未启动时上传接口返回 503
echo "  启动摄入 worker (arq)..."
cd "$DIR/backend"
$PYTHON -m arq app.worker.WorkerSettings &
WORKER_PID=$!
cd "$DIR"

# 启动前端 (后台)
echo "  启动前端 (port 3000)..."
cd "$DIR/frontend"
npm run dev &
FRONTEND_PID=$!
cd "$DIR"

# 等待启动
sleep 5

echo ""
echo "╔══════════════════════════════════════╗"
echo "║  系统已启动!                        ║"
echo "║  前端: http://localhost:3000        ║"
echo "║  后端: http://localhost:8000/docs   ║"
echo "║                                      ║"
echo "║  按 Ctrl+C 停止所有服务              ║"
echo "╚══════════════════════════════════════╝"
echo ""

# 打开浏览器 (可选)
if command -v start &>/dev/null; then
    start http://localhost:3000 2>/dev/null || true
elif command -v open &>/dev/null; then
    open http://localhost:3000 2>/dev/null || true
elif command -v xdg-open &>/dev/null; then
    xdg-open http://localhost:3000 2>/dev/null || true
fi

# 等待子进程
trap "kill $BACKEND_PID $WORKER_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait
