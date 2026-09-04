"""
阶段 2 双跑验收共享引导：纯 ASCII 临时数据目录 + sys.path 引导。

项目路径含中文，FAISS C 扩展无法写中文路径，因此 VECTOR_PERSIST_DIR /
UPLOAD_DIR / SESSION_DIR 必须指向纯 ASCII 目录，且在 import 任何 app 模块
前设置（settings 是 import 期单例）。目录固定在 %TEMP%/dxrag_dual_run，
保证 prepare / compare / degrade / contract 四个脚本共享同一个知识库。
"""
import os
import sys
import tempfile
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
SMOKE_DIR = PROJECT_ROOT / ".smoke2"

DUAL_RUN_DIR = Path(tempfile.gettempdir()) / "dxrag_dual_run"
COLLECTION = "dual_run_kb"


def setup_env() -> None:
    """设置数据目录 env 与 sys.path，必须在 import 任何 app 模块前调用"""
    if not str(DUAL_RUN_DIR).isascii():
        raise RuntimeError(f"临时目录含非 ASCII 字符，FAISS 无法写入: {DUAL_RUN_DIR}")
    for sub in ("vector_db", "uploads", "sessions"):
        (DUAL_RUN_DIR / sub).mkdir(parents=True, exist_ok=True)
    os.environ["VECTOR_PERSIST_DIR"] = str(DUAL_RUN_DIR / "vector_db")
    os.environ["UPLOAD_DIR"] = str(DUAL_RUN_DIR / "uploads")
    os.environ["SESSION_DIR"] = str(DUAL_RUN_DIR / "sessions")
    os.environ.setdefault("STORAGE_BACKEND", "file")
    SMOKE_DIR.mkdir(exist_ok=True)
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))
