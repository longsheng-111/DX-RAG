@echo off
setlocal enabledelayedexpansion
title DX-RAG Launcher

echo.
echo ========================================
echo   DX-RAG Knowledge Base QA System v1.0
echo ========================================
echo.

cd /d "%~dp0"

REM ============================================================
REM [1/6] Pick a Python interpreter that can actually run the app
REM   Preferred: backend\venv_pg  (project venv, all deps installed)
REM   Fallback : py -3.12 / py / python
REM   Why this matters: a plain "py -3.12" interpreter can import
REM   fastapi/faiss but NOT redis/arq, so uvicorn crashed on import
REM   while the --reload supervisor kept port 8000 bound, and every
REM   /api call from the frontend returned HTTP 500.
REM ============================================================
echo [1/6] Locating Python...

set "PYCMD="
set "PYDESC="

if exist "backend\venv_pg\Scripts\python.exe" (
    REM Absolute path: later steps pushd/cd into backend, so a relative
    REM "backend\venv_pg\..." would resolve to backend\backend\... and die.
    set "PYCMD=%~dp0backend\venv_pg\Scripts\python.exe"
    set "PYDESC=project venv (backend\venv_pg)"
    goto :found_python
)

py -3.12 --version >nul 2>&1
if %errorlevel% equ 0 (
    set "PYCMD=py -3.12"
    set "PYDESC=py -3.12"
    goto :found_python
)

py --version >nul 2>&1
if %errorlevel% equ 0 (
    set "PYCMD=py"
    set "PYDESC=py"
    goto :found_python
)

python --version >nul 2>&1
if %errorlevel% equ 0 (
    set "PYCMD=python"
    set "PYDESC=python"
    goto :found_python
)

echo   [ERROR] Python not found. Install Python 3.9+ from https://python.org
pause
exit /b 1

:found_python
echo   Using %PYDESC%
%PYCMD% --version

REM ---- Check Node.js ----
echo [2/6] Checking Node.js...
where node >nul 2>&1
if %errorlevel% neq 0 (
    echo   [ERROR] Node.js not found. Install Node.js 18+
    pause
    exit /b 1
)
node --version
echo   Node OK

REM ---- Check .env ----
echo [3/6] Checking config...
if not exist ".env" (
    echo   .env not found, creating from template...
    copy .env.example .env >nul
    echo   Please edit .env with your API keys and re-run.
    start notepad .env
    pause
    exit /b 1
)
echo   Config OK

REM ---- Backend deps ----
REM   redis + arq are imported at startup (stage-4 ingest queue),
REM   so they must be part of this check.
echo [4/6] Checking Python dependencies...
%PYCMD% -c "import fastapi, uvicorn, faiss, sentence_transformers, redis, arq, langgraph" >nul 2>&1
if %errorlevel% neq 0 (
    echo   Installing backend dependencies from backend\requirements.txt ...
    pushd backend
    %PYCMD% -m pip install -r requirements.txt -q
    popd
    %PYCMD% -c "import fastapi, uvicorn, faiss, sentence_transformers, redis, arq, langgraph" >nul 2>&1
    if !errorlevel! neq 0 (
        echo   [ERROR] Interpreter "%PYDESC%" still cannot import every dependency.
        echo   Install them manually, or delete backend\venv_pg and re-run this script.
        pause
        exit /b 1
    )
)
echo   Backend deps OK

REM ---- Data folder must be writable by this interpreter ----
REM   Windows caps a process at the integrity level of its own image file.
REM   If this project folder carries a "Low" mandatory label, programs that
REM   live inside it (backend\venv_pg\Scripts\python.exe) also start at Low
REM   integrity, and a Low process cannot write to a normal folder such as
REM   %USERPROFILE%\.dx-rag. Symptoms: sessions never persist and
REM   /api/chat answers with HTTP 500.
REM   Detect it here and restore a normal (Medium) level automatically.
%PYCMD% -c "import os,pathlib;d=pathlib.Path(os.path.expanduser('~'))/'.dx-rag';d.mkdir(parents=True,exist_ok=True);p=d/'_write_test.tmp';p.write_text('x');p.unlink()" >nul 2>&1
if !errorlevel! neq 0 (
    echo   [FIX] Interpreter cannot write to the data folder - raising integrity level...
    icacls "%PYCMD%" /setintegritylevel Medium >nul 2>&1
    %PYCMD% -c "import os,pathlib;d=pathlib.Path(os.path.expanduser('~'))/'.dx-rag';d.mkdir(parents=True,exist_ok=True);p=d/'_write_test.tmp';p.write_text('x');p.unlink()" >nul 2>&1
    if !errorlevel! neq 0 (
        echo   [ERROR] Interpreter "%PYDESC%" cannot write to the data folder.
        echo   Run this once in an admin prompt ^(adjust the path^), then re-run:
        echo       icacls "backend\venv_pg\Scripts\python.exe" /setintegritylevel Medium
        pause
        exit /b 1
    )
    echo   Integrity level fixed.
)

REM ---- Frontend deps ----
if not exist "frontend\node_modules\" (
    echo   Installing frontend dependencies...
    pushd frontend
    call npm install --silent
    popd
)
echo   Frontend deps OK

echo.
echo ========================================
echo   Starting services...
echo ========================================
echo.

REM ============================================================
REM [5/6] Free ports + kill stale backend/worker
REM   A hung "uvicorn --reload" keeps port 8000 bound WITHOUT
REM   showing up as LISTENING, so a port-only cleanup cannot see it
REM   and the next start silently fails. Match the command line too.
REM ============================================================
echo [5/6] Cleaning ports and stale processes...

for /f "delims=" %%P in ('powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and $_.CommandLine -match 'uvicorn app.main:app|arq app.worker.WorkerSettings' } | Select-Object -ExpandProperty ProcessId"') do (
    taskkill /F /PID %%P >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000.*LISTENING" 2^>nul') do (
    taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":3000.*LISTENING" 2^>nul') do (
    taskkill /F /PID %%a >nul 2>&1
)
echo   Ports cleaned

REM ============================================================
REM [6/6] Start services
REM ============================================================
echo [6/6] Starting services...

REM --- Redis: upload queue broker. Without it POST /api/upload -> 503.
netstat -ano | findstr ":6379.*LISTENING" >nul 2>&1
if %errorlevel% neq 0 (
    if exist ".tools\redis\redis-server.exe" (
        echo   Starting bundled Redis on 6379...
        start "DX-RAG-Redis" cmd /k "cd /d %cd%\.tools\redis && redis-server.exe redis.windows.conf"
        ping -n 3 127.0.0.1 >nul
    ) else (
        echo   [WARN] Redis is not running and .tools\redis was not found.
        echo          Uploads will return 503 until Redis is available.
    )
) else (
    echo   Redis already running on 6379
)

REM --- Backend API (append --reload if you want hot reload while editing)
start "DX-RAG-Backend" cmd /k "cd /d %cd%\backend && %PYCMD% -m uvicorn app.main:app --host 0.0.0.0 --port 8000"

REM --- Ingest worker (consumes the upload queue)
start "DX-RAG-Worker" cmd /k "cd /d %cd%\backend && %PYCMD% -m arq app.worker.WorkerSettings"

echo   Waiting for backend to start...
ping -n 6 127.0.0.1 >nul

REM --- Frontend
start "DX-RAG-Frontend" cmd /k "cd /d %cd%\frontend && npm run dev"

echo   Opening browser...
ping -n 8 127.0.0.1 >nul
start http://localhost:3000

echo.
echo ========================================
echo   System started!
echo   Frontend : http://localhost:3000
echo   API Docs : http://localhost:8000/docs
echo ========================================
echo.
pause
