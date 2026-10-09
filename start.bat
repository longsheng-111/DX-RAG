@echo off
title DX-RAG Launcher

echo.
echo ========================================
echo   DX-RAG Knowledge Base QA System v1.0
echo ========================================
echo.

cd /d "%~dp0"

REM ---- Find Python ----
echo [1/5] Finding Python...
set PYCMD=

py -3.12 --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYCMD=py -3.12
    goto :found_python
)

py --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYCMD=py
    goto :found_python
)

python --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYCMD=python
    goto :found_python
)

echo   [ERROR] Python not found.
echo   Install Python 3.10+ from https://python.org
pause
exit /b 1

:found_python
%PYCMD% --version 2>&1
echo   Python OK

REM ---- Check Node.js ----
echo [2/5] Checking Node.js...
where node >nul 2>&1
if %errorlevel% neq 0 (
    echo   [ERROR] Node.js not found. Install Node.js 18+
    pause
    exit /b 1
)
node --version 2>&1
echo   Node OK

REM ---- Check .env ----
echo [3/5] Checking config...
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
echo [4/5] Checking backend deps...
%PYCMD% -c "import fastapi, faiss, sentence_transformers" >nul 2>&1
if %errorlevel% neq 0 (
    echo   Installing backend dependencies...
    cd backend
    %PYCMD% -m pip install -r requirements.txt -q
    cd ..
)
echo   Backend deps OK

REM ---- Frontend deps ----
echo [5/5] Checking frontend deps...
if not exist "frontend\node_modules\" (
    echo   Installing frontend dependencies...
    cd frontend
    call npm install --silent
    cd ..
)
echo   Frontend deps OK

echo.
echo ========================================
echo   Starting services...
echo ========================================
echo.

REM ---- Kill existing processes on ports ----
echo   Cleaning ports 8000 and 3000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000.*LISTENING" 2^>nul') do (
    taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":3000.*LISTENING" 2^>nul') do (
    taskkill /F /PID %%a >nul 2>&1
)
echo   Ports cleaned

REM Start Backend in new window
start "DX-RAG-Backend" cmd /k "cd /d %cd%\backend && %PYCMD% -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"

REM Start ingest worker in new window (stage 4: upload queue consumer)
REM 需要 Redis：docker compose up -d redis，或本地 redis-server；未启动时上传接口返回 503
start "DX-RAG-Worker" cmd /k "cd /d %cd%\backend && %PYCMD% -m arq app.worker.WorkerSettings"

echo   Waiting for backend to start...
ping -n 6 127.0.0.1 >nul

REM Start Frontend in new window
start "DX-RAG-Frontend" cmd /k "cd /d %cd%\frontend && npm run dev"

echo   Opening browser...
ping -n 5 127.0.0.1 >nul
start http://localhost:3000

echo.
echo ========================================
echo   System started!
echo   Frontend : http://localhost:3000
echo   API Docs : http://localhost:8000/docs
echo ========================================
echo.
pause
