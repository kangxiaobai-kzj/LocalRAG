@echo off
chcp 936 > nul
title LocalRAG - LocalRAG Agent

:: ============ LocalRAG launcher (source / portable / desktop) ============
:: switch to script dir
cd /d "%~dp0"

:: pick python: portable runtime > venv > system
if exist "%~dp0runtime\python\python.exe" (
    set "PYTHON=%~dp0runtime\python\python.exe"
) else if exist "%~dp0venv\Scripts\python.exe" (
    set "PYTHON=%~dp0venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

:: portable: model cache inside package; source: ocr binaries on PATH
if exist "%~dp0models\fastembed" set "FASTEMBED_CACHE_PATH=%~dp0models\fastembed"
if exist "%~dp0models\hf" set "HF_HOME=%~dp0models\hf"
if exist "%~dp0bin" set "PATH=%~dp0bin;%PATH%"

echo ==========================================
echo   [OK] LocalRAG launcher
echo ==========================================
echo.

:: 2. check python
"%PYTHON%" -c "import sys" >nul 2>&1
if errorlevel 1 goto :no_python

:: 3. check python version >= 3.10
"%PYTHON%" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if errorlevel 1 goto :low_python

:: 4. core deps (portable/venv bundle them; source-mode auto installs)
"%PYTHON%" -c "import streamlit, chromadb, langchain, fastembed, sentence_transformers" >nul 2>&1
if errorlevel 1 goto :install_deps

goto :check_models

:no_python
echo [ERR] python not found. install Python 3.10+ (check PATH), or keep runtime/ or venv/.
pause
exit /b 1

:low_python
echo [ERR] python too old, need 3.10+.
pause
exit /b 1

:install_deps
echo [SETUP] installing deps (need network, first time ~3-8 min)...
"%PYTHON%" -m pip install --upgrade pip
"%PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERR] dependency install failed, check network and rerun.
    pause
    exit /b 1
)
echo [DONE] deps installed

:check_models
:: 5. check model cache (embedding / rerank / ocr)
echo.
echo [CHECK] checking model cache...
"%PYTHON%" scripts\check_models.py
if not errorlevel 1 goto :check_port

echo.
echo [WARN] model cache incomplete, offline retrieval unavailable.
echo    option A: run install_models.bat to download (~2.3GB, slow)
echo    option B: skip, only chat works now
echo.
if defined LOCALRAG_DESKTOP (
    echo   [desktop] model download skipped, run install_models.bat later.
    goto :check_port
)
set /p DOWNLOAD_CHOICE=download models now? (Y/N) :
if /i "%DOWNLOAD_CHOICE%"=="Y" (
    "%PYTHON%" scripts\download_models.py
    if errorlevel 1 (
        echo [ERR] model download failed, rerun install_models.bat.
        pause
        exit /b 1
    )
) else (
    echo model download skipped, run install_models.bat later.
)

:check_port
:: 6. check port 8501
netstat -ano | findstr ":8501" | findstr "LISTENING" >nul 2>&1
if errorlevel 1 goto :start_service

echo.
echo [WARN] port 8501 in use, maybe another LocalRAG instance.
if defined LOCALRAG_DESKTOP (
    echo   [desktop] existing instance detected, connecting http://127.0.0.1:8501.
    exit /b 0
)
echo    check http://127.0.0.1:8501 in browser first.
set /p CONTINUE_ANYWAY=start anyway? (Y/N) :
if /i "%CONTINUE_ANYWAY%"=="Y" goto :start_service
echo cancelled.
pause
exit /b 0

:start_service
:: 7. start streamlit (localhost only; no auth, do not expose publicly)
echo.
echo ==========================================
echo   [WEB] starting LocalRAG web UI...
echo   [STOP] close this window to stop
echo ==========================================
if not defined LOCALRAG_DESKTOP (
    echo   browser will open: http://127.0.0.1:8501
    start "" /b powershell -WindowStyle Hidden -Command "Start-Sleep -Seconds 5; Start-Process 'http://127.0.0.1:8501'"
)
"%PYTHON%" -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.headless true

:: 8. pause on exit to show errors (desktop hidden window, end directly)
echo.
echo [STOP] service stopped.
if not defined LOCALRAG_DESKTOP (
    pause
)
exit /b 0
