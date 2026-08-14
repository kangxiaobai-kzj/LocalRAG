@echo off
chcp 65001 > nul
title LocalRAG · 模型下载工具
cd /d "%~dp0"

echo ==========================================
echo   📥 LocalRAG 模型下载工具
echo   用于补齐 Embedding / 重排模型（需联网，约 2.3GB）
echo ==========================================
echo.

:: 1. 检查虚拟环境是否存在（不存在则提示先运行 start_agent.bat）
if not exist ".\venv\Scripts\python.exe" (
    echo ❌ 未找到虚拟环境，请先运行一次 start_agent.bat 完成环境初始化。
    pause
    exit /b 1
)

call .\venv\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ 虚拟环境激活失败。
    pause
    exit /b 1
)

:: 2. 执行模型下载脚本
python scripts\download_models.py
if errorlevel 1 (
    echo ❌ 模型下载失败，请检查网络后重试。
    pause
    exit /b 1
)

echo.
echo ✅ 模型安装完成，现在可以离线使用检索功能了。
pause
