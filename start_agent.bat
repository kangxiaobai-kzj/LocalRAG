@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion
title LocalRAG · 可定制本地 RAG 智能体

:: 1. 切换到脚本所在目录（双击运行也能定位正确）
cd /d "%~dp0"

echo ==========================================
echo   🚀 LocalRAG 一键启动
echo   首次运行会自动准备环境（创建 venv / 安装依赖 / 下载模型）
echo ==========================================
echo.

:: 2. 检查 Python 是否可用
where python >nul 2>&1
if errorlevel 1 (
    echo ❌ 未检测到 Python。
    echo    请先安装 Python 3.10 或更高版本：https://www.python.org/downloads/
    echo    安装时务必勾选 "Add python.exe to PATH"。
    pause
    exit /b 1
)

:: 3. 检查 Python 版本 >= 3.10
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if errorlevel 1 (
    echo ❌ Python 版本过低，需要 3.10+。请升级后重试。
    pause
    exit /b 1
)

:: 4. 虚拟环境不存在则自动创建
if not exist ".\venv\Scripts\python.exe" (
    echo 📦 首次运行：正在创建虚拟环境 venv...
    python -m venv venv
    if errorlevel 1 (
        echo ❌ 创建虚拟环境失败，请检查 Python 安装是否完整。
        pause
        exit /b 1
    )
    echo ✅ 虚拟环境创建完成
)

:: 5. 激活虚拟环境
call .\venv\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ 虚拟环境激活失败，请删除 venv 目录后重试。
    pause
    exit /b 1
)

:: 6. 校验核心依赖，缺失则自动安装（需要联网）
python -c "import streamlit, chromadb, langchain, fastembed, sentence_transformers" >nul 2>&1
if errorlevel 1 (
    echo 📦 核心依赖未就绪，正在安装（需要联网，首次约 3~8 分钟）...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ 依赖安装失败，请检查网络后重新运行本脚本。
        pause
        exit /b 1
    )
    echo ✅ 依赖安装完成
)

:: 7. 校验模型缓存（Embedding / 重排 / OCR）
echo.
echo 🔍 检查模型缓存...
python scripts\check_models.py
if errorlevel 1 (
    echo.
    echo ⚠️ 模型缓存不完整，离线检索不可用。
    echo    方案 A：运行 install_models.bat 联网下载（约 2.3GB，耗时较长）
    echo    方案 B：跳过，仅聊天功能可用，检索功能后续再安装
    echo.
    set /p DOWNLOAD_CHOICE=是否现在下载模型？(Y/N)：
    if /i "!DOWNLOAD_CHOICE!"=="Y" (
        python scripts\download_models.py
        if errorlevel 1 (
            echo ❌ 模型下载失败，请检查网络后运行 install_models.bat 重试。
            pause
            exit /b 1
        )
    ) else (
        echo 已跳过模型下载，可稍后运行 install_models.bat。
    )
)

:: 8. 检查 8501 端口占用
netstat -ano | findstr ":8501" | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo.
    echo ⚠️ 端口 8501 已被占用，可能已有 LocalRAG 实例在运行。
    echo    可先在浏览器访问 http://127.0.0.1:8501 确认。
    set /p CONTINUE_ANYWAY=仍然继续启动？(Y/N)：
    if /i not "!CONTINUE_ANYWAY!"=="Y" (
        echo 已取消启动。
        pause
        exit /b 0
    )
)

:: 9. 启动 Streamlit（仅本机可访问；应用无鉴权且含内部资料，禁止暴露公网/局域网）
echo.
echo ==========================================
echo   🌐 正在启动 LocalRAG Web 界面...
echo   浏览器即将自动打开：http://127.0.0.1:8501
echo   ⏹️  关闭本窗口即可停止服务
echo ==========================================
start "" /b powershell -WindowStyle Hidden -Command "Start-Sleep -Seconds 5; Start-Process 'http://127.0.0.1:8501'"
python -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.headless true

:: 10. 程序退出后暂停，显示报错方便排查
echo.
echo ⏹️ 服务已停止。
pause
