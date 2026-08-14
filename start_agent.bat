@echo off
chcp 65001 > nul
title LocalRAG · 可定制本地 RAG 智能体

:: 1. 将命令行路径切换到脚本所在的目录（确保路径绝对正确）
cd /d "%~dp0"

echo ==========================================
echo   🚀 正在启动智能体 Web 界面...
echo ==========================================

:: 2. 激活虚拟环境
echo 📦 激活虚拟环境 (venv)...
call .\venv\Scripts\activate.bat

:: 3. 判断激活是否成功，如果出错则暂停
if errorlevel 1 (
    echo ❌ 错误：未找到虚拟环境，请确保 venv 文件夹存在。
    pause
    exit /b
)

:: 4. 启动 Streamlit（这里用了 python -m 方式，兼容性更好，防止找不到 streamlit 命令）
::    --server.address 127.0.0.1：仅本机可访问（应用无鉴权且含内部资料，禁止暴露到公网/局域网）
echo 🌐 启动 Streamlit 服务（仅本机访问）...
python -m streamlit run streamlit_app.py --server.address 127.0.0.1

:: 5. 如果程序意外退出，暂停窗口显示报错（方便调试）
pause