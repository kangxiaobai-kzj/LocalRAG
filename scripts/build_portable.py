# ============================================
# 项目：LocalRAG · 可定制本地 RAG 智能体
# 开发者：kangxiaobai-kzj
# 开发时间：2026-08-14
# ============================================

# scripts/build_portable.py
# 便携版打包：将项目打包为「免安装 Python 的 Windows 便携包」zip。
#   结构：runtime/python（自包含解释器 + 全部依赖）
#         models/fastembed（Embedding 模型缓存，随包走）
#         bin/（OCR 二进制）
#         项目源码 + 便携版启动脚本
# 用法（用 venv 的 python 运行，保证 sys.base_prefix 正确）：
#   python scripts/build_portable.py [--version 1.0.0] [--no-zip] [--keep-build]
# 输出：dist/LocalRAG-v{version}-portable-win64.zip
import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = REPO_ROOT / "VERSION"
EMBED_REPO_DIR = "models--Qdrant--bge-small-zh-v1.5"   # 与 check_models.py 一致
EMBED_CACHE_SRC = Path(os.environ.get(
    "FASTEMBED_CACHE_PATH",
    os.path.join(os.environ.get("TEMP", os.getcwd()), "fastembed_cache"))) / EMBED_REPO_DIR

# 源码复制时排除的顶层条目（本地运行时数据 / 构建产物 / 隐私文件）
EXCLUDE_TOP = {
    "venv", ".git", "build", "dist", "bin", "models",
    "__pycache__", ".pytest_cache", ".idea", ".vscode",
    "chroma_db", "chroma_db_old_1", "knowledge_base", "chat_sessions",
    "logs", "archive", "corpus.txt", "config.json",
    "eval/questions.json",
}

# 源码复制时全局忽略的目录名 / 文件名
# 注意：仅用于项目源码复制，绝不能用于 runtime（site-packages 里存在同名目录，如
# opentelemetry/proto/collector/logs，会被误删）
IGNORE_DIRS = {"__pycache__", ".pytest_cache", ".git", "chroma_db", "chroma_db_old_1",
               "knowledge_base", "chat_sessions", "logs", "archive", "node_modules"}
IGNORE_FILES = {"config.json", ".env", "corpus.txt", "questions.json"}


def read_version() -> str:
    if VERSION_FILE.exists():
        return VERSION_FILE.read_text(encoding="utf-8").strip()
    return "1.0.0"


def ignore_pyc(directory, names):
    """copytree 忽略规则（runtime 用）：仅跳过 __pycache__ 与 *.pyc / *.pyo。"""
    return {n for n in names
            if n == "__pycache__"
            or n.endswith(".pyc") or n.endswith(".pyo")}


def ignore_source(directory, names):
    """copytree 忽略规则（项目源码用）：额外跳过本地运行时目录。"""
    return {n for n in names
            if n in IGNORE_DIRS
            or n.endswith(".pyc") or n.endswith(".pyo")}


def build_runtime(pkg_dir: Path) -> None:
    """自包含 Python：复制 base 解释器，把 venv 的 site-packages 合并进去。"""
    base = Path(sys.base_prefix)          # venv 对应的基础 Python 安装目录
    venv_site = Path(sys.prefix) / "Lib" / "site-packages"   # venv 的依赖
    if not venv_site.is_dir():
        print(f"❌ 未找到 venv site-packages：{venv_site}")
        sys.exit(1)

    py_dir = pkg_dir / "runtime" / "python"
    print(f"📦 复制基础 Python（{base}）→ runtime/python ...")
    # 基础解释器整体复制，但站点包目录由 venv 的替换（base 的 site-packages 基本为空）
    base_site = base / "Lib" / "site-packages"

    def ignore_base(directory, names):
        skip = ignore_pyc(directory, names)
        p = Path(directory)
        if p == base:
            skip |= {"include", "tcl", "Tools"}
        elif p == base / "Lib":
            skip |= {"site-packages", "test", "tkinter", "idlelib", "turtledemo"}
        return skip

    # 先复制基础解释器（含标准库 Lib/，但站点包目录由 venv 的替换）
    shutil.copytree(base, py_dir, ignore=ignore_base, dirs_exist_ok=True)
    # 再把 venv 的依赖复制进站点包目录
    print("📦 复制 venv 依赖（Lib/site-packages）...")
    shutil.copytree(venv_site, py_dir / "Lib" / "site-packages",
                    ignore=ignore_pyc, dirs_exist_ok=True)
    # pip 等命令行脚本
    base_scripts = base / "Scripts"
    if base_scripts.is_dir():
        shutil.copytree(base_scripts, py_dir / "Scripts", ignore=ignore_pyc, dirs_exist_ok=True)
    # 写入 python310._pth：让解释器完全以本目录为 home（无视注册表），实现随包走
    # （与官方 embeddable 发行版同机制；末尾 import site 以启用 site-packages）
    pth = py_dir / "python310._pth"
    pth.write_text("python310.zip\nDLLs\nLib\nLib\\site-packages\nimport site\n",
                   encoding="utf-8", newline="\r\n")


def build_source(pkg_dir: Path) -> None:
    print("📦 复制项目源码 ...")
    for item in sorted(REPO_ROOT.iterdir()):
        if item.name in EXCLUDE_TOP:
            continue
        dst = pkg_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dst, ignore=ignore_source, dirs_exist_ok=True)
        elif item.name not in IGNORE_FILES:
            if item.name.endswith((".pyc", ".pyo")):
                continue
            shutil.copy2(item, dst)
    # 保留 eval 评测脚本（questions.json 已排除）
    if (REPO_ROOT / "eval" / "questions.json").exists():
        print("  ℹ️ 已排除 eval/questions.json（含内部资料，不入包）")


def build_bin(pkg_dir: Path) -> None:
    src = REPO_ROOT / "bin"
    if src.is_dir():
        print("📦 复制 OCR 二进制（bin/）...")
        shutil.copytree(src, pkg_dir / "bin", dirs_exist_ok=True)


def build_models(pkg_dir: Path) -> None:
    """复制 Embedding 模型缓存；重排模型体积大，不在主包内（首次运行引导下载）。"""
    if EMBED_CACHE_SRC.is_dir():
        print(f"📦 复制 Embedding 模型缓存 → models/fastembed/ ...")
        shutil.copytree(EMBED_CACHE_SRC,
                        pkg_dir / "models" / "fastembed" / EMBED_REPO_DIR,
                        ignore=ignore_pyc, dirs_exist_ok=True)
    else:
        print(f"⚠️ 未找到 Embedding 模型缓存：{EMBED_CACHE_SRC}\n"
              f"   主包将不含 Embedding 模型，首次启动会提示联网下载。")


def gen_launchers(pkg_dir: Path) -> None:
    """生成便携版启动脚本（python 走 runtime，模型缓存走包内 models/）。"""
    launcher = pkg_dir / "LocalRAG.bat"
    launcher.write_text(PORTABLE_LAUNCHER, encoding="utf-8", newline="\r\n")
    installer = pkg_dir / "install_models.bat"
    installer.write_text(PORTABLE_INSTALLER, encoding="utf-8", newline="\r\n")
    readme = pkg_dir / "启动说明.txt"
    readme.write_text(PORTABLE_README, encoding="utf-8", newline="\r\n")


def make_zip(pkg_name: str, pkg_dir: Path, version: str) -> Path:
    out_dir = REPO_ROOT / "dist"
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"LocalRAG-v{version}-portable-win64.zip"
    if zip_path.exists():
        zip_path.unlink()
    print(f"📦 压缩中（体积较大，请稍候）：{zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=True) as zf:
        root = Path(pkg_dir)
        for fp in sorted(root.rglob("*")):
            if not fp.is_file():
                continue
            if any(part in ("__pycache__",) for part in fp.parts):
                continue
            arc = str(fp.relative_to(root))
            zf.write(fp, arc)
    size_mb = zip_path.stat().st_size / 1024 / 1024
    print(f"✅ 便携包已生成：{zip_path}（{size_mb:.0f} MB）")
    return zip_path


def main() -> int:
    parser = argparse.ArgumentParser(description="构建 LocalRAG 便携版")
    parser.add_argument("--version", default=None, help="版本号（默认读 VERSION 文件）")
    parser.add_argument("--no-zip", action="store_true", help="只生成解压目录，不压缩")
    args = parser.parse_args()

    version = args.version or read_version()
    pkg_name = f"LocalRAG-v{version}"
    build_root = REPO_ROOT / "build" / "portable"
    pkg_dir = build_root / pkg_name

    if pkg_dir.exists():
        shutil.rmtree(pkg_dir)
    pkg_dir.mkdir(parents=True)

    print(f"== 构建 LocalRAG 便携版 v{version} ==")
    build_runtime(pkg_dir)
    build_source(pkg_dir)
    build_bin(pkg_dir)
    build_models(pkg_dir)
    gen_launchers(pkg_dir)

    if args.no_zip:
        print(f"✅ 解压目录已生成：{pkg_dir}")
        return 0
    make_zip(pkg_name, pkg_dir, version)
    return 0


PORTABLE_LAUNCHER = r"""@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion
title LocalRAG · 可定制本地 RAG 智能体（便携版）

:: 1. 切换到脚本所在目录
cd /d "%~dp0"

:: 2. 使用包内自带的 Python（无需安装 Python）
set PYTHON=%~dp0runtime\python\python.exe
if not exist "%PYTHON%" (
    echo ❌ 未找到包内 Python，请确认文件完整（缺少 runtime 目录）。
    pause
    exit /b 1
)

:: 3. 模型缓存指向包内 models/（随包走，不占用系统目录）
set FASTEMBED_CACHE_PATH=%~dp0models\fastembed
set HF_HOME=%~dp0models\hf

echo ==========================================
echo   🚀 LocalRAG 一键启动（便携版）
echo ==========================================
echo.

:: 4. 检查模型缓存（Embedding 缺失时检索不可用）
echo 🔍 检查模型缓存...
"%PYTHON%" scripts\check_models.py
if errorlevel 1 (
    echo.
    echo ⚠️ 模型缓存不完整。
    echo    方案 A：运行 install_models.bat 联网下载（约 2.3GB，仅首次需要）
    echo    方案 B：跳过，聊天功能可用，检索功能后续再安装
    echo.
    set /p DOWNLOAD_CHOICE=是否现在下载模型？(Y/N)：
    if /i "!DOWNLOAD_CHOICE!"=="Y" (
        call install_models.bat
        if errorlevel 1 (
            echo ❌ 模型下载失败，请检查网络后重试。
            pause
            exit /b 1
        )
    ) else (
        echo 已跳过模型下载。
    )
)

:: 5. 检查 8501 端口占用
netstat -ano | findstr ":8501" | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo.
    echo ⚠️ 端口 8501 已被占用，可能已有 LocalRAG 在运行。
    set /p CONTINUE_ANYWAY=仍然继续启动？(Y/N)：
    if /i not "!CONTINUE_ANYWAY!"=="Y" (
        echo 已取消启动。
        pause
        exit /b 0
    )
)

:: 6. 启动（仅本机可访问）
echo.
echo ==========================================
echo   🌐 正在启动 LocalRAG Web 界面...
echo   浏览器即将自动打开：http://127.0.0.1:8501
echo   ⏹️  关闭本窗口即可停止服务
echo ==========================================
start "" /b powershell -WindowStyle Hidden -Command "Start-Sleep -Seconds 5; Start-Process 'http://127.0.0.1:8501'"
"%PYTHON%" -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.headless true

echo.
echo ⏹️ 服务已停止。
pause
"""

PORTABLE_INSTALLER = r"""@echo off
chcp 65001 > nul
title LocalRAG · 模型下载工具（便携版）
cd /d "%~dp0"

set PYTHON=%~dp0runtime\python\python.exe
if not exist "%PYTHON%" (
    echo ❌ 未找到包内 Python，请确认文件完整。
    pause
    exit /b 1
)

set FASTEMBED_CACHE_PATH=%~dp0models\fastembed
set HF_HOME=%~dp0models\hf

echo ==========================================
echo   📥 LocalRAG 模型下载（需联网，约 2.3GB）
echo ==========================================
"%PYTHON%" scripts\download_models.py
if errorlevel 1 (
    echo ❌ 模型下载失败，请检查网络后重试。
    pause
    exit /b 1
)
echo.
echo ✅ 模型安装完成，现在可以离线使用检索功能了。
pause
"""

PORTABLE_README = """LocalRAG · 可定制本地 RAG 智能体（便携版）
===================================================

一、如何启动
  1. 双击「LocalRAG.bat」即可一键启动（无需安装 Python）。
  2. 启动后浏览器会自动打开 http://127.0.0.1:8501。
  3. 关闭黑色窗口即可停止服务。

二、首次使用
  - 聊天功能开箱即用。
  - 检索功能依赖本地模型（Embedding 约 90MB，重排约 2.2GB）。
    主包已内置 Embedding 模型；重排模型缺失时自动降级（检索质量略降）。
    如需完整离线检索，双击「install_models.bat」联网下载（仅首次需要）。

三、使用提示
  - 本应用仅允许本机访问（数据不出本机）。
  - 首次使用请到「设置」页填入大模型 API Key（支持 DeepSeek / OpenAI 兼容接口）。
  - 知识库数据、会话记录均保存在本包目录内，删除本目录即完全清理。

四、目录说明
  runtime/python   自带的 Python 解释器与依赖（勿删）
  models/          模型缓存（随包走）
  bin/             OCR 识别二进制（扫描版 PDF 转文字用）
  项目源码         在包根目录（streamlit_app.py 等）
"""


if __name__ == "__main__":
    sys.exit(main())
