# ============================================
# 项目：LocalRAG · 可定制本地 RAG 智能体
# 开发者：kangxiaobai-kzj
# 开发时间：2026-08-14
# ============================================

# utils/logger.py
# 统一日志：控制台 + 滚动文件（logs/app.log，5MB×3 份），进程内只初始化一次。
# 说明：控制台输出走 stderr（logging.StreamHandler 默认），避免污染 MCP stdio 协议通道（stdout）。
import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = "./logs"
LOG_FILE = os.path.join(LOG_DIR, "app.log")

_configured = False


def setup_logging(level: int = logging.INFO) -> None:
    """初始化根日志器（幂等，进程内只执行一次）。"""
    global _configured
    if _configured:
        return
    os.makedirs(LOG_DIR, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(level)
    # 控制台（stderr）：Streamlit 终端 / 命令行可见
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)
    # 滚动文件：5MB × 3 份
    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024,
                                       backupCount=3, encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)
    _configured = True


def get_logger(name: str = "app") -> logging.Logger:
    """获取模块日志器（自动完成根日志初始化）。"""
    setup_logging()
    return logging.getLogger(name)
