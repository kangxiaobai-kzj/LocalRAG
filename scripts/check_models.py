# ============================================
# 项目：LocalRAG · 可定制本地 RAG 智能体
# 开发者：kangxiaobai-kzj
# 开发时间：2026-08-14
# ============================================

# scripts/check_models.py
# 检查本地模型缓存是否就绪（Embedding + 重排），供 start_agent.bat 启动前校验。
# 用法：python scripts/check_models.py
# 退出码：0 = 就绪；1 = 缺失（提示运行 scripts/download_models.py）
import os
import sys
import tempfile
from pathlib import Path

# 与 retriever.py 中实际使用的模型一致
EMBED_REPO_DIR = "models--Qdrant--bge-small-zh-v1.5"   # fastembed 将 BAAI/bge-small-zh-v1.5 映射到 Qdrant 仓库
RERANK_REPO_DIR = "models--BAAI--bge-reranker-v2-m3"


def fastembed_cache_dir() -> Path:
    """fastembed 缓存目录：优先 FASTEMBED_CACHE_PATH，默认 %TEMP%/fastembed_cache。"""
    return Path(os.environ.get("FASTEMBED_CACHE_PATH",
                               os.path.join(tempfile.gettempdir(), "fastembed_cache")))


def hf_hub_dir() -> Path:
    """HuggingFace 缓存目录：优先 HF_HOME，默认 ~/.cache/huggingface/hub。"""
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def check_embed() -> bool:
    repo = fastembed_cache_dir() / EMBED_REPO_DIR
    ready = (repo / "snapshots").is_dir()
    print(f"[{'OK' if ready else '缺失'}] Embedding 模型 bge-small-zh-v1.5\n      {repo}")
    return ready


def check_rerank() -> bool:
    repo = hf_hub_dir() / RERANK_REPO_DIR
    ready = (repo / "snapshots").is_dir()
    print(f"[{'OK' if ready else '缺失'}] 重排模型 bge-reranker-v2-m3\n      {repo}")
    return ready


def check_ocr() -> bool:
    """OCR 二进制（可选，缺失仅提示，不影响文字版 PDF）。"""
    poppler = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
    ready = os.path.isdir(poppler) and os.listdir(poppler)
    print(f"[{'OK' if ready else '缺失(可选)'}] OCR 二进制（Poppler/Tesseract）\n      {poppler}")
    return ready


def main() -> int:
    print("== LocalRAG 模型缓存检查 ==")
    ok = check_embed() and check_rerank()
    check_ocr()
    if not ok:
        print("\n⚠️ 存在缺失模型。联网环境下运行：python scripts/download_models.py 预下载。")
        return 1
    print("\n✅ 模型缓存就绪，可离线使用。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
