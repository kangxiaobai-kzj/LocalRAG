# ============================================
# 项目：LocalRAG · 可定制本地 RAG 智能体
# 开发者：kangxiaobai-kzj
# 开发时间：2026-08-14
# ============================================

# mcp_server.py - MCP Server：把本地私有知识库暴露为标准 MCP 工具
# 本服务可被任意 MCP 客户端连接（Claude Desktop、Cursor、本项目的 MCP 客户端等）。
#
# 启动（项目根目录、虚拟环境内）：
#   python mcp_server.py
# 进程内测试：
#   python -c "from mcp.server.fastmcp import FastMCP; import mcp_server; print(mcp_server.mcp.list_tools())"
import asyncio
import contextlib
import os
import sys

# 允许以脚本方式从项目根目录导入包
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 兼容 mcp SDK 1.x（FastMCP）与 2.x（MCPServer）：两者工具注册与运行 API 一致
try:
    from mcp.server.fastmcp import FastMCP as _MCPClass
except ImportError:  # mcp >= 2.0：FastMCP 已重构为 mcp.server.MCPServer
    from mcp.server import MCPServer as _MCPClass

# ⚠️ stdio 协议独占 stdout：启动期所有日志（含 tqdm 进度条）改走 stderr，
# 否则非 JSON 字节会污染协议通道，导致客户端握手失败。
with contextlib.redirect_stdout(sys.stderr):
    from config import CHROMA_DB_DIR, KNOWLEDGE_BASE_DIR, RERANK_TOP_K, RERANK_MIN_DOCS, CONFIG_FILE
    from utils import load_config
    from rag.retriever import load_retriever_and_reranker
    from rag.reranker import rerank_documents

    # 启动时加载一次检索组件（模型常驻进程，避免每次工具调用重复加载）
    _retriever, _reranker = load_retriever_and_reranker(CHROMA_DB_DIR)

# 创建 MCP Server
mcp = _MCPClass("knowledge-base")


@mcp.tool()
def search_knowledge_base(question: str, top_k: int = None, source: str = None) -> str:
    """检索本地私有知识库，返回与问题最相关的文档片段（含来源文件名与页码）。可用 source 指定仅检索某一文档（按文件名精确匹配）。"""
    if _retriever is None:
        return "知识库未构建，无法检索。请先上传 PDF 并重建知识库。"
    question = question.strip()
    if not question:
        return "缺少参数 question。"
    if top_k is None:
        # 未显式指定时，使用设置页「最相关条数」
        top_k = int(load_config(CONFIG_FILE).get("rerank_top_k", RERANK_TOP_K))
    top_k = max(1, int(top_k))
    source = (source or "").strip() or None
    docs = _retriever.invoke(question, source=source)
    # 相似度阈值跟随设置页「检索与切片设置」（0 表示关闭过滤）
    min_score = float(load_config(CONFIG_FILE).get("min_score", 0.0) or 0.0)
    docs = rerank_documents(question, docs, _reranker, top_k, RERANK_MIN_DOCS, min_score=min_score)
    if not docs:
        return "未检索到相关内容。"
    lines = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "未知")
        page = d.metadata.get("page", "?")
        lines.append(f"[{i}] 来源:{src} 页码:P{page}\n{d.page_content}")
    return "\n\n".join(lines)


@mcp.tool()
def list_documents() -> str:
    """列出知识库中已收录的文档文件。用于回答"有哪些资料/文档"这类全局性问题。"""
    from rag.parsers import is_supported_file
    files = [f for f in os.listdir(KNOWLEDGE_BASE_DIR) if is_supported_file(f)]
    if not files:
        return "知识库为空，暂无文档。"
    return "知识库文档列表：\n" + "\n".join(f"- {f}" for f in files)


@mcp.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """搜索互联网（必应，免 Key），返回与问题相关的网页标题/链接/摘要。用于回答知识库中没有的最新资讯或公开资料（如行业动态、最新政策）。结果发送前会本地脱敏。"""
    from core.mcp.executor import ToolExecutor
    return ToolExecutor()._web_search({"query": query, "max_results": max_results})


if __name__ == "__main__":
    print("🚀 MCP Server 'knowledge-base' 启动（stdio 传输），等待客户端连接...", file=sys.stderr)
    print("   💡 stdio 服务器由客户端拉起；直接运行时会阻塞等待握手，被取消即退出（属预期行为）。", file=sys.stderr)
    try:
        mcp.run(transport="stdio")
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n👋 MCP Server 已退出", file=sys.stderr)
