# ============================================
# 项目：LocalRAG · 可定制本地 RAG 智能体
# 开发者：kangxiaobai-kzj
# 开发时间：2026-08-14
# ============================================

# core/mcp/executor.py
# 工具执行器：统一执行 registry 中注册的工具。
# backend 支持两种模式：
#   direct（默认）：进程内直调检索器（轻量稳定，复用调用方已加载的模型）
#   mcp：通过 MCP 协议（stdio）调用 mcp_server.py（演示真实 MCP 链路，每次调用会拉起独立进程）
import os

from config import (
    CHROMA_DB_DIR,
    KNOWLEDGE_BASE_DIR,
    RERANK_TOP_K,
    RERANK_MIN_DOCS,
    MCP_TOOL_BACKEND,
)
from rag.retriever import load_retriever_and_reranker
from rag.reranker import rerank_documents


class ToolExecutor:
    """执行工具调用。可复用外部已加载的 (retriever, reranker) 以避免重复加载模型。"""

    def __init__(self, backend: str = None, retriever=None, reranker=None,
                 latest_only: bool = False, top_k: int = RERANK_TOP_K, min_score: float = 0.0):
        self.backend = (backend or os.getenv("MCP_TOOL_BACKEND") or MCP_TOOL_BACKEND).strip().lower()
        self._retriever = retriever
        self._reranker = reranker
        self._latest_only = latest_only
        self._top_k = max(1, int(top_k or RERANK_TOP_K))
        self._min_score = float(min_score or 0.0)
        # 知识库检索器改为按需加载：仅真正调用 search_knowledge_base 时才加载，
        # 避免闲聊/联网等仅使用 web_search 的场景无谓加载本地检索模型。

    def call(self, name: str, args: dict = None) -> str:
        """执行工具并返回文本结果（与 MCP 工具返回一致，纯字符串）。"""
        args = args or {}
        if self.backend == "mcp":
            from core.mcp.client import call_mcp_tool
            return call_mcp_tool(name, args)
        if name == "search_knowledge_base":
            return self._search(args)
        if name == "list_documents":
            return self._list_documents()
        if name == "web_search":
            return self._web_search(args)
        return f"未知工具：{name}"

    # ================= direct 实现 =================
    def _search(self, args: dict) -> str:
        # 按需加载：未被调用方注入检索器时（如独立工具调用），首次使用才加载
        if self._retriever is None:
            self._retriever, self._reranker = load_retriever_and_reranker(CHROMA_DB_DIR)
        if self._retriever is None:
            return "知识库未构建，无法检索。请先上传 PDF 并重建知识库。"
        question = str(args.get("question", "")).strip()
        if not question:
            return "缺少参数 question。"
        source = (args.get("source") or "").strip() or None  # 可选：按来源文件过滤
        top_k = max(1, int(args.get("top_k", self._top_k)))
        docs = self._retriever.invoke(question, source=source, latest_only=self._latest_only)
        docs = rerank_documents(question, docs, self._reranker, top_k, RERANK_MIN_DOCS, self._min_score)
        if not docs:
            return "未检索到相关内容。"
        lines = []
        for i, d in enumerate(docs, start=1):
            src = d.metadata.get("source", "未知")
            page = d.metadata.get("page", "?")
            lines.append(f"[{i}] 来源:{src} 页码:P{page}\n{d.page_content}")
        return "\n\n".join(lines)

    def _list_documents(self) -> str:
        from rag.parsers import is_supported_file
        files = [f for f in os.listdir(KNOWLEDGE_BASE_DIR) if is_supported_file(f)]
        if not files:
            return "知识库为空，暂无文档。"
        return "知识库文档列表：\n" + "\n".join(f"- {f}" for f in files)

    def _web_search(self, args: dict) -> str:
        """
        必应网页搜索（免 Key）：requests 抓取 + bs4 解析，返回标题/链接/摘要。
        结果会随工具返回发往云端 LLM，故先做本地脱敏。
        网络异常 / 被反爬时返回友好错误（不影响 Agent 主流程）。
        """
        from security.desensitize import desensitize_text
        query = str(args.get("query", "")).strip()
        if not query:
            return "缺少参数 query。"
        max_results = max(1, min(10, int(args.get("max_results", 5))))
        try:
            import requests
            from bs4 import BeautifulSoup
            headers = {
                "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                               "AppleWebKit/537.36 (KHTML, like Gecko) "
                               "Chrome/126.0 Safari/537.36"),
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            }
            resp = requests.get(
                "https://cn.bing.com/search", params={"q": query},
                headers=headers, timeout=15,
            )
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            results = []
            for li in soup.select("li.b_algo")[:max_results]:
                a = li.select_one("h2 a")
                if not a:
                    continue
                title = a.get_text(" ", strip=True)
                url = a.get("href", "")
                caption = li.select_one(".b_caption p") or li.select_one("p")
                snippet = caption.get_text(" ", strip=True) if caption else ""
                if title and url:
                    results.append(f"{len(results) + 1}. {title}\n   链接: {url}\n   摘要: {snippet}")
            if not results:
                return "未找到相关网页结果，可尝试更换关键词。"
            return "Web 搜索结果：\n\n" + desensitize_text("\n\n".join(results))
        except Exception as e:
            return f"Web 搜索失败（网络异常或搜索服务不可用）：{e}"
