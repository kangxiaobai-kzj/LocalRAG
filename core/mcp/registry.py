# ============================================
# 项目：LocalRAG · 可定制本地 RAG 智能体
# 开发者：kangxiaobai-kzj
# 开发时间：2026-08-14
# ============================================

# core/mcp/registry.py
# MCP 工具契约：与 mcp_server.py 中 FastMCP 注册的工具一一对应。
# 这里使用 OpenAI function-calling 的 schema 格式，可直接传给 llm.bind_tools()，
# 使 LLM 能"看到可用工具并自主决策调用"。
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "检索本地私有知识库，返回与问题最相关的文档片段（含来源文件名与页码）。"
                "用于回答具体的业务、技术、产品问题。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "用户的问题"},
                    "top_k": {"type": "integer", "description": "返回的片段数量，默认 3"},
                    "source": {"type": "string", "description": "按来源文件名精确过滤（可选），仅检索该文档"},
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": (
                "列出知识库中已收录的文档文件。"
                "用于回答‘有哪些资料/文档’这类全局性问题。"
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "搜索互联网（必应），返回与问题相关的网页标题、链接与摘要。"
                "用于回答知识库中【没有】的、需要最新外部资讯或公开资料的问题"
                "（如行业动态、最新政策、公开技术资料等）。仅当用户在「设置」中启用"
                "Web 联网搜索时此工具才可用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "要搜索的关键词或问题"},
                    "max_results": {"type": "integer", "description": "返回结果数量，默认 5，最多 10"},
                },
                "required": ["query"],
            },
        },
    },
]


def get_tool_schemas(enable_web_search: bool = False):
    """
    返回工具 schema 列表（OpenAI function-calling 格式）。
    Web 联网搜索为可选能力：默认关闭，仅在设置页开启后对 LLM 暴露。
    """
    if enable_web_search:
        return TOOLS
    return [t for t in TOOLS if t["function"]["name"] != "web_search"]


def get_tool_names():
    """返回可用工具名列表。"""
    return [t["function"]["name"] for t in TOOLS]
