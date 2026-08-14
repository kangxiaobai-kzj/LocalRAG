# ============================================
# 项目：LocalRAG · 可定制本地 RAG 智能体
# 开发者：kangxiaobai-kzj
# 开发时间：2026-08-14
# ============================================

# core/mcp/client.py
# MCP 客户端：通过 stdio 协议调用 mcp_server.py 中注册的工具。
# 注意：stdio 每次调用会拉起一个独立 server 进程（模型需重新加载），
# 适合演示/外部调用；应用内部默认使用 direct 后端（core/mcp/executor.py）。
import asyncio
import os
import sys

# mcp_server.py 位于项目根目录
SERVER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "mcp_server.py",
)


def call_mcp_tool(name: str, args: dict) -> str:
    """同步包装：通过 stdio 调用 MCP Server 的工具。失败时返回错误文本而非抛异常。"""
    try:
        return asyncio.run(_call_tool(name, args))
    except BaseException as e:  # noqa: BLE001 - 任何异常都转成文本返回，避免中断调用方
        subs = getattr(e, "exceptions", None)  # ExceptionGroup 展开子异常，便于定位真实错误
        if subs:
            return f"[MCP 调用失败] {' | '.join(str(x) for x in subs)}"
        return f"[MCP 调用失败] {e}"


async def _call_tool(name: str, args: dict) -> str:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=sys.executable, args=[SERVER_PATH])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments=args)
            texts = [c.text for c in result.content if hasattr(c, "text")]
            return "\n".join(texts) if texts else str(result)
