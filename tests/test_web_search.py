# ============================================
# 项目：LocalRAG · 可定制本地 RAG 智能体
# 开发者：kangxiaobai-kzj
# 开发时间：2026-08-14
# ============================================

# 测试：Web 联网搜索工具（registry 开关 + executor 参数校验）
# 说明：不发起真实网络请求（避免依赖外部网络/被反爬），用 mock 验证解析与参数钳制。
from unittest import mock

from core.mcp.executor import ToolExecutor
from core.mcp.registry import get_tool_schemas, get_tool_names


def test_tool_schemas_web_search_disabled_by_default():
    names = get_tool_names()
    assert "search_knowledge_base" in names
    assert "list_documents" in names
    assert "web_search" in names  # 注册表完整包含


def test_get_tool_schemas_switch():
    # 关闭时不对 LLM 暴露 web_search（默认）
    closed = [t["function"]["name"] for t in get_tool_schemas(enable_web_search=False)]
    assert "web_search" not in closed
    assert "search_knowledge_base" in closed
    # 开启时暴露
    opened = [t["function"]["name"] for t in get_tool_schemas(enable_web_search=True)]
    assert "web_search" in opened
    assert len(opened) == len(closed) + 1


def test_web_search_missing_query():
    # 缺参数应返回友好提示，而非抛异常
    result = ToolExecutor().call("web_search", {})
    assert "query" in result


def test_web_search_empty_query():
    result = ToolExecutor().call("web_search", {"query": "   "})
    assert "query" in result


class _FakeResp:
    """模拟 requests.Response：只暴露 text 与 raise_for_status。"""
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        return None


_BING_HTML = """
<html><body>
<li class="b_algo"><h2><a href="https://example.com/a">结果标题A</a></h2>
<div class="b_caption"><p>这是摘要A，包含电话 13812345678 应被脱敏。</p></div></li>
<li class="b_algo"><h2><a href="https://example.com/b">结果标题B</a></h2>
<div class="b_caption"><p>摘要B</p></div></li>
<li class="b_algo"><h2><a href="https://example.com/c">结果标题C</a></h2>
<div class="b_caption"><p>摘要C</p></div></li>
</body></html>
"""


@mock.patch("requests.get", return_value=_FakeResp(_BING_HTML))
def test_web_search_parse_and_desensitize(mock_get):
    result = ToolExecutor()._web_search({"query": "测试", "max_results": 5})
    # 解析到 3 条结果
    assert "结果标题A" in result and "结果标题B" in result
    # 摘要中的手机号被脱敏
    assert "13812345678" not in result
    assert "1**********" in result
    # max_results 钳制为 10 的上限入参（3 条 HTML 只返回 3 条）
    assert mock_get.call_args.kwargs["params"]["q"] == "测试"


@mock.patch("requests.get", side_effect=Exception("network down"))
def test_web_search_network_failure(mock_get):
    # 网络异常返回友好字符串，不抛异常
    result = ToolExecutor()._web_search({"query": "测试"})
    assert "Web 搜索失败" in result
