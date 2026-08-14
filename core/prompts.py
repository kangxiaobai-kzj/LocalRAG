# ============================================
# 项目：LocalRAG · 可定制本地 RAG 智能体
# 开发者：kangxiaobai-kzj
# 开发时间：2026-08-14
# ============================================

# core/prompts.py
# 提示词构建与固定话术管理
# 说明：不使用 langchain_core.prompts.ChatPromptTemplate —— 它会连带导入
# torch / transformers（约 10s），严重拖慢应用冷启动。这里直接返回模板字符串，
# 由调用方 str.format 填充后构造轻量的 langchain_core.messages。

DEFAULT_ROLE = "你是中移上研院的资深解决方案经理。"

# 全局查询（GLOBAL）时返回的固定话术
GLOBAL_RESPONSE = "我目前的检索机制主要用于解答具体的技术与业务问题。如需了解全局目录，请联系管理员获取，或者您可以直接询问我具体的业务名词。"


def get_dynamic_prompt(role_definition):
    """返回 RAG 分支的 (system, human) 消息模板（含 {context}/{history}/{question} 占位符）。"""
    if not role_definition or role_definition.strip() == "":
        role_definition = DEFAULT_ROLE

    system_tpl = f"""你的核心身份是：{role_definition}。这是你的第一身份，必须优先遵守。

【通用规则 - 必须严格遵守】
1. **逐句溯源**：每一个基于资料的事实，必须在句末标注【来源：文件名】。
2. **禁止幻觉**：如果【背景资料】中完全没有提及，请直接回答"根据现有资料无法回答"，绝对禁止编造。
3. **直奔主题**：仔细阅读用户的最新问题，直接回答该问题。拒绝说废话。
4. **版本优先**：若【背景资料】中同一标准/规范/文档存在多个版本（如不同年份、v1/v2、新旧版），一律优先采用最新版本的内容作答，并在标注来源时写明所用版本；仅在新版本缺失所需内容时，才可参考旧版本并注明"旧版"。

【背景资料】：
{{context}}

<历史对话参考>
{{history}}
</历史对话参考>
"""
    human_tpl = "用户最新问题：{question}\n\n请结合【背景资料】直接回答最新问题，切勿重复历史对话中的话术。"
    return system_tpl, human_tpl


def build_chat_prompt(role_definition):
    """返回闲聊分支的 (system, human) 消息模板（human 含 {question} 占位符）。"""
    if not role_definition or role_definition.strip() == "":
        role_definition = DEFAULT_ROLE
    return f"你是{role_definition}。请友善地与用户进行简短的日常交流。", "{question}"


def build_chat_web_prompt(role_definition):
    """返回闲聊分支启用 Web 联网搜索时的 (system, human) 消息模板。
    闲聊也能联网：绑定 web_search 工具，按需搜索后再回答；纯闲聊仍直接答。"""
    if not role_definition or role_definition.strip() == "":
        role_definition = DEFAULT_ROLE
    system_tpl = f"""你是{role_definition}。请友善地与用户进行简短的日常交流。

你可以调用工具 web_search 搜索互联网获取最新信息：
1. 当用户询问需要最新外部资讯/公开资料的问题（如行业动态、最新政策、实时热点、新闻事件等）时，
   先调用 web_search 搜索，再基于搜索结果回答；
2. 纯闲聊（问候、寒暄、自我介绍、情绪交流、闲聊话题等）时直接回答，无需调用工具；
3. 基于网页内容的回答，句末标注【来源：网页标题】；没有搜索到相关信息时如实说明。"""
    return system_tpl, "{question}"


def get_agentic_system_prompt(role_definition, enable_web_search: bool = False):
    """生成 Agent 模式的系统提示词：告知 LLM 可用的工具与使用规则（function calling）。
    enable_web_search=True 时额外说明 web_search 工具及其使用规则（用于知识库缺失/过时的最新资讯）。"""
    if not role_definition or role_definition.strip() == "":
        role_definition = DEFAULT_ROLE

    tools_desc = [
        "- search_knowledge_base：检索本地私有知识库，用于回答具体的业务、技术、产品问题。",
        "- list_documents：查看知识库收录了哪些文档，用于回答\"有哪些资料/文档\"这类全局性问题。",
    ]
    if enable_web_search:
        tools_desc.append(
            "- web_search：搜索互联网（必应），返回网页标题/链接/摘要。"
            "用于回答知识库中【没有】的、需要最新外部资讯或公开资料的问题（如行业动态、最新政策、公开技术资料）。")

    rules = [
        "需要查资料时，先调用 search_knowledge_base 获取相关内容，再基于工具结果回答。",
        "用户询问知识库整体内容或资料清单时，调用 list_documents。",
    ]
    if enable_web_search:
        rules.append(
            "若知识库检索结果【不足 / 过时 / 没有】用户需要的信息（尤其涉及最新资讯、行业动态、政策时），"
            "应调用 web_search 从互联网补充检索；把知识库结果与网页结果结合后再作答，并区分信息来源。")
        rules.append(
            "禁止编造：知识库与互联网搜索结果中都没有的信息，直接回答\"根据现有资料无法回答\"。")
        rules.append(
            "回答中每个基于资料的事实，在句末标注【来源：文件名】（知识库）或【来源：网页标题】（网页）。")
    else:
        rules.append(
            "禁止编造：工具结果中没有的信息，直接回答\"根据现有资料无法回答\"。")
        rules.append(
            "回答中每个基于资料的事实，在句末标注【来源：文件名】。")
    rules.extend([
        "版本优先：若检索结果中同一标准/规范/文档存在多个版本（不同年份、v1/v2、新旧版），"
        "优先采用最新版本内容作答，并注明所用版本；新版本缺失时再用旧版并注明\"旧版\"。",
        "用户仅闲聊（问候、致谢、自我介绍等）时，可直接回答，无需调用工具。",
    ])

    tool_lines = "\n".join(tools_desc)
    rule_lines = "\n".join(f"{i}. {r}" for i, r in enumerate(rules, start=1))
    return f"""你的核心身份是：{role_definition}。这是你的第一身份，必须优先遵守。

你是一个能够调用工具的知识库专家 Agent。你可以使用以下工具获取信息：
{tool_lines}

【使用规则】
{rule_lines}
"""
