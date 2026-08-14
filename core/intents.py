# ============================================
# 项目：LocalRAG · 可定制本地 RAG 智能体
# 开发者：kangxiaobai-kzj
# 开发时间：2026-08-14
# ============================================

# core/intents.py
# 意图初判：本地关键词快速通道。
# 极简问候/致谢直接判定为 CHAT（闲聊），其余输入进入 Agent 模式（AGENTIC），
# 由 LLM 通过 function calling 自主决策是否需要调用工具。
import re

# 极简问候/致谢：无需检索也无需工具调用
_CHAT_FASTLANE_RE = re.compile(
    r"^(你好|您好|嗨|哈喽|hello|hi|在吗|谢谢|多谢|感谢|辛苦了|再见|拜拜|"
    r"你是谁|你能做什么|你会什么|介绍下你自己)[!！。？?~～\s]*$",
    re.IGNORECASE,
)


def detect_fastlane_intent(question: str) -> str:
    """
    本地快速通道：极短问候/致谢直接判定为 CHAT，避免无谓的 LLM 工具调用。
    其余输入返回 AGENTIC（由编排器进入 function calling 模式）。
    """
    q = (question or "").strip()
    if not q:
        return "CHAT"  # 空输入兜底为闲聊
    if _CHAT_FASTLANE_RE.match(q):
        return "CHAT"
    return "AGENTIC"
