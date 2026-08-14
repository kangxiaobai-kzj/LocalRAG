# ============================================
# 项目：LocalRAG · 可定制本地 RAG 智能体
# 开发者：kangxiaobai-kzj
# 开发时间：2026-08-14
# ============================================

# 测试：意图初判（本地快速通道）
from core.intents import detect_fastlane_intent


def test_greeting_is_chat():
    assert detect_fastlane_intent("你好") == "CHAT"
    assert detect_fastlane_intent("您好！") == "CHAT"
    assert detect_fastlane_intent("Hi!") == "CHAT"
    assert detect_fastlane_intent("hello") == "CHAT"
    assert detect_fastlane_intent("谢谢") == "CHAT"
    assert detect_fastlane_intent("你是谁") == "CHAT"


def test_question_is_agentic():
    assert detect_fastlane_intent("石化行业5G智慧巡检方案是什么？") == "AGENTIC"
    assert detect_fastlane_intent("帮我查一下高精度的授时精度") == "AGENTIC"


def test_empty_input():
    assert detect_fastlane_intent("") == "CHAT"
    assert detect_fastlane_intent("   ") == "CHAT"
    assert detect_fastlane_intent(None) == "CHAT"
