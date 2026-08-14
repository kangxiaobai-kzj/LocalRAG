# ============================================
# 项目：LocalRAG · 可定制本地 RAG 智能体
# 开发者：kangxiaobai-kzj
# 开发时间：2026-08-14
# ============================================

# llm/tokenizer.py
# 基于 tiktoken 的 Token 计数
import tiktoken


def _get_encoding():
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return tiktoken.get_encoding("p50k_base")


def count_tokens(text: str, role: str) -> int:
    """按 '用户：/助手：{text}\n' 的格式计算单条消息的 token 数。"""
    enc = _get_encoding()
    line = f"{'用户' if role == 'user' else '助手'}：{text}\n"
    return len(enc.encode(line))
