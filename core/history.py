# ============================================
# 项目：LocalRAG · 可定制本地 RAG 智能体
# 开发者：kangxiaobai-kzj
# 开发时间：2026-08-14
# ============================================

# core/history.py
# 历史对话截断：滑动窗口保留最近 N 个 token 内的消息
import tiktoken
from config import MAX_HISTORY_TOKENS
from utils.logger import get_logger

logger = get_logger("history")


def truncate_history(history_messages, max_tokens=MAX_HISTORY_TOKENS, token_counts=None):
    """
    截断历史消息至指定 token 数。
    如果提供 token_counts（与 history_messages 长度一致），则直接使用，否则重新计算。
    返回 (截断后的消息列表, 截断后的 token_counts)
    若发生异常，返回 (原始消息, 原始 token_counts) 以保底。
    """
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        try:
            enc = tiktoken.get_encoding("p50k_base")
        except Exception:
            # 如果 tiktoken 完全不可用，直接返回原数据
            logger.warning("tiktoken 编码加载失败，跳过历史截断")
            return history_messages, token_counts if token_counts is not None else []

    # 若未提供 token_counts，则计算
    if token_counts is None:
        token_counts = []
        for msg in history_messages:
            line = f"{'用户' if msg['role'] == 'user' else '助手'}：{msg['content']}\n"
            token_counts.append(len(enc.encode(line)))
    else:
        # 确保长度一致
        if len(token_counts) != len(history_messages):
            token_counts = []
            for msg in history_messages:
                line = f"{'用户' if msg['role'] == 'user' else '助手'}：{msg['content']}\n"
                token_counts.append(len(enc.encode(line)))

    kept_msgs = []
    kept_counts = []
    total = 0
    for msg, cnt in zip(reversed(history_messages), reversed(token_counts)):
        if total + cnt > max_tokens:
            break
        kept_msgs.append(msg)
        kept_counts.append(cnt)
        total += cnt
    return list(reversed(kept_msgs)), list(reversed(kept_counts))
