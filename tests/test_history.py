# ============================================
# 项目：LocalRAG · 可定制本地 RAG 智能体
# 开发者：kangxiaobai-kzj
# 开发时间：2026-08-14
# ============================================

# 测试：历史对话截断
from core.history import truncate_history


def test_short_history_fully_kept():
    msgs = [{"role": "user", "content": f"问题{i}"} for i in range(5)]
    kept, counts = truncate_history(msgs, max_tokens=500)
    assert kept == msgs
    assert len(counts) == len(msgs)


def test_long_history_truncated_to_recent():
    msgs = [{"role": "user", "content": "x" * 200} for _ in range(10)]
    kept, _ = truncate_history(msgs, max_tokens=100)
    assert len(kept) < len(msgs)


def test_token_counts_provided_are_used():
    msgs = [{"role": "user", "content": "a" * 50} for _ in range(4)]
    counts = [1, 1, 1, 1]
    kept, kept_counts = truncate_history(msgs, max_tokens=2, token_counts=counts)
    # 最多保留 2 条
    assert 1 <= len(kept) <= 2
    assert len(kept_counts) == len(kept)
