# ============================================
# 项目：LocalRAG · 可定制本地 RAG 智能体
# 开发者：kangxiaobai-kzj
# 开发时间：2026-08-14
# ============================================

# 测试：切片策略
from rag.chunker import split_into_chunks


def test_sentence_chunks_reassemble():
    # 文本约 520 字符 > chunk_size=200，必然拆成多块
    text = ("这是第一句话。这是第二句话！这是第三句话？这是第四句话；" * 20)
    chunks = split_into_chunks(text, strategy="sentence", chunk_size=200)
    assert len(chunks) > 1
    # 空格归一后应能拼回原文
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")


def test_sentence_never_cuts_mid_sentence():
    text = "句A。" + "句B。" * 60
    chunks = split_into_chunks(text, strategy="sentence", chunk_size=50)
    assert chunks
    for c in chunks:
        if c.strip():
            assert c.rstrip().endswith("。")


def test_sentence_respects_chunk_size():
    text = "这是第一句话。" * 100
    chunks = split_into_chunks(text, strategy="sentence", chunk_size=100)
    for c in chunks:
        assert len(c) <= 110  # 允许少量余量


def test_heading_keeps_heading_with_content():
    text = "一、产品概述\n概述内容甲。\n二、技术架构\n架构内容乙。"
    chunks = split_into_chunks(text, strategy="heading", chunk_size=500)
    assert any(c.startswith("一、产品概述") for c in chunks)
    assert any(c.startswith("二、技术架构") for c in chunks)


def test_empty_text():
    assert split_into_chunks("", strategy="sentence") == []
    assert split_into_chunks("   ", strategy="sentence") == []
