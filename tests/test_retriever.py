# ============================================
# 项目：LocalRAG · 可定制本地 RAG 智能体
# 开发者：kangxiaobai-kzj
# 开发时间：2026-08-14
# ============================================

# 测试：RRF 融合
from rag.retriever import rrf_fuse


class FakeDoc:
    def __init__(self, content):
        self.page_content = content
        self.metadata = {}


def test_rrf_ranks_by_dual_rank():
    v = [FakeDoc("a"), FakeDoc("b"), FakeDoc("c")]
    b = [FakeDoc("c"), FakeDoc("a"), FakeDoc("d")]
    result = [d.page_content for d in rrf_fuse(v, b)]
    # a:1/61+1/62  c:1/63+1/61  b:1/62  d:1/63  → a > c > b > d
    assert result == ["a", "c", "b", "d"]


def test_rrf_dedupes_duplicates():
    v = [FakeDoc("a"), FakeDoc("a"), FakeDoc("b")]
    b = [FakeDoc("b")]
    result = [d.page_content for d in rrf_fuse(v, b)]
    assert len(set(result)) == len(result)


def test_rrf_empty_second_list():
    v = [FakeDoc("a"), FakeDoc("b")]
    result = [d.page_content for d in rrf_fuse(v, [])]
    assert result == ["a", "b"]
