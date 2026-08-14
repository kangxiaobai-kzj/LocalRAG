# ============================================
# 项目：LocalRAG · 可定制本地 RAG 智能体
# 开发者：kangxiaobai-kzj
# 开发时间：2026-08-14
# ============================================

# 测试：检索调试（search_debug）与 RRF 分数透出
from rag.retriever import rrf_fuse_scored
from rag.search_debug import search_debug


class FakeDoc:
    def __init__(self, content, source="a.pdf", page=1):
        self.page_content = content
        self.metadata = {"source": source, "page": page}


class FakeRetriever:
    def invoke_with_scores(self, question, source=None, latest_only=False):
        # 4 条 > RERANK_MIN_DOCS(3)，确保重排分支可达
        return [
            (FakeDoc("内容甲", "a.pdf", 1), 0.5),
            (FakeDoc("内容乙", "b.pdf", 2), 0.3),
            (FakeDoc("内容丙", "c.pdf", 3), 0.2),
            (FakeDoc("内容丁", "d.pdf", 4), 0.1),
        ]


class FakeReranker:
    def score(self, pairs):
        # 第 4 对最相关
        return [1.0, 2.0, 3.0, 4.0]


def test_rrf_fuse_scored_orders_and_scores():
    v = [FakeDoc("a"), FakeDoc("b")]
    b = [FakeDoc("b"), FakeDoc("c")]
    ranked = rrf_fuse_scored(v, b)
    names = [d.page_content for d, _ in ranked]
    assert names == ["b", "a", "c"]  # b 两路命中位次最高
    assert all(isinstance(s, float) for _, s in ranked)


def test_search_debug_without_rerank():
    hits = search_debug("问题", FakeRetriever(), None, top_k=2, use_rerank=False)
    assert len(hits) == 2
    assert hits[0].source == "a.pdf"
    assert hits[0].page == "1"
    assert hits[0].rerank_score is None
    assert hits[0].score > 0


def test_search_debug_with_rerank():
    hits = search_debug("问题", FakeRetriever(), FakeReranker(), top_k=2, use_rerank=True)
    # 重排后最高分（内容丁，4.0）排第一
    assert hits[0].content == "内容丁"
    assert hits[0].rerank_score == 4.0
