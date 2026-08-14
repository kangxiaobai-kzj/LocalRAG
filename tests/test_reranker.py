# ============================================
# 项目：LocalRAG · 可定制本地 RAG 智能体
# 开发者：kangxiaobai-kzj
# 开发时间：2026-08-14
# ============================================

# 测试：重排序封装
from rag.reranker import rerank_documents


class FakeDoc:
    def __init__(self, content):
        self.page_content = content
        self.metadata = {}


class FakeReranker:
    def score(self, pairs):
        # 分数与文档顺序相反：doc0 最低、doc4 最高
        return [float(i) for i in range(len(pairs))]


def test_rerank_sorts_by_score():
    docs = [FakeDoc(f"内容{i}") for i in range(5)]
    result = rerank_documents("问题", docs, FakeReranker(), top_k=3, min_docs=3)
    assert [d.page_content for d in result] == ["内容4", "内容3", "内容2"]


def test_rerank_skipped_when_reranker_none():
    docs = [FakeDoc("a"), FakeDoc("b")]
    result = rerank_documents("问题", docs, None, top_k=3, min_docs=3)
    assert result == docs


def test_rerank_truncates_when_below_min_docs():
    docs = [FakeDoc(f"内容{i}") for i in range(2)]
    result = rerank_documents("问题", docs, FakeReranker(), top_k=1, min_docs=3)
    assert len(result) == 1


def test_rerank_degrades_on_error():
    class BrokenReranker:
        def score(self, pairs):
            raise RuntimeError("模型异常")

    docs = [FakeDoc(f"内容{i}") for i in range(5)]
    result = rerank_documents("问题", docs, BrokenReranker(), top_k=2, min_docs=3)
    assert len(result) == 2


def test_rerank_min_score_filters_low_scores():
    """min_score 生效时，低于阈值的片段被过滤（分数 0..4，阈值 2.5 → 仅保留 3、4）。"""
    docs = [FakeDoc(f"内容{i}") for i in range(5)]
    result = rerank_documents("问题", docs, FakeReranker(), top_k=5, min_docs=3, min_score=2.5)
    assert [d.page_content for d in result] == ["内容4", "内容3"]


def test_rerank_min_score_zero_keeps_all():
    """min_score=0（默认）表示关闭过滤，全部保留。"""
    docs = [FakeDoc(f"内容{i}") for i in range(5)]
    result = rerank_documents("问题", docs, FakeReranker(), top_k=5, min_docs=3, min_score=0.0)
    assert len(result) == 5


def test_rerank_min_score_high_threshold_returns_empty():
    """阈值高于所有分数时返回空列表（低质量内容全部过滤）。"""
    docs = [FakeDoc(f"内容{i}") for i in range(5)]
    result = rerank_documents("问题", docs, FakeReranker(), top_k=5, min_docs=3, min_score=99.0)
    assert result == []
