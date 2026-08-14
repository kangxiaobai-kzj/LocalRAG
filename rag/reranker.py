# ============================================
# 项目：LocalRAG · 可定制本地 RAG 智能体
# 开发者：kangxiaobai-kzj
# 开发时间：2026-08-14
# ============================================

# rag/reranker.py
# 交叉编码器重排序封装：对初步召回文档精排，异常时降级为基础截断
from typing import List

from config import RERANK_TOP_K, RERANK_MIN_DOCS


def rerank_documents(question: str, docs: List, reranker, top_k: int = RERANK_TOP_K,
                     min_docs: int = RERANK_MIN_DOCS, min_score: float = 0.0) -> List:
    """
    对召回文档执行重排序。
    - reranker 为 None 或文档数不足 min_docs 时，直接截断返回前 top_k。
    - min_score > 0 时，仅保留重排分 >= min_score 的片段（低于阈值的低质量片段被过滤）。
    - 重排异常时降级为基础截断（保证系统可用）。
    """
    if reranker is None or len(docs) <= min_docs:
        return docs[:top_k]

    try:
        pairs = [[question, doc.page_content] for doc in docs]
        if hasattr(reranker, "score"):
            scores = reranker.score(pairs)
        elif hasattr(reranker, "predict"):
            scores = reranker.predict(pairs)
        elif hasattr(reranker, "rank"):
            return reranker.rank(question, docs, top_k=top_k)
        else:
            scores = None

        if scores is not None:
            scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            if min_score > 0:
                scored = [(doc, s) for doc, s in scored if s >= min_score]
            return [doc for doc, _ in scored[:top_k]]
        return docs[:top_k]
    except Exception:
        # 重排序异常，降级取前 top_k
        return docs[:top_k]
