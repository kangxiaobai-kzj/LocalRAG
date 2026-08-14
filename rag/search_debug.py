# ============================================
# 项目：LocalRAG · 可定制本地 RAG 智能体
# 开发者：kangxiaobai-kzj
# 开发时间：2026-08-14
# ============================================

# rag/search_debug.py
# 检索调试：返回带分数的检索结果，供 UI 检索测试面板展示（对标 RAGFlow 检索透明化）。
from dataclasses import dataclass
from typing import List, Optional

from config import RERANK_TOP_K, RERANK_MIN_DOCS
from utils.logger import get_logger

logger = get_logger("search_debug")


@dataclass
class SearchHit:
    rank: int
    content: str
    source: str
    page: str
    score: float                     # 融合分数（RRF 融合值；重排后仍保留）
    rerank_score: Optional[float] = None   # 交叉编码器分数（未重排时为 None）


def search_debug(question: str, retriever, reranker, top_k: int = RERANK_TOP_K,
                 use_rerank: bool = True, source: Optional[str] = None,
                 latest_only: bool = False, min_score: float = 0.0, on_stage=None) -> List[SearchHit]:
    """
    执行检索并返回带分数的命中列表。
    - retriever 需提供 invoke_with_scores(query, source) → [(doc, rrf_score)]
    - use_rerank=True 且 reranker 可用时，用交叉编码器重新打分排序
    - latest_only=True 时同主题只保留最新版本文件的命中
    - min_score > 0 时，仅保留重排分 >= min_score 的命中（低质量片段被过滤）
    - on_stage(stage, **kw) 可选回调，用于 UI 展示检索阶段（retrieved / reranking / reranked）
    """
    if on_stage:
        on_stage("retrieving")
    ranked = retriever.invoke_with_scores(question, source=source, latest_only=latest_only)
    if on_stage:
        on_stage("retrieved", count=len(ranked))
    if use_rerank and reranker is not None and len(ranked) > RERANK_MIN_DOCS:
        if on_stage:
            on_stage("reranking", count=len(ranked))
        pairs = [[question, doc.page_content] for doc, _ in ranked]
        try:
            if hasattr(reranker, "score"):
                scores = reranker.score(pairs)
            elif hasattr(reranker, "predict"):
                scores = reranker.predict(pairs)
            else:
                scores = None
            if scores is not None:
                scored = sorted(zip([d for d, _ in ranked], scores), key=lambda x: x[1], reverse=True)
                if min_score > 0:
                    scored = [(doc, s) for doc, s in scored if s >= min_score]
                # 保留重排前的 RRF 融合分，与重排分一同展示（doc 按 page_content 关联）
                fusion = {d.page_content: s for d, s in ranked}
                if on_stage:
                    on_stage("reranked", count=len(scored[:top_k]))
                return [
                    SearchHit(i + 1, doc.page_content, doc.metadata.get("source", "未知"),
                              str(doc.metadata.get("page", "?")),
                              float(fusion.get(doc.page_content, 0.0)), float(s))
                    for i, (doc, s) in enumerate(scored[:top_k])
                ]
        except Exception as e:
            logger.warning("重排失败，展示原始检索结果: %s", e)
    return _from_ranked(ranked, top_k)


def _from_ranked(ranked, top_k: int) -> List[SearchHit]:
    return [
        SearchHit(i + 1, doc.page_content, doc.metadata.get("source", "未知"),
                  str(doc.metadata.get("page", "?")), float(score), None)
        for i, (doc, score) in enumerate(ranked[:top_k])
    ]
