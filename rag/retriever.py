# ============================================
# 项目：LocalRAG · 可定制本地 RAG 智能体
# 开发者：kangxiaobai-kzj
# 开发时间：2026-08-14
# ============================================

# rag/retriever.py
# 向量库加载与混合检索器（向量 + BM25），RRF 融合 + 结果缓存 + 来源过滤
import os
import pickle
from collections import OrderedDict
from typing import List, Optional

# 必须先于 langchain 导入 config（其顶部会设置 HF_ENDPOINT 镜像）
from config import (
    BM25_CACHE_FILENAME,
    RETRIEVER_K,
    RRF_K,
    RETRIEVER_CACHE_SIZE,
    get_embedding_model,
)

from langchain_core.documents import Document

from rag.version import parse_version
from utils.logger import get_logger

logger = get_logger("retriever")

# BM25 依赖（缺失时混合检索自动降级为纯向量）
try:
    from rank_bm25 import BM25Okapi
    import jieba
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("rank_bm25 或 jieba 未安装，混合检索降级")


def rrf_fuse(vector_docs: List, bm25_docs: List, k: int = RRF_K) -> List:
    """
    Reciprocal Rank Fusion（倒数排名融合）：
    多路召回按"位次"加权融合，文档在每路的得分 = 1/(k + rank)，汇总后降序。
    相比"按位交叉去重"，RRF 对检索深度更鲁棒，且天然去重。
    """
    scores = {}
    doc_map = {}
    for doc in vector_docs:
        doc_map.setdefault(doc.page_content, doc)
    for doc in bm25_docs:
        doc_map.setdefault(doc.page_content, doc)

    for rank, doc in enumerate(vector_docs, start=1):
        scores[doc.page_content] = scores.get(doc.page_content, 0.0) + 1.0 / (k + rank)
    for rank, doc in enumerate(bm25_docs, start=1):
        scores[doc.page_content] = scores.get(doc.page_content, 0.0) + 1.0 / (k + rank)

    return sorted(doc_map.values(), key=lambda d: scores.get(d.page_content, 0.0), reverse=True)


def rrf_fuse_scored(vector_docs: List, bm25_docs: List, k: int = RRF_K) -> List:
    """RRF 融合并返回 [(doc, score)]，按分数降序，doc 天然去重（供调试面板展示分数）。"""
    scores = {}
    doc_map = {}
    for doc in vector_docs:
        doc_map.setdefault(doc.page_content, doc)
    for doc in bm25_docs:
        doc_map.setdefault(doc.page_content, doc)

    for rank, doc in enumerate(vector_docs, start=1):
        scores[doc.page_content] = scores.get(doc.page_content, 0.0) + 1.0 / (k + rank)
    for rank, doc in enumerate(bm25_docs, start=1):
        scores[doc.page_content] = scores.get(doc.page_content, 0.0) + 1.0 / (k + rank)

    ranked = sorted(doc_map.values(), key=lambda d: scores.get(d.page_content, 0.0), reverse=True)
    return [(d, scores.get(d.page_content, 0.0)) for d in ranked]


def load_retriever_and_reranker(chroma_db_dir):
    """加载向量库和重排序模型；重排模型加载失败时降级为仅混合检索。
    重依赖（chromadb / fastembed / sentence-transformers）延迟到本函数内导入，
    以加快应用冷启动（首屏不再加载 torch / transformers）。"""
    if not os.path.exists(chroma_db_dir):
        return None, None
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    # 防御性加载：索引目录存在但损坏/不完整（如异常删除残留）时，优雅降级为"无知识库"
    try:
        embeddings = FastEmbedEmbeddings(model_name=get_embedding_model())
        vectorstore = Chroma(
            persist_directory=chroma_db_dir,
            embedding_function=embeddings,
        )
    except Exception as e:
        logger.warning("向量库加载失败（索引可能损坏或为空），按空知识库处理: %s", e)
        return None, None
    hybrid_retriever = HybridRetriever(vectorstore, k=RETRIEVER_K)
    try:
        encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
        return hybrid_retriever, encoder
    except Exception as e:
        logger.warning("Rerank 模型加载失败，降级为普通检索（仅使用混合检索）: %s", e)
        return hybrid_retriever, None


class HybridRetriever:
    """向量检索 + BM25 关键词检索的混合检索器（RRF 融合 + LRU 缓存 + 来源过滤）。"""

    def __init__(self, vectorstore, k=20):
        self.vectorstore = vectorstore
        self.k = k
        self.base_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        self.cache_path = os.path.join(vectorstore._persist_directory, BM25_CACHE_FILENAME)
        self.docs = []
        self.bm25 = None
        self._cache = OrderedDict()  # (query, source) → 检索结果 的 LRU 缓存
        self._init_bm25()

    # ================= 结果缓存 =================
    def _cache_get(self, key):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def _cache_put(self, key, docs):
        self._cache[key] = docs
        self._cache.move_to_end(key)
        if len(self._cache) > RETRIEVER_CACHE_SIZE:
            self._cache.popitem(last=False)

    # ================= BM25 索引持久化 =================
    def _init_bm25(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "rb") as f:
                    cached_data = pickle.load(f)
                content_list = cached_data["contents"]
                meta_list = cached_data["metadatas"]
                tokenized_corpus = cached_data["tokenized_corpus"]
                self.docs = [Document(page_content=c, metadata=m) for c, m in zip(content_list, meta_list)]
                all_data = self.vectorstore.get(include=["documents"])
                current_count = len(all_data.get("documents", []))
                if current_count == len(self.docs) and BM25_AVAILABLE:
                    self.bm25 = BM25Okapi(tokenized_corpus)
                    logger.info("BM25 缓存加载成功（%d 个文档）", len(self.docs))
                    return
                else:
                    logger.warning("缓存文档数不一致，重新构建")
            except Exception as e:
                logger.warning("缓存加载失败: %s", e)

        try:
            all_data = self.vectorstore.get(include=["documents", "metadatas"])
            docs = all_data.get("documents", [])
            metas = all_data.get("metadatas", [])
            if docs and metas and BM25_AVAILABLE:
                self.docs = [Document(page_content=d, metadata=m) for d, m in zip(docs, metas)]
                tokenized_corpus = [list(jieba.cut(doc.page_content)) for doc in self.docs]
                self.bm25 = BM25Okapi(tokenized_corpus)
                logger.info("BM25 构建成功（%d 个文档）", len(self.docs))
                self._save_cache(tokenized_corpus)
            else:
                logger.warning("向量库为空或 BM25 不可用")
        except Exception as e:
            logger.warning("BM25 初始化失败: %s", e)

    def _save_cache(self, tokenized_corpus):
        try:
            cache_data = {
                "contents": [doc.page_content for doc in self.docs],
                "metadatas": [doc.metadata for doc in self.docs],
                "tokenized_corpus": tokenized_corpus,
            }
            with open(self.cache_path, "wb") as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("BM25 缓存已保存至 %s", self.cache_path)
        except Exception as e:
            logger.warning("缓存保存失败: %s", e)

    # ================= 混合检索 =================
    def invoke(self, query: str, source: Optional[str] = None, latest_only: bool = False) -> List:
        """
        混合检索：向量 + BM25，RRF 融合后取 Top-k。
        source 非空时按来源文件名过滤（metadata.source 精确匹配）。
        latest_only=True 时同一主题（文件名去版本标识）只保留最新版本文件的切片。
        相同 (query, source, latest_only) 命中缓存直接返回。
        """
        cache_key = (query, source, latest_only)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        # 1. 向量检索（可按来源过滤）
        if source:
            filtered_retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.k, "filter": {"source": source}}
            )
            vector_docs = filtered_retriever.invoke(query)
        else:
            vector_docs = self.base_retriever.invoke(query)

        # 2. BM25 检索（可按来源过滤）
        if self.bm25 and BM25_AVAILABLE:
            tokenized_query = list(jieba.cut(query))
            try:
                if source:
                    pool = [d for d in self.docs if d.metadata.get("source") == source]
                else:
                    pool = self.docs
                bm25_docs = self.bm25.get_top_n(tokenized_query, pool, n=self.k)
            except Exception:
                bm25_docs = []
        else:
            bm25_docs = []

        # 3. 融合：BM25 不可用时退化为纯向量结果
        if not self.bm25 or not BM25_AVAILABLE or not bm25_docs:
            docs = vector_docs
        else:
            docs = rrf_fuse(vector_docs, bm25_docs)

        docs = self._reorder_latest_first(docs)
        if latest_only:
            docs = self._filter_latest_only(docs)
        docs = docs[: self.k]
        self._cache_put(cache_key, docs)
        return docs

    def list_sources(self) -> List[str]:
        """知识库中包含的文档来源列表（用于过滤提示）。"""
        return sorted({d.metadata.get("source", "") for d in self.docs if d.metadata.get("source")})

    def _reorder_latest_first(self, docs: List) -> List:
        """
        同主题多版本优先：标准/规范更新后，知识库可能同时存在新旧版本。
        将同主题（文件名去掉版本标识后归一化）的最新版本排在前，旧版本后置，
        使 Top-K 截断时优先命中最新版。无版本号时以上传时间(upload_time)兜底。
        """
        if not docs:
            return docs
        groups = {}
        for idx, doc in enumerate(docs):
            src = doc.metadata.get("source", "")
            topic, _, vkey = parse_version(src)
            ut = doc.metadata.get("upload_time", "")
            groups.setdefault(topic, []).append((idx, vkey, ut))
        latest_idx = set()
        for _, items in groups.items():
            # 组内：版本号新者优先；版本号相同/缺失时，上传时间新者优先
            items.sort(key=lambda x: (x[1], x[2]), reverse=True)
            if items:
                latest_idx.add(items[0][0])
        # 稳定排序：最新版本主题的文档在前（保持原相对顺序），旧版本后置
        ranked = sorted(range(len(docs)), key=lambda i: (0 if i in latest_idx else 1, i))
        return [docs[i] for i in ranked]

    def _filter_latest_only(self, docs: List) -> List:
        """仅保留每个主题（文件名去版本标识归一化）最新版本文件的全部切片。
        版本号（年份 / vX.Y / 第X版 / 修订）优先，无法识别版本时以上传时间(upload_time)兜底。"""
        if not docs:
            return docs
        groups = {}  # topic -> {source: {"vkey", "ut", "idx": [..]}}
        for idx, doc in enumerate(docs):
            src = doc.metadata.get("source", "")
            topic, _, vkey = parse_version(src)
            ut = doc.metadata.get("upload_time", "")
            g = groups.setdefault(topic, {})
            g.setdefault(src, {"vkey": vkey, "ut": ut, "idx": []})["idx"].append(idx)
        keep_idx = set()
        for _, files in groups.items():
            if not files:
                continue
            best = max(files.values(), key=lambda f: (f["vkey"], f["ut"]))
            keep_idx.update(best["idx"])
        return [d for i, d in enumerate(docs) if i in keep_idx]

    def invoke_with_scores(self, query: str, source: Optional[str] = None,
                           latest_only: bool = False) -> List:
        """返回 [(doc, rrf_score)]，按融合分数降序（供检索测试面板展示分数）。"""
        cache_key = ("debug", query, source, latest_only)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        # 向量路：similarity_search_with_score 返回 [(doc, distance)]，distance 越小越相关，顺序即相似度降序
        kwargs = {"k": self.k}
        if source:
            kwargs["filter"] = {"source": source}
        vs = self.vectorstore.similarity_search_with_score(query, **kwargs)
        vector_docs = [doc for doc, _ in vs]

        # BM25 路
        bm25_docs = []
        if self.bm25 and BM25_AVAILABLE:
            pool = [d for d in self.docs if d.metadata.get("source") == source] if source else self.docs
            try:
                bm25_docs = self.bm25.get_top_n(list(jieba.cut(query)), pool, n=self.k)
            except Exception:
                bm25_docs = []

        fused = rrf_fuse_scored(vector_docs, bm25_docs) if bm25_docs else [(d, 0.0) for d in vector_docs]
        score_map = {d.page_content: s for d, s in fused}
        docs = self._reorder_latest_first([d for d, _ in fused])
        if latest_only:
            docs = self._filter_latest_only(docs)
        docs = docs[: self.k]
        ranked = [(d, score_map[d.page_content]) for d in docs]
        self._cache_put(cache_key, ranked)
        return ranked
