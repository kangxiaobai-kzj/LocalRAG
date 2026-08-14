# ============================================
# 项目：LocalRAG · 可定制本地 RAG 智能体
# 开发者：kangxiaobai-kzj
# 开发时间：2026-08-14
# ============================================

# rag/builder.py
# 知识库构建：解析引擎分层（rag/parsers.py）→ 切片策略（rag/chunker.py）→ 向量化入库
# 支持多格式（PDF/TXT/MD/DOCX/XLSX），并优先"增量构建"（仅重建新增/变更/删除的文件）。
import asyncio
import gc
import json
import os
import pickle
import re
import shutil
import time
from datetime import datetime
from typing import List, Tuple

# 必须先于 langchain 导入 config（其顶部会设置 HF_ENDPOINT 镜像）
from config import (
    CHUNK_OVERLAP,
    TEXT_SPLITTER_SEPARATORS,
    BM25_CACHE_FILENAME,
    get_chunk_config,
    get_embedding_model,
)

from langchain_core.documents import Document

from rag.chunker import split_into_chunks
from rag.parsers import is_supported_file, parse_document
from utils.logger import get_logger

logger = get_logger("builder")

MANIFEST_FILENAME = "manifest.json"


def rebuild_vector_store_sync(
    kb_dir, chroma_db_dir, progress_placeholder=None, force_full: bool = False
) -> Tuple[int, List[Tuple[str, str]], dict]:
    """
    知识库构建入口：优先增量（仅重建变更文件，速度快）；force_full=True 或首次构建时全量重建。
    返回 (文本块总数, 失败文件列表, {文件名: (解析引擎, 切片数)})。
    """
    if force_full or not _has_existing_store(chroma_db_dir):
        return _full_rebuild(kb_dir, chroma_db_dir, progress_placeholder)
    return _incremental_build(kb_dir, chroma_db_dir, progress_placeholder)


# ============================================
# 工具函数：文件扫描 / manifest / 缓存
# ============================================
def _scan_kb_files(kb_dir) -> dict:
    """返回 {文件名: {"size": 字节数, "mtime": 修改时间}}，仅统计可解析文件。"""
    files = {}
    for f in os.listdir(kb_dir):
        if is_supported_file(f):
            fp = os.path.join(kb_dir, f)
            st_ = os.stat(fp)
            files[f] = {"size": st_.st_size, "mtime": st_.st_mtime}
    return files


def _load_manifest(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_manifest(path: str, files: dict, embedding_model: str,
                   chunk_size: int, chunk_strategy: str) -> None:
    data = {
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_strategy": chunk_strategy,
        "files": files,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _has_existing_store(chroma_db_dir) -> bool:
    """判断是否具备增量构建条件：向量库存在且有过往构建记录（manifest）。"""
    if not os.path.exists(chroma_db_dir):
        return False
    if not os.path.exists(os.path.join(chroma_db_dir, "chroma.sqlite3")):
        return False
    return os.path.exists(os.path.join(chroma_db_dir, MANIFEST_FILENAME))


def _rebuild_bm25_cache(chroma_db_dir, vectorstore) -> None:
    """用向量库全量内容重建 BM25 缓存，保证知识库页统计与检索加载即时可用。"""
    try:
        from rank_bm25 import BM25Okapi
        import jieba
        all_data = vectorstore.get(include=["documents", "metadatas"])
        docs = all_data.get("documents", [])
        metas = all_data.get("metadatas", [])
        if not docs:
            return
        tokenized = [list(jieba.cut(d)) for d in docs]
        cache = {
            "contents": docs,
            "metadatas": metas,
            "tokenized_corpus": tokenized,
        }
        with open(os.path.join(chroma_db_dir, BM25_CACHE_FILENAME), "wb") as f:
            pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("BM25 缓存已重建（%d 个文档）", len(docs))
    except Exception as e:
        logger.warning("BM25 缓存重建失败（检索首次加载时自动重建）: %s", e)


def _make_text_splitter(chunk_size: int):
    """fixed 策略使用的递归切分器（sentence/heading 走 rag/chunker.py）。"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=CHUNK_OVERLAP,
        separators=TEXT_SPLITTER_SEPARATORS,
    )


def _parse_and_chunk(file_path: str, filename: str, chunk_size: int,
                     chunk_strategy: str, text_splitter, failed_files: List[Tuple[str, str]]):
    """
    解析单个文件并按其策略切片。
    返回 (docs 列表, 切片数)；解析失败时记录到 failed_files 并返回 ([], 0)。
    """
    docs = []
    # 上传时间 = 文件修改时间（作为同主题多版本时的"新旧"兜底依据）
    upload_time = datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
    try:
        # 1. 按扩展名分发解析（PDF 分级调度 / TXT / MD / DOCX / XLSX）
        page_texts, parser_name, char_count = parse_document(file_path)
        if not page_texts:
            failed_files.append((filename, "无法提取任何文本"))
            return docs, 0

        # 2. 按策略切片
        chunk_count = 0
        for page_num, text_list in page_texts.items():
            raw_text = "\n".join(text_list)
            clean_text = re.sub(r"[ \t]+", " ", raw_text).strip()
            if not clean_text:
                continue
            if chunk_strategy == "fixed":
                chunks = text_splitter.split_text(clean_text)
            else:
                chunks = split_into_chunks(clean_text, chunk_strategy, chunk_size)
            for chunk in chunks:
                if not chunk or not chunk.strip():
                    continue
                chunk_count += 1
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": filename,
                            "page": page_num,
                            "parser": parser_name,
                            "chunk_strategy": chunk_strategy,
                            "upload_time": upload_time,
                        },
                    )
                )
        return docs, chunk_count
    except Exception as e:
        failed_files.append((filename, str(e)))
        logger.error("读取失败: %s", e)
        return docs, 0


# ============================================
# 全量重建：备份旧库 → 解析全部文件 → 重建向量库
# ============================================
def _full_rebuild(kb_dir, chroma_db_dir, progress_placeholder=None):
    gc.collect()
    time.sleep(0.5)

    old_dir = chroma_db_dir
    backup_dir = chroma_db_dir + "_old_" + str(int(time.time()))
    if os.path.exists(old_dir):
        try:
            os.rename(old_dir, backup_dir)
        except Exception:
            shutil.rmtree(old_dir, ignore_errors=True)
    os.makedirs(old_dir, exist_ok=True)

    chunk_cfg = get_chunk_config()
    chunk_size = chunk_cfg["chunk_size"]
    chunk_strategy = chunk_cfg["chunk_strategy"]
    embedding_model = get_embedding_model()

    # 重依赖延迟导入，加快应用冷启动
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

    text_splitter = _make_text_splitter(chunk_size)

    all_docs = []
    doc_files = [f for f in os.listdir(kb_dir) if is_supported_file(f)]
    if not doc_files:
        logger.info("知识库目录下没有可解析的文件")
        return 0, [], {}

    failed_files = []
    file_chunks = {}  # filename -> (parser_name, chunk_count)
    total_files = len(doc_files)

    for idx, filename in enumerate(doc_files):
        file_path = os.path.join(kb_dir, filename)
        if progress_placeholder:
            progress_placeholder.progress(idx / total_files,
                                          text=f"🔄 正在解析 [{idx + 1}/{total_files}]：{filename}")

        logger.info("处理文件 [%d/%d]: %s", idx + 1, total_files, filename)
        docs, chunk_count = _parse_and_chunk(file_path, filename, chunk_size,
                                             chunk_strategy, text_splitter, failed_files)
        all_docs.extend(docs)
        parser_name = docs[0].metadata["parser"] if docs else "unknown"
        file_chunks[filename] = (parser_name, chunk_count)
        if progress_placeholder:
            progress_placeholder.progress((idx + 1) / total_files,
                                          text=f"✅ 完成 {filename}（{chunk_count} 个切片）")
        logger.info("处理完成 %s（解析引擎：%s，%d 个切片）", filename, parser_name, chunk_count)

    if progress_placeholder:
        progress_placeholder.progress(1.0, text="🧠 正在向量化（本地 BGE 模型）...")

    if all_docs:
        logger.info("总计生成 %d 个文本块，正在向量化...", len(all_docs))
        embeddings = FastEmbedEmbeddings(model_name=embedding_model)
        store = Chroma.from_documents(
            documents=all_docs,
            embedding=embeddings,
            persist_directory=old_dir,
        )
        _save_manifest(os.path.join(old_dir, MANIFEST_FILENAME),
                       _scan_kb_files(kb_dir), embedding_model, chunk_size, chunk_strategy)
        _rebuild_bm25_cache(old_dir, store)
        logger.info("向量库构建完成，已存入 %s", old_dir)
        return len(all_docs), failed_files, file_chunks
    else:
        logger.error("未生成任何文本块，知识库构建失败")
        return 0, failed_files, file_chunks


# ============================================
# 增量构建：仅对新增 / 变更 / 删除的文件更新索引
# ============================================
def _incremental_build(kb_dir, chroma_db_dir, progress_placeholder=None):
    gc.collect()
    time.sleep(0.3)

    manifest_path = os.path.join(chroma_db_dir, MANIFEST_FILENAME)
    old_manifest = _load_manifest(manifest_path)
    old_files = old_manifest.get("files", {})

    chunk_cfg = get_chunk_config()
    chunk_size = chunk_cfg["chunk_size"]
    chunk_strategy = chunk_cfg["chunk_strategy"]
    embedding_model = get_embedding_model()

    files_now = _scan_kb_files(kb_dir)
    if not files_now:
        # 全部文件被删除 → 清空向量库
        shutil.rmtree(chroma_db_dir, ignore_errors=True)
        return 0, [], {}

    # 向量模型 / 切片参数变更 → 旧索引与新配置不兼容，降级为全量重建
    if (old_manifest.get("embedding_model") != embedding_model
            or old_manifest.get("chunk_size") != chunk_size
            or old_manifest.get("chunk_strategy") != chunk_strategy):
        logger.info("检测到向量模型或切片参数变更，转为全量重建")
        return _full_rebuild(kb_dir, chroma_db_dir, progress_placeholder)

    added = {f for f in files_now if f not in old_files}
    changed = {f for f in files_now if f in old_files
               and (old_files[f].get("size") != files_now[f]["size"]
                    or old_files[f].get("mtime") != files_now[f]["mtime"])}
    removed = {f for f in old_files if f not in files_now}

    if not (added or changed or removed):
        logger.info("文件无变化，跳过重建")
        from langchain_community.vectorstores import Chroma
        from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
        store = Chroma(persist_directory=chroma_db_dir,
                       embedding_function=FastEmbedEmbeddings(model_name=embedding_model))
        count = len(store.get(include=["documents"]).get("documents", []))
        return count, [], {}

    logger.info("增量更新：新增 %d · 变更 %d · 删除 %d", len(added), len(changed), len(removed))

    # 重依赖延迟导入
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

    text_splitter = _make_text_splitter(chunk_size)
    store = Chroma(persist_directory=chroma_db_dir,
                   embedding_function=FastEmbedEmbeddings(model_name=embedding_model))

    # 1. 删除被移除 / 变更文件的旧向量
    for src in sorted(removed | changed):
        try:
            store.delete(where={"source": src})
            logger.info("已移除 %s 的旧索引", src)
        except Exception as e:
            logger.warning("删除 %s 旧向量失败: %s", src, e)

    # 2. 解析并写入新增 / 变更文件
    failed_files = []
    file_chunks = {}
    targets = sorted(added | changed)
    total_files = len(targets)
    all_docs = []

    for idx, filename in enumerate(targets):
        file_path = os.path.join(kb_dir, filename)
        if progress_placeholder:
            progress_placeholder.progress(idx / total_files,
                                          text=f"🔄 正在解析 [{idx + 1}/{total_files}]：{filename}")
        logger.info("处理文件 [%d/%d]: %s", idx + 1, total_files, filename)
        docs, chunk_count = _parse_and_chunk(file_path, filename, chunk_size,
                                             chunk_strategy, text_splitter, failed_files)
        all_docs.extend(docs)
        parser_name = docs[0].metadata["parser"] if docs else "unknown"
        file_chunks[filename] = (parser_name, chunk_count)
        if progress_placeholder:
            progress_placeholder.progress((idx + 1) / total_files,
                                          text=f"✅ 完成 {filename}（{chunk_count} 个切片）")

    if all_docs:
        if progress_placeholder:
            progress_placeholder.progress(1.0, text="🧠 正在向量化变更文件...")
        store.add_documents(all_docs)
        logger.info("已写入 %d 个新文本块", len(all_docs))

    # 3. 保存 manifest + 重建 BM25 缓存
    _save_manifest(manifest_path, files_now, embedding_model, chunk_size, chunk_strategy)
    _rebuild_bm25_cache(chroma_db_dir, store)

    total = len(store.get(include=["documents"]).get("documents", []))
    return total, failed_files, file_chunks


async def rebuild_vector_store_async(kb_dir, chroma_db_dir):
    return await asyncio.to_thread(rebuild_vector_store_sync, kb_dir, chroma_db_dir)


def get_kb_stats(chroma_db_dir) -> dict:
    """从 BM25 缓存读取每份文档的切片数（不加载模型）。返回 {source: count}。"""
    stats = {}
    cache_path = os.path.join(chroma_db_dir, BM25_CACHE_FILENAME)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            for meta in data.get("metadatas", []):
                src = meta.get("source", "未知")
                stats[src] = stats.get(src, 0) + 1
        except Exception as e:
            logger.warning("读取切片统计失败: %s", e)
    return stats
