# ============================================
# 项目：LocalRAG · 可定制本地 RAG 智能体
# 开发者：kangxiaobai-kzj
# 开发时间：2026-08-14
# ============================================

# rag/chunker.py
# 切片策略：sentence=句级感知（推荐）/ heading=标题感知 / fixed=固定长度（回退）
# 目标：避免固定长度硬切把完整语义（如一个安全措施、一个方案要点）拦腰截断。
import re
from typing import List

from config import CHUNK_SIZE, CHUNK_STRATEGY

_SENT_END_RE = re.compile(r"(?<=[。！？；!?])")

# 标题行识别（中文编号 / 第X章 / 括号编号），用于 heading 策略
_HEADING_RE = re.compile(
    r"^(第[一二三四五六七八九十百零\d]+[章节篇部分]|"
    r"[\d一二三四五六七八九十]{1,3}[、.．]\s*\S+|"
    r"\([一二三四五六七八九十\d]{1,2}\)\s*\S+)\s*$"
)

# heading 策略下，标题行最长字符数（防止把普通短行误判为标题）
_HEADING_MAX_LEN = 40


def split_into_chunks(text: str, strategy: str = None, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """按策略对单页文本切片。未知策略回退为句级感知。"""
    strategy = (strategy or CHUNK_STRATEGY).lower()
    if strategy == "sentence":
        return _sentence_chunks(text, chunk_size)
    if strategy == "heading":
        return _heading_chunks(text, chunk_size)
    return _fixed_chunks(text, chunk_size)


# ================= 句级感知（推荐） =================
def _sentence_chunks(text: str, chunk_size: int) -> List[str]:
    """以 。！？； 等结束符为界切句，再按 chunk_size 贪心合并；句子永不截断。"""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    units = [u.strip() for u in _SENT_END_RE.split(text) if u.strip()]

    chunks: List[str] = []
    current = ""
    for u in units:
        if len(u) > chunk_size:
            # 单句超长（罕见），先落盘当前句组，再按更细粒度兜底切分
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_long(u, chunk_size))
            continue
        if current and len(current) + len(u) + 1 > chunk_size:
            chunks.append(current)
            current = u
        else:
            current = f"{current} {u}".strip()
    if current:
        chunks.append(current)
    return chunks


# ================= 标题感知 =================
def _heading_chunks(text: str, chunk_size: int) -> List[str]:
    """识别标题行，标题与紧随其后的内容组成一个组；组内再按句级合并。"""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    groups: List[List[str]] = []
    for ln in lines:
        if len(ln) <= _HEADING_MAX_LEN and _HEADING_RE.match(ln):
            groups.append([ln])          # 新标题组
        elif groups:
            groups[-1].append(ln)        # 归入最近标题组
        else:
            groups.append([ln])          # 无标题前缀的正文，独立成组

    chunks: List[str] = []
    for group in groups:
        content = " ".join(group)
        chunks.extend(_sentence_chunks(content, chunk_size))
    return chunks


# ================= 固定长度（回退） =================
def _fixed_chunks(text: str, chunk_size: int) -> List[str]:
    """固定长度回退：按句级合并近似切分（builder 的 'fixed' 策略仍走 RecursiveCharacterTextSplitter）。"""
    return _sentence_chunks(text, chunk_size)


def _split_long(unit: str, chunk_size: int) -> List[str]:
    """超长单句的兜底切分：优先按中文逗号/英文逗号/空格合并。"""
    parts = unit
    for sep in ["，", ",", " "]:
        if sep in unit:
            parts = unit.split(sep)
            break
    if isinstance(parts, str):
        return [parts]

    chunks: List[str] = []
    current = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if current and len(current) + len(p) + 1 > chunk_size:
            chunks.append(current)
            current = p
        else:
            current = f"{current} {p}".strip()
    if current:
        chunks.append(current)
    return chunks
