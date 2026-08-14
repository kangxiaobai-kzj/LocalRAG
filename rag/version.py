# ============================================
# 项目：LocalRAG · 可定制本地 RAG 智能体
# 开发者：kangxiaobai-kzj
# 开发时间：2026-08-14
# ============================================

# rag/version.py
# 文件名版本解析：识别新旧版本，供检索结果"同主题多版本时优先最新版"使用。
# 支持模式：年份（GB/T 1234-2018 / 2024版）、v1.2 / V2、第X版、修订版等。
import re

# 常见版本标识的正则（按优先级匹配）
_PATTERNS = [
    # 年份：1990-2099，前后不能紧跟数字（避免把 20260703 这类日期截断成 2026）
    ("year", re.compile(r"(?<!\d)(19|20)\d{2}(?!\d)")),
    # 语义版本：v1.2 / V2.0 / v3
    ("semver", re.compile(r"[vV]\d+(?:\.\d+)*")),
    # 第X版
    ("edition", re.compile(r"第\s*(\d+)\s*版")),
    # XXXX版
    ("year_version", re.compile(r"(\d{4})\s*版")),
    # 修订版 / 修订稿 / Rev
    ("revision", re.compile(r"(修订[稿版次]|Rev\.?\s*\d*)", re.IGNORECASE)),
]


def _sort_key(pattern_type: str, match) -> float:
    """把不同模式的版本号映射为可比数值（越大越新）。"""
    if pattern_type == "year":
        return float(match.group(0))
    if pattern_type == "semver":
        parts = match.group(0)[1:].split(".")
        return sum(int(p) * (1000 ** (2 - i)) for i, p in enumerate(parts[:3]))
    if pattern_type == "edition":
        return float(match.group(1))
    if pattern_type == "year_version":
        return float(match.group(1))
    if pattern_type == "revision":
        text = match.group(0).lower()
        if "rev" in text:
            digits = re.search(r"rev\.?\s*(\d+)", text)
            if digits:
                return 0.5 + float(digits.group(1))
        return 0.5  # 修订稿等：略高于"无版本"，低于任何数字版本
    return 0.0


def parse_version(filename: str):
    """
    从文件名提取版本信息。
    返回 (topic, version_label, version_key)：
    - topic：去掉版本标识后的主题键（同主题多文件用于分组）
    - version_label：展示用字符串（如 "2018"、"v2.0"、"第3版"、"修订版"），无则 ""
    - version_key：数值比较键，越大越新；无版本时为 0
    """
    if not filename:
        return "", "", 0.0
    base = filename.rsplit(".", 1)[0] if "." in filename else filename

    topic = base
    label = ""
    key = 0.0
    for ptype, pattern in _PATTERNS:
        m = pattern.search(topic)
        if m:
            k = _sort_key(ptype, m)
            if k > key:
                key = k
                label = m.group(0) if ptype in ("year", "semver") else m.group(0)
            topic = topic.replace(m.group(0), "")

    # 归一化主题键：去掉空白与常见分隔符，便于跨文件匹配
    topic_norm = re.sub(r"[\s_\-—·()（）\[\]]+", "", topic).strip()
    return (topic_norm or base), (label or ""), key
